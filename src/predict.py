"""
BreastGuard AI — Clinical Prediction Engine & Inference Service
================================================================================
Implements production-grade inference, input validation, SHAP explainability,
batch CSV processing, and individual PDF report generation:
  1. Loads the saved final model (models/final_breastguard_model.pkl).
  2. Dynamically verifies class ordering using `classes_` attribute.
  3. Validates incoming numerical features (finite, non-negative, complete).
  4. Applies the clinical decision threshold (0.37).
  5. Computes local SHAP explanations with graceful fallback if unavailable.
  6. Executes robust batch prediction on uploaded CSV datasets.
  7. Generates publication-ready PDF clinical prediction reports.
================================================================================
"""
from __future__ import annotations

import datetime
import json
import logging
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from config.config import (
    CLASS_LABELS,
    EVALUATION_DIR,
    FIGURES_DIR,
    LABEL_BENIGN,
    LABEL_MALIGNANT,
    MODELS_DIR,
    N_FEATURES,
    PROJECT_NAME,
    PROJECT_ROOT,
    PROJECT_TAGLINE,
    RANDOM_SEED,
    RISK_THRESHOLD_HIGH,
    RISK_THRESHOLD_LOW,
    RISK_THRESHOLD_MODERATE,
    VERSION,
)
from src.explain import (
    LocalExplanation,
    ShapExplainerManager,
    ShapValueContainer,
    explain_single_sample,
)

logger = logging.getLogger(__name__)

PREDICTIONS_DIR: Path = PROJECT_ROOT / "reports" / "predictions"
PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)

CLINICAL_DISCLAIMER: str = (
    "EDUCATIONAL & RESEARCH NOTICE: This is an educational research prototype, "
    "not a clinically validated diagnostic system. It must not be used to diagnose "
    "cancer, make treatment decisions, or replace professional medical evaluation."
)


@dataclass
class PredictionResult:
    """Structured result of a single patient prediction."""
    model_name: str
    predicted_label: str
    predicted_class: int
    prob_malignant: float
    prob_benign: float
    classification_threshold: float
    risk_tier: str
    risk_color: str
    explanation: Any | None = None
    explanation_available: bool = False
    top_10_features: list[tuple[str, float, float]] = field(default_factory=list)  # (name, val, shap)
    positive_drivers: list[tuple[str, float, float]] = field(default_factory=list)
    negative_drivers: list[tuple[str, float, float]] = field(default_factory=list)
    narrative_explanation: list[str] = field(default_factory=list)
    feature_values: dict[str, float] = field(default_factory=dict)
    warning_assessment: Any | None = None
    ood_result: Any | None = None
    disclaimer: str = CLINICAL_DISCLAIMER
    timestamp: str = field(default_factory=lambda: datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))


@dataclass
class BatchPredictionResult:
    """Results of a batch CSV prediction run."""
    total_records: int
    valid_records: int
    invalid_records: int
    predictions_df: pd.DataFrame
    output_path: Path
    invalid_rows_summary: list[dict[str, Any]]
    cancer_detected_count: int = 0
    no_cancer_count: int = 0
    cancer_percentage: float = 0.0
    high_risk_count: int = 0
    borderline_count: int = 0
    column_mapping_used: dict[str, str] = field(default_factory=dict)
    disclaimer: str = CLINICAL_DISCLAIMER


class PredictionEngine:
    """
    High-performance prediction engine wrapping the trained final champion model.
    """
    _instance: PredictionEngine | None = None

    def __init__(
        self,
        model_path: Path | None = None,
        data_path: Path | None = None,
    ):
        self.model_path = model_path or (MODELS_DIR / "final_breastguard_model.pkl")
        self.data_path = data_path or (PROJECT_ROOT / "data" / "processed" / "breast_cancer_cleaned.csv")
        
        self.model_bundle: dict[str, Any] = {}
        self.pipeline: Pipeline | None = None
        self.feature_names: list[str] = []
        self.threshold: float = 0.37
        self.model_name: str = "Support Vector Machine"
        self.benign_idx: int = 0
        self.malignant_idx: int = 1
        self.default_values: dict[str, float] = {}
        self.feature_bounds: dict[str, tuple[float, float]] = {}
        
        self.explainer_manager: ShapExplainerManager | None = None
        self._load_artifacts()

    def _load_artifacts(self) -> None:
        """Load and validate model pipeline, feature specifications, and training medians."""
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Trained final model artifact not found at {self.model_path}!"
            )

        self.model_bundle = joblib.load(self.model_path)
        self.pipeline = self.model_bundle["pipeline"]
        self.feature_names = list(self.model_bundle["feature_names"])
        self.threshold = float(self.model_bundle.get("classification_threshold", 0.37))
        self.model_name = str(self.model_bundle.get("model_name", "Support Vector Machine"))

        # Dynamically inspect classes_ attribute on the classifier step
        model_step = self.pipeline.named_steps.get("model")
        classes = getattr(model_step, "classes_", np.array([0, 1]))
        
        # Verify class indices without assuming fixed order
        classes_list = list(classes)
        if 0 in classes_list and 1 in classes_list:
            self.benign_idx = classes_list.index(0)
            self.malignant_idx = classes_list.index(1)
        else:
            self.benign_idx = 0
            self.malignant_idx = 1

        logger.info(
            "PredictionEngine initialized with model '%s' (Benign idx=%d, Malignant idx=%d, Threshold=%.2f)",
            self.model_name, self.benign_idx, self.malignant_idx, self.threshold,
        )

        # Compute sensible default values (medians) from processed dataset
        if self.data_path.exists():
            df = pd.read_csv(self.data_path)
            for f in self.feature_names:
                if f in df.columns:
                    med = float(df[f].median())
                    p01 = float(df[f].quantile(0.01))
                    p99 = float(df[f].quantile(0.99))
                    self.default_values[f] = round(med, 4)
                    self.feature_bounds[f] = (round(p01, 4), round(p99, 4))
                else:
                    self.default_values[f] = 0.0
                    self.feature_bounds[f] = (0.0, 100.0)
        else:
            for f in self.feature_names:
                self.default_values[f] = 1.0
                self.feature_bounds[f] = (0.0, 100.0)

        # Initialize SHAP explainer
        try:
            self.explainer_manager = ShapExplainerManager(
                model_path=self.model_path,
                n_background=50,
                random_state=RANDOM_SEED,
            )
            self.explainer_manager.load_model()
            if self.data_path.exists():
                df = pd.read_csv(self.data_path)
                X_bg = df[self.feature_names].values
                self.explainer_manager.initialize_explainer(X_bg)
        except Exception as e:
            logger.warning("SHAP Explainer initialization deferred or failed: %s", e)
            self.explainer_manager = None

    def get_feature_names(self) -> list[str]:
        """Return the exact 30 feature names expected by the model."""
        return list(self.feature_names)

    def get_default_values(self) -> dict[str, float]:
        """Return sensible default values (medians) for all 30 features."""
        return dict(self.default_values)

    def _determine_risk_tier(self, prob_malignant: float) -> tuple[str, str]:
        """Classify malignancy probability into clinical risk tiers and hex colors."""
        if prob_malignant >= RISK_THRESHOLD_HIGH:
            return "Critical Risk (>80%)", "#FF4B4B"
        elif prob_malignant >= RISK_THRESHOLD_MODERATE:
            return "High Risk (50-80%)", "#FF7675"
        elif prob_malignant >= RISK_THRESHOLD_LOW:
            return "Moderate Risk (20-50%)", "#FFB347"
        else:
            return "Low Risk (<20%)", "#00C9A7"

    def validate_inputs(
        self,
        feature_values: dict[str, float] | pd.Series | np.ndarray,
    ) -> np.ndarray:
        """
        Validate incoming feature values.
        Ensures all 30 features are present, non-null, finite, and non-negative.
        
        Returns:
            np.ndarray of shape (1, 30) ordered correctly.
        """
        if isinstance(feature_values, dict):
            missing = set(self.feature_names) - set(feature_values.keys())
            if missing:
                raise ValueError(f"Missing required features ({len(missing)}): {list(missing)[:5]}")
            
            raw_vals = []
            for f in self.feature_names:
                v = feature_values[f]
                if v is None:
                    raise ValueError(f"Feature '{f}' cannot be None.")
                try:
                    f_val = float(v)
                except (ValueError, TypeError):
                    raise ValueError(f"Feature '{f}' has non-numeric value: {v}")
                if np.isnan(f_val) or np.isinf(f_val):
                    raise ValueError(f"Feature '{f}' cannot be NaN or infinite.")
                if f_val < 0.0:
                    raise ValueError(f"Feature '{f}' must be non-negative, got {f_val}")
                raw_vals.append(f_val)
            x_arr = np.array(raw_vals, dtype=float).reshape(1, -1)

        elif isinstance(feature_values, pd.Series):
            return self.validate_inputs(feature_values.to_dict())

        elif isinstance(feature_values, np.ndarray):
            x_arr = np.asarray(feature_values, dtype=float)
            if x_arr.ndim == 1:
                x_arr = x_arr.reshape(1, -1)
            if x_arr.shape[1] != N_FEATURES:
                raise ValueError(f"Expected {N_FEATURES} features, got {x_arr.shape[1]}")
            if np.isnan(x_arr).any() or np.isinf(x_arr).any():
                raise ValueError("Input array contains NaN or Infinite values.")
            if (x_arr < 0.0).any():
                raise ValueError("Input array contains negative values.")
        else:
            raise TypeError("feature_values must be a dict, pd.Series, or np.ndarray")

        return x_arr

    def predict_single(
        self,
        feature_values: dict[str, float] | pd.Series | np.ndarray,
        generate_explanation: bool = True,
    ) -> PredictionResult:
        """
        Execute prediction pipeline and local SHAP explainability on a single patient biopsy.
        """
        x_2d = self.validate_inputs(feature_values)
        feat_dict = {self.feature_names[i]: float(x_2d[0, i]) for i in range(N_FEATURES)}

        # Pipeline probability inference
        probs = self.pipeline.predict_proba(x_2d)[0]
        prob_benign = float(probs[self.benign_idx])
        prob_malignant = float(probs[self.malignant_idx])

        # Apply calibrated clinical threshold
        pred_class = int(prob_malignant >= self.threshold)
        pred_label = CLASS_LABELS[pred_class]
        risk_tier, risk_color = self._determine_risk_tier(prob_malignant)

        # Local Explainability
        explanation = None
        explanation_avail = False
        top_10 = []
        pos_drivers = []
        neg_drivers = []
        narratives = []

        if generate_explanation and self.explainer_manager is not None:
            try:
                explanation = explain_single_sample(
                    pipeline=self.pipeline,
                    explainer=self.explainer_manager.explainer,
                    sample=x_2d.ravel(),
                    feature_names=self.feature_names,
                    threshold=self.threshold,
                    nsamples=100,
                )
                explanation_avail = True
                top_10 = [
                    (f, feat_dict[f], float(explanation.shap_values[f]))
                    for f in explanation.top_features[:10]
                ] if hasattr(explanation, "top_features") else [
                    (f, feat_dict[f], float(explanation.shap_values[f]))
                    for f in self.feature_names[:10]
                ]
                pos_drivers = explanation.top_positive_contributions[:5]
                neg_drivers = explanation.top_negative_contributions[:5]
                narratives = explanation.narrative_explanation
            except Exception as e:
                logger.warning("Local SHAP explanation generation failed: %s", e)
                explanation = None
                explanation_avail = False
                narratives = [
                    "Local SHAP explainability is temporarily unavailable for this prediction. "
                    "The model classification and calibrated probabilities remain fully valid."
                ]

        # Transparent safety and OOD warnings
        warning_assessment = None
        ood_result = None
        try:
            from src.prediction_warnings import load_warning_system
            warning_sys = load_warning_system()
            warning_assessment = warning_sys.inspect_prediction(
                feature_values=feat_dict,
                prob_malignant=prob_malignant,
                threshold=self.threshold,
                explanation_available=explanation_avail,
            )
            ood_result = warning_assessment.ood_result
        except Exception as e:
            logger.debug(f"Warning inspection skipped: {e}")

        return PredictionResult(
            model_name=self.model_name,
            predicted_label=pred_label,
            predicted_class=pred_class,
            prob_malignant=round(prob_malignant, 4),
            prob_benign=round(prob_benign, 4),
            classification_threshold=self.threshold,
            risk_tier=risk_tier,
            risk_color=risk_color,
            explanation=explanation,
            explanation_available=explanation_avail,
            top_10_features=top_10,
            positive_drivers=pos_drivers,
            negative_drivers=neg_drivers,
            narrative_explanation=narratives,
            feature_values=feat_dict,
            warning_assessment=warning_assessment,
            ood_result=ood_result,
        )

    def standardize_feature_columns(
        self,
        df: pd.DataFrame,
    ) -> tuple[pd.DataFrame, dict[str, str]]:
        """
        Normalize and map common breast cancer dataset column name variations
        (e.g., radius_mean -> mean_radius, radius_worst -> worst_radius,
        concave points_mean -> mean_concave_points, radius_se -> radius_se)
        to canonical feature names expected by the model.
        """
        df_mapped = df.copy()
        col_mapping: dict[str, str] = {}
        existing_cols = list(df.columns)

        def simplify(name: str) -> str:
            return str(name).strip().lower().replace(" ", "_").replace(".", "_").replace("-", "_")

        norm_to_orig = {simplify(c): c for c in existing_cols}

        for exp in self.feature_names:
            if exp in existing_cols:
                continue
            exp_norm = simplify(exp)
            if exp_norm in norm_to_orig:
                col_mapping[norm_to_orig[exp_norm]] = exp
                continue

            parts = exp.split("_")
            prefix = parts[0]
            candidates = []
            if prefix in ("mean", "worst"):
                attr = "_".join(parts[1:])
                candidates.append(f"{attr}_{prefix}")
                candidates.append(f"{attr}_{prefix}s")
                candidates.append(f"{attr}{prefix}")
            elif exp.endswith("_se"):
                attr = exp[:-3]
                candidates.append(f"{attr}_se")
                candidates.append(f"{attr}_standard_error")
                candidates.append(f"{attr}_std_error")
                candidates.append(f"{attr}_error")
                candidates.append(f"se_{attr}")

            for cand in candidates:
                cand_norm = simplify(cand)
                if cand_norm in norm_to_orig:
                    col_mapping[norm_to_orig[cand_norm]] = exp
                    break

        if col_mapping:
            df_mapped = df_mapped.rename(columns=col_mapping)

        return df_mapped, col_mapping

    def predict_batch(
        self,
        csv_file_or_df: str | Path | pd.DataFrame,
        output_path: Path | None = None,
    ) -> BatchPredictionResult:
        """
        Execute robust batch prediction on a CSV file or DataFrame.
        Validates columns, tracks invalid rows, and never silently discards records.
        """
        if isinstance(csv_file_or_df, (str, Path)):
            df_in = pd.read_csv(csv_file_or_df)
        elif isinstance(csv_file_or_df, pd.DataFrame):
            df_in = csv_file_or_df.copy()
        else:
            raise TypeError("csv_file_or_df must be a file path or pd.DataFrame")

        if output_path is None:
            PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
            output_path = PREDICTIONS_DIR / "batch_prediction_results.csv"

        results_rows = []
        invalid_rows = []
        valid_count = 0

        # Standardize and map column variations (e.g. radius_mean -> mean_radius)
        df_mapped, col_mapping = self.standardize_feature_columns(df_in)

        # Check required columns
        missing_cols = set(self.feature_names) - set(df_mapped.columns)
        if missing_cols:
            raise ValueError(f"Uploaded CSV is missing {len(missing_cols)} required columns: {list(missing_cols)[:5]}")

        for row_idx, row in df_mapped.iterrows():
            record_id = row.get("id", row.get("patient_id", row.get("ID", row_idx + 1)))
            row_dict = {f: row[f] for f in self.feature_names}

            try:
                pred = self.predict_single(row_dict, generate_explanation=False)
                has_cancer = "YES (Cancer Detected)" if pred.predicted_label == "Malignant" else "NO (No Cancer Detected)"
                cancer_status = "CANCER DETECTED" if pred.predicted_label == "Malignant" else "NO CANCER"
                confidence_score = max(pred.prob_malignant, pred.prob_benign)
                is_borderline = (0.30 <= pred.prob_malignant <= 0.45)

                results_rows.append({
                    "record_id": record_id,
                    "has_cancer": has_cancer,
                    "cancer_status": cancer_status,
                    "predicted_class": pred.predicted_label,
                    "cancer_probability": f"{pred.prob_malignant * 100:.1f}%",
                    "confidence": f"{confidence_score * 100:.1f}%",
                    "risk_tier": pred.risk_tier,
                    "is_borderline": is_borderline,
                    "prob_benign": pred.prob_benign,
                    "prob_malignant": pred.prob_malignant,
                    "classification_threshold": pred.classification_threshold,
                    "prediction_status": "SUCCESS",
                })
                valid_count += 1
            except Exception as err:
                error_msg = f"INVALID_ROW: {str(err)}"
                results_rows.append({
                    "record_id": record_id,
                    "has_cancer": "ERROR",
                    "cancer_status": "ERROR",
                    "predicted_class": "ERROR",
                    "cancer_probability": "N/A",
                    "confidence": "N/A",
                    "risk_tier": "UNKNOWN",
                    "is_borderline": False,
                    "prob_benign": np.nan,
                    "prob_malignant": np.nan,
                    "classification_threshold": self.threshold,
                    "prediction_status": error_msg,
                })
                invalid_rows.append({"row_number": row_idx + 1, "error": error_msg})

        df_out = pd.DataFrame(results_rows)
        df_out.to_csv(output_path, index=False)

        cancer_detected_count = sum(1 for r in results_rows if r.get("predicted_class") == "Malignant")
        no_cancer_count = sum(1 for r in results_rows if r.get("predicted_class") == "Benign")
        cancer_percentage = (cancer_detected_count / valid_count * 100.0) if valid_count > 0 else 0.0
        high_risk_count = sum(
            1 for r in results_rows
            if "Critical" in str(r.get("risk_tier", "")) or "High" in str(r.get("risk_tier", ""))
        )
        borderline_count = sum(1 for r in results_rows if r.get("is_borderline", False))

        logger.info(
            "Batch prediction completed: %d total, %d valid, %d invalid. Cancer detected: %d (%.1f%%). Exported to %s",
            len(df_in), valid_count, len(invalid_rows), cancer_detected_count, cancer_percentage, output_path,
        )

        return BatchPredictionResult(
            total_records=len(df_in),
            valid_records=valid_count,
            invalid_records=len(invalid_rows),
            predictions_df=df_out,
            output_path=output_path,
            invalid_rows_summary=invalid_rows,
            cancer_detected_count=cancer_detected_count,
            no_cancer_count=no_cancer_count,
            cancer_percentage=round(cancer_percentage, 1),
            high_risk_count=high_risk_count,
            borderline_count=borderline_count,
            column_mapping_used=col_mapping,
        )

    def generate_pdf_report(
        self,
        prediction_result: PredictionResult,
        output_path: Path | None = None,
    ) -> Path:
        """
        Generate a downloadable, publication-grade clinical prediction report in PDF format.
        """
        if output_path is None:
            PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
            timestamp_slug = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = PREDICTIONS_DIR / f"prediction_report_{timestamp_slug}.pdf"

        pred = prediction_result
        is_mal = (pred.predicted_class == 1)
        tag_color = "#e63946" if is_mal else "#2a9d8f"

        fig, ax = plt.subplots(figsize=(8.5, 11), dpi=300)
        ax.axis("off")

        # 1. Header Banner
        ax.text(0.5, 0.96, PROJECT_NAME, ha="center", fontsize=20, fontweight="bold", color="#1d3557")
        ax.text(0.5, 0.93, PROJECT_TAGLINE, ha="center", fontsize=11, color="#457b9d", style="italic")
        ax.axhline(0.915, color="#1d3557", linewidth=1.5)

        # 2. Metadata Strip
        ax.text(0.05, 0.885, f"Report Date: {pred.timestamp}", fontsize=9, color="#555555")
        ax.text(0.50, 0.885, f"Model Architecture: {pred.model_name}", fontsize=9, color="#555555")
        ax.text(0.05, 0.865, f"Decision Threshold: {pred.classification_threshold:.2f}", fontsize=9, color="#555555")
        ax.text(0.50, 0.865, f"Risk Stratification: {pred.risk_tier}", fontsize=9, color="#555555")

        # 3. Diagnosis Card
        rect = plt.Rectangle((0.05, 0.74), 0.90, 0.10, fill=True, facecolor="#f8f9fa", edgecolor=tag_color, linewidth=2, transform=ax.transAxes)
        ax.add_patch(rect)
        ax.text(0.08, 0.80, "PREDICTED DIAGNOSIS:", fontsize=11, fontweight="bold", color="#333333")
        ax.text(0.08, 0.76, pred.predicted_label.upper(), fontsize=20, fontweight="bold", color=tag_color)
        
        prob_str = f"P(Malignant): {pred.prob_malignant * 100:.1f}%\nP(Benign): {pred.prob_benign * 100:.1f}%"
        ax.text(0.55, 0.77, prob_str, fontsize=12, fontweight="bold", color="#1d3557")

        # 4. Feature Contributions Chart (Inner Plot)
        ax.text(0.05, 0.70, "TOP CONTRIBUTING CYTOLOGICAL FEATURES (SHAP ATTRIBUTION):", fontsize=11, fontweight="bold", color="#1d3557")
        
        if pred.top_10_features:
            feats = [f[0].replace("_", " ").title() for f in pred.top_10_features[:8]][::-1]
            s_vals = [f[2] for f in pred.top_10_features[:8]][::-1]
            c_colors = ["#e63946" if s > 0 else "#2a9d8f" for s in s_vals]

            # Inset bar axes
            inset_ax = fig.add_axes([0.15, 0.38, 0.70, 0.28])
            bars = inset_ax.barh(feats, s_vals, color=c_colors, height=0.6, edgecolor="#333333", linewidth=0.5)
            inset_ax.axvline(0, color="black", linewidth=1.0)
            inset_ax.set_xlabel("SHAP Contribution toward Malignancy", fontsize=9)
            inset_ax.tick_params(labelsize=8)
            inset_ax.grid(True, linestyle="--", alpha=0.3, axis="x")
            for bar, s in zip(bars, s_vals):
                offset = 0.002 if s >= 0 else -0.002
                ha = "left" if s >= 0 else "right"
                sign = "+" if s >= 0 else ""
                inset_ax.text(s + offset, bar.get_y() + bar.get_height() / 2, f"{sign}{s:.3f}", va="center", ha=ha, fontsize=7, fontweight="bold")
        else:
            ax.text(0.10, 0.50, "Local feature attribution not available for this prediction.", fontsize=10, style="italic")

        # 5. Narrative Explanation Summary
        ax.text(0.05, 0.32, "NARRATIVE EXPLANATION:", fontsize=10, fontweight="bold", color="#1d3557")
        y_pos = 0.29
        for line in pred.narrative_explanation[:3]:
            cleaned_line = line.replace("**", "").replace("- ", "• ")
            ax.text(0.05, y_pos, cleaned_line[:110], fontsize=8, color="#333333")
            y_pos -= 0.025

        # 6. Medical Safety Disclaimer
        ax.axhline(0.12, color="#e63946", linewidth=1.2)
        ax.text(0.5, 0.09, "MANDATORY CLINICAL & REGULATORY DISCLAIMER", ha="center", fontsize=9, fontweight="bold", color="#e63946")
        ax.text(
            0.5, 0.04,
            "\"This is an educational research prototype, not a clinically validated diagnostic system. "
            "It must not be used to diagnose cancer, make treatment decisions, or replace professional "
            "medical evaluation.\"",
            ha="center", fontsize=8, color="#555555", style="italic", wrap=True,
        )

        with PdfPages(output_path) as pdf:
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

        logger.info("Saved PDF clinical prediction report to: %s", output_path)
        return output_path


# ==============================================================================
# BACKWARD COMPATIBILITY LAYER FOR EXISTING TESTS
# ==============================================================================
class Predictor:
    """
    Backward-compatible wrapper preserving the signature expected by earlier tests:
    Predictor(trained_model, X_train, feature_names)
    """

    def __init__(
        self,
        trained_model: Any,
        X_train: np.ndarray,
        feature_names: list[str],
    ):
        self.model = trained_model
        self.X_train = np.asarray(X_train)
        self.feature_names = list(feature_names)

    def _determine_risk_tier(self, prob_malignant: float) -> tuple[str, str]:
        if prob_malignant >= RISK_THRESHOLD_HIGH:
            return "Critical Risk (>80%)", "#FF4B4B"
        elif prob_malignant >= RISK_THRESHOLD_MODERATE:
            return "High Risk (50-80%)", "#FF7675"
        elif prob_malignant >= RISK_THRESHOLD_LOW:
            return "Moderate Risk (20-50%)", "#FFB347"
        else:
            return "Low Risk (<20%)", "#00C9A7"

    def predict(self, feature_values: dict[str, float] | np.ndarray) -> PredictionResult:
        if isinstance(feature_values, dict):
            missing = set(self.feature_names) - set(feature_values.keys())
            if missing:
                raise ValueError(f"Missing required features: {list(missing)[:5]}")
            x_arr = np.array([[float(feature_values[f]) for f in self.feature_names]])
        elif isinstance(feature_values, np.ndarray):
            x_arr = feature_values.reshape(1, -1)
            if x_arr.shape[1] != len(self.feature_names):
                raise ValueError(f"Expected {len(self.feature_names)} features, got {x_arr.shape[1]}")
        else:
            raise TypeError("feature_values must be a dict or numpy ndarray")

        probs = self.model.pipeline.predict_proba(x_arr)[0]
        prob_ben = float(probs[0])
        prob_mal = float(probs[1])
        pred_class = int(prob_mal >= 0.37)
        label = CLASS_LABELS.get(pred_class, "Unknown")
        risk_tier, risk_color = self._determine_risk_tier(prob_mal)

        explanation = explain_single_sample(
            pipeline=self.model.pipeline,
            X_train=self.X_train,
            x_sample=x_arr,
            feature_names=self.feature_names,
            model_name=getattr(self.model, "name", "Model"),
        )

        return PredictionResult(
            model_name=getattr(self.model, "name", "Model"),
            predicted_label=label,
            predicted_class=pred_class,
            prob_malignant=round(prob_mal, 4),
            prob_benign=round(prob_ben, 4),
            classification_threshold=0.37,
            risk_tier=risk_tier,
            risk_color=risk_color,
            explanation=explanation,
            explanation_available=True,
            feature_values={self.feature_names[i]: float(x_arr[0, i]) for i in range(len(self.feature_names))},
        )


def load_prediction_engine() -> PredictionEngine:
    """Factory helper to obtain a singleton or new PredictionEngine instance."""
    if PredictionEngine._instance is None:
        PredictionEngine._instance = PredictionEngine()
    return PredictionEngine._instance
