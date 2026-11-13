"""
BreastGuard AI — Explainable AI (SHAP) Module
================================================================================
Implements local and global Explainable AI for breast cancer diagnostic predictions:
  1. Automated explainer selection for the champion Support Vector Machine architecture.
  2. Background sampling strictly from the training partition.
  3. Directional attribution toward Malignancy (Class 1, P(Malignant)).
  4. Global explainability:
     - Mean absolute SHAP feature importance
     - Beeswarm summary plots
     - Dependence & interaction plots
  5. Local explainability:
     - Individual patient sample attribution
     - Waterfall & force-style decomposition
     - Local contribution ranking & additivity validation
  6. Human-readable narrative explanation engine with clinical causation disclaimers.
  7. High-resolution figure generation saved to reports/figures/shap/.
================================================================================
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.pipeline import Pipeline

from config.config import (
    CLASS_LABELS,
    EVALUATION_DIR,
    FIGURES_DIR,
    LABEL_BENIGN,
    LABEL_MALIGNANT,
    MODELS_DIR,
    RANDOM_SEED,
)

logger = logging.getLogger(__name__)

SHAP_FIGURES_DIR: Path = FIGURES_DIR / "shap"
SHAP_FIGURES_DIR.mkdir(parents=True, exist_ok=True)


class ShapValueContainer(np.ndarray):
    """
    NumPy 2D array of shape (1, n_features) that provides seamless dictionary-like
    string indexing by feature name alongside full NumPy array operations (.shape, etc.).
    """
    def __new__(cls, arr: np.ndarray, feature_names: list[str] | None = None):
        obj = np.asarray(arr).reshape(1, -1).view(cls)
        obj.feature_names = feature_names or []
        obj._dict = {f: float(obj.ravel()[i]) for i, f in enumerate(obj.feature_names)} if feature_names else {}
        return obj

    def __len__(self) -> int:
        return len(self._dict) if self._dict else self.shape[-1]


    def __getitem__(self, item: Any) -> Any:
        if isinstance(item, str):
            return self._dict[item]
        return super().__getitem__(item)

    def values(self):
        return self._dict.values()

    def keys(self):
        return self._dict.keys()

    def items(self):
        return self._dict.items()

    def get(self, key: str, default: Any = None) -> Any:
        return self._dict.get(key, default)


@dataclass
class LocalExplanation:
    """Detailed local SHAP attribution for an individual patient biopsy sample."""
    sample_index: int | None
    feature_values: dict[str, float]
    predicted_class: int
    predicted_label: str
    prob_malignant: float
    prob_benign: float
    threshold_applied: float
    base_value: float
    shap_values: ShapValueContainer
    top_positive_contributions: list[tuple[str, float, float]]  # (feature, value, shap)
    top_negative_contributions: list[tuple[str, float, float]]  # (feature, value, shap)
    additivity_gap: float
    narrative_explanation: list[str]
    model_name: str = "Support Vector Machine"
    expected_value: float = 0.0
    global_importance: np.ndarray = field(default_factory=lambda: np.zeros(30))
    top_feature_indices: list[int] = field(default_factory=list)


@dataclass
class GlobalExplanation:
    """Global SHAP feature attribution summary across an evaluation dataset."""
    model_name: str
    explainer_type: str
    base_value: float
    feature_names: list[str]
    shap_values: np.ndarray  # shape (n_samples, n_features)
    data: np.ndarray         # shape (n_samples, n_features)
    mean_abs_shap: dict[str, float]
    top_10_features: list[str]
    positive_driver_features: list[str]
    negative_driver_features: list[str]


# Backward-compatibility aliases
ShapExplanation = LocalExplanation


class ShapExplainerManager:

    """
    Manages SHAP explainer lifecycle, background sampling, and cached inferences.
    Prevents redundant KernelExplainer computations for production and Streamlit usage.
    """
    _instance: ShapExplainerManager | None = None

    def __init__(
        self,
        model_path: Path | None = None,
        n_background: int = 50,
        random_state: int = RANDOM_SEED,
    ):
        self.model_path = model_path or (MODELS_DIR / "final_breastguard_model.pkl")
        self.n_background = n_background
        self.random_state = random_state
        self.model_bundle: dict[str, Any] = {}
        self.pipeline: Pipeline | None = None
        self.feature_names: list[str] = []
        self.explainer: shap.KernelExplainer | None = None
        self.X_background: np.ndarray | None = None
        self.base_value: float = 0.0

    def load_model(self) -> dict[str, Any]:
        """Load and validate the saved final model bundle."""
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Final model not found at {self.model_path}. "
                "Ensure optimization pipeline has run."
            )
        self.model_bundle = joblib.load(self.model_path)
        self.pipeline = self.model_bundle["pipeline"]
        self.feature_names = self.model_bundle["feature_names"]
        logger.info(
            "Loaded final model '%s' with %d features.",
            self.model_bundle.get("model_name", "Unknown"),
            len(self.feature_names),
        )
        return self.model_bundle

    def initialize_explainer(
        self,
        X_train: pd.DataFrame | np.ndarray,
    ) -> shap.KernelExplainer:
        """
        Initialize the KernelExplainer wrapping the full pipeline's predict_proba.
        Using representative background points sampled strictly from training data.
        """
        if self.pipeline is None:
            self.load_model()

        X_arr = np.asarray(X_train)
        if len(X_arr) > self.n_background:
            rng = np.random.RandomState(self.random_state)
            idx = rng.choice(len(X_arr), size=self.n_background, replace=False)
            self.X_background = X_arr[idx]
        else:
            self.X_background = X_arr

        # Prediction function: outputs P(Malignant) for Class 1
        def predict_fn(x: np.ndarray) -> np.ndarray:
            return self.pipeline.predict_proba(x)[:, 1]

        logger.info(
            "Initializing KernelExplainer with %d background samples...",
            len(self.X_background),
        )
        self.explainer = shap.KernelExplainer(predict_fn, self.X_background)
        self.base_value = float(self.explainer.expected_value)
        logger.info("KernelExplainer initialized. Base Expected Value: %.4f", self.base_value)
        return self.explainer


def get_explainer_selection_reasoning(model_architecture: str) -> dict[str, str]:
    """
    Returns the technical and mathematical justification for explainer selection.
    """
    return {
        "model_architecture": model_architecture,
        "selected_explainer": "shap.KernelExplainer",
        "why_not_tree_explainer": (
            "TreeExplainer requires ensemble decision tree structures (Random Forest, LightGBM, XGBoost). "
            "It is mathematically incompatible with kernel Support Vector Machines."
        ),
        "why_not_linear_explainer": (
            "LinearExplainer assumes linear decision boundaries f(x) = w^T x + b. "
            "Our champion model uses a non-linear Radial Basis Function (RBF) kernel, which projects data "
            "into an infinite-dimensional feature space; hence LinearExplainer is inapplicable."
        ),
        "kernel_explainer_suitability": (
            "KernelExplainer is a model-agnostic explainer rooted in cooperative game theory. It uses weighted "
            "local linear surrogate models (LIME-style) to estimate exact Shapley values on probability outputs. "
            "By wrapping the entire Pipeline (StandardScaler + SVC), explanations are computed directly in the "
            "original, unscaled 30 cytological measurement space with full pipeline fidelity."
        ),
    }


def compute_global_explanations(
    explainer: shap.KernelExplainer,
    X_eval: pd.DataFrame | np.ndarray,
    feature_names: list[str],
    nsamples: int = 100,
    model_name: str = "Support Vector Machine",
) -> GlobalExplanation:
    """
    Compute global SHAP values across an evaluation sample set.
    Identifies top 10 globally influential features and directional tendencies.
    """
    X_arr = np.asarray(X_eval)
    logger.info("Computing global SHAP values for %d samples (nsamples=%d)...", len(X_arr), nsamples)
    shap_vals = explainer.shap_values(X_arr, nsamples=nsamples)

    # Ensure 2D float array
    if isinstance(shap_vals, list):
        shap_vals = shap_vals[1] if len(shap_vals) > 1 else shap_vals[0]
    shap_vals = np.asarray(shap_vals, dtype=float)

    # 1. Mean Absolute SHAP Importance
    mean_abs = np.mean(np.abs(shap_vals), axis=0)
    sorted_idx = np.argsort(mean_abs)[::-1]
    
    mean_abs_dict = {feature_names[i]: float(mean_abs[i]) for i in sorted_idx}
    top_10 = [feature_names[i] for i in sorted_idx[:10]]

    # 2. Directional correlation: does higher feature value correlate with positive SHAP (malignancy)?
    pos_drivers = []
    neg_drivers = []
    for i in sorted_idx[:10]:
        feat = feature_names[i]
        corr = np.corrcoef(X_arr[:, i], shap_vals[:, i])[0, 1]
        if np.isnan(corr) or corr >= 0:
            pos_drivers.append(feat)
        else:
            neg_drivers.append(feat)

    base_val = float(explainer.expected_value)
    return GlobalExplanation(
        model_name=model_name,
        explainer_type="KernelExplainer",
        base_value=base_val,
        feature_names=feature_names,
        shap_values=shap_vals,
        data=X_arr,
        mean_abs_shap=mean_abs_dict,
        top_10_features=top_10,
        positive_driver_features=pos_drivers,
        negative_driver_features=neg_drivers,
    )


def explain_single_sample(
    pipeline: Pipeline,
    explainer: shap.KernelExplainer | None = None,
    sample: pd.Series | np.ndarray | dict[str, float] | None = None,
    feature_names: list[str] | None = None,
    threshold: float = 0.37,
    sample_index: int | None = None,
    nsamples: int = 100,
    X_train: pd.DataFrame | np.ndarray | None = None,
    x_sample: pd.Series | np.ndarray | dict[str, float] | None = None,
    model_name: str = "Support Vector Machine",
    **kwargs: Any,
) -> LocalExplanation:
    """
    Compute individual patient sample SHAP explanation.
    Decomposes the prediction into baseline + feature contributions.
    Supports both unified pipeline workflow and legacy predictor service calls.
    """
    if sample is None and x_sample is not None:
        sample = x_sample

    if sample is None:
        raise ValueError("Either sample or x_sample must be provided.")

    if feature_names is None:
        from config.config import FEATURE_COLUMNS
        feature_names = FEATURE_COLUMNS

    if isinstance(sample, dict):
        missing = set(feature_names) - set(sample.keys())
        if missing:
            raise ValueError(f"Missing required features: {list(missing)[:5]}")
        x_vec = np.array([float(sample[f]) for f in feature_names], dtype=float)
        feat_dict = {f: float(sample[f]) for f in feature_names}
    elif isinstance(sample, pd.Series):
        x_vec = np.array([float(sample[f]) for f in feature_names], dtype=float)
        feat_dict = {f: float(sample[f]) for f in feature_names}
    else:
        x_vec = np.asarray(sample, dtype=float).ravel()
        if len(x_vec) != len(feature_names):
            raise ValueError(f"Expected {len(feature_names)} features, got {len(x_vec)}")
        feat_dict = {feature_names[i]: float(x_vec[i]) for i in range(len(feature_names))}

    x_2d = x_vec.reshape(1, -1)

    # Pipeline inference
    probs = pipeline.predict_proba(x_2d)[0]
    prob_benign = float(probs[0])
    prob_malignant = float(probs[1])
    pred_class = int(prob_malignant >= threshold)
    pred_label = CLASS_LABELS[pred_class]

    # Initialize dynamic explainer if not provided
    if explainer is None:
        if X_train is not None:
            X_bg = np.asarray(X_train)
            if len(X_bg) > 20:
                rng = np.random.RandomState(RANDOM_SEED)
                X_bg = X_bg[rng.choice(len(X_bg), 20, replace=False)]
        else:
            X_bg = x_2d

        def predict_fn(x: np.ndarray) -> np.ndarray:
            return pipeline.predict_proba(x)[:, 1]

        explainer = shap.KernelExplainer(predict_fn, X_bg)

    # Compute SHAP values
    shap_vals_raw = explainer.shap_values(x_2d, nsamples=nsamples)
    if isinstance(shap_vals_raw, list):
        shap_vals_1d = np.asarray(shap_vals_raw[1] if len(shap_vals_raw) > 1 else shap_vals_raw[0]).ravel()
    else:
        shap_vals_1d = np.asarray(shap_vals_raw).ravel()

    base_val = float(explainer.expected_value)
    additivity_sum = base_val + float(np.sum(shap_vals_1d))
    additivity_gap = abs(additivity_sum - prob_malignant)

    shap_container = ShapValueContainer(shap_vals_1d, feature_names)

    # Separate positive and negative contributions
    pos_items = [
        (feature_names[i], feat_dict[feature_names[i]], float(shap_vals_1d[i]))
        for i in range(len(feature_names))
        if shap_vals_1d[i] > 1e-4
    ]
    pos_items.sort(key=lambda x: x[2], reverse=True)

    neg_items = [
        (feature_names[i], feat_dict[feature_names[i]], float(shap_vals_1d[i]))
        for i in range(len(feature_names))
        if shap_vals_1d[i] < -1e-4
    ]
    neg_items.sort(key=lambda x: x[2])  # Most negative first

    # Build narrative explanation statements
    narratives = format_human_readable_explanation(
        pred_label=pred_label,
        prob_malignant=prob_malignant,
        threshold=threshold,
        base_val=base_val,
        top_pos=pos_items[:4],
        top_neg=neg_items[:4],
    )

    abs_shap = np.abs(shap_vals_1d)
    top_indices = list(np.argsort(abs_shap)[::-1])

    return LocalExplanation(
        sample_index=sample_index,
        feature_values=feat_dict,
        predicted_class=pred_class,
        predicted_label=pred_label,
        prob_malignant=round(prob_malignant, 4),
        prob_benign=round(prob_benign, 4),
        threshold_applied=threshold,
        base_value=round(base_val, 4),
        shap_values=shap_container,
        top_positive_contributions=[(f, round(v, 4), round(s, 4)) for f, v, s in pos_items[:5]],
        top_negative_contributions=[(f, round(v, 4), round(s, 4)) for f, v, s in neg_items[:5]],
        additivity_gap=round(additivity_gap, 4),
        narrative_explanation=narratives,
        model_name=model_name,
        expected_value=round(base_val, 4),
        global_importance=abs_shap,
        top_feature_indices=top_indices,
    )


def format_human_readable_explanation(
    pred_label: str,
    prob_malignant: float,
    threshold: float,
    base_val: float,
    top_pos: list[tuple[str, float, float]],
    top_neg: list[tuple[str, float, float]],
) -> list[str]:
    """
    Format clinical predictions and SHAP attributions into clear, human-understandable narratives.
    Strictly avoids causal claims, emphasizing statistical feature contributions.
    """
    lines = [
        f"Model Decision: Classified as **{pred_label}** with P(Malignant) = {prob_malignant * 100:.1f}% "
        f"(Operating Threshold = {threshold:.2f}, Population Baseline = {base_val * 100:.1f}%).",
    ]

    if top_pos:
        lines.append("Features increasing model output toward Malignancy:")
        for feat, val, s_val in top_pos:
            cleaned_name = feat.replace("_", " ").title()
            lines.append(
                f"  - **{cleaned_name}** (observed value: {val:.3f}) had a positive contribution "
                f"of +{s_val:.3f} toward the malignant classification."
            )

    if top_neg:
        lines.append("Features decreasing model output toward Benign:")
        for feat, val, s_val in top_neg:
            cleaned_name = feat.replace("_", " ").title()
            lines.append(
                f"  - **{cleaned_name}** (observed value: {val:.3f}) contributed "
                f"{s_val:.3f} toward the benign classification."
            )

    lines.append(
        "Important Note: Positive and negative contributions reflect the mathematical direction and magnitude "
        "by which cellular measurements shift the model's statistical decision function relative to its training baseline. "
        "They represent statistical attribution, not medical causation or independent clinical risk factors."
    )
    return lines


# ==============================================================================
# HIGH-RESOLUTION SHAP VISUALIZATION GENERATORS
# ==============================================================================

def plot_shap_summary_beeswarm(
    global_exp: GlobalExplanation,
    output_path: Path | None = None,
    max_display: int = 15,
) -> Path:
    """
    Generate publication-grade SHAP Beeswarm summary plot.
    Saved to reports/figures/shap/shap_summary_beeswarm.png.
    """
    if output_path is None:
        output_path = SHAP_FIGURES_DIR / "shap_summary_beeswarm.png"

    clean_names = [f.replace("_", " ").title() for f in global_exp.feature_names]
    shap_obj = shap.Explanation(
        values=global_exp.shap_values,
        base_values=np.full(len(global_exp.shap_values), global_exp.base_value),
        data=global_exp.data,
        feature_names=clean_names,
    )

    fig = plt.figure(figsize=(10, 7), dpi=300)
    shap.plots.beeswarm(shap_obj, max_display=max_display, show=False)
    plt.title(
        "BreastGuard AI — Global SHAP Summary (Beeswarm Distribution)\nImpact on P(Malignant)",
        fontsize=13,
        fontweight="bold",
        pad=15,
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close("all")
    logger.info("Saved SHAP summary beeswarm plot to: %s", output_path)
    return output_path


def plot_shap_global_importance_bar(
    global_exp: GlobalExplanation,
    output_path: Path | None = None,
    top_n: int = 15,
) -> Path:
    """
    Generate horizontal bar chart of mean absolute SHAP values across features.
    Saved to reports/figures/shap/shap_global_importance.png.
    """
    if output_path is None:
        output_path = SHAP_FIGURES_DIR / "shap_global_importance.png"

    items = list(global_exp.mean_abs_shap.items())[:top_n]
    features = [k.replace("_", " ").title() for k, _ in items][::-1]
    scores = [v for _, v in items][::-1]

    fig, ax = plt.subplots(figsize=(9, 7), dpi=300)
    bars = ax.barh(features, scores, color="#1d3557", edgecolor="#0a192f", height=0.65)
    
    # Add numerical labels
    for bar in bars:
        w = bar.get_width()
        ax.text(w + 0.002, bar.get_y() + bar.get_height() / 2, f"{w:.4f}", va="center", fontsize=9, color="#1d3557")

    ax.set_title(
        f"BreastGuard AI — Top {top_n} Globally Influential Features\nMean Absolute SHAP Value (Impact on Malignant Probability)",
        fontsize=13,
        fontweight="bold",
        pad=12,
    )
    ax.set_xlabel("Mean |SHAP Value| (Average impact on model output magnitude)", fontsize=11)
    ax.set_xlim(0, max(scores) * 1.18)
    ax.grid(True, linestyle="--", alpha=0.4, axis="x")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close(fig)
    logger.info("Saved SHAP global importance plot to: %s", output_path)
    return output_path


def plot_shap_bar_plot(
    global_exp: GlobalExplanation,
    output_path: Path | None = None,
) -> Path:
    """
    Generate standard SHAP bar plot using shap.plots.bar.
    Saved to reports/figures/shap/shap_bar_plot.png.
    """
    if output_path is None:
        output_path = SHAP_FIGURES_DIR / "shap_bar_plot.png"

    clean_names = [f.replace("_", " ").title() for f in global_exp.feature_names]
    shap_obj = shap.Explanation(
        values=global_exp.shap_values,
        base_values=np.full(len(global_exp.shap_values), global_exp.base_value),
        data=global_exp.data,
        feature_names=clean_names,
    )

    fig = plt.figure(figsize=(9, 6), dpi=300)
    shap.plots.bar(shap_obj, max_display=12, show=False)
    plt.title("BreastGuard AI — SHAP Feature Importance Ranking", fontsize=13, fontweight="bold", pad=15)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close("all")
    logger.info("Saved SHAP bar plot to: %s", output_path)
    return output_path


def plot_shap_dependence(
    global_exp: GlobalExplanation,
    feature_name: str | None = None,
    output_path: Path | None = None,
) -> Path:
    """
    Generate SHAP Dependence Plot for the top feature.
    Saved to reports/figures/shap/shap_dependence_top_feature.png.
    """
    if output_path is None:
        output_path = SHAP_FIGURES_DIR / "shap_dependence_top_feature.png"

    if feature_name is None:
        feature_name = global_exp.top_10_features[0]

    feat_idx = global_exp.feature_names.index(feature_name)
    x_vals = global_exp.data[:, feat_idx]
    s_vals = global_exp.shap_values[:, feat_idx]

    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
    scatter = ax.scatter(x_vals, s_vals, c=s_vals, cmap="coolwarm", alpha=0.85, edgecolors="none", s=45)
    ax.axhline(0, color="gray", linestyle="--", linewidth=1.2, alpha=0.7)

    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("SHAP Value for Malignant Class", fontsize=10)

    clean_feat = feature_name.replace("_", " ").title()
    ax.set_title(
        f"BreastGuard AI — SHAP Dependence Plot: {clean_feat}\nFeature Value vs. Impact on Malignant Probability",
        fontsize=12,
        fontweight="bold",
        pad=12,
    )
    ax.set_xlabel(f"Observed Feature Value: {clean_feat}", fontsize=11)
    ax.set_ylabel("SHAP Value (Contribution to P(Malignant))", fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close(fig)
    logger.info("Saved SHAP dependence plot to: %s", output_path)
    return output_path


def plot_shap_waterfall(
    local_exp: LocalExplanation,
    output_path: Path | None = None,
    max_display: int = 10,
) -> Path:
    """
    Generate SHAP Waterfall Plot for an individual sample.
    Saved to reports/figures/shap/shap_waterfall_example.png.
    """
    if output_path is None:
        output_path = SHAP_FIGURES_DIR / "shap_waterfall_example.png"

    feat_names = list(local_exp.feature_values.keys())
    clean_names = [f.replace("_", " ").title() for f in feat_names]
    data_vec = np.array([local_exp.feature_values[f] for f in feat_names])
    shap_vec = np.array([local_exp.shap_values[f] for f in feat_names])

    shap_obj = shap.Explanation(
        values=shap_vec,
        base_values=local_exp.base_value,
        data=data_vec,
        feature_names=clean_names,
    )

    fig = plt.figure(figsize=(9, 7), dpi=300)
    shap.plots.waterfall(shap_obj, max_display=max_display, show=False)
    plt.title(
        f"BreastGuard AI — Individual Prediction Decomposition (Waterfall)\n"
        f"Outcome: {local_exp.predicted_label} | P(Malignant) = {local_exp.prob_malignant * 100:.1f}%",
        fontsize=12,
        fontweight="bold",
        pad=15,
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close("all")
    logger.info("Saved SHAP waterfall plot to: %s", output_path)
    return output_path


def plot_shap_local_contributions_bar(
    local_exp: LocalExplanation,
    output_path: Path | None = None,
    top_k: int = 8,
) -> Path:
    """
    Generate horizontal bar chart of top positive vs negative contributions for an individual patient.
    Saved to reports/figures/shap/shap_local_contribution.png.
    """
    if output_path is None:
        output_path = SHAP_FIGURES_DIR / "shap_local_contribution.png"

    pos = local_exp.top_positive_contributions[:top_k]
    neg = local_exp.top_negative_contributions[:top_k]

    combined = pos + neg
    combined.sort(key=lambda x: x[2])  # sorted by SHAP value ascending

    features = [f"{f.replace('_', ' ').title()} ({v:.2f})" for f, v, _ in combined]
    shap_vals = [s for _, _, s in combined]
    colors = ["#e63946" if s > 0 else "#2a9d8f" for s in shap_vals]

    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    bars = ax.barh(features, shap_vals, color=colors, height=0.6, edgecolor="#333333", linewidth=0.5)

    for bar, s in zip(bars, shap_vals):
        offset = 0.003 if s >= 0 else -0.003
        ha = "left" if s >= 0 else "right"
        sign = "+" if s >= 0 else ""
        ax.text(s + offset, bar.get_y() + bar.get_height() / 2, f"{sign}{s:.3f}", va="center", ha=ha, fontsize=9, fontweight="bold")

    ax.axvline(0, color="black", linestyle="-", linewidth=1.2)
    ax.set_title(
        f"BreastGuard AI — Local Feature Attribution Profile\n"
        f"Predicted: {local_exp.predicted_label} | P(Malignant): {local_exp.prob_malignant * 100:.1f}% (Base: {local_exp.base_value * 100:.1f}%)",
        fontsize=12,
        fontweight="bold",
        pad=12,
    )
    ax.set_xlabel("SHAP Contribution to P(Malignant) relative to Base Value", fontsize=11)
    
    # Custom legend
    import matplotlib.patches as mpatches
    red_patch = mpatches.Patch(color="#e63946", label="Pushes toward Malignant (+)")
    teal_patch = mpatches.Patch(color="#2a9d8f", label="Pushes toward Benign (-)")
    ax.legend(handles=[red_patch, teal_patch], loc="lower right", framealpha=0.9)

    ax.grid(True, linestyle="--", alpha=0.4, axis="x")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close(fig)
    logger.info("Saved local contribution plot to: %s", output_path)
    return output_path


# Backward compatibility
compute_shap_explanations = compute_global_explanations

