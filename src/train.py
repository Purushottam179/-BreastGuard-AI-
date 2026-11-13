"""
BreastGuard AI — Machine Learning Training & Cross-Validation Engine
================================================================================
Implements five candidate classification algorithms:
  1. Logistic Regression (L2 regularization, lbfgs solver) — Scaled
  2. Random Forest (300 estimators, bagged ensemble)       — Unscaled
  3. Support Vector Machine (RBF kernel, probability=True) — Scaled
  4. Gradient Boosting (GradientBoostingClassifier)        — Unscaled
  5. K-Nearest Neighbors (k=5, Euclidean metric)           — Scaled

Performs Stratified 5-Fold Cross-Validation on the training split only.
Prevents data leakage by encapsulating scaling inside scikit-learn Pipelines.
Exports serialized models (.pkl), metadata, and CV comparison tables.
================================================================================
"""
from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

from config.config import (
    CLASS_LABELS,
    CV_FOLDS,
    EVALUATION_DIR,
    FIGURES_DIR,
    MODEL_FILENAMES,
    MODEL_GRADIENT_BOOSTING,
    MODEL_KNN,
    MODEL_LOGISTIC_REGRESSION,
    MODEL_RANDOM_FOREST,
    MODEL_SVM,
    MODELS_DIR,
    RANDOM_SEED,
)
from src.preprocessing import SplitData, build_pipeline

logger = logging.getLogger(__name__)

# Scoring metrics evaluated during cross-validation
CV_SCORING = {
    "accuracy": "accuracy",
    "precision": "precision",
    "recall": "recall",
    "f1": "f1",
    "roc_auc": "roc_auc",
}


@dataclass
class ModelConfig:
    """Configuration definition for a candidate model."""
    name: str
    estimator: Any
    use_scaler: bool
    filename: str
    description: str


@dataclass
class ModelCVResult:
    """Holds 5-fold cross-validation performance for a single model."""
    name: str
    metrics_mean: dict[str, float]
    metrics_std: dict[str, float]
    fold_scores: dict[str, list[float]]


@dataclass
class TrainedModelBundle:
    """Holds a fully fitted model pipeline, configuration, and CV statistics."""
    name: str
    pipeline: Pipeline
    cv_result: Any = None
    filename: str = ""
    filepath: Path | None = None
    cv_scores: Any = None
    cv_mean: Any = None
    cv_std: Any = None


def get_model_specifications() -> list[ModelConfig]:
    """
    Define all five candidate algorithms with appropriate scaling and hyperparameter configurations.
    Returns:
      List of ModelConfig instances.
    """
    return [
        ModelConfig(
            name=MODEL_LOGISTIC_REGRESSION,
            estimator=LogisticRegression(
                C=1.0,
                max_iter=1000,
                solver="lbfgs",
                random_state=RANDOM_SEED,
            ),
            use_scaler=True,
            filename=MODEL_FILENAMES[MODEL_LOGISTIC_REGRESSION],
            description="L2-regularized linear baseline with StandardScaler.",
        ),
        ModelConfig(
            name=MODEL_RANDOM_FOREST,
            estimator=RandomForestClassifier(
                n_estimators=300,
                max_depth=None,
                min_samples_split=2,
                min_samples_leaf=1,
                random_state=RANDOM_SEED,
                n_jobs=-1,
            ),
            use_scaler=False,  # Tree models are scale-invariant
            filename=MODEL_FILENAMES[MODEL_RANDOM_FOREST],
            description="Bagged ensemble of 300 decision trees without feature scaling.",
        ),
        ModelConfig(
            name=MODEL_SVM,
            estimator=SVC(
                C=1.0,
                kernel="rbf",
                gamma="scale",
                probability=True,  # Required for ROC-AUC and probability calibration
                random_state=RANDOM_SEED,
            ),
            use_scaler=True,  # SVM is distance-based, requires scaling
            filename=MODEL_FILENAMES[MODEL_SVM],
            description="RBF-kernel Support Vector Machine with probability estimates and scaling.",
        ),
        ModelConfig(
            name=MODEL_GRADIENT_BOOSTING,
            estimator=GradientBoostingClassifier(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=3,
                subsample=0.8,
                random_state=RANDOM_SEED,
            ),
            use_scaler=False,  # Gradient boosting is scale-invariant
            filename=MODEL_FILENAMES[MODEL_GRADIENT_BOOSTING],
            description="Sequential boosted decision tree ensemble without scaling.",
        ),
        ModelConfig(
            name=MODEL_KNN,
            estimator=KNeighborsClassifier(
                n_neighbors=5,
                weights="uniform",
                metric="minkowski",
                p=2,  # Euclidean distance
            ),
            use_scaler=True,  # KNN requires strict normalization
            filename=MODEL_FILENAMES[MODEL_KNN],
            description="5-Nearest Neighbors classifier with Euclidean distance and scaling.",
        ),
    ]


def run_cross_validation(
    model_configs: list[ModelConfig],
    split: SplitData,
) -> dict[str, ModelCVResult]:
    """
    Perform Stratified 5-Fold Cross-Validation on the training split only.
    Test set is strictly excluded to prevent data leakage.

    Parameters
    ----------
    model_configs : list[ModelConfig]
    split : SplitData

    Returns
    -------
    dict[str, ModelCVResult]
    """
    logger.info("Executing Stratified %d-Fold CV on training split (%d samples) ...", CV_FOLDS, split.train_size)
    skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    cv_results = {}

    for cfg in model_configs:
        logger.info("Running CV for: %s (use_scaler=%s)", cfg.name, cfg.use_scaler)
        pipe = build_pipeline(cfg.estimator, use_scaler=cfg.use_scaler, feature_names=split.feature_names)

        raw_scores = cross_validate(
            pipe,
            split.X_train,
            split.y_train,
            cv=skf,
            scoring=CV_SCORING,
            return_train_score=False,
            n_jobs=-1,
        )

        metrics_mean = {}
        metrics_std = {}
        fold_scores = {}

        for metric_name in CV_SCORING.keys():
            fold_key = f"test_{metric_name}"
            scores = raw_scores[fold_key].tolist()
            fold_scores[metric_name] = [round(float(s), 5) for s in scores]
            metrics_mean[metric_name] = round(float(np.mean(scores)), 5)
            metrics_std[metric_name] = round(float(np.std(scores)), 5)

        cv_res = ModelCVResult(
            name=cfg.name,
            metrics_mean=metrics_mean,
            metrics_std=metrics_std,
            fold_scores=fold_scores,
        )
        cv_results[cfg.name] = cv_res

        logger.info(
            "  -> %-24s | Acc: %.4f | Rec: %.4f | F1: %.4f | AUC: %.4f",
            cfg.name,
            metrics_mean["accuracy"],
            metrics_mean["recall"],
            metrics_mean["f1"],
            metrics_mean["roc_auc"],
        )

    return cv_results


def export_cv_results_table(
    cv_results: dict[str, ModelCVResult],
    output_path: Path | None = None,
) -> pd.DataFrame:
    """
    Compile and export a comprehensive cross-validation results table.
    Saved to reports/evaluation/cross_validation_results.csv.
    """
    if output_path is None:
        EVALUATION_DIR.mkdir(parents=True, exist_ok=True)
        output_path = EVALUATION_DIR / "cross_validation_results.csv"

    rows = []
    for name, res in cv_results.items():
        row = {"Model": name}
        for metric in CV_SCORING.keys():
            mean_val = res.metrics_mean[metric]
            std_val = res.metrics_std[metric]
            row[f"{metric}_mean"] = mean_val
            row[f"{metric}_std"] = std_val
            # Record individual folds
            for fold_i, score in enumerate(res.fold_scores[metric], 1):
                row[f"{metric}_fold_{fold_i}"] = score
        rows.append(row)

    df_cv = pd.DataFrame(rows)
    df_cv.to_csv(output_path, index=False)
    logger.info("Saved cross-validation results table to: %s", output_path)
    return df_cv


def train_and_save_all_models(
    model_configs: list[ModelConfig],
    split: SplitData,
    cv_results: dict[str, ModelCVResult],
) -> dict[str, TrainedModelBundle]:
    """
    Fit each candidate model pipeline on the complete training split and serialize to models/.

    Parameters
    ----------
    model_configs : list[ModelConfig]
    split : SplitData
    cv_results : dict[str, ModelCVResult]

    Returns
    -------
    dict[str, TrainedModelBundle]
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    bundles = {}

    for cfg in model_configs:
        logger.info("Fitting final pipeline on full training partition: %s", cfg.name)
        pipe = build_pipeline(cfg.estimator, use_scaler=cfg.use_scaler, feature_names=split.feature_names)
        pipe.fit(split.X_train, split.y_train)

        # Save model pipeline using Joblib (.pkl format)
        model_path = MODELS_DIR / cfg.filename
        joblib.dump(pipe, model_path)
        logger.info("Saved trained pipeline: %s", model_path)

        bundle = TrainedModelBundle(
            name=cfg.name,
            pipeline=pipe,
            cv_result=cv_results[cfg.name],
            filename=cfg.filename,
            filepath=model_path,
        )
        bundles[cfg.name] = bundle

    # Save centralized training metadata artifact
    metadata = {
        "random_seed": RANDOM_SEED,
        "test_size": 0.20,
        "cv_folds": CV_FOLDS,
        "class_labels": CLASS_LABELS,
        "n_features": len(split.feature_names),
        "feature_names": split.feature_names,
        "models": {
            cfg.name: {
                "filename": cfg.filename,
                "use_scaler": cfg.use_scaler,
                "description": cfg.description,
                "cv_metrics_mean": cv_results[cfg.name].metrics_mean,
                "cv_metrics_std": cv_results[cfg.name].metrics_std,
            }
            for cfg in model_configs
        },
    }

    meta_json_path = MODELS_DIR / "training_metadata.json"
    with open(meta_json_path, "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info("Training metadata saved to: %s", meta_json_path)

    return bundles


def plot_cv_comparisons(
    cv_results: dict[str, ModelCVResult],
    output_dir: Path | None = None,
) -> dict[str, Path]:
    """
    Generate the 4 required cross-validation comparison charts:
      1. Cross-validation accuracy comparison
      2. Cross-validation recall (sensitivity) comparison
      3. Cross-validation F1-score comparison
      4. Cross-validation ROC-AUC comparison

    Saved to reports/figures/.
    """
    if output_dir is None:
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        output_dir = FIGURES_DIR

    metrics_to_plot = [
        ("accuracy", "Cross-Validation Accuracy Comparison", "Accuracy Score", "cv_accuracy_comparison.png"),
        ("recall", "Cross-Validation Recall (Sensitivity) Comparison", "Recall Score (Malignant)", "cv_recall_comparison.png"),
        ("f1", "Cross-Validation F1-Score Comparison", "F1 Score (Malignant)", "cv_f1_comparison.png"),
        ("roc_auc", "Cross-Validation ROC-AUC Comparison", "Area Under ROC Curve", "cv_roc_auc_comparison.png"),
    ]

    saved_charts = {}
    models = list(cv_results.keys())
    palette = ["#4C6EF5", "#12B886", "#FA5252", "#FD7E14", "#7950F2"]

    for metric_key, title, ylabel, filename in metrics_to_plot:
        means = [cv_results[m].metrics_mean[metric_key] for m in models]
        stds = [cv_results[m].metrics_std[metric_key] for m in models]

        plt.figure(figsize=(10, 6), dpi=300)
        bars = plt.bar(models, means, yerr=stds, capsize=6, color=palette, edgecolor="#2D3748", width=0.55, alpha=0.9)

        plt.title(f"{title} (5-Fold Stratified CV)", fontsize=14, fontweight="bold", pad=15)
        plt.ylabel(ylabel, fontsize=12, labelpad=10)
        plt.xlabel("Candidate Model Architecture", fontsize=12, labelpad=10)
        plt.ylim(max(0.80, min(means) - 0.10), 1.01)
        plt.grid(axis="y", linestyle="--", alpha=0.4)

        # Annotate mean value on each bar
        for bar, mean_val, std_val in zip(bars, means, stds):
            yval = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width() / 2.0,
                yval + std_val + 0.008,
                f"{mean_val:.4f} ± {std_val:.4f}",
                ha="center",
                va="bottom",
                fontsize=10,
                fontweight="semibold",
            )

        plt.xticks(rotation=15, ha="right", fontsize=10)
        plt.tight_layout()

        chart_path = output_dir / filename
        plt.savefig(chart_path, dpi=300)
        plt.close()

        saved_charts[metric_key] = chart_path
        logger.info("Saved comparison chart: %s", chart_path)

    return saved_charts


def select_best_model_from_cv(cv_results: dict[str, ModelCVResult]) -> tuple[str, str]:
    """
    Perform principled model selection based on training cross-validation results only.
    Selection criteria (clinical priority):
      1. Malignant-class recall (Sensitivity) — minimizing missed cancers is critical.
      2. ROC-AUC and F1-Score — overall discriminatory power and harmonic balance.
      3. Cross-validation standard deviation (Stability) — lower variance implies better generalization.

    Returns
    -------
    selected_model_name : str
    justification : str
    """
    scores = []
    for name, res in cv_results.items():
        rec = res.metrics_mean["recall"]
        auc = res.metrics_mean["roc_auc"]
        f1 = res.metrics_mean["f1"]
        acc = res.metrics_mean["accuracy"]
        var = (res.metrics_std["recall"] + res.metrics_std["roc_auc"]) / 2.0

        # Clinical composite ranking: 40% Recall + 30% ROC-AUC + 20% F1 + 10% Accuracy - penalty for high variance
        composite = (0.40 * rec) + (0.30 * auc) + (0.20 * f1) + (0.10 * acc) - (0.10 * var)
        scores.append((name, composite, rec, auc, f1, acc, var))

    # Sort descending by composite score
    scores.sort(key=lambda x: x[1], reverse=True)
    best_name, best_comp, best_rec, best_auc, best_f1, best_acc, best_var = scores[0]

    justification = (
        f"Selected '{best_name}' as the primary model. In clinical oncology, prioritizing "
        f"Malignant-class recall (Sensitivity={best_rec:.4f}) is essential to minimize fatal false negatives. "
        f"Combined with exceptional ROC-AUC ({best_auc:.4f}), balanced F1-score ({best_f1:.4f}), and high stability "
        f"across folds (std={best_var:.4f}), this architecture demonstrates superior clinical reliability "
        f"over naive accuracy ({best_acc:.4f})."
    )

    logger.info("Model Selection: %s\nJustification: %s", best_name, justification)
    return best_name, justification


# Compatibility aliases for package exports
TrainedModel = TrainedModelBundle


@dataclass
class TrainingSummary:
    """Container holding all fitted models and CV results."""
    models: dict[str, TrainedModel]
    cv_results: dict[str, ModelCVResult]


def train_models(split: SplitData, save_artifacts: bool = True) -> TrainingSummary:
    """
    High-level orchestrator: runs CV, plots comparisons, and fits all 5 models.
    """
    configs = get_model_specifications()
    cv_results = run_cross_validation(configs, split)
    export_cv_results_table(cv_results)
    models = train_and_save_all_models(configs, split, cv_results)
    plot_cv_comparisons(cv_results)
    return TrainingSummary(models=models, cv_results=cv_results)
