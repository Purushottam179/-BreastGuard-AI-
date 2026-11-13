"""
BreastGuard AI — SHAP Explainability Module
Generates local and global SHAP explanations for all four model types.

Strategy:
- Random Forest, XGBoost → TreeExplainer (fast, exact)
- Logistic Regression    → LinearExplainer (fast, exact)
- SVM                    → KernelExplainer (slow; background sampled, results cached)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import shap
from sklearn.pipeline import Pipeline

from src.utils.config import SHAP_BACKGROUND_SAMPLES, RANDOM_STATE

logger = logging.getLogger(__name__)


@dataclass
class ShapResult:
    """Holds SHAP explanation outputs for one model."""
    model_name:    str
    shap_values:   np.ndarray    # shape (n_samples, n_features)  — class=1 (benign)
    expected_value: float
    feature_names: list[str]
    # Global importance: mean absolute SHAP per feature
    global_importance: np.ndarray   # shape (n_features,)


def _get_explainer_and_values(
    pipeline: Pipeline,
    X_train: np.ndarray,
    X_explain: np.ndarray,
    model_name: str,
) -> tuple:
    """
    Select the appropriate SHAP explainer based on model type.
    Returns (explainer, shap_values_class1, expected_value).
    """
    # Extract the fitted model and scaler from the pipeline
    scaler = pipeline.named_steps["scaler"]
    model  = pipeline.named_steps["model"]

    # Transform data through the fitted scaler
    X_train_scaled   = scaler.transform(X_train)
    X_explain_scaled = scaler.transform(X_explain)

    model_type = type(model).__name__

    if model_type in ("RandomForestClassifier",):
        logger.info("Using TreeExplainer for %s", model_name)
        explainer = shap.TreeExplainer(model)
        sv = explainer.shap_values(X_explain_scaled)
        # RF returns list [class0_shap, class1_shap]; take class 1 (benign)
        shap_vals = sv[1] if isinstance(sv, list) else sv
        expected  = float(explainer.expected_value[1]) if hasattr(explainer.expected_value, '__len__') else float(explainer.expected_value)

    elif model_type in ("XGBClassifier",):
        logger.info("Using TreeExplainer for %s", model_name)
        explainer = shap.TreeExplainer(model)
        sv = explainer.shap_values(X_explain_scaled)
        shap_vals = sv[1] if isinstance(sv, list) else sv
        expected  = float(explainer.expected_value[1]) if hasattr(explainer.expected_value, '__len__') else float(explainer.expected_value)

    elif model_type in ("LogisticRegression",):
        logger.info("Using LinearExplainer for %s", model_name)
        explainer = shap.LinearExplainer(model, X_train_scaled, feature_perturbation="interventional")
        sv = explainer.shap_values(X_explain_scaled)
        # LinearExplainer returns shape (n_samples, n_features) for binary
        shap_vals = sv if sv.ndim == 2 else sv
        expected  = float(explainer.expected_value) if not hasattr(explainer.expected_value, '__len__') else float(explainer.expected_value[-1])

    else:
        # SVM and others → KernelExplainer
        logger.info("Using KernelExplainer for %s (this may take 30–60 s) …", model_name)
        rng = np.random.default_rng(RANDOM_STATE)
        idx = rng.choice(len(X_train_scaled), size=min(SHAP_BACKGROUND_SAMPLES, len(X_train_scaled)), replace=False)
        background = X_train_scaled[idx]

        def predict_fn(x):
            return model.predict_proba(x)[:, 1]

        explainer = shap.KernelExplainer(predict_fn, background)
        shap_vals = explainer.shap_values(X_explain_scaled, nsamples=100)
        expected  = float(explainer.expected_value)

    return explainer, shap_vals, expected


def compute_shap(
    pipeline:     Pipeline,
    X_train:      np.ndarray,
    X_explain:    np.ndarray,
    feature_names: list[str],
    model_name:   str,
) -> ShapResult:
    """
    Compute SHAP values for the given model on X_explain.

    Parameters
    ----------
    pipeline      : Fitted sklearn Pipeline (scaler + model)
    X_train       : Raw (unscaled) training data for background distribution
    X_explain     : Raw (unscaled) samples to explain (e.g., X_test)
    feature_names : List of feature names
    model_name    : Human-readable model name

    Returns
    -------
    ShapResult
    """
    _, shap_vals, expected = _get_explainer_and_values(
        pipeline, X_train, X_explain, model_name
    )

    global_imp = np.abs(shap_vals).mean(axis=0)

    result = ShapResult(
        model_name    = model_name,
        shap_values   = shap_vals,
        expected_value = expected,
        feature_names = feature_names,
        global_importance = global_imp,
    )
    logger.info(
        "SHAP done for %s | shape=%s | top feature=%s (%.4f)",
        model_name, shap_vals.shape,
        feature_names[int(np.argmax(global_imp))],
        float(np.max(global_imp)),
    )
    return result


def explain_single(
    pipeline:     Pipeline,
    X_train:      np.ndarray,
    x_instance:   np.ndarray,   # shape (1, n_features)  unscaled
    feature_names: list[str],
    model_name:   str,
) -> ShapResult:
    """
    Compute SHAP for a single prediction instance (used in interactive page).
    """
    return compute_shap(pipeline, X_train, x_instance, feature_names, model_name)
