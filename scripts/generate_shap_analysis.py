"""
BreastGuard AI — SHAP Global and Local Explainability Generation Script
================================================================================
Generates all global and local Explainable AI artifacts:
  1. Loads final model bundle (models/final_breastguard_model.pkl)
  2. Uses representative background data sampled strictly from X_train
  3. Computes 30-feature global SHAP attributions toward P(Malignant)
  4. Generates publication-ready figures in reports/figures/shap/
  5. Computes individual sample explanations for clinical test cases
  6. Validates SHAP mathematical consistency and additivity
  7. Exports reports/evaluation/shap_analysis_summary.json
================================================================================
"""
from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".matplotlib_cache"))
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import numpy as np
import pandas as pd
from config.config import EVALUATION_DIR, FIGURES_DIR, RANDOM_SEED
from src.data_loader import load_and_validate_processed_data
from src.explain import (
    SHAP_FIGURES_DIR,
    ShapExplainerManager,
    compute_global_explanations,
    explain_single_sample,
    get_explainer_selection_reasoning,
    plot_shap_bar_plot,
    plot_shap_dependence,
    plot_shap_global_importance_bar,
    plot_shap_local_contributions_bar,
    plot_shap_summary_beeswarm,
    plot_shap_waterfall,
)
from src.preprocessing import create_stratified_split
from src.utils import setup_logger

logger = setup_logger("shap_pipeline")


def run_shap_pipeline() -> dict:
    logger.info("=" * 80)
    logger.info("BreastGuard AI — Global and Local Explainability Pipeline (SHAP)")
    logger.info("=" * 80)

    # 1. Load Data & Prepare Train/Test Partitions
    X, y, feature_names = load_and_validate_processed_data()
    split = create_stratified_split(X, y, test_size=0.20, random_seed=RANDOM_SEED, feature_names=feature_names)
    logger.info("Dataset loaded: Train N=%d, Test N=%d", len(split.y_train), len(split.y_test))

    # 2. Initialize Explainer Manager with Trained Champion Model
    manager = ShapExplainerManager(n_background=50, random_state=RANDOM_SEED)
    model_bundle = manager.load_model()
    explainer = manager.initialize_explainer(split.X_train)
    
    threshold = float(model_bundle.get("classification_threshold", 0.37))
    model_name = model_bundle.get("model_name", "Support Vector Machine")
    arch_desc = model_bundle.get("model_architecture", "RBF SVM with StandardScaler")
    logger.info("Model: %s | Clinical Threshold: %.2f", model_name, threshold)

    # 3. Explainer Selection Justification
    selection_reasoning = get_explainer_selection_reasoning(arch_desc)
    logger.info("Explainer selected: %s", selection_reasoning["selected_explainer"])

    # 4. Global SHAP Analysis
    # Sample 100 representative observations from training set for global evaluation
    rng = np.random.RandomState(RANDOM_SEED)
    eval_indices = rng.choice(len(split.X_train), size=min(100, len(split.X_train)), replace=False)
    X_eval = np.asarray(split.X_train)[eval_indices]

    logger.info("Computing global SHAP values across %d samples...", len(X_eval))
    global_exp = compute_global_explanations(
        explainer=explainer,
        X_eval=X_eval,
        feature_names=feature_names,
        nsamples=100,
        model_name=model_name,
    )

    logger.info("Top 10 Globally Influential Features (Mean |SHAP|):")
    for rank, feat in enumerate(global_exp.top_10_features, 1):
        score = global_exp.mean_abs_shap[feat]
        logger.info("  %2d. %-26s : %.4f", rank, feat, score)

    # 5. Generate and Save Global Visualizations
    SHAP_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    p_beeswarm = plot_shap_summary_beeswarm(global_exp)
    p_importance = plot_shap_global_importance_bar(global_exp)
    p_bar = plot_shap_bar_plot(global_exp)
    p_dependence = plot_shap_dependence(global_exp, feature_name=global_exp.top_10_features[0])

    logger.info("Global plots saved to: %s", SHAP_FIGURES_DIR)

    # 6. Local Explainability on Exemplary Samples from Test Partition
    # Select a true malignant sample and a true benign sample from test data
    y_test_arr = np.asarray(split.y_test)
    test_mal_idx = int(np.where(y_test_arr == 1)[0][0])
    test_ben_idx = int(np.where(y_test_arr == 0)[0][0])

    sample_mal = np.asarray(split.X_test)[test_mal_idx]
    sample_ben = np.asarray(split.X_test)[test_ben_idx]

    logger.info("Computing local explanation for Malignant test patient (idx=%d)...", test_mal_idx)
    local_mal = explain_single_sample(
        pipeline=manager.pipeline,
        explainer=explainer,
        sample=sample_mal,
        feature_names=feature_names,
        threshold=threshold,
        sample_index=test_mal_idx,
        nsamples=150,
    )

    logger.info("Computing local explanation for Benign test patient (idx=%d)...", test_ben_idx)
    local_ben = explain_single_sample(
        pipeline=manager.pipeline,
        explainer=explainer,
        sample=sample_ben,
        feature_names=feature_names,
        threshold=threshold,
        sample_index=test_ben_idx,
        nsamples=150,
    )

    # Generate Local Visualizations
    p_waterfall = plot_shap_waterfall(local_mal)
    p_local_bar = plot_shap_local_contributions_bar(local_mal)

    logger.info("Local plots saved to: %s", SHAP_FIGURES_DIR)

    # 7. Model Explanation Validation
    all_shap = global_exp.shap_values
    is_finite = bool(np.all(np.isfinite(all_shap)))
    dim_match = (all_shap.shape == (len(X_eval), len(feature_names)))
    additivity_verified = (local_mal.additivity_gap <= 0.05 and local_ben.additivity_gap <= 0.05)

    validation_summary = {
        "shap_values_finite": is_finite,
        "dimension_match": dim_match,
        "feature_count": len(feature_names),
        "evaluation_samples": len(X_eval),
        "target_class_explained": "Class 1 (Malignant)",
        "base_expected_value": round(global_exp.base_value, 4),
        "malignant_sample_additivity_gap": round(local_mal.additivity_gap, 4),
        "benign_sample_additivity_gap": round(local_ben.additivity_gap, 4),
        "additivity_verified_within_tolerance": additivity_verified,
    }

    logger.info("Validation Summary: %s", validation_summary)

    # 8. Export Structured JSON Artifact
    summary_artifact = {
        "model_architecture": arch_desc,
        "explainer_selection": selection_reasoning,
        "validation_summary": validation_summary,
        "global_importance_ranking": [
            {"rank": i + 1, "feature": f, "mean_abs_shap": round(global_exp.mean_abs_shap[f], 5)}
            for i, f in enumerate(global_exp.top_10_features)
        ],
        "positive_driver_features": global_exp.positive_driver_features,
        "negative_driver_features": global_exp.negative_driver_features,
        "example_local_malignant": {
            "predicted_label": local_mal.predicted_label,
            "prob_malignant": local_mal.prob_malignant,
            "prob_benign": local_mal.prob_benign,
            "threshold": local_mal.threshold_applied,
            "base_value": local_mal.base_value,
            "top_positive_contributions": local_mal.top_positive_contributions,
            "top_negative_contributions": local_mal.top_negative_contributions,
            "narrative_explanation": local_mal.narrative_explanation,
        },
        "example_local_benign": {
            "predicted_label": local_ben.predicted_label,
            "prob_malignant": local_ben.prob_malignant,
            "prob_benign": local_ben.prob_benign,
            "threshold": local_ben.threshold_applied,
            "base_value": local_ben.base_value,
            "top_positive_contributions": local_ben.top_positive_contributions,
            "top_negative_contributions": local_ben.top_negative_contributions,
            "narrative_explanation": local_ben.narrative_explanation,
        },
        "figure_paths": {
            "beeswarm": str(p_beeswarm),
            "global_importance": str(p_importance),
            "bar_plot": str(p_bar),
            "dependence": str(p_dependence),
            "waterfall": str(p_waterfall),
            "local_contribution": str(p_local_bar),
        },
    }

    EVALUATION_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = EVALUATION_DIR / "shap_analysis_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_artifact, f, indent=2)
    logger.info("Saved SHAP analysis summary to: %s", summary_path)

    logger.info("=" * 80)
    logger.info("SHAP Analysis Pipeline Completed Successfully!")
    logger.info("=" * 80)

    return summary_artifact


if __name__ == "__main__":
    run_shap_pipeline()
