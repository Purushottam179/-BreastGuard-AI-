"""
BreastGuard AI — Machine Learning Pipeline Orchestration Script
================================================================================
Executes the end-to-end training and evaluation workflow:
  1. Load & validate data from data/processed/breast_cancer_cleaned.csv
  2. Stratified train/test splitting (80/20, seed=42)
  3. Preprocessing pipelines with and without scaling
  4. 5-Fold Stratified Cross-Validation for 5 candidate algorithms
  5. Cross-validation comparisons and visualization generation
  6. Model fitting and serialization to models/*.pkl
  7. Principled model selection using training CV only
  8. Preliminary test evaluation of selected model on held-out test split
================================================================================
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import load_and_validate_processed_data
from src.evaluate import evaluate_on_test_set
from src.preprocessing import create_stratified_split, export_processed_splits
from src.train import (
    export_cv_results_table,
    get_model_specifications,
    plot_cv_comparisons,
    run_cross_validation,
    select_best_model_from_cv,
    train_and_save_all_models,
)
from src.utils import setup_logger

logger = setup_logger("training_pipeline")


def main() -> None:
    logger.info("=" * 70)
    logger.info("BreastGuard AI — Machine Learning Training & Evaluation Pipeline")
    logger.info("=" * 70)

    # ─────────────────────────────────────────────────────────────────────────
    # PART 1: Load and Validate Data
    # ─────────────────────────────────────────────────────────────────────────
    logger.info("[PART 1] Loading and validating processed dataset ...")
    X, y, feature_names = load_and_validate_processed_data()

    # ─────────────────────────────────────────────────────────────────────────
    # PART 2 & 3: Stratified Splitting & Preprocessing
    # ─────────────────────────────────────────────────────────────────────────
    logger.info("[PART 2 & 3] Performing stratified 80/20 train-test split ...")
    split = create_stratified_split(X, y, test_size=0.20, random_seed=42, feature_names=feature_names)
    export_processed_splits(split)

    # ─────────────────────────────────────────────────────────────────────────
    # PART 4: Candidate Model Specifications
    # ─────────────────────────────────────────────────────────────────────────
    logger.info("[PART 4] Configuring candidate model specifications ...")
    model_configs = get_model_specifications()
    for cfg in model_configs:
        logger.info("  Configured: %-26s | Scaling: %-5s | File: %s", cfg.name, str(cfg.use_scaler), cfg.filename)

    # ─────────────────────────────────────────────────────────────────────────
    # PART 5: 5-Fold Stratified Cross-Validation
    # ─────────────────────────────────────────────────────────────────────────
    logger.info("[PART 5] Executing 5-Fold Stratified Cross-Validation on training data ...")
    cv_results = run_cross_validation(model_configs, split)

    # Export CV table to reports/evaluation/cross_validation_results.csv
    cv_df = export_cv_results_table(cv_results)

    # ─────────────────────────────────────────────────────────────────────────
    # PART 6: Train & Serialize All 5 Models
    # ─────────────────────────────────────────────────────────────────────────
    logger.info("[PART 6] Fitting all 5 candidate pipelines on full training set ...")
    trained_bundles = train_and_save_all_models(model_configs, split, cv_results)

    # ─────────────────────────────────────────────────────────────────────────
    # PART 7: Model Comparison & Visualizations
    # ─────────────────────────────────────────────────────────────────────────
    logger.info("[PART 7] Generating cross-validation comparison charts ...")
    plot_cv_comparisons(cv_results)

    # Select best model based strictly on CV results
    selected_name, justification = select_best_model_from_cv(cv_results)
    selected_bundle = trained_bundles[selected_name]

    # ─────────────────────────────────────────────────────────────────────────
    # PART 8: Preliminary Evaluation on Held-Out Test Set
    # ─────────────────────────────────────────────────────────────────────────
    logger.info("[PART 8] Evaluating selected model '%s' on held-out test partition ...", selected_name)
    test_report = evaluate_on_test_set(
        model_name=selected_name,
        pipeline=selected_bundle.pipeline,
        split=split,
        generate_plots=True,
    )

    # ─────────────────────────────────────────────────────────────────────────
    # Summary Output
    # ─────────────────────────────────────────────────────────────────────────
    sep = "=" * 70
    print("\n" + sep)
    print("BREASTGUARD AI — ML TRAINING & EVALUATION PIPELINE COMPLETED")
    print(sep)
    print(f"Dataset             : 569 samples (Train: {split.train_size}, Test: {split.test_size})")
    print(f"Features            : {split.n_features} numerical cytological measurements")
    print(f"Candidate Models    : {len(model_configs)} algorithms trained and evaluated")
    print("\nCross-Validation Comparison Table (5-Fold Stratified CV on Training Split):")
    print("-" * 70)
    summary_cols = ["Model", "accuracy_mean", "accuracy_std", "recall_mean", "recall_std", "f1_mean", "roc_auc_mean"]
    print(cv_df[summary_cols].to_string(index=False))

    print("\n" + "-" * 70)
    print(f"SELECTED MODEL      : {selected_name}")
    print(f"SELECTION REASON    : {justification}")
    print("-" * 70)

    print("\nHELD-OUT TEST EVALUATION RESULTS (114 samples, 20% untouched partition):")
    print(f"  Accuracy           : {test_report.accuracy:.4f}")
    print(f"  Precision          : {test_report.precision:.4f}")
    print(f"  Sensitivity (Recall): {test_report.malignant_recall:.4f} (Malignant correctly detected)")
    print(f"  Specificity        : {test_report.benign_recall:.4f} (Benign correctly cleared)")
    print(f"  F1-Score           : {test_report.f1_score:.4f}")
    print(f"  ROC-AUC            : {test_report.roc_auc:.4f}")
    print(f"  Brier Score        : {test_report.brier_score:.4f}")
    print("\nConfusion Matrix (Test Set):")
    cm = test_report.confusion_matrix
    print(f"  True Positive (TP) : {cm.true_positive:2d} (Malignant identified)")
    print(f"  True Negative (TN) : {cm.true_negative:2d} (Benign identified)")
    print(f"  False Positive (FP): {cm.false_positive:2d} (False Alarm)")
    print(f"  False Negative (FN): {cm.false_negative:2d} (Missed Malignancy)")
    print(sep + "\n")


if __name__ == "__main__":
    main()

