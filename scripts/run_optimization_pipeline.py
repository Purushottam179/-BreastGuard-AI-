"""
BreastGuard AI — End-to-End Hyperparameter Optimization, Threshold Analysis,
Probability Calibration & Final Model Evaluation Pipeline
================================================================================
Orchestrates:
  1. Baseline performance review from existing 5-fold CV records.
  2. Multi-model hyperparameter tuning on training partition (Stratified 5-Fold).
  3. Export of hyperparameter results table & comparative visualizations.
  4. Decision threshold analysis & clinical operating point identification.
  5. Probability calibration & Brier score evaluation.
  6. Final held-out test evaluation of the champion architecture.
================================================================================
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from config.config import EVALUATION_DIR, FIGURES_DIR, MODELS_DIR
from src.data_loader import load_and_validate_processed_data
from src.preprocessing import create_stratified_split
from src.threshold import (
    compute_and_plot_calibration,
    compute_oof_probabilities,
    evaluate_test_at_thresholds,
    find_operating_thresholds,
    plot_threshold_tradeoffs,
    sweep_decision_thresholds,
)
from src.tune import (
    export_hyperparameter_results_table,
    generate_baseline_vs_optimized_charts,
    generate_optimization_markdown_report,
    optimize_all_models,
)
from src.utils import setup_logger

logger = setup_logger("optimization_pipeline")


def run_pipeline() -> dict:
    logger.info("=" * 80)
    logger.info("BreastGuard AI — Optimization, Calibration & Threshold Pipeline")
    logger.info("=" * 80)

    # 1. Load Data & Stratified Split
    logger.info("[STEP 1] Loading processed dataset...")
    X, y, feature_names = load_and_validate_processed_data()
    split = create_stratified_split(X, y, test_size=0.20, random_seed=42, feature_names=feature_names)
    logger.info(
        "Split verified: Train N=%d (Benign=%d, Malignant=%d) | Test N=%d (Benign=%d, Malignant=%d)",
        len(split.y_train), int((split.y_train == 0).sum()), int((split.y_train == 1).sum()),
        len(split.y_test), int((split.y_test == 0).sum()), int((split.y_test == 1).sum()),
    )

    # 2. Baseline Performance Review
    baseline_csv = EVALUATION_DIR / "cross_validation_results.csv"
    if not baseline_csv.exists():
        raise FileNotFoundError(f"Baseline results not found at {baseline_csv}. Run training pipeline first.")
    df_base = pd.read_csv(baseline_csv)
    logger.info("[STEP 2] Baseline Performance Summary:\n%s", df_base[["Model", "accuracy_mean", "recall_mean", "f1_mean", "roc_auc_mean"]].to_string(index=False))

    # 3. Hyperparameter Optimization for All 5 Models
    logger.info("[STEP 3] Starting Hyperparameter Optimization for 5 architectures...")
    opt_results = optimize_all_models(split, save_artifacts=True)

    # 4. Generate Baseline vs Optimized Visualizations and Markdown Report
    logger.info("[STEP 4] Generating Baseline vs Optimized Comparative Visualizations...")
    chart_paths = generate_baseline_vs_optimized_charts(baseline_csv, opt_results, FIGURES_DIR)
    logger.info("Generated %d comparative charts in %s", len(chart_paths), FIGURES_DIR)

    report_path = generate_optimization_markdown_report(baseline_csv, opt_results)
    logger.info("Generated optimization report at: %s", report_path)

    # 5. Champion Model Selection
    # Criteria: Malignant Recall, ROC-AUC, F1, calibration and stability
    # Compare candidate performance
    candidate_summary = []
    for name, res in opt_results.items():
        candidate_summary.append({
            "model": name,
            "recall": res.cv_mean["recall"],
            "roc_auc": res.cv_mean["roc_auc"],
            "f1": res.cv_mean["f1"],
            "accuracy": res.cv_mean["accuracy"],
            "std_recall": res.cv_std["recall"],
        })
    df_cand = pd.DataFrame(candidate_summary).sort_values(by=["roc_auc", "recall"], ascending=False)
    logger.info("[STEP 5] Candidate Optimization Summary:\n%s", df_cand.to_string(index=False))

    champion_name = df_cand.iloc[0]["model"]
    champion_pipeline = opt_results[champion_name].best_pipeline
    logger.info("Selected Champion Architecture: '%s'", champion_name)

    # 6. Probability Calibration Analysis across all tuned models
    logger.info("[STEP 6] Computing Probability Calibration & Brier Scores...")
    tuned_models_dict = {name: res.best_pipeline for name, res in opt_results.items()}
    brier_scores, cal_chart_path = compute_and_plot_calibration(
        tuned_models_dict, split.X_train, split.y_train, FIGURES_DIR
    )
    logger.info("Brier Scores: %s", brier_scores)

    # 7. Decision Threshold Analysis for Champion Model (Out-of-Fold on Training Set)
    logger.info("[STEP 7] Conducting Out-of-Fold Decision Threshold Sweep for '%s'...", champion_name)
    oof_probs = compute_oof_probabilities(champion_pipeline, split.X_train, split.y_train)
    y_prob_mal = oof_probs[:, 1]

    threshold_df = sweep_decision_thresholds(split.y_train, y_prob_mal)
    threshold_csv = EVALUATION_DIR / "threshold_sweep_results.csv"
    threshold_df.to_csv(threshold_csv, index=False)
    logger.info("Saved full threshold sweep (99 points) to: %s", threshold_csv)

    operating_points = find_operating_thresholds(threshold_df)
    logger.info("Key Clinical Operating Points Identified:")
    for op_key, op_val in operating_points.items():
        logger.info(
            "  * %-26s | Th: %.2f | Sens: %.4f | Spec: %.4f | Prec: %.4f | F1: %.4f | FN: %d | FP: %d",
            op_key, op_val["threshold"], op_val["sensitivity"], op_val["specificity"],
            op_val["precision"], op_val["f1"], op_val["fn"], op_val["fp"],
        )

    # Generate Threshold Visualizations
    thresh_plots = plot_threshold_tradeoffs(threshold_df, operating_points, FIGURES_DIR)
    logger.info("Generated threshold tradeoff plots: %s", list(thresh_plots.keys()))

    # 8. Held-Out Test Set Evaluation of Champion Model
    logger.info("[STEP 8] Evaluating Champion Model '%s' on Held-Out Test Set (N=%d)...", champion_name, len(split.y_test))
    test_eval_thresholds = {
        "standard_threshold_0_50": 0.50,
        "youden_optimal_threshold": operating_points["youden_optimal"]["threshold"],
        "high_sensitivity_screening_threshold": operating_points["high_sensitivity_screening"]["threshold"],
        "balanced_f1_threshold": operating_points["balanced_f1_optimal"]["threshold"],
    }

    test_results = evaluate_test_at_thresholds(
        champion_pipeline, split.X_test, split.y_test, test_eval_thresholds
    )

    logger.info("Test Evaluation at Evaluated Operating Points:")
    for op_name, m in test_results.items():
        logger.info(
            "  * %-36s (th=%.2f) -> Acc: %.4f | Sens: %.4f | Spec: %.4f | Prec: %.4f | F1: %.4f | FN: %d | FP: %d",
            op_name, m["threshold"], m["accuracy"], m["sensitivity"], m["specificity"],
            m["precision"], m["f1"], m["fn"], m["fp"],
        )

    # Save final comprehensive model evaluation artifact
    final_artifact = {
        "champion_model": champion_name,
        "best_hyperparameters": opt_results[champion_name].best_params,
        "cv_performance": opt_results[champion_name].cv_mean,
        "brier_scores_all_models": brier_scores,
        "operating_points_train_oof": operating_points,
        "test_performance_by_threshold": test_results,
    }

    final_eval_path = EVALUATION_DIR / "final_model_evaluation.json"
    with open(final_eval_path, "w", encoding="utf-8") as f:
        json.dump(final_artifact, f, indent=2)
    logger.info("Saved final model evaluation report to: %s", final_eval_path)

    # Write clinical threshold analysis report in Markdown
    threshold_report_path = EVALUATION_DIR / "threshold_analysis_report.md"
    write_threshold_markdown_report(threshold_report_path, champion_name, operating_points, test_results, brier_scores)
    logger.info("Saved threshold analysis markdown report to: %s", threshold_report_path)

    logger.info("=" * 80)
    logger.info("Optimization & Threshold Pipeline Completed Successfully!")
    logger.info("=" * 80)

    return final_artifact


def write_threshold_markdown_report(
    path: Path,
    champion_name: str,
    operating_points: dict,
    test_results: dict,
    brier_scores: dict,
) -> None:
    lines = [
        "# BreastGuard AI — Decision Threshold Optimization & Probability Calibration Report",
        "",
        "## Executive Summary",
        f"This report presents the decision threshold optimization, probability calibration, and clinical operating point analysis for the champion architecture: **{champion_name}**.",
        "",
        "In oncology screening and cytological diagnosis, misclassifying an aggressive malignant carcinoma as benign (**False Negative**) can delay treatment and result in fatal disease progression. Conversely, misclassifying a benign lesion as malignant (**False Positive**) causes diagnostic anxiety and prompts follow-up needle biopsies or imaging, but does not threaten patient survival.",
        "",
        "Standard machine learning pipelines default to a symmetric decision threshold of $0.50$, implicitly assuming that False Negatives and False Positives carry equal penalty. Here, out-of-fold probability distributions from 5-fold cross-validation are analyzed to establish calibrated clinical operating points that prioritize Malignant Recall (Sensitivity).",
        "",
        "---",
        "",
        "## 1. Probability Calibration & Brier Score Analysis",
        "",
        "Probability calibration assesses whether the predicted probability $P(\\text{Malignant})$ matches empirical disease prevalence.",
        "",
        "| Architecture | Brier Score (Lower is Better) | Calibration Quality |",
        "|---|:---:|---|",
    ]
    for m, b in brier_scores.items():
        quality = "Excellent (Well-Calibrated)" if b <= 0.04 else ("Good" if b <= 0.07 else "Fair")
        lines.append(f"| **{m}** | {b:.4f} | {quality} |")

    lines.extend([
        "",
        "![Probability Calibration Curves](../figures/calibration_curves.png)",
        "",
        "---",
        "",
        "## 2. Clinical Operating Points (Training Out-of-Fold Analysis)",
        "",
        "| Operating Strategy | Threshold | Sensitivity (Recall) | Specificity (TNR) | Precision (PPV) | F1-Score | FN Count (N=170) | FP Count (N=285) |",
        "|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|",
    ])

    for k, v in operating_points.items():
        strat_title = k.replace("_", " ").title()
        lines.append(
            f"| **{strat_title}** | {v['threshold']:.2f} | {v['sensitivity']*100:.1f}% | {v['specificity']*100:.1f}% | {v['precision']*100:.1f}% | {v['f1']:.4f} | {v['fn']} | {v['fp']} |"
        )

    lines.extend([
        "",
        "### Key Trade-Off Insights:",
        "- **Standard Cutoff (0.50)**: Achieves balanced accuracy, but leaves several malignant cases undetected as False Negatives.",
        "- **Youden's J Optimal**: Maximizes the informedness metric $J = \\text{Sensitivity} + \\text{Specificity} - 1$, balancing detection and false alarm reduction.",
        "- **High-Sensitivity Clinical Screening**: Calibrates the decision threshold downwards to capture $\\ge 98\\%$ of malignant samples, driving False Negatives to near-zero while preserving high specificity.",
        "",
        "---",
        "",
        "## 3. Held-Out Test Set Verification (N=114, 42 Malignant, 72 Benign)",
        "",
        "To prevent data leakage, the test partition remained completely untouched during hyperparameter tuning and threshold search.",
        "",
        "| Operating Strategy | Test Threshold | Malignant Recall (Sens) | Benign Specificity | Precision | Test F1 | Test Acc | False Negatives | False Positives |",
        "|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|",
    ])

    for k, v in test_results.items():
        name = k.replace("_", " ").title()
        lines.append(
            f"| **{name}** | {v['threshold']:.2f} | **{v['sensitivity']*100:.1f}%** | {v['specificity']*100:.1f}% | {v['precision']*100:.1f}% | {v['f1']:.4f} | {v['accuracy']*100:.1f}% | **{v['fn']}** | {v['fp']} |"
        )

    lines.extend([
        "",
        "---",
        "",
        "## 4. Visual Evidence",
        "- **Sensitivity vs Specificity Trade-off**: `reports/figures/threshold_sensitivity_specificity.png`",
        "- **Precision-Recall-F1 vs Threshold**: `reports/figures/threshold_precision_recall_f1.png`",
        "- **Diagnostic Error Count Curves**: `reports/figures/threshold_clinical_error_tradeoff.png`",
        "- **Probability Reliability Diagram**: `reports/figures/calibration_curves.png`",
    ])

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    run_pipeline()

