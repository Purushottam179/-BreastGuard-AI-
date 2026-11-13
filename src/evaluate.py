"""
BreastGuard AI — Model Evaluation & Clinical Diagnostics Suite
================================================================================
Evaluates trained models on the held-out test partition (20% of dataset).
Ensures zero data leakage: test set is evaluated strictly AFTER model training and
selection is completed.

Clinical Confusion Matrix interpretation (0 = Benign, 1 = Malignant):
  - True Positive (TP) : Malignant correctly identified.
  - True Negative (TN) : Benign correctly identified.
  - False Positive (FP): Benign incorrectly classified as malignant (biopsy anxiety).
  - False Negative (FN): Malignant incorrectly classified as benign (missed cancer).

Generates:
  - Clinical metrics: Accuracy, Precision, Malignant-class recall (Sensitivity),
    Benign-class recall (Specificity), F1-Score, ROC-AUC.
  - Visualizations: Confusion Matrix heatmap, ROC Curve, Precision-Recall Curve.
  - Comprehensive classification reports saved to reports/evaluation/.
================================================================================
"""
from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.pipeline import Pipeline

from config.config import (
    CLASS_LABELS,
    CLASS_NAMES,
    EVALUATION_DIR,
    FIGURES_DIR,
    LABEL_BENIGN,
    LABEL_MALIGNANT,
)
from src.preprocessing import SplitData

logger = logging.getLogger(__name__)


@dataclass
class ClinicalConfusionMatrix:
    """Explicitly mapped 2x2 clinical confusion matrix (0=Benign, 1=Malignant)."""
    true_positive: int    # TP: Malignant correctly identified
    true_negative: int    # TN: Benign correctly identified
    false_positive: int   # FP: Benign incorrectly classified as malignant
    false_negative: int   # FN: Malignant incorrectly classified as benign
    raw_matrix: list[list[int]]


@dataclass
class TestEvaluationReport:
    """Complete evaluation report for a model evaluated on held-out test partition."""
    __test__ = False  # Instruct pytest not to treat this dataclass as a test case
    model_name: str
    accuracy: float
    precision: float
    malignant_recall: float  # Sensitivity (Recall of class 1)
    benign_recall: float     # Specificity (Recall of class 0)
    f1_score: float          # F1 score for Malignant class
    macro_f1: float          # Macro-averaged F1 across both classes
    roc_auc: float           # Area under ROC curve
    brier_score: float       # Probability calibration error
    log_loss_value: float    # Cross-entropy loss
    confusion_matrix: ClinicalConfusionMatrix
    classification_report_str: str
    classification_report_dict: dict[str, Any]


def evaluate_on_test_set(
    model_name: str,
    pipeline: Pipeline,
    split: SplitData,
    generate_plots: bool = True,
    output_eval_dir: Path | None = None,
    output_fig_dir: Path | None = None,
) -> TestEvaluationReport:
    """
    Perform rigorous evaluation of the selected model on the held-out test partition.

    Parameters
    ----------
    model_name : str
    pipeline : Pipeline
        Fitted Scikit-learn Pipeline.
    split : SplitData
        Contains X_test and y_test.
    generate_plots : bool, default=True
        Whether to generate and save Confusion Matrix, ROC, and PR curves.
    output_eval_dir : Path, optional
    output_fig_dir : Path, optional

    Returns
    -------
    TestEvaluationReport
    """
    if output_eval_dir is None:
        EVALUATION_DIR.mkdir(parents=True, exist_ok=True)
        output_eval_dir = EVALUATION_DIR

    if output_fig_dir is None:
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        output_fig_dir = FIGURES_DIR

    logger.info("Evaluating selected model '%s' on held-out test set (%d samples) ...", model_name, split.test_size)

    y_test = split.y_test
    X_test = split.X_test

    y_pred = pipeline.predict(X_test)

    # Probabilities for positive class (1 = Malignant)
    if hasattr(pipeline, "predict_proba"):
        probs = pipeline.predict_proba(X_test)
        p_malignant = probs[:, 1]
        p_benign = probs[:, 0]
    elif hasattr(pipeline, "decision_function"):
        decision = pipeline.decision_function(X_test)
        # Min-max scale decision function as pseudo-probabilities if predict_proba is unavailable
        p_malignant = (decision - decision.min()) / (decision.max() - decision.min() + 1e-9)
        p_benign = 1.0 - p_malignant
        probs = np.column_stack([p_benign, p_malignant])
    else:
        p_malignant = y_pred.astype(float)
        p_benign = 1.0 - p_malignant
        probs = np.column_stack([p_benign, p_malignant])

    # Confusion matrix with standard ordering: rows=Actual, cols=Predicted
    # labels=[0, 1] -> Row 0: Actual Benign (0), Row 1: Actual Malignant (1)
    cm = confusion_matrix(y_test, y_pred, labels=[LABEL_BENIGN, LABEL_MALIGNANT])
    tn = int(cm[0, 0])
    fp = int(cm[0, 1])
    fn = int(cm[1, 0])
    tp = int(cm[1, 1])

    clin_cm = ClinicalConfusionMatrix(
        true_positive=tp,
        true_negative=tn,
        false_positive=fp,
        false_negative=fn,
        raw_matrix=[[tn, fp], [fn, tp]],
    )

    # Metrics
    acc = float(accuracy_score(y_test, y_pred))
    prec = float(precision_score(y_test, y_pred, pos_label=LABEL_MALIGNANT, zero_division=0))
    sensitivity = float(recall_score(y_test, y_pred, pos_label=LABEL_MALIGNANT, zero_division=0))
    specificity = float(recall_score(y_test, y_pred, pos_label=LABEL_BENIGN, zero_division=0))
    f1 = float(f1_score(y_test, y_pred, pos_label=LABEL_MALIGNANT, zero_division=0))
    macro_f1 = float(f1_score(y_test, y_pred, average="macro", zero_division=0))
    auc_score = float(roc_auc_score(y_test, p_malignant))
    brier = float(brier_score_loss(y_test, p_malignant))
    logloss = float(log_loss(y_test, probs))

    # Classification report
    cr_dict = classification_report(
        y_test, y_pred,
        target_names=["Benign", "Malignant"],
        output_dict=True,
    )
    cr_str = classification_report(
        y_test, y_pred,
        target_names=["Benign", "Malignant"],
        digits=4,
    )

    report = TestEvaluationReport(
        model_name=model_name,
        accuracy=acc,
        precision=prec,
        malignant_recall=sensitivity,
        benign_recall=specificity,
        f1_score=f1,
        macro_f1=macro_f1,
        roc_auc=auc_score,
        brier_score=brier,
        log_loss_value=logloss,
        confusion_matrix=clin_cm,
        classification_report_str=cr_str,
        classification_report_dict=cr_dict,
    )

    logger.info(
        "Held-Out Test Results [%s]: Accuracy=%.4f | Precision=%.4f | Sensitivity=%.4f | Specificity=%.4f | ROC-AUC=%.4f",
        model_name, acc, prec, sensitivity, specificity, auc_score,
    )
    logger.info("Confusion Matrix: TP=%d, TN=%d, FP=%d, FN=%d", tp, tn, fp, fn)

    # Save classification report text and json
    cr_txt_path = output_eval_dir / "classification_report.txt"
    with open(cr_txt_path, "w") as f:
        f.write(f"BreastGuard AI — Classification Report for Selected Model: {model_name}\n")
        f.write("Held-Out Test Set Evaluation (114 samples, 20% split)\n")
        f.write("=" * 65 + "\n\n")
        f.write(cr_str + "\n\n")
        f.write("Clinical Summary Metrics:\n")
        f.write(f"  Accuracy           : {acc:.4f}\n")
        f.write(f"  Precision          : {prec:.4f}\n")
        f.write(f"  Sensitivity (Recall): {sensitivity:.4f}\n")
        f.write(f"  Specificity        : {specificity:.4f}\n")
        f.write(f"  F1-Score           : {f1:.4f}\n")
        f.write(f"  ROC-AUC            : {auc_score:.4f}\n")
        f.write(f"  True Positives (TP): {tp} (Malignant correctly identified)\n")
        f.write(f"  True Negatives (TN): {tn} (Benign correctly identified)\n")
        f.write(f"  False Positives(FP): {fp} (Benign classified as Malignant)\n")
        f.write(f"  False Negatives(FN): {fn} (Malignant classified as Benign - MISSED)\n")
    logger.info("Saved text classification report: %s", cr_txt_path)

    # Save JSON summary
    summary_json_path = output_eval_dir / "test_evaluation_selected_model.json"
    json_data = {
        "model_name": model_name,
        "accuracy": acc,
        "precision": prec,
        "malignant_recall_sensitivity": sensitivity,
        "benign_recall_specificity": specificity,
        "f1_score": f1,
        "macro_f1": macro_f1,
        "roc_auc": auc_score,
        "brier_score": brier,
        "log_loss": logloss,
        "confusion_matrix": asdict(clin_cm),
        "classification_report": cr_dict,
    }
    with open(summary_json_path, "w") as f:
        json.dump(json_data, f, indent=2)
    logger.info("Saved test evaluation JSON: %s", summary_json_path)

    if generate_plots:
        _plot_test_figures(y_test, y_pred, p_malignant, clin_cm, model_name, auc_score, output_fig_dir)

    return report


def _plot_test_figures(
    y_test: np.ndarray,
    y_pred: np.ndarray,
    p_malignant: np.ndarray,
    clin_cm: ClinicalConfusionMatrix,
    model_name: str,
    auc_score: float,
    output_dir: Path,
) -> None:
    """Generate and save publication-grade test figures."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Confusion Matrix Plot
    plt.figure(figsize=(7, 6), dpi=300)
    cm_matrix = np.array([
        [clin_cm.true_negative, clin_cm.false_positive],
        [clin_cm.false_negative, clin_cm.true_positive],
    ])

    labels = np.array([
        [f"TN\n{clin_cm.true_negative}\n(Benign Correct)", f"FP\n{clin_cm.false_positive}\n(False Alarm)"],
        [f"FN\n{clin_cm.false_negative}\n(Missed Cancer)", f"TP\n{clin_cm.true_positive}\n(Malignant Correct)"],
    ])

    sns.heatmap(
        cm_matrix,
        annot=labels,
        fmt="",
        cmap="Blues",
        cbar=False,
        xticklabels=["Predicted Benign", "Predicted Malignant"],
        yticklabels=["Actual Benign", "Actual Malignant"],
        annot_kws={"size": 11, "weight": "bold"},
        linewidths=1.5,
        linecolor="#E2E8F0",
    )
    plt.title(f"Clinical Confusion Matrix — {model_name}\nHeld-Out Test Partition (N=114)", fontsize=13, fontweight="bold", pad=12)
    plt.tight_layout()
    cm_chart_path = output_dir / "confusion_matrix_test.png"
    plt.savefig(cm_chart_path, dpi=300)
    plt.close()
    logger.info("Saved: %s", cm_chart_path)

    # 2. ROC Curve Plot
    fpr, tpr, _ = roc_curve(y_test, p_malignant, pos_label=LABEL_MALIGNANT)
    plt.figure(figsize=(7, 6), dpi=300)
    plt.plot(fpr, tpr, color="#4C6EF5", lw=2.5, label=f"{model_name} (AUC = {auc_score:.4f})")
    plt.plot([0, 1], [0, 1], color="#868E96", lw=1.5, linestyle="--", label="Random Chance (AUC = 0.5000)")
    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.05])
    plt.xlabel("False Positive Rate (1 - Specificity)", fontsize=11, labelpad=8)
    plt.ylabel("True Positive Rate (Sensitivity / Recall)", fontsize=11, labelpad=8)
    plt.title(f"ROC Curve — {model_name}\nHeld-Out Test Partition", fontsize=13, fontweight="bold", pad=12)
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    roc_chart_path = output_dir / "roc_curve_test.png"
    plt.savefig(roc_chart_path, dpi=300)
    plt.close()
    logger.info("Saved: %s", roc_chart_path)

    # 3. Precision-Recall Curve Plot
    prec_curve, rec_curve, _ = precision_recall_curve(y_test, p_malignant, pos_label=LABEL_MALIGNANT)
    plt.figure(figsize=(7, 6), dpi=300)
    plt.plot(rec_curve, prec_curve, color="#12B886", lw=2.5, label=f"{model_name}")
    prevalence = (y_test == 1).mean()
    plt.axhline(prevalence, color="#FA5252", linestyle="--", lw=1.5, label=f"Baseline Prevalence ({prevalence:.1%})")
    plt.xlim([-0.02, 1.02])
    plt.ylim([max(0.0, prevalence - 0.1), 1.05])
    plt.xlabel("Malignant Recall (Sensitivity)", fontsize=11, labelpad=8)
    plt.ylabel("Precision (Positive Predictive Value)", fontsize=11, labelpad=8)
    plt.title(f"Precision-Recall Curve — {model_name}\nHeld-Out Test Partition", fontsize=13, fontweight="bold", pad=12)
    plt.legend(loc="lower left", fontsize=10)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    pr_chart_path = output_dir / "precision_recall_curve_test.png"
    plt.savefig(pr_chart_path, dpi=300)
    plt.close()
    logger.info("Saved: %s", pr_chart_path)


# Compatibility aliases for package exports
EvaluationReport = TestEvaluationReport
evaluate_model = evaluate_on_test_set


def evaluate_all_models(
    models_dict: dict[str, Any],
    split: SplitData,
) -> dict[str, TestEvaluationReport]:
    """Evaluate multiple models on the held-out test split."""
    reports = {}
    for name, item in models_dict.items():
        pipe = getattr(item, "pipeline", item)
        reports[name] = evaluate_on_test_set(name, pipe, split, generate_plots=False)
    return reports
