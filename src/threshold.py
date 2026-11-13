"""
BreastGuard AI — Decision Threshold Analysis & Probability Calibration Module
================================================================================
Implements rigorous decision threshold optimization and probability calibration
for clinical breast cancer diagnostic decision support:

1. Out-of-fold (OOF) cross-validation probability generation on training data.
2. Full threshold sweep from 0.01 to 0.99 evaluating:
   - Sensitivity (Malignant Recall — minimizing fatal False Negatives)
   - Specificity (Benign True Negative rate — avoiding unnecessary biopsies)
   - Precision (Positive Predictive Value)
   - F1-Score
   - Youden's J Statistic (Sensitivity + Specificity - 1)
3. Identification of key clinical operating points:
   - Standard Diagnostic Threshold (0.50)
   - Youden's J Optimal Operating Point
   - High-Sensitivity Clinical Screening Operating Point (Recall >= 98%)
   - Maximum F1 Operating Point
4. Probability Calibration Analysis:
   - Reliability diagrams (calibration curves)
   - Brier Score calculation across all models
5. Final held-out test set evaluation at default (0.50) and tuned clinical thresholds.
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
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline

from config.config import (
    CLASS_LABELS,
    CV_FOLDS,
    EVALUATION_DIR,
    FIGURES_DIR,
    LABEL_BENIGN,
    LABEL_MALIGNANT,
    RANDOM_SEED,
)

logger = logging.getLogger(__name__)


@dataclass
class ThresholdMetrics:
    """Clinical diagnostic performance metrics at a specific decision threshold."""
    threshold: float
    tp: int
    fp: int
    tn: int
    fn: int
    sensitivity: float  # Malignant Recall (TP / (TP + FN))
    specificity: float  # Benign Recall (TN / (TN + FP))
    precision: float    # PPV (TP / (TP + FP))
    f1: float
    accuracy: float
    youden_j: float     # Sensitivity + Specificity - 1


def compute_oof_probabilities(
    pipeline: Pipeline,
    X_train: pd.DataFrame | np.ndarray,
    y_train: pd.Series | np.ndarray,
    cv: int = CV_FOLDS,
    random_state: int = RANDOM_SEED,
) -> np.ndarray:
    """
    Generate out-of-fold predicted probabilities for training samples using Stratified K-Fold.
    Prevents overconfidence bias from in-sample predictions.
    
    Returns:
        np.ndarray of shape (N, 2) with [P(Benign), P(Malignant)].
    """
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
    logger.info("Computing out-of-fold probabilities with %d-fold Stratified CV...", cv)
    oof_probs = cross_val_predict(
        pipeline,
        X_train,
        y_train,
        cv=skf,
        method="predict_proba",
        n_jobs=-1,
    )
    return oof_probs


def sweep_decision_thresholds(
    y_true: np.ndarray | pd.Series,
    y_prob_malignant: np.ndarray,
    thresholds: np.ndarray | None = None,
) -> pd.DataFrame:
    """
    Evaluate diagnostic performance across decision thresholds.
    
    Args:
        y_true: True binary labels (0 = Benign, 1 = Malignant).
        y_prob_malignant: Predicted probability of Malignant class.
        thresholds: Array of thresholds to evaluate. Defaults to 0.01 to 0.99 (step 0.01).
        
    Returns:
        pd.DataFrame indexed by threshold with sensitivity, specificity, precision, f1, etc.
    """
    if thresholds is None:
        thresholds = np.linspace(0.01, 0.99, 99)
        
    y_true_arr = np.asarray(y_true).astype(int)
    records = []
    
    for th in thresholds:
        y_pred = (y_prob_malignant >= th).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true_arr, y_pred, labels=[0, 1]).ravel()
        
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        f1 = (2 * prec * sens) / (prec + sens) if (prec + sens) > 0 else 0.0
        acc = (tp + tn) / (tp + tn + fp + fn)
        youden = sens + spec - 1.0
        
        records.append({
            "threshold": round(float(th), 4),
            "tp": int(tp),
            "fp": int(fp),
            "tn": int(tn),
            "fn": int(fn),
            "sensitivity": round(float(sens), 4),
            "specificity": round(float(spec), 4),
            "precision": round(float(prec), 4),
            "f1": round(float(f1), 4),
            "accuracy": round(float(acc), 4),
            "youden_j": round(float(youden), 4),
        })
        
    df = pd.DataFrame(records)
    return df


def find_operating_thresholds(threshold_df: pd.DataFrame) -> dict[str, dict[str, Any]]:
    """
    Identify key diagnostic operating points from the threshold sweep:
      1. Default (threshold = 0.50)
      2. Youden's J Optimal (maximizes Sensitivity + Specificity - 1)
      3. High-Sensitivity Clinical Operating Point (Sensitivity >= 98% with highest Specificity)
      4. Balanced F1 Optimal (maximizes F1-score)
    """
    # 1. Default
    idx_default = (threshold_df["threshold"] - 0.50).abs().idxmin()
    row_default = threshold_df.loc[idx_default].to_dict()
    
    # 2. Youden's J Optimal
    idx_youden = threshold_df["youden_j"].idxmax()
    row_youden = threshold_df.loc[idx_youden].to_dict()
    
    # 3. High Sensitivity (>= 0.98, max specificity)
    high_sens_df = threshold_df[threshold_df["sensitivity"] >= 0.98]
    if not high_sens_df.empty:
        idx_high_sens = high_sens_df["specificity"].idxmax()
        row_high_sens = high_sens_df.loc[idx_high_sens].to_dict()
    else:
        # Fallback to max sensitivity
        idx_high_sens = threshold_df["sensitivity"].idxmax()
        row_high_sens = threshold_df.loc[idx_high_sens].to_dict()
        
    # 4. Balanced F1
    idx_f1 = threshold_df["f1"].idxmax()
    row_f1 = threshold_df.loc[idx_f1].to_dict()
    
    return {
        "standard_0_50": row_default,
        "youden_optimal": row_youden,
        "high_sensitivity_screening": row_high_sens,
        "balanced_f1_optimal": row_f1,
    }


def plot_threshold_tradeoffs(
    threshold_df: pd.DataFrame,
    operating_points: dict[str, dict[str, Any]],
    output_dir: Path = FIGURES_DIR,
) -> dict[str, Path]:
    """
    Create publication-grade threshold analysis visualizations:
      1. Sensitivity vs. Specificity Trade-off Curve
      2. Precision, Recall, F1 vs. Threshold Curve
      3. Clinical Error Counts (False Negatives vs False Positives)
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    plots = {}
    
    # Plot 1: Sensitivity vs. Specificity Trade-off
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    ax.plot(
        threshold_df["threshold"],
        threshold_df["sensitivity"],
        label="Sensitivity (Malignant Recall / TPR)",
        color="#e63946",
        linewidth=2.5,
    )
    ax.plot(
        threshold_df["threshold"],
        threshold_df["specificity"],
        label="Specificity (Benign True Negative Rate / TNR)",
        color="#1d3557",
        linewidth=2.5,
    )
    ax.plot(
        threshold_df["threshold"],
        threshold_df["youden_j"],
        label="Youden's J Statistic (Sens + Spec - 1)",
        color="#457b9d",
        linestyle="--",
        linewidth=1.8,
    )
    
    # Mark operating points
    y_opt_th = operating_points["youden_optimal"]["threshold"]
    ax.axvline(x=y_opt_th, color="#2a9d8f", linestyle=":", label=f"Youden Optimal (th={y_opt_th:.2f})")
    
    hs_opt_th = operating_points["high_sensitivity_screening"]["threshold"]
    ax.axvline(x=hs_opt_th, color="#e76f51", linestyle="-.", label=f"High-Sensitivity (th={hs_opt_th:.2f})")
    
    ax.axvline(x=0.50, color="#6c757d", linestyle="--", alpha=0.7, label="Standard Default (th=0.50)")
    
    ax.set_title("BreastGuard AI — Decision Threshold vs. Sensitivity & Specificity", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Decision Threshold (P(Malignant) Cutoff)", fontsize=12)
    ax.set_ylabel("Metric Value (0.0 to 1.0)", fontsize=12)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="lower center", framealpha=0.95, fontsize=10)
    plt.tight_layout()
    
    path_sens_spec = output_dir / "threshold_sensitivity_specificity.png"
    plt.savefig(path_sens_spec, dpi=300)
    plt.close(fig)
    plots["sensitivity_specificity"] = path_sens_spec
    
    # Plot 2: Precision, Recall, and F1 Curve
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    ax.plot(
        threshold_df["threshold"],
        threshold_df["precision"],
        label="Precision (PPV)",
        color="#2a9d8f",
        linewidth=2.5,
    )
    ax.plot(
        threshold_df["threshold"],
        threshold_df["sensitivity"],
        label="Recall / Sensitivity",
        color="#e63946",
        linewidth=2.5,
    )
    ax.plot(
        threshold_df["threshold"],
        threshold_df["f1"],
        label="F1-Score",
        color="#f4a261",
        linewidth=2.5,
    )
    
    f1_th = operating_points["balanced_f1_optimal"]["threshold"]
    ax.axvline(x=f1_th, color="#264653", linestyle=":", label=f"Peak F1 (th={f1_th:.2f})")
    ax.axvline(x=0.50, color="#6c757d", linestyle="--", alpha=0.7, label="Default (th=0.50)")
    
    ax.set_title("BreastGuard AI — Precision-Recall-F1 vs. Decision Threshold", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Decision Threshold", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="best", framealpha=0.95, fontsize=10)
    plt.tight_layout()
    
    path_pr_f1 = output_dir / "threshold_precision_recall_f1.png"
    plt.savefig(path_pr_f1, dpi=300)
    plt.close(fig)
    plots["precision_recall_f1"] = path_pr_f1
    
    # Plot 3: Clinical Diagnostic Error Tradeoff (False Negatives vs False Positives)
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    ax.plot(
        threshold_df["threshold"],
        threshold_df["fn"],
        label="False Negatives (FN: Missed Malignant Cancer)",
        color="#d62828",
        linewidth=3.0,
    )
    ax.plot(
        threshold_df["threshold"],
        threshold_df["fp"],
        label="False Positives (FP: False Alarms / Extra Biopsies)",
        color="#f77f00",
        linewidth=2.5,
    )
    ax.axvline(x=0.50, color="#6c757d", linestyle="--", label="Default Cutoff (0.50)")
    ax.axvline(x=hs_opt_th, color="#2a9d8f", linestyle="-.", label=f"Clinical Screening ({hs_opt_th:.2f})")
    
    ax.set_title("BreastGuard AI — Clinical Diagnostic Errors across Thresholds (N_train=455)", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Decision Threshold", fontsize=12)
    ax.set_ylabel("Number of Misclassified Patients", fontsize=12)
    ax.set_xlim(0.0, 1.0)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="upper center", framealpha=0.95, fontsize=10)
    plt.tight_layout()
    
    path_errors = output_dir / "threshold_clinical_error_tradeoff.png"
    plt.savefig(path_errors, dpi=300)
    plt.close(fig)
    plots["clinical_errors"] = path_errors
    
    return plots


def compute_and_plot_calibration(
    models_dict: dict[str, Pipeline],
    X_train: pd.DataFrame | np.ndarray,
    y_train: pd.Series | np.ndarray,
    output_dir: Path = FIGURES_DIR,
) -> tuple[dict[str, float], Path]:
    """
    Compute calibration curves and Brier scores for all models using out-of-fold predictions.
    Generates calibration curve figure: reports/figures/calibration_curves.png.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    brier_scores: dict[str, float] = {}
    
    fig, ax = plt.subplots(figsize=(9, 7), dpi=300)
    ax.plot([0, 1], [0, 1], "k--", label="Perfectly Calibrated", alpha=0.7)
    
    colors = {
        "Logistic Regression": "#2b5c8f",
        "Random Forest": "#2a9d8f",
        "Support Vector Machine": "#e76f51",
        "Gradient Boosting": "#9d4edd",
        "K-Nearest Neighbors": "#e63946",
    }
    
    skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    
    for name, pipeline in models_dict.items():
        oof_probs = cross_val_predict(
            pipeline,
            X_train,
            y_train,
            cv=skf,
            method="predict_proba",
            n_jobs=-1,
        )[:, 1]
        
        brier = float(brier_score_loss(y_train, oof_probs))
        brier_scores[name] = round(brier, 4)
        
        frac_pos, mean_pred = calibration_curve(y_train, oof_probs, n_bins=10, strategy="uniform")
        color = colors.get(name, "#333333")
        ax.plot(
            mean_pred,
            frac_pos,
            marker="o",
            linewidth=2,
            label=f"{name} (Brier: {brier:.4f})",
            color=color,
        )
        
    ax.set_title("BreastGuard AI — Probability Calibration Reliability Diagram (5-Fold CV)", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Mean Predicted Probability P(Malignant)", fontsize=11)
    ax.set_ylabel("Observed Fraction of Malignant Tumors", fontsize=11)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="upper left", framealpha=0.95, fontsize=9)
    plt.tight_layout()
    
    cal_path = output_dir / "calibration_curves.png"
    plt.savefig(cal_path, dpi=300)
    plt.close(fig)
    
    logger.info("Saved calibration curves to %s", cal_path)
    return brier_scores, cal_path


def evaluate_test_at_thresholds(
    pipeline: Pipeline,
    X_test: pd.DataFrame | np.ndarray,
    y_test: pd.Series | np.ndarray,
    thresholds_dict: dict[str, float],
) -> dict[str, dict[str, Any]]:
    """
    Evaluate the final champion model on held-out test data (N=114) at specified thresholds.
    """
    y_test_arr = np.asarray(y_test).astype(int)
    y_probs = pipeline.predict_proba(X_test)[:, 1]
    roc_auc = float(roc_auc_score(y_test_arr, y_probs))
    brier = float(brier_score_loss(y_test_arr, y_probs))
    
    results = {}
    for op_name, th in thresholds_dict.items():
        y_pred = (y_probs >= th).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_test_arr, y_pred, labels=[0, 1]).ravel()
        
        sens = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
        prec = float(precision_score(y_test_arr, y_pred, zero_division=0))
        f1 = float(f1_score(y_test_arr, y_pred, zero_division=0))
        acc = float(accuracy_score(y_test_arr, y_pred))
        
        results[op_name] = {
            "threshold": round(float(th), 4),
            "tp": int(tp),
            "fp": int(fp),
            "tn": int(tn),
            "fn": int(fn),
            "sensitivity": round(sens, 4),
            "specificity": round(spec, 4),
            "precision": round(prec, 4),
            "f1": round(f1, 4),
            "accuracy": round(acc, 4),
            "roc_auc": round(roc_auc, 4),
            "brier_score": round(brier, 4),
        }
        
    return results

