"""
BreastGuard AI — Model Evaluator & Clinical Metrics Suite
Computes oncological classification metrics, confusion matrices, ROC curves,
and probabilistic calibration scores on the held-out test set.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from src.models.trainer import TrainedModel
from src.preprocessing.pipeline import SplitData
from src.utils.config import LABEL_BENIGN, LABEL_MALIGNANT

logger = logging.getLogger(__name__)


@dataclass
class ClinicalConfusionMatrix:
    """Explicitly mapped clinical confusion matrix."""
    true_malignant: int   # TP (Malignant correctly identified)
    missed_malignant: int # FN (Malignant misclassified as Benign - high clinical danger)
    false_alarm: int      # FP (Benign misclassified as Malignant - emotional distress/biopsy)
    true_benign: int      # TN (Benign correctly identified)
    raw_matrix: np.ndarray# shape (2, 2)


@dataclass
class ModelReport:
    """Comprehensive evaluation report for one model on the held-out test set."""
    name: str
    accuracy: float
    sensitivity: float        # Recall for Malignant: TP / (TP + FN)
    specificity: float        # TNR for Benign: TN / (TN + FP)
    precision_malignant: float# PPV for Malignant: TP / (TP + FP)
    npv: float                # Negative Predictive Value: TN / (TN + FN)
    f1_malignant: float       # F1 harmonic mean for Malignant class
    macro_f1: float           # Balanced macro F1 across both classes
    roc_auc: float            # Area under the ROC curve (malignancy discrimination)
    brier_score: float        # Probabilistic calibration error (lower is better)
    log_loss_val: float       # Cross-entropy loss
    confusion: ClinicalConfusionMatrix
    fpr: np.ndarray           # False Positive Rate points for ROC
    tpr: np.ndarray           # True Positive Rate points for ROC
    thresholds: np.ndarray    # Decision thresholds for ROC
    cv_auc_mean: float        # Stratified 5-fold CV AUC mean (from training)
    cv_auc_std: float         # Stratified 5-fold CV AUC standard deviation

    # Aliases for compatibility
    @property
    def recall(self) -> float:
        return self.sensitivity

    @property
    def precision(self) -> float:
        return self.precision_malignant

    @property
    def f1(self) -> float:
        return self.f1_malignant


def evaluate(trained: TrainedModel, split: SplitData) -> ModelReport:
    """
    Evaluate a fitted pipeline on the held-out test partition.
    Strictly clinical: Malignant (0) is treated as the clinical positive.

    Parameters
    ----------
    trained : TrainedModel
        Output of trainer.train_all()
    split : SplitData
        SplitData instance containing untouched X_test and y_test.

    Returns
    -------
    ModelReport
    """
    pipe = trained.pipeline
    y_test = split.y_test
    X_test = split.X_test

    y_pred = pipe.predict(X_test)
    probs = pipe.predict_proba(X_test)

    # In sklearn: index 0 is Malignant (0), index 1 is Benign (1)
    p_malignant = probs[:, 0]
    p_benign = probs[:, 1]

    # Clinical confusion matrix: rows=Actual, cols=Predicted; [0=Malignant, 1=Benign]
    cm = confusion_matrix(y_test, y_pred, labels=[LABEL_MALIGNANT, LABEL_BENIGN])
    tp_mal = int(cm[0, 0])  # True Malignant
    fn_mal = int(cm[0, 1])  # Missed Malignant (FN)
    fp_mal = int(cm[1, 0])  # False Alarm (FP)
    tn_mal = int(cm[1, 1])  # True Benign (TN)

    clin_cm = ClinicalConfusionMatrix(
        true_malignant=tp_mal,
        missed_malignant=fn_mal,
        false_alarm=fp_mal,
        true_benign=tn_mal,
        raw_matrix=cm,
    )

    # Clinical performance ratios
    sensitivity = tp_mal / (tp_mal + fn_mal) if (tp_mal + fn_mal) > 0 else 0.0
    specificity = tn_mal / (tn_mal + fp_mal) if (tn_mal + fp_mal) > 0 else 0.0
    precision_mal = tp_mal / (tp_mal + fp_mal) if (tp_mal + fp_mal) > 0 else 0.0
    npv = tn_mal / (tn_mal + fn_mal) if (tn_mal + fn_mal) > 0 else 0.0

    if (precision_mal + sensitivity) > 0:
        f1_mal = 2.0 * (precision_mal * sensitivity) / (precision_mal + sensitivity)
    else:
        f1_mal = 0.0

    acc = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)

    # ROC curve with Malignant (0) as Positive
    # Create binary indicator: 1 if Malignant, 0 if Benign
    y_test_mal_binary = (y_test == LABEL_MALIGNANT).astype(int)
    auc_score = roc_auc_score(y_test_mal_binary, p_malignant)
    fpr, tpr, thresholds = roc_curve(y_test_mal_binary, p_malignant, pos_label=1)

    # Probabilistic quality
    brier = brier_score_loss(y_test_mal_binary, p_malignant)
    ll = log_loss(y_test, probs)

    cv_auc = trained.cv_mean.get("test_roc_auc", 0.0)
    cv_auc_sd = trained.cv_std.get("test_roc_auc", 0.0)

    report = ModelReport(
        name=trained.name,
        accuracy=float(acc),
        sensitivity=float(sensitivity),
        specificity=float(specificity),
        precision_malignant=float(precision_mal),
        npv=float(npv),
        f1_malignant=float(f1_mal),
        macro_f1=float(macro_f1),
        roc_auc=float(auc_score),
        brier_score=float(brier),
        log_loss_val=float(ll),
        confusion=clin_cm,
        fpr=fpr,
        tpr=tpr,
        thresholds=thresholds,
        cv_auc_mean=cv_auc,
        cv_auc_std=cv_auc_sd,
    )

    logger.info(
        "Evaluated [%s] -> AUC: %.4f | Sensitivity: %.4f | Specificity: %.4f | Missed: %d",
        report.name,
        report.roc_auc,
        report.sensitivity,
        report.specificity,
        report.confusion.missed_malignant,
    )

    return report


def evaluate_all(
    training_results,
    split: SplitData,
) -> dict[str, ModelReport]:
    """Evaluate all trained models and return {model_name: ModelReport}."""
    return {
        name: evaluate(tm, split)
        for name, tm in training_results.models.items()
    }
