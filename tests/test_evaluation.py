"""
Tests for Model Evaluation, Clinical Metrics, and Confusion Matrix Formulation
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pytest
from sklearn.linear_model import LogisticRegression

from src.data_loader import load_and_validate_processed_data
from src.evaluate import ClinicalConfusionMatrix, TestEvaluationReport, evaluate_on_test_set
from src.preprocessing import build_pipeline, create_stratified_split


@pytest.fixture(scope="module")
def eval_setup():
    X, y, feature_names = load_and_validate_processed_data()
    split = create_stratified_split(X, y, test_size=0.20, random_seed=42, feature_names=feature_names)

    # Train a baseline model on training set only
    pipe = build_pipeline(
        LogisticRegression(max_iter=1000, random_state=42),
        use_scaler=True,
        feature_names=feature_names,
    )
    pipe.fit(split.X_train, split.y_train)

    report = evaluate_on_test_set(
        model_name="Logistic Regression",
        pipeline=pipe,
        split=split,
        generate_plots=False,  # Skip file write in quick unit test
    )
    return split, report


class TestClinicalEvaluation:
    def test_report_type(self, eval_setup):
        _, report = eval_setup
        assert isinstance(report, TestEvaluationReport)

    def test_metric_ranges(self, eval_setup):
        _, report = eval_setup
        assert 0.85 <= report.accuracy <= 1.0
        assert 0.85 <= report.precision <= 1.0
        assert 0.85 <= report.malignant_recall <= 1.0
        assert 0.85 <= report.benign_recall <= 1.0
        assert 0.85 <= report.f1_score <= 1.0
        assert 0.90 <= report.roc_auc <= 1.0
        assert 0.0 <= report.brier_score <= 0.20

    def test_confusion_matrix_sum(self, eval_setup):
        split, report = eval_setup
        cm = report.confusion_matrix
        assert isinstance(cm, ClinicalConfusionMatrix)

        total_samples = cm.true_positive + cm.true_negative + cm.false_positive + cm.false_negative
        assert total_samples == split.test_size == 114

    def test_confusion_matrix_interpretations(self, eval_setup):
        split, report = eval_setup
        cm = report.confusion_matrix

        # Actual Malignant in test set: 42 (37.2% of 114)
        n_actual_malignant = int((split.y_test == 1).sum())
        n_actual_benign = int((split.y_test == 0).sum())

        assert cm.true_positive + cm.false_negative == n_actual_malignant
        assert cm.true_negative + cm.false_positive == n_actual_benign

        # Verify sensitivity calculation matches TP / (TP + FN)
        calculated_sens = cm.true_positive / (cm.true_positive + cm.false_negative)
        assert abs(report.malignant_recall - calculated_sens) < 1e-5

        # Verify specificity calculation matches TN / (TN + FP)
        calculated_spec = cm.true_negative / (cm.true_negative + cm.false_positive)
        assert abs(report.benign_recall - calculated_spec) < 1e-5

    def test_classification_report_presence(self, eval_setup):
        _, report = eval_setup
        assert "Benign" in report.classification_report_str
        assert "Malignant" in report.classification_report_str
        assert "accuracy" in report.classification_report_dict

