"""Tests for model training and evaluation metrics."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
import pytest

from src.data.loader import load_data
from src.preprocessing.pipeline import make_split
from src.models.trainer import train_all, TrainedModel, TrainingResults
from src.models.evaluator import evaluate, evaluate_all, ModelReport


@pytest.fixture(scope="module")
def trained_data():
    b       = load_data()
    split   = make_split(b.X, b.y)
    trained = train_all(split)
    reports = evaluate_all(trained, split)
    return split, trained, reports


class TestTrainer:
    def test_returns_training_results(self, trained_data):
        _, trained, _ = trained_data
        assert isinstance(trained, TrainingResults)

    def test_all_models_present(self, trained_data):
        _, trained, _ = trained_data
        assert len(trained.models) >= 3   # LR, RF, SVM (+ XGB if available)

    def test_each_model_is_trained_model(self, trained_data):
        _, trained, _ = trained_data
        for name, tm in trained.models.items():
            assert isinstance(tm, TrainedModel), f"{name} is not TrainedModel"

    def test_cv_scores_present(self, trained_data):
        _, trained, _ = trained_data
        for tm in trained.models.values():
            assert "test_roc_auc" in tm.cv_scores
            assert len(tm.cv_scores["test_roc_auc"]) == 5   # 5-fold

    def test_cv_auc_reasonable(self, trained_data):
        _, trained, _ = trained_data
        for tm in trained.models.values():
            mean_auc = tm.cv_mean["test_roc_auc"]
            assert 0.85 <= mean_auc <= 1.0, f"{tm.name}: CV AUC={mean_auc}"


class TestEvaluator:
    def test_all_reports_present(self, trained_data):
        _, _, reports = trained_data
        assert len(reports) >= 3

    def test_report_type(self, trained_data):
        _, _, reports = trained_data
        for r in reports.values():
            assert isinstance(r, ModelReport)

    def test_metric_ranges(self, trained_data):
        """All metrics must be in [0, 1]."""
        _, _, reports = trained_data
        for name, r in reports.items():
            for metric, val in [
                ("accuracy",    r.accuracy),
                ("precision",   r.precision),
                ("recall",      r.recall),
                ("specificity", r.specificity),
                ("f1",          r.f1),
                ("roc_auc",     r.roc_auc),
            ]:
                assert 0.0 <= val <= 1.0, f"{name} {metric}={val} out of range"

    def test_high_auc(self, trained_data):
        """Best model should achieve AUC > 0.97 on this dataset."""
        _, _, reports = trained_data
        best_auc = max(r.roc_auc for r in reports.values())
        assert best_auc > 0.97, f"Best AUC={best_auc} below expected threshold"

    def test_confusion_matrix_shape(self, trained_data):
        _, _, reports = trained_data
        for r in reports.values():
            assert r.confusion.shape == (2, 2)

    def test_roc_curve_monotone(self, trained_data):
        _, _, reports = trained_data
        for r in reports.values():
            # FPR should be non-decreasing (sklearn guarantees this)
            assert np.all(np.diff(r.fpr) >= -1e-10), f"{r.name}: FPR not monotone"

    def test_probabilities_sum_to_one(self, trained_data):
        split, trained, _ = trained_data
        for name, tm in trained.models.items():
            proba = tm.pipeline.predict_proba(split.X_test)
            sums  = proba.sum(axis=1)
            np.testing.assert_allclose(sums, np.ones_like(sums), atol=1e-6)
