"""
BreastGuard AI — Unit & Integration Tests for Hyperparameter Optimization,
Decision Threshold Sweeps, and Probability Calibration
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.threshold import (
    compute_oof_probabilities,
    evaluate_test_at_thresholds,
    find_operating_thresholds,
    sweep_decision_thresholds,
)
from src.tune import get_tuning_spaces


@pytest.fixture
def synthetic_binary_data():
    """Generates small reproducible synthetic dataset for fast threshold testing."""
    X, y = make_classification(
        n_samples=100,
        n_features=10,
        n_informative=6,
        n_redundant=2,
        n_clusters_per_class=1,
        weights=[0.63, 0.37],  # similar class balance to WDBC
        random_state=42,
    )
    return pd.DataFrame(X, columns=[f"f_{i}" for i in range(10)]), pd.Series(y, name="target")


@pytest.fixture
def dummy_pipeline():
    """Returns a simple scaled Logistic Regression pipeline."""
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(random_state=42)),
    ])


class TestHyperparameterSearchSpaces:
    def test_five_models_configured(self):
        spaces = get_tuning_spaces()
        expected = [
            "Logistic Regression",
            "Random Forest",
            "Support Vector Machine",
            "Gradient Boosting",
            "K-Nearest Neighbors",
        ]
        for name in expected:
            assert name in spaces, f"Missing tuning space for {name}"

    def test_scaling_specifications(self):
        spaces = get_tuning_spaces()
        assert spaces["Logistic Regression"]["use_scaler"] is True
        assert spaces["Support Vector Machine"]["use_scaler"] is True
        assert spaces["K-Nearest Neighbors"]["use_scaler"] is True
        assert spaces["Random Forest"]["use_scaler"] is False
        assert spaces["Gradient Boosting"]["use_scaler"] is False

    def test_logistic_regression_param_compatibility(self):
        spaces = get_tuning_spaces()
        lr_grid = spaces["Logistic Regression"]["param_grid"]
        assert isinstance(lr_grid, list)
        for grid in lr_grid:
            penalty = grid["model__penalty"]
            solver = grid["model__solver"]
            if "l1" in penalty:
                assert "liblinear" in solver, "L1 penalty must be paired with liblinear"
            if "l2" in penalty:
                assert "lbfgs" in solver, "L2 penalty can be paired with lbfgs"


class TestThresholdAnalysis:
    def test_oof_probabilities_shape_and_sum(self, synthetic_binary_data, dummy_pipeline):
        X, y = synthetic_binary_data
        oof = compute_oof_probabilities(dummy_pipeline, X, y, cv=3, random_state=42)
        assert oof.shape == (len(X), 2)
        np.testing.assert_allclose(oof.sum(axis=1), 1.0, atol=1e-5)

    def test_threshold_sweep_monotonic_properties(self, synthetic_binary_data, dummy_pipeline):
        X, y = synthetic_binary_data
        oof = compute_oof_probabilities(dummy_pipeline, X, y, cv=3, random_state=42)
        sweep_df = sweep_decision_thresholds(y, oof[:, 1], thresholds=np.linspace(0.1, 0.9, 9))

        assert "sensitivity" in sweep_df.columns
        assert "specificity" in sweep_df.columns
        assert "f1" in sweep_df.columns
        assert "youden_j" in sweep_df.columns

        # Sensitivity should be non-increasing with increasing threshold
        sens_diffs = np.diff(sweep_df["sensitivity"].values)
        assert np.all(sens_diffs <= 0.001), "Sensitivity must decrease or stay constant as threshold increases"

        # Specificity should be non-decreasing with increasing threshold
        spec_diffs = np.diff(sweep_df["specificity"].values)
        assert np.all(spec_diffs >= -0.001), "Specificity must increase or stay constant as threshold increases"

    def test_find_operating_thresholds(self, synthetic_binary_data, dummy_pipeline):
        X, y = synthetic_binary_data
        oof = compute_oof_probabilities(dummy_pipeline, X, y, cv=3, random_state=42)
        sweep_df = sweep_decision_thresholds(y, oof[:, 1])
        operating_points = find_operating_thresholds(sweep_df)

        required_keys = [
            "standard_0_50",
            "youden_optimal",
            "high_sensitivity_screening",
            "balanced_f1_optimal",
        ]
        for k in required_keys:
            assert k in operating_points
            pt = operating_points[k]
            assert 0.0 <= pt["threshold"] <= 1.0
            assert 0.0 <= pt["sensitivity"] <= 1.0
            assert 0.0 <= pt["specificity"] <= 1.0
            assert pt["fn"] >= 0
            assert pt["fp"] >= 0

    def test_evaluate_test_at_thresholds(self, synthetic_binary_data, dummy_pipeline):
        X, y = synthetic_binary_data
        dummy_pipeline.fit(X, y)
        evals = evaluate_test_at_thresholds(
            dummy_pipeline,
            X,
            y,
            {"standard": 0.50, "high_sens": 0.30},
        )
        assert "standard" in evals
        assert "high_sens" in evals
        for m in evals.values():
            assert 0.0 <= m["accuracy"] <= 1.0
            assert 0.0 <= m["sensitivity"] <= 1.0
            assert 0.0 <= m["roc_auc"] <= 1.0
            assert 0.0 <= m["brier_score"] <= 1.0

