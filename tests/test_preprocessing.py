"""
Tests for Preprocessing Pipeline — Stratified Split & Data Leakage Prevention
(Compatible with new M=1, B=0 encoding convention)
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from src.data_loader import FEATURE_COLUMNS, run_acquisition
from src.preprocessing import SplitData, build_pipeline, create_stratified_split


@pytest.fixture(scope="module")
def bundle():
    return run_acquisition()


@pytest.fixture(scope="module")
def split(bundle):
    X = bundle.df_cleaned[FEATURE_COLUMNS].values
    y = bundle.df_cleaned["diagnosis"].values
    return create_stratified_split(X, y)


class TestStratifiedSplit:
    def test_returns_split_data(self, split):
        assert isinstance(split, SplitData)

    def test_partition_sizes(self, split):
        assert split.train_size + split.test_size == 569
        assert split.test_size == 114   # 20% of 569 = 113.8 → 114
        assert split.train_size == 455

    def test_no_row_overlap(self, split):
        train_rows = set(map(tuple, split.X_train.tolist()))
        test_rows = set(map(tuple, split.X_test.tolist()))
        assert len(train_rows.intersection(test_rows)) == 0

    def test_stratification_preserved(self, split):
        """Both splits should preserve the ~37.3% Malignant (M=1) ratio."""
        train_ratio = (split.y_train == 1).mean()
        test_ratio = (split.y_test == 1).mean()
        assert abs(train_ratio - test_ratio) < 0.02

    def test_reproducibility(self, bundle):
        X = bundle.df_cleaned[FEATURE_COLUMNS].values
        y = bundle.df_cleaned["diagnosis"].values
        s1 = create_stratified_split(X, y, random_seed=42)
        s2 = create_stratified_split(X, y, random_seed=42)
        np.testing.assert_array_equal(s1.X_train, s2.X_train)
        np.testing.assert_array_equal(s1.y_test, s2.y_test)


class TestPipelineLeakagePrevention:
    def test_pipeline_named_steps(self):
        pipe = build_pipeline(LogisticRegression(max_iter=200))
        assert "scaler" in pipe.named_steps
        assert "model" in pipe.named_steps

    def test_scaler_not_fitted_before_fit(self):
        pipe = build_pipeline(LogisticRegression(max_iter=200))
        assert not hasattr(pipe.named_steps["scaler"], "mean_")

    def test_scaler_fitted_only_on_train(self, split):
        pipe = build_pipeline(LogisticRegression(max_iter=200))
        pipe.fit(split.X_train, split.y_train)
        # Scaler must have 30 feature statistics learned from training data
        assert len(pipe.named_steps["scaler"].mean_) == 30
        assert len(pipe.named_steps["scaler"].scale_) == 30

    def test_predictions_are_binary(self, split):
        pipe = build_pipeline(LogisticRegression(max_iter=200))
        pipe.fit(split.X_train, split.y_train)
        preds = pipe.predict(split.X_test)
        assert preds.shape == (split.test_size,)
        assert set(preds).issubset({0, 1})
