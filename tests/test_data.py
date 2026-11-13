"""
Tests for Dataset Acquisition, Cleaning & Validation
(Compatible with new src/data_loader.py API — M=1, B=0 encoding)
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pytest

from src.data_loader import (
    AcquisitionBundle,
    DataValidationError,
    FEATURE_COLUMNS,
    FEATURE_GROUPS,
    run_acquisition,
    validate_cleaned_dataset,
)


@pytest.fixture(scope="module")
def bundle() -> AcquisitionBundle:
    return run_acquisition()


class TestAcquisitionBundle:
    def test_returns_acquisition_bundle(self, bundle):
        assert isinstance(bundle, AcquisitionBundle)

    def test_raw_dataframe_shape(self, bundle):
        # Raw has 32 columns: id + diagnosis + 30 features
        assert bundle.df_raw.shape == (569, 32)
        assert "id" in bundle.df_raw.columns
        assert "diagnosis" in bundle.df_raw.columns

    def test_raw_diagnosis_is_string(self, bundle):
        unique_diag = set(bundle.df_raw["diagnosis"].unique())
        assert unique_diag == {"M", "B"}

    def test_cleaned_dataframe_shape(self, bundle):
        assert bundle.df_cleaned.shape == (569, 31)  # 30 features + diagnosis

    def test_cleaned_no_id_column(self, bundle):
        assert "id" not in bundle.df_cleaned.columns

    def test_target_encoding_M_equals_1(self, bundle):
        # PROJECT CONVENTION: M=1, B=0
        assert int((bundle.df_cleaned["diagnosis"] == 1).sum()) == 212
        assert int((bundle.df_cleaned["diagnosis"] == 0).sum()) == 357

    def test_no_missing_values(self, bundle):
        assert not bundle.df_cleaned.isnull().any().any()

    def test_no_duplicate_rows(self, bundle):
        assert not bundle.df_cleaned.duplicated().any()

    def test_all_features_float64(self, bundle):
        for col in FEATURE_COLUMNS:
            assert bundle.df_cleaned[col].dtype == np.float64, f"{col} is not float64"

    def test_all_features_non_negative(self, bundle):
        for col in FEATURE_COLUMNS:
            assert (bundle.df_cleaned[col] >= 0).all(), f"{col} has negative values"

    def test_feature_column_count(self, bundle):
        assert len(bundle.feature_names) == 30
        assert len(FEATURE_COLUMNS) == 30

    def test_feature_groups_complete(self, bundle):
        assert len(FEATURE_GROUPS["mean"]) == 10
        assert len(FEATURE_GROUPS["se"]) == 10
        assert len(FEATURE_GROUPS["worst"]) == 10

    def test_dataset_info_correct(self, bundle):
        info = bundle.info
        assert info.n_samples == 569
        assert info.n_malignant == 212
        assert info.n_benign == 357
        assert info.n_missing == 0
        assert info.n_duplicates == 0
        assert info.source in ("UCI", "sklearn_fallback")


class TestValidation:
    def test_valid_cleaned_data_passes(self, bundle):
        validate_cleaned_dataset(bundle.df_cleaned)  # Must not raise

    def test_wrong_encoding_raises(self, bundle):
        bad_df = bundle.df_cleaned.copy()
        bad_df["diagnosis"] = bad_df["diagnosis"].map({1: 0, 0: 2})
        with pytest.raises(DataValidationError):
            validate_cleaned_dataset(bad_df)

    def test_missing_value_raises(self, bundle):
        bad_df = bundle.df_cleaned.copy()
        bad_df.loc[0, "mean_radius"] = np.nan
        with pytest.raises(DataValidationError):
            validate_cleaned_dataset(bad_df)

    def test_negative_feature_raises(self, bundle):
        bad_df = bundle.df_cleaned.copy()
        bad_df.loc[0, "mean_area"] = -1.0
        with pytest.raises(DataValidationError):
            validate_cleaned_dataset(bad_df)
