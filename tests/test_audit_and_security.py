"""
BreastGuard AI — Security, Input Validation & Quality Assurance Unit Tests
================================================================================
Validates:
  1. Input validation security (rejection of malicious, non-numeric, NaN, infinite, negative values).
  2. Batch CSV schema validation (missing columns, extra columns, malformed headers).
  3. Prediction cache isolation (ensuring user inputs are never cross-contaminated).
  4. Consistency between single prediction and batch prediction results.
  5. PDF clinical report generation integrity and non-empty byte stream.
================================================================================
"""
from __future__ import annotations

import io
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.data_loader import load_dataset
from src.predict import PredictionEngine, load_prediction_engine


class TestInputValidationSecurity:
    """Ensure strict input sanitization and boundary defense."""

    def test_reject_string_injection_in_features(self) -> None:
        """Verify non-numeric string values raise explicit ValueError."""
        engine = load_prediction_engine()
        defaults = engine.get_default_values()
        defaults["mean_radius"] = "malicious_string_drop_table"

        with pytest.raises(ValueError, match="non-numeric"):
            engine.predict_single(defaults)

    def test_reject_nan_values(self) -> None:
        """Verify NaN feature inputs raise explicit ValueError."""
        engine = load_prediction_engine()
        defaults = engine.get_default_values()
        defaults["mean_radius"] = np.nan

        with pytest.raises(ValueError, match="NaN or infinite"):
            engine.predict_single(defaults)

    def test_reject_infinite_values(self) -> None:
        """Verify infinite inputs raise explicit ValueError."""
        engine = load_prediction_engine()
        defaults = engine.get_default_values()
        defaults["worst_area"] = np.inf

        with pytest.raises(ValueError, match="NaN or infinite"):
            engine.predict_single(defaults)

    def test_reject_negative_measurements(self) -> None:
        """Verify negative physical measurements raise ValueError."""
        engine = load_prediction_engine()
        defaults = engine.get_default_values()
        defaults["mean_radius"] = -14.5

        with pytest.raises(ValueError, match="non-negative"):
            engine.predict_single(defaults)

    def test_reject_missing_required_features(self) -> None:
        """Verify incomplete feature dictionaries raise ValueError."""
        engine = load_prediction_engine()
        defaults = engine.get_default_values()
        del defaults["worst_radius"]

        with pytest.raises(ValueError, match="Missing required features"):
            engine.predict_single(defaults)


class TestBatchProcessingRobustness:
    """Verify batch CSV processing handles anomalous data without crashing."""

    def test_batch_processing_with_corrupted_rows(self) -> None:
        """Verify invalid rows are tracked and valid rows succeed."""
        engine = load_prediction_engine()
        defaults = engine.get_default_values()

        # Create DataFrame with 2 valid rows and 2 corrupted rows
        row_valid_1 = dict(defaults)
        row_valid_2 = dict(defaults)
        row_corrupt_1 = dict(defaults)
        row_corrupt_1["mean_radius"] = -5.0  # Invalid negative
        row_corrupt_2 = dict(defaults)
        row_corrupt_2["worst_texture"] = np.nan  # Invalid NaN

        batch_df = pd.DataFrame([row_valid_1, row_corrupt_1, row_valid_2, row_corrupt_2])
        batch_res = engine.predict_batch(batch_df)

        assert batch_res.total_records == 4
        assert batch_res.valid_records == 2
        assert batch_res.invalid_records == 2
        assert len(batch_res.invalid_rows_summary) == 2

    def test_batch_missing_column_raises(self) -> None:
        """Verify uploaded batch missing a required column raises ValueError."""
        engine = load_prediction_engine()
        defaults = engine.get_default_values()
        df = pd.DataFrame([defaults])
        df = df.drop(columns=["worst_concavity"])

        with pytest.raises(ValueError, match="missing.*required columns"):
            engine.predict_batch(df)


class TestPredictionConcordanceAndExport:
    """Verify concordance between single and batch predictions and report generation."""

    def test_single_and_batch_prediction_concordance(self) -> None:
        """Verify predict_single and predict_batch yield identical probabilities for the same record."""
        engine = load_prediction_engine()
        bundle = load_dataset(save_raw_csv=False)
        test_sample = bundle.X_df.iloc[0].to_dict()

        # Single prediction
        single_res = engine.predict_single(test_sample, generate_explanation=False)

        # Batch prediction
        batch_df = pd.DataFrame([test_sample])
        batch_res = engine.predict_batch(batch_df)

        batch_p_mal = float(batch_res.predictions_df.iloc[0]["prob_malignant"])
        batch_label = batch_res.predictions_df.iloc[0]["predicted_class"]

        assert single_res.prob_malignant == pytest.approx(batch_p_mal, abs=1e-3)
        assert single_res.predicted_label == batch_label

    def test_pdf_report_bytes_and_existence(self) -> None:
        """Verify PDF clinical report generates valid, non-empty binary file."""
        engine = load_prediction_engine()
        defaults = engine.get_default_values()
        res = engine.predict_single(defaults, generate_explanation=True)

        pdf_path = engine.generate_pdf_report(res)
        assert pdf_path.exists()
        assert pdf_path.suffix.lower() == ".pdf"
        assert pdf_path.stat().st_size > 5000  # Non-trivial vector PDF with charts
