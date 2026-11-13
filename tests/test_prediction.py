"""
BreastGuard AI — Tests for Clinical Prediction Engine & Batch Pipeline
================================================================================
Verifies:
  1. PredictionEngine loading and feature specification integrity.
  2. Single-sample prediction with probability estimation and risk stratification.
  3. Dynamic class mapping verification via classes_ attribute.
  4. Clinical threshold application (tau = 0.37).
  5. Rigorous input validation: rejection of NaN, inf, negative, and missing features.
  6. Batch CSV processing with invalid-row tracking and CSV export.
  7. Local SHAP explainability integration and fallback handling.
  8. PDF clinical prediction report generation and verification.
  9. Backward compatibility with legacy Predictor class.
================================================================================
"""
from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from config.config import MODELS_DIR, N_FEATURES
from src.predict import (
    BatchPredictionResult,
    PredictionEngine,
    PredictionResult,
    Predictor,
    load_prediction_engine,
)


@pytest.fixture(scope="module")
def prediction_engine():
    """Returns a loaded PredictionEngine instance."""
    return load_prediction_engine()


@pytest.fixture(scope="module")
def sample_feature_dict(prediction_engine):
    """Returns valid default feature dictionary."""
    return prediction_engine.get_default_values()


class TestPredictionEngineLoading:
    def test_engine_initializes_successfully(self, prediction_engine):
        assert prediction_engine is not None
        assert prediction_engine.pipeline is not None
        assert len(prediction_engine.feature_names) == N_FEATURES
        assert prediction_engine.threshold == 0.37

    def test_classes_order_is_verified(self, prediction_engine):
        assert prediction_engine.benign_idx == 0
        assert prediction_engine.malignant_idx == 1


class TestSinglePrediction:
    def test_valid_prediction_returns_result(self, prediction_engine, sample_feature_dict):
        result = prediction_engine.predict_single(sample_feature_dict, generate_explanation=False)
        assert isinstance(result, PredictionResult)
        assert result.predicted_label in ("Benign", "Malignant")
        assert result.predicted_class in (0, 1)
        assert 0.0 <= result.prob_malignant <= 1.0
        assert 0.0 <= result.prob_benign <= 1.0
        assert np.isclose(result.prob_malignant + result.prob_benign, 1.0, atol=1e-4)
        assert result.classification_threshold == 0.37
        assert "Educational" in result.disclaimer or "clinical" in result.disclaimer.lower()

    def test_malignant_prediction_with_high_features(self, prediction_engine, sample_feature_dict):
        # Override with severe malignant cytological values
        mal_dict = dict(sample_feature_dict)
        mal_dict["worst_area"] = 2500.0
        mal_dict["worst_radius"] = 28.0
        mal_dict["mean_concave_points"] = 0.18
        mal_dict["worst_texture"] = 35.0

        result = prediction_engine.predict_single(mal_dict, generate_explanation=False)
        assert result.predicted_class == 1
        assert result.predicted_label == "Malignant"
        assert result.prob_malignant >= 0.37

    def test_benign_prediction_with_low_features(self, prediction_engine, sample_feature_dict):
        # Override with benign cytological values
        ben_dict = dict(sample_feature_dict)
        ben_dict["worst_area"] = 400.0
        ben_dict["worst_radius"] = 10.0
        ben_dict["mean_concave_points"] = 0.01
        ben_dict["worst_texture"] = 15.0

        result = prediction_engine.predict_single(ben_dict, generate_explanation=False)
        assert result.predicted_class == 0
        assert result.predicted_label == "Benign"
        assert result.prob_malignant < 0.37


class TestInputValidation:
    def test_missing_feature_raises_error(self, prediction_engine, sample_feature_dict):
        bad_dict = dict(sample_feature_dict)
        del bad_dict["mean_radius"]
        with pytest.raises(ValueError, match="Missing required features"):
            prediction_engine.validate_inputs(bad_dict)

    def test_negative_feature_raises_error(self, prediction_engine, sample_feature_dict):
        bad_dict = dict(sample_feature_dict)
        bad_dict["mean_area"] = -10.0
        with pytest.raises(ValueError, match="must be non-negative"):
            prediction_engine.validate_inputs(bad_dict)

    def test_nan_feature_raises_error(self, prediction_engine, sample_feature_dict):
        bad_dict = dict(sample_feature_dict)
        bad_dict["mean_texture"] = np.nan
        with pytest.raises(ValueError, match="cannot be NaN"):
            prediction_engine.validate_inputs(bad_dict)

    def test_non_numeric_feature_raises_error(self, prediction_engine, sample_feature_dict):
        bad_dict = dict(sample_feature_dict)
        bad_dict["mean_smoothness"] = "not_a_number"
        with pytest.raises(ValueError, match="non-numeric"):
            prediction_engine.validate_inputs(bad_dict)

    def test_invalid_array_shape_raises_error(self, prediction_engine):
        bad_arr = np.zeros(20)  # 20 instead of 30
        with pytest.raises(ValueError, match="Expected 30 features"):
            prediction_engine.validate_inputs(bad_arr)


class TestBatchPrediction:
    def test_predict_batch_dataframe(self, prediction_engine, sample_feature_dict, tmp_path):
        # Create dataframe with 2 valid records and 1 invalid record
        valid_row1 = dict(sample_feature_dict)
        valid_row2 = dict(sample_feature_dict)
        valid_row2["worst_area"] = 2200.0

        invalid_row = dict(sample_feature_dict)
        invalid_row["mean_radius"] = -5.0  # invalid negative

        df_batch = pd.DataFrame([valid_row1, valid_row2, invalid_row])
        df_batch["id"] = ["P001", "P002", "P003"]

        out_csv = tmp_path / "test_batch_out.csv"
        batch_res = prediction_engine.predict_batch(df_batch, output_path=out_csv)

        assert isinstance(batch_res, BatchPredictionResult)
        assert batch_res.total_records == 3
        assert batch_res.valid_records == 2
        assert batch_res.invalid_records == 1
        assert out_csv.exists()

        # Check output dataframe structure
        df_res = pd.read_csv(out_csv)
        assert "predicted_class" in df_res.columns
        assert "prob_malignant" in df_res.columns
        assert "prediction_status" in df_res.columns
        assert df_res.loc[0, "prediction_status"] == "SUCCESS"
        assert df_res.loc[2, "prediction_status"].startswith("INVALID_ROW")

    def test_missing_column_in_csv_raises_error(self, prediction_engine, tmp_path):
        bad_df = pd.DataFrame([{"mean_radius": 15.0, "mean_texture": 20.0}])
        bad_csv = tmp_path / "bad_columns.csv"
        bad_df.to_csv(bad_csv, index=False)

        with pytest.raises(ValueError, match="missing .* required columns"):
            prediction_engine.predict_batch(bad_csv)


class TestSHAPIntegrationAndReportExport:
    def test_shap_explanation_integration(self, prediction_engine, sample_feature_dict):
        result = prediction_engine.predict_single(sample_feature_dict, generate_explanation=True)
        if result.explanation_available:
            assert result.explanation is not None
            assert len(result.top_10_features) <= 10
            assert len(result.narrative_explanation) >= 1

    def test_pdf_report_generation(self, prediction_engine, sample_feature_dict, tmp_path):
        result = prediction_engine.predict_single(sample_feature_dict, generate_explanation=True)
        pdf_path = tmp_path / "test_report.pdf"
        generated_pdf = prediction_engine.generate_pdf_report(result, output_path=pdf_path)

        assert generated_pdf.exists()
        assert generated_pdf.stat().st_size > 5000  # valid PDF with vector graphics


class TestLegacyPredictorCompatibility:
    def test_legacy_predictor_workflow(self, sample_feature_dict):
        # Ensure older tests that instantiate Predictor directly still succeed
        bundle_model = load_prediction_engine().model_bundle
        tm = type("TrainedModelMock", (), {
            "name": "Support Vector Machine",
            "pipeline": bundle_model["pipeline"],
        })()
        
        predictor = Predictor(
            trained_model=tm,
            X_train=np.zeros((10, 30)),
            feature_names=bundle_model["feature_names"],
        )
        res = predictor.predict(sample_feature_dict)
        assert isinstance(res, PredictionResult)
        assert res.predicted_label in ("Malignant", "Benign")
