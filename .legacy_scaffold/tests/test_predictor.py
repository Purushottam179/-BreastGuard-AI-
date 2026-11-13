"""Tests for the end-to-end PredictionService."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
import pytest

from src.data.loader import load_data
from src.preprocessing.pipeline import make_split
from src.models.trainer import train_all
from src.prediction.predictor import PredictionService, PredictionResult
from src.utils.config import MODEL_RF


@pytest.fixture(scope="module")
def predictor_setup():
    b       = load_data()
    split   = make_split(b.X, b.y)
    trained = train_all(split)
    model   = trained.models.get(MODEL_RF) or list(trained.models.values())[0]
    svc     = PredictionService(model, split.X_train, b.feature_names)
    return b, split, svc


class TestPredictionService:
    def _make_feature_dict(self, bundle, x_row):
        return dict(zip(bundle.feature_names, x_row.tolist()))

    def test_predict_returns_result(self, predictor_setup):
        b, split, svc = predictor_setup
        fv     = self._make_feature_dict(b, split.X_test[0])
        result = svc.predict(fv)
        assert isinstance(result, PredictionResult)

    def test_label_is_valid(self, predictor_setup):
        b, split, svc = predictor_setup
        fv     = self._make_feature_dict(b, split.X_test[0])
        result = svc.predict(fv)
        assert result.label in {"Benign", "Malignant"}

    def test_probabilities_sum_to_one(self, predictor_setup):
        b, split, svc = predictor_setup
        fv     = self._make_feature_dict(b, split.X_test[0])
        result = svc.predict(fv)
        total  = result.prob_benign + result.prob_malignant
        assert abs(total - 1.0) < 1e-5

    def test_shap_values_in_result(self, predictor_setup):
        b, split, svc = predictor_setup
        fv     = self._make_feature_dict(b, split.X_test[0])
        result = svc.predict(fv)
        assert result.shap_result is not None
        assert result.shap_result.shap_values.shape[1] == 30

    def test_known_malignant_sample(self, predictor_setup):
        """A typical mean-malignant sample should predict Malignant."""
        b, split, svc = predictor_setup
        # Use mean of malignant training samples
        mal_mean = split.X_train[split.y_train == 0].mean(axis=0)
        fv       = dict(zip(b.feature_names, mal_mean.tolist()))
        result   = svc.predict(fv)
        assert result.label == "Malignant", (
            f"Expected Malignant for mean malignant input, got {result.label} "
            f"(P_malignant={result.prob_malignant:.3f})"
        )

    def test_known_benign_sample(self, predictor_setup):
        """A typical mean-benign sample should predict Benign."""
        b, split, svc = predictor_setup
        ben_mean = split.X_train[split.y_train == 1].mean(axis=0)
        fv       = dict(zip(b.feature_names, ben_mean.tolist()))
        result   = svc.predict(fv)
        assert result.label == "Benign", (
            f"Expected Benign for mean benign input, got {result.label} "
            f"(P_benign={result.prob_benign:.3f})"
        )
