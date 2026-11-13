"""Tests for SHAP explainability outputs."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
import pytest

from src.data.loader import load_data
from src.preprocessing.pipeline import make_split
from src.models.trainer import train_all
from src.explainability.shap_explainer import compute_shap, ShapResult


@pytest.fixture(scope="module")
def shap_setup():
    b       = load_data()
    split   = make_split(b.X, b.y)
    trained = train_all(split)
    return b, split, trained


class TestShapExplainer:
    def test_rf_shap_shape(self, shap_setup):
        b, split, trained = shap_setup
        from src.utils.config import MODEL_RF
        if MODEL_RF not in trained.models:
            pytest.skip("RF model not trained")
        result = compute_shap(
            trained.models[MODEL_RF].pipeline,
            split.X_train, split.X_test,
            b.feature_names, MODEL_RF,
        )
        assert isinstance(result, ShapResult)
        assert result.shap_values.shape == (len(split.X_test), 30)

    def test_lr_shap_shape(self, shap_setup):
        b, split, trained = shap_setup
        from src.utils.config import MODEL_LR
        if MODEL_LR not in trained.models:
            pytest.skip("LR model not trained")
        result = compute_shap(
            trained.models[MODEL_LR].pipeline,
            split.X_train, split.X_test,
            b.feature_names, MODEL_LR,
        )
        assert result.shap_values.shape[1] == 30

    def test_global_importance_shape(self, shap_setup):
        b, split, trained = shap_setup
        from src.utils.config import MODEL_RF
        if MODEL_RF not in trained.models:
            pytest.skip("RF model not trained")
        result = compute_shap(
            trained.models[MODEL_RF].pipeline,
            split.X_train, split.X_test,
            b.feature_names, MODEL_RF,
        )
        assert result.global_importance.shape == (30,)
        assert np.all(result.global_importance >= 0)

    def test_feature_names_match(self, shap_setup):
        b, split, trained = shap_setup
        from src.utils.config import MODEL_RF
        if MODEL_RF not in trained.models:
            pytest.skip("RF model not trained")
        result = compute_shap(
            trained.models[MODEL_RF].pipeline,
            split.X_train, split.X_test,
            b.feature_names, MODEL_RF,
        )
        assert result.feature_names == b.feature_names
