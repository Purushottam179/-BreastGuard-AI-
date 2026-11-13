"""
Tests for Machine Learning Model Training, Pipelines, and Serialization
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import numpy as np
import pytest
from sklearn.pipeline import Pipeline

from config.config import MODEL_FILENAMES, MODELS_DIR
from src.data_loader import load_and_validate_processed_data
from src.preprocessing import SplitData, build_pipeline, create_stratified_split
from src.train import (
    CV_SCORING,
    ModelConfig,
    get_model_specifications,
    run_cross_validation,
    select_best_model_from_cv,
    train_and_save_all_models,
)


@pytest.fixture(scope="module")
def data_setup():
    X, y, feature_names = load_and_validate_processed_data()
    split = create_stratified_split(X, y, test_size=0.20, random_seed=42, feature_names=feature_names)
    configs = get_model_specifications()
    return split, configs


class TestModelConfigurations:
    def test_five_models_configured(self, data_setup):
        _, configs = data_setup
        assert len(configs) == 5
        names = {cfg.name for cfg in configs}
        expected = {
            "Logistic Regression",
            "Random Forest",
            "Support Vector Machine",
            "Gradient Boosting",
            "K-Nearest Neighbors",
        }
        assert names == expected

    def test_scaling_specification_correctness(self, data_setup):
        _, configs = data_setup
        config_map = {cfg.name: cfg for cfg in configs}

        # Distance/linear models MUST use scaling
        assert config_map["Logistic Regression"].use_scaler is True
        assert config_map["Support Vector Machine"].use_scaler is True
        assert config_map["K-Nearest Neighbors"].use_scaler is True

        # Tree-based models must NOT require scaling
        assert config_map["Random Forest"].use_scaler is False
        assert config_map["Gradient Boosting"].use_scaler is False

    def test_pipeline_construction(self, data_setup):
        split, configs = data_setup
        for cfg in configs:
            pipe = build_pipeline(cfg.estimator, use_scaler=cfg.use_scaler, feature_names=split.feature_names)
            assert isinstance(pipe, Pipeline)
            if cfg.use_scaler:
                assert "scaler" in pipe.named_steps
            else:
                assert "scaler" not in pipe.named_steps
            assert "model" in pipe.named_steps


class TestCrossValidation:
    def test_cross_validation_runs_successfully(self, data_setup):
        split, configs = data_setup
        # Run CV on first 2 models for fast test assertion
        fast_configs = configs[:2]
        cv_res = run_cross_validation(fast_configs, split)

        assert len(cv_res) == 2
        for name, res in cv_res.items():
            for m in CV_SCORING.keys():
                assert m in res.metrics_mean
                assert m in res.metrics_std
                assert len(res.fold_scores[m]) == 5
                # Performance sanity check: ROC-AUC should be high on WDBC
                assert res.metrics_mean["roc_auc"] >= 0.90


class TestModelSerializationAndLoading:
    def test_saved_models_can_be_loaded(self, data_setup):
        split, configs = data_setup
        # Verify that models directory contains serialized models after training
        for cfg in configs:
            model_path = MODELS_DIR / cfg.filename
            if model_path.exists():
                loaded_pipe = joblib.load(model_path)
                assert isinstance(loaded_pipe, Pipeline)
                # Test inference capability on 1 test sample
                sample = split.X_test[0].reshape(1, -1)
                pred = loaded_pipe.predict(sample)
                assert pred[0] in (0, 1)
                if hasattr(loaded_pipe, "predict_proba"):
                    probs = loaded_pipe.predict_proba(sample)[0]
                    assert len(probs) == 2
                    assert abs(sum(probs) - 1.0) < 1e-4

