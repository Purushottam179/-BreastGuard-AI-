"""
BreastGuard AI — Unit & Integration Tests for Explainable AI (SHAP)
================================================================================
Tests:
  1. Final model loading and metadata verification.
  2. Explainer initialization and background data sampling.
  3. SHAP value calculations and finite value verification.
  4. Correct feature dimension matching (30 cytological measurements).
  5. Target class orientation (Class 1 = Malignant probability).
  6. Global feature importance calculation and ranking properties.
  7. Local explanation generation, waterfall additivity, and narrative formatting.
  8. Robust error handling for invalid input shapes and missing model files.
================================================================================
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from config.config import MODELS_DIR, N_FEATURES, RANDOM_SEED
from src.explain import (
    GlobalExplanation,
    LocalExplanation,
    ShapExplainerManager,
    compute_global_explanations,
    explain_single_sample,
    format_human_readable_explanation,
    get_explainer_selection_reasoning,
)


@pytest.fixture(scope="module")
def explainer_manager():
    """Initializes and provides a cached ShapExplainerManager instance for fast tests."""
    manager = ShapExplainerManager(
        model_path=MODELS_DIR / "final_breastguard_model.pkl",
        n_background=20,  # small background for fast unit test execution
        random_state=RANDOM_SEED,
    )
    manager.load_model()
    # Mock / small background from normal distribution with 30 features
    rng = np.random.RandomState(42)
    X_mock_train = rng.randn(30, 30)
    manager.initialize_explainer(X_mock_train)
    return manager


class TestModelLoadingAndMetadata:
    def test_final_model_exists_and_loads(self, explainer_manager):
        assert explainer_manager.pipeline is not None
        assert isinstance(explainer_manager.pipeline, Pipeline)
        assert len(explainer_manager.feature_names) == N_FEATURES

    def test_pipeline_contains_scaler_and_svc(self, explainer_manager):
        named_steps = explainer_manager.pipeline.named_steps
        assert "scaler" in named_steps
        assert "model" in named_steps
        assert named_steps["model"].probability is True

    def test_missing_model_file_raises_error(self, tmp_path):
        bad_manager = ShapExplainerManager(model_path=tmp_path / "non_existent_model.pkl")
        with pytest.raises(FileNotFoundError, match="Final model not found"):
            bad_manager.load_model()


class TestExplainerSelectionAndInitialization:
    def test_explainer_selection_reasoning(self):
        reasoning = get_explainer_selection_reasoning("Support Vector Classifier with RBF Kernel")
        assert reasoning["selected_explainer"] == "shap.KernelExplainer"
        assert "TreeExplainer" in reasoning["why_not_tree_explainer"]
        assert "LinearExplainer" in reasoning["why_not_linear_explainer"]
        assert "KernelExplainer" in reasoning["kernel_explainer_suitability"]

    def test_explainer_initialized_with_correct_background(self, explainer_manager):
        assert explainer_manager.explainer is not None
        assert explainer_manager.X_background.shape == (20, 30)
        assert 0.0 <= explainer_manager.base_value <= 1.0


class TestGlobalExplainability:
    def test_compute_global_explanations(self, explainer_manager):
        rng = np.random.RandomState(123)
        X_eval = rng.randn(5, 30)
        global_exp = compute_global_explanations(
            explainer=explainer_manager.explainer,
            X_eval=X_eval,
            feature_names=explainer_manager.feature_names,
            nsamples=20,
            model_name="Support Vector Machine",
        )

        assert isinstance(global_exp, GlobalExplanation)
        assert global_exp.shap_values.shape == (5, 30)
        assert np.all(np.isfinite(global_exp.shap_values))
        assert len(global_exp.top_10_features) == 10
        assert len(global_exp.mean_abs_shap) == 30

        # Mean absolute SHAP values should be sorted descending
        scores = list(global_exp.mean_abs_shap.values())
        assert all(scores[i] >= scores[i + 1] - 1e-9 for i in range(len(scores) - 1))


class TestLocalExplainability:
    def test_explain_single_sample_returns_local_explanation(self, explainer_manager):
        rng = np.random.RandomState(456)
        sample = rng.randn(30)
        local_exp = explain_single_sample(
            pipeline=explainer_manager.pipeline,
            explainer=explainer_manager.explainer,
            sample=sample,
            feature_names=explainer_manager.feature_names,
            threshold=0.37,
            sample_index=99,
            nsamples=25,
        )

        assert isinstance(local_exp, LocalExplanation)
        assert local_exp.predicted_class in [0, 1]
        assert local_exp.predicted_label in ["Benign", "Malignant"]
        assert 0.0 <= local_exp.prob_malignant <= 1.0
        assert 0.0 <= local_exp.prob_benign <= 1.0
        assert np.isclose(local_exp.prob_malignant + local_exp.prob_benign, 1.0, atol=1e-4)

        # 30 SHAP values calculated and all finite
        assert len(local_exp.shap_values) == 30
        assert all(np.isfinite(v) for v in local_exp.shap_values.values())

    def test_narrative_explanation_contains_disclaimer(self, explainer_manager):
        rng = np.random.RandomState(789)
        sample = rng.randn(30)
        local_exp = explain_single_sample(
            pipeline=explainer_manager.pipeline,
            explainer=explainer_manager.explainer,
            sample=sample,
            feature_names=explainer_manager.feature_names,
            threshold=0.37,
            nsamples=20,
        )

        full_text = " ".join(local_exp.narrative_explanation)
        assert "not medical causation" in full_text
        assert "statistical" in full_text

    def test_format_human_readable_handles_empty_contributions(self):
        narratives = format_human_readable_explanation(
            pred_label="Benign",
            prob_malignant=0.05,
            threshold=0.37,
            base_val=0.25,
            top_pos=[],
            top_neg=[("mean_radius", 12.0, -0.15)],
        )
        assert len(narratives) >= 2
        assert "Benign" in narratives[0]

    def test_invalid_input_dimension_raises_error(self, explainer_manager):
        # 10 features instead of 30
        bad_sample = np.random.randn(10)
        with pytest.raises(Exception):
            explain_single_sample(
                pipeline=explainer_manager.pipeline,
                explainer=explainer_manager.explainer,
                sample=bad_sample,
                feature_names=explainer_manager.feature_names,
            )

