"""
BreastGuard AI — Unit & Integration Tests for Advanced Explainability Engine
================================================================================
Tests:
  1. XAIExplorer initialization and model loading.
  2. Individual sample explanation and feature contribution ranking.
  3. Cohort comparison calculations (Benign vs. Malignant statistics).
  4. Counterfactual explanation search, sparsity, and bounded perturbations.
  5. What-if single-feature trajectory sweeps and tipping point detection.
  6. Explanation caching and deterministic reproducibility.
  7. Robust error handling for out-of-bounds indices and invalid feature names.
================================================================================
"""
from __future__ import annotations

import numpy as np
import pytest

from src.counterfactual import CounterfactualExplainer, CounterfactualResult
from src.xai_explorer import (
    CohortComparisonResult,
    WhatIfSensitivityResult,
    XAIExplorer,
)


@pytest.fixture(scope="module")
def xai_explorer():
    """Initializes and returns shared XAIExplorer instance."""
    return XAIExplorer()


class TestIndividualSampleExploration:
    def test_explain_sample_by_index_valid(self, xai_explorer):
        # Sample #0 (Malignant)
        exp = xai_explorer.explain_sample_by_index(0, nsamples=40)
        assert exp.predicted_label in ("Malignant", "Benign")
        assert 0.0 <= exp.prob_malignant <= 1.0
        assert 0.0 <= exp.base_value <= 1.0
        assert len(exp.shap_values) == 30
        assert len(exp.top_positive_contributions) <= 5
        assert len(exp.top_negative_contributions) <= 5

    def test_explain_sample_by_index_out_of_bounds_raises(self, xai_explorer):
        with pytest.raises(IndexError, match="out of bounds"):
            xai_explorer.explain_sample_by_index(999999)

    def test_explain_custom_sample_valid(self, xai_explorer):
        row = xai_explorer.df_data.iloc[10][xai_explorer.feature_names].to_dict()
        exp = xai_explorer.explain_custom_sample(row, nsamples=40)
        assert exp.predicted_label in ("Malignant", "Benign")
        assert len(exp.shap_values) == 30


class TestCohortComparison:
    def test_compare_benign_vs_malignant(self, xai_explorer):
        comp = xai_explorer.compare_benign_vs_malignant(["worst_area", "worst_texture"])
        assert isinstance(comp, CohortComparisonResult)
        assert "worst_area" in comp.benign_stats
        assert "worst_area" in comp.malignant_stats

        # Malignant mean worst area should be substantially higher than benign
        assert comp.malignant_stats["worst_area"]["mean"] > comp.benign_stats["worst_area"]["mean"]
        assert comp.relative_difference_pct["worst_area"] > 50.0  # observed > 150%


class TestCounterfactualExplainer:
    def test_counterfactual_generation_flips_borderline_sample(self, xai_explorer):
        # Patient 99 is borderline malignant (P ~ 0.539)
        sample_99 = xai_explorer.df_data.iloc[99][xai_explorer.feature_names]
        cf_res = xai_explorer.run_counterfactual_analysis(sample_99, sample_id=99)

        assert isinstance(cf_res, CounterfactualResult)
        assert cf_res.counterfactual_found is True
        assert cf_res.best_counterfactual is not None
        assert cf_res.best_counterfactual.predicted_label == "Benign"
        assert cf_res.best_counterfactual.prob_malignant < xai_explorer.threshold

        # Verify sparsity is small and modifications respect bounds
        assert cf_res.best_counterfactual.sparsity >= 1
        for feat, details in cf_res.best_counterfactual.modified_features.items():
            low, high = cf_res.feature_bounds_used[feat]
            assert low <= details["counterfactual"] <= high

    def test_counterfactual_safety_disclaimer_present(self, xai_explorer):
        sample_0 = xai_explorer.df_data.iloc[0][xai_explorer.feature_names]
        cf_res = xai_explorer.run_counterfactual_analysis(sample_0)
        assert "NOT" in cf_res.safety_disclaimer or "NEVER" in cf_res.safety_disclaimer
        assert "medical" in cf_res.safety_disclaimer.lower()


class TestWhatIfAnalysis:
    def test_what_if_sweeps_correctly(self, xai_explorer):
        sample_99 = xai_explorer.df_data.iloc[99][xai_explorer.feature_names]
        what_if_res = xai_explorer.run_what_if_analysis(sample_99, "worst_area", n_points=25)

        assert isinstance(what_if_res, WhatIfSensitivityResult)
        assert len(what_if_res.sweep_values) == 25
        assert len(what_if_res.predicted_probs_malignant) == 25
        assert what_if_res.min_prob <= what_if_res.max_prob
        assert what_if_res.tipping_point_value is not None

    def test_what_if_invalid_feature_raises(self, xai_explorer):
        sample_0 = xai_explorer.df_data.iloc[0][xai_explorer.feature_names]
        with pytest.raises(ValueError, match="not in recognized feature list"):
            xai_explorer.run_what_if_analysis(sample_0, "non_existent_feature_xyz")


class TestExplanationConsistencyAndCaching:
    def test_reproducibility_via_caching(self, xai_explorer):
        exp1 = xai_explorer.explain_sample_by_index(5, nsamples=40, use_cache=True)
        exp2 = xai_explorer.explain_sample_by_index(5, nsamples=40, use_cache=True)

        for f in xai_explorer.feature_names:
            assert exp1.shap_values[f] == exp2.shap_values[f]

    def test_audit_consistency_passes(self, xai_explorer):
        audit = xai_explorer.audit_explanation_consistency(test_samples=[0, 19])
        assert audit["all_audits_passed"] is True
        assert audit["reproducibility_verified"] is True
        assert audit["additivity_verified"] is True

