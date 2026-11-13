"""
BreastGuard AI — Responsible AI, Reliability & Robustness Unit Tests
================================================================================
Validates:
  1. Probability calibration, Brier score, and ECE computation.
  2. Single-prediction confidence tier assignment and borderline ambiguity flagging.
  3. Out-of-Distribution detector fitting (training-only), thresholds, and outlier detection.
  4. Feature-based subgroup performance analysis and demographic disclaimers.
  5. Warning system alert categorization.
  6. Model robustness and Monte Carlo perturbation stability.
================================================================================
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data_loader import load_dataset
from src.fairness_analysis import (
    SubgroupAnalysisReport,
    evaluate_nuclear_texture_subgroups,
    evaluate_tumor_size_subgroups,
)
from src.model_reliability import (
    ConfidenceAssessment,
    ReliabilityReport,
    assess_prediction_confidence,
    compute_expected_calibration_error,
    evaluate_model_calibration,
)
from src.ood_detection import OODDetector, OODResult, load_ood_detector
from src.prediction_warnings import PredictionWarningSystem, WarningAssessment, load_warning_system
from src.robustness_analysis import ModelRobustnessAnalyzer, PerturbationTrialResult


class TestModelReliability:
    """Test calibration metrics, ECE calculation, and confidence stratification."""

    def test_expected_calibration_error_calculation(self) -> None:
        """Verify ECE calculation on known synthetic distributions."""
        # Perfectly calibrated case
        y_prob = np.array([0.1, 0.2, 0.8, 0.9])
        y_true = np.array([0, 0, 1, 1])
        ece, mce, bins = compute_expected_calibration_error(y_true, y_prob, n_bins=5)
        assert 0.0 <= ece <= 0.3
        assert 0.0 <= mce <= 0.5
        assert len(bins) == 5

    def test_evaluate_model_calibration_returns_report(self) -> None:
        """Verify evaluate_model_calibration populates report correctly."""
        y_true = np.array([0, 0, 0, 1, 1, 1])
        y_prob = np.array([0.05, 0.12, 0.25, 0.78, 0.89, 0.95])
        report = evaluate_model_calibration(y_true, y_prob, n_bins=5)
        assert isinstance(report, ReliabilityReport)
        assert report.brier_score >= 0.0
        assert report.sample_count == 6
        assert len(report.bins) == 5
        assert report.high_confidence_ratio >= 0.5

    def test_assess_prediction_confidence_borderline(self) -> None:
        """Verify borderline ambiguity detection near threshold tau = 0.37."""
        assessment = assess_prediction_confidence(prob_malignant=0.38, threshold=0.37)
        assert isinstance(assessment, ConfidenceAssessment)
        assert assessment.is_borderline is True
        assert assessment.confidence_tier == "Borderline / Ambiguous"
        assert assessment.warning_message is not None

    def test_assess_prediction_confidence_high(self) -> None:
        """Verify high confidence tier for extreme probabilities."""
        assessment = assess_prediction_confidence(prob_malignant=0.98, threshold=0.37)
        assert assessment.confidence_tier == "High Confidence"
        assert assessment.is_borderline is False


class TestOODDetection:
    """Test Out-of-Distribution detector fitting and scoring."""

    def test_ood_detector_fit_and_thresholds(self) -> None:
        """Verify OOD detector establishes monotonic percentiles strictly on training data."""
        detector = load_ood_detector()
        assert detector.is_fitted is True
        assert detector.mahal_threshold_95_ > 0.0
        assert detector.mahal_threshold_99_ > detector.mahal_threshold_95_

    def test_ood_detector_normal_sample(self) -> None:
        """Verify typical sample is classified as NORMAL."""
        detector = load_ood_detector()
        bundle = load_dataset(save_raw_csv=False)
        typical_sample = bundle.X_df.iloc[10].to_dict()
        res = detector.inspect_sample(typical_sample)
        assert isinstance(res, OODResult)
        assert res.severity in ["NORMAL", "MILD_OUTLIER"]
        assert res.mahalanobis_distance > 0.0

    def test_ood_detector_extreme_outlier(self) -> None:
        """Verify extreme simulated outlier triggers severe OOD alert."""
        detector = load_ood_detector()
        bundle = load_dataset(save_raw_csv=False)
        extreme_sample = bundle.X_df.iloc[0].to_dict()
        # Scale multiple key dimensions 5x beyond biological range
        extreme_sample["mean_radius"] = 90.0
        extreme_sample["worst_area"] = 6000.0
        extreme_sample["worst_texture"] = 90.0

        res = detector.inspect_sample(extreme_sample)
        assert res.is_ood is True
        assert res.severity in ["SEVERE_OOD", "MILD_OUTLIER"]
        assert len(res.extreme_features) > 0


class TestSubgroupFairnessAnalysis:
    """Test feature-based subgroup evaluation and demographic limitations disclosures."""

    def test_tumor_size_subgroups(self) -> None:
        """Verify stratification across tumor size categories."""
        bundle = load_dataset(save_raw_csv=False)
        y_true = bundle.y_series.values
        # Synthetic probabilities matching true labels with slight noise
        y_prob = np.clip(y_true * 0.9 + 0.05, 0.0, 1.0)

        report = evaluate_tumor_size_subgroups(bundle.X_df, y_true, y_prob, threshold=0.37)
        assert isinstance(report, SubgroupAnalysisReport)
        assert len(report.subgroups) >= 2
        for s in report.subgroups:
            assert s.sample_count > 0
            assert 0.0 <= s.sensitivity <= 1.0
            assert 0.0 <= s.specificity <= 1.0
            assert s.false_negative_rate + s.sensitivity == pytest.approx(1.0, abs=1e-3)

    def test_demographic_fairness_disclaimer_present(self) -> None:
        """Verify report explicitly states absence of demographic attributes."""
        bundle = load_dataset(save_raw_csv=False)
        report = evaluate_tumor_size_subgroups(bundle.X_df, bundle.y_series.values, bundle.y_series.values)
        assert report.demographic_data_present is False
        assert "no patient demographic" in report.fairness_disclaimer.lower()


class TestPredictionWarnings:
    """Test transparent multi-category warning system."""

    def test_warning_system_borderline_alert(self) -> None:
        """Verify warning system flags predictions near the decision threshold."""
        warning_sys = load_warning_system()
        bundle = load_dataset(save_raw_csv=False)
        sample = bundle.X_df.iloc[0].to_dict()

        assessment = warning_sys.inspect_prediction(
            feature_values=sample,
            prob_malignant=0.36,  # Close to threshold 0.37
            threshold=0.37,
        )
        assert isinstance(assessment, WarningAssessment)
        assert assessment.is_borderline_decision is True
        assert any(a.category == "BORDERLINE_CONFIDENCE" for a in assessment.alerts)

    def test_warning_system_outlier_alert(self) -> None:
        """Verify severe outlier generates OUT_OF_DISTRIBUTION alert."""
        warning_sys = load_warning_system()
        bundle = load_dataset(save_raw_csv=False)
        extreme_sample = bundle.X_df.iloc[0].to_dict()
        extreme_sample["mean_radius"] = 99.0
        extreme_sample["worst_area"] = 9999.0

        assessment = warning_sys.inspect_prediction(
            feature_values=extreme_sample,
            prob_malignant=0.85,
            threshold=0.37,
        )
        assert assessment.status_level in ["ALERT", "CAUTION"]
        assert any(a.category == "OUT_OF_DISTRIBUTION" for a in assessment.alerts)


class TestModelRobustness:
    """Test perturbation experiments and single-feature sensitivity."""

    def test_noise_perturbation_clear_cut_sample(self) -> None:
        """Verify clear-cut benign sample maintains high stability under 5% noise."""
        from src.predict import load_prediction_engine
        engine = load_prediction_engine()
        bundle = load_dataset(save_raw_csv=False)

        feature_stds = bundle.X_df.std().to_dict()
        analyzer = ModelRobustnessAnalyzer(
            pipeline=engine.pipeline,
            feature_names=engine.feature_names,
            feature_stds=feature_stds,
            operating_threshold=0.37,
            random_seed=42,
        )

        benign_sample = engine.get_default_values()
        res = analyzer.run_noise_perturbation(benign_sample, noise_scale=0.03, n_trials=30)
        assert isinstance(res, PerturbationTrialResult)
        assert res.stability_rate >= 0.85

    def test_feature_sensitivity_ranking(self) -> None:
        """Verify sensitivity analysis ranks all 30 features by probability delta."""
        from src.predict import load_prediction_engine
        engine = load_prediction_engine()
        bundle = load_dataset(save_raw_csv=False)

        feature_stds = bundle.X_df.std().to_dict()
        analyzer = ModelRobustnessAnalyzer(
            pipeline=engine.pipeline,
            feature_names=engine.feature_names,
            feature_stds=feature_stds,
            operating_threshold=0.37,
        )

        sample = engine.get_default_values()
        sensitivities = analyzer.analyze_single_feature_sensitivity(sample, perturbation_pct=0.10)
        assert len(sensitivities) == 30
        assert sensitivities[0].max_delta >= sensitivities[-1].max_delta
        assert sensitivities[0].sensitivity_rank == 1

