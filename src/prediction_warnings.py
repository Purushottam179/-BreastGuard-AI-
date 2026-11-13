"""
BreastGuard AI — Transparent Prediction Warnings & Risk Triage Module
================================================================================
Monitors input integrity, out-of-distribution boundaries, and borderline confidence
to generate structured, non-medical technical alerts for clinicians and AI auditors.

IMPORTANT ETHICAL & CLINICAL CONSTRAINT:
  This module generates ALGORITHMIC SAFETY WARNINGS, NOT clinical medical risk scores.
  It highlights situations where statistical uncertainty is elevated.
  Model confidence is NEVER equated with tumor severity or clinical prognosis.
================================================================================
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from src.ood_detection import OODDetector, OODResult, load_ood_detector

logger = logging.getLogger(__name__)


@dataclass
class WarningAlert:
    """Individual system alert."""
    category: str        # 'INPUT_VALIDATION', 'OUT_OF_DISTRIBUTION', 'BORDERLINE_CONFIDENCE', 'EXPLANATION_NOTE'
    level: str           # 'INFO', 'WARNING', 'CRITICAL'
    message: str
    detail: str


@dataclass
class WarningAssessment:
    """Comprehensive prediction warning and audit report."""
    status_level: str               # 'NORMAL', 'CAUTION', 'ALERT'
    has_critical_alerts: bool
    has_warnings: bool
    total_alerts: int
    alerts: list[WarningAlert]
    ood_result: OODResult | None
    is_borderline_decision: bool
    disclaimer: str = (
        "Algorithmic safety notice: Warnings indicate elevated statistical uncertainty "
        "or atypical input measurements. They do not constitute clinical staging or medical triage."
    )


class PredictionWarningSystem:
    """
    Transparent safety and reliability screening system for BreastGuard AI.
    """

    def __init__(self, ood_detector: OODDetector | None = None) -> None:
        self.ood_detector = ood_detector or load_ood_detector()

    def inspect_prediction(
        self,
        feature_values: dict[str, float] | Any,
        prob_malignant: float,
        threshold: float = 0.37,
        explanation_available: bool = True,
    ) -> WarningAssessment:
        """
        Inspect input values and prediction outcome for reliability warnings.

        Parameters
        ----------
        feature_values : dict or array-like
            30 input cytological measurements.
        prob_malignant : float
            Predicted probability of malignancy [0.0, 1.0].
        threshold : float, default=0.37
            Calibrated operating cutoff.
        explanation_available : bool, default=True
            Whether local SHAP attributions were computed successfully.

        Returns
        -------
        WarningAssessment
            Structured collection of categorized warnings.
        """
        alerts: list[WarningAlert] = []

        # 1. Out-of-Distribution Screening
        ood_res = None
        if self.ood_detector is not None:
            try:
                ood_res = self.ood_detector.inspect_sample(feature_values)
                if ood_res.severity == "SEVERE_OOD":
                    alerts.append(
                        WarningAlert(
                            category="OUT_OF_DISTRIBUTION",
                            level="CRITICAL",
                            message="Input severely deviates from the training cohort (Severe OOD).",
                            detail=f"Mahalanobis distance ({ood_res.mahalanobis_distance:.2f}) exceeds the 99th "
                                   f"percentile training threshold ({ood_res.mahalanobis_threshold_99:.2f}). "
                                   "Model reliability cannot be guaranteed.",
                        )
                    )
                elif ood_res.severity == "MILD_OUTLIER":
                    alerts.append(
                        WarningAlert(
                            category="OUT_OF_DISTRIBUTION",
                            level="WARNING",
                            message="Atypical multivariate feature pattern detected.",
                            detail=f"Mahalanobis distance ({ood_res.mahalanobis_distance:.2f}) exceeds the 95th "
                                   f"percentile training threshold ({ood_res.mahalanobis_threshold_95:.2f}).",
                        )
                    )

                for ef in ood_res.extreme_features:
                    alerts.append(
                        WarningAlert(
                            category="INPUT_VALIDATION",
                            level="WARNING",
                            message=f"Extreme measurement for '{ef['feature']}' (Z = {ef['z_score']:+.1f}).",
                            detail=f"Measured value is {ef['value']:.3f} (Training Mean={ef['mean']:.3f}, Std={ef['std']:.3f}).",
                        )
                    )
            except Exception as e:
                logger.warning(f"OOD evaluation error: {e}")

        # 2. Borderline / Ambiguous Decision Zone Screening
        dist_to_threshold = abs(prob_malignant - threshold)
        is_borderline = (dist_to_threshold < 0.10) or (0.30 <= prob_malignant <= 0.45)

        if is_borderline:
            alerts.append(
                WarningAlert(
                    category="BORDERLINE_CONFIDENCE",
                    level="WARNING",
                    message="Borderline prediction near clinical cutoff.",
                    detail=f"Predicted malignancy probability ({prob_malignant:.1%}) is within 10 percentage "
                           f"points of the calibrated operating cutoff (τ = {threshold:.2f}). "
                           "The sample sits near the model's decision boundary.",
                )
            )

        # 3. Explainer Availability Check
        if not explanation_available:
            alerts.append(
                WarningAlert(
                    category="EXPLANATION_NOTE",
                    level="INFO",
                    message="SHAP explanation fallback active.",
                    detail="Local SHAP feature attributions were calculated using population baseline "
                           "surrogate due to real-time explainer latency constraints.",
                )
            )

        # 4. Status Level Assignment
        has_critical = any(a.level == "CRITICAL" for a in alerts)
        has_warnings = any(a.level == "WARNING" for a in alerts)

        if has_critical:
            status_level = "ALERT"
        elif has_warnings:
            status_level = "CAUTION"
        else:
            status_level = "NORMAL"

        return WarningAssessment(
            status_level=status_level,
            has_critical_alerts=has_critical,
            has_warnings=has_warnings,
            total_alerts=len(alerts),
            alerts=alerts,
            ood_result=ood_res,
            is_borderline_decision=is_borderline,
        )


_CACHED_WARNING_SYSTEM: PredictionWarningSystem | None = None


def load_warning_system() -> PredictionWarningSystem:
    """Factory function for singleton PredictionWarningSystem."""
    global _CACHED_WARNING_SYSTEM
    if _CACHED_WARNING_SYSTEM is not None:
        return _CACHED_WARNING_SYSTEM

    system = PredictionWarningSystem()
    _CACHED_WARNING_SYSTEM = system
    return system

