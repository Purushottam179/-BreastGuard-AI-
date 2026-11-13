"""
BreastGuard AI — Model Confidence & Prediction Reliability Module
================================================================================
Evaluates model calibration, confidence distributions, and reliability metrics:
  - Brier Score: Mean squared error of probabilistic outputs
  - Expected Calibration Error (ECE): Frequency-weighted discrepancy between confidence and accuracy
  - Maximum Calibration Error (MCE): Worst-case bin calibration gap
  - Reliability Diagrams (Calibration Curves): Visual comparison against perfect calibration
  - Single-prediction confidence assessment and borderline ambiguity flagging

IMPORTANT CLINICAL SAFETY NOTICE:
  Model confidence represents the statistical likelihood under the training distribution.
  It MUST NOT be interpreted as clinical certainty, disease severity, or a replacement
  for histopathological confirmation.
================================================================================
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.metrics import brier_score_loss

logger = logging.getLogger(__name__)


@dataclass
class ReliabilityBin:
    """Individual calibration bin metrics."""
    bin_index: int
    bin_lower: float
    bin_upper: float
    bin_center: float
    sample_count: int
    mean_confidence: float
    empirical_accuracy: float
    calibration_gap: float  # |empirical_accuracy - mean_confidence|


@dataclass
class ReliabilityReport:
    """Comprehensive calibration and reliability analysis summary."""
    brier_score: float
    ece: float  # Expected Calibration Error
    mce: float  # Maximum Calibration Error
    n_bins: int
    sample_count: int
    bins: list[ReliabilityBin]
    mean_confidence: float
    observed_prevalence: float
    high_confidence_ratio: float
    low_confidence_ratio: float
    binning_method: str = "equal_width"
    disclaimer: str = (
        "Statistical calibration metrics reflect empirical alignment on evaluation partitions. "
        "They do not guarantee individual patient outcome certainty."
    )


@dataclass
class ConfidenceAssessment:
    """Single-prediction reliability and confidence evaluation."""
    predicted_label: str
    prob_malignant: float
    prob_benign: float
    confidence_score: float  # max(prob_malignant, prob_benign)
    operating_threshold: float
    confidence_tier: str     # 'High Confidence', 'Moderate Confidence', 'Borderline / Ambiguous'
    distance_to_threshold: float
    is_borderline: bool
    warning_message: str | None = None
    calibrated_guidance: str = ""


def compute_expected_calibration_error(
    y_true: np.ndarray | list[int],
    y_prob: np.ndarray | list[float],
    n_bins: int = 10,
) -> tuple[float, float, list[ReliabilityBin]]:
    """
    Compute Expected Calibration Error (ECE) and Maximum Calibration Error (MCE)
    using equal-width probability binning.

    Formula:
        ECE = sum_{m=1}^{M} (|B_m| / N) * |acc(B_m) - conf(B_m)|
        MCE = max_{m=1}^{M} |acc(B_m) - conf(B_m)|

    Parameters
    ----------
    y_true : np.ndarray or list of int
        True binary labels (0 = Benign, 1 = Malignant).
    y_prob : np.ndarray or list of float
        Predicted probabilities of the positive class (Malignant).
    n_bins : int, default=10
        Number of equal-width probability bins spanning [0.0, 1.0].

    Returns
    -------
    tuple[float, float, list[ReliabilityBin]]
        (ece, mce, list of ReliabilityBin objects)
    """
    y_true_arr = np.asarray(y_true, dtype=int)
    y_prob_arr = np.clip(np.asarray(y_prob, dtype=float), 0.0, 1.0)
    n_samples = len(y_true_arr)

    if n_samples == 0:
        return 0.0, 0.0, []

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins_data: list[ReliabilityBin] = []
    weighted_error_sum = 0.0
    max_error = 0.0

    for i in range(n_bins):
        lower = bin_edges[i]
        upper = bin_edges[i + 1]

        # Right-inclusive for the last bin
        if i == n_bins - 1:
            in_bin = (y_prob_arr >= lower) & (y_prob_arr <= upper)
        else:
            in_bin = (y_prob_arr >= lower) & (y_prob_arr < upper)

        bin_count = int(np.sum(in_bin))

        if bin_count > 0:
            mean_conf = float(np.mean(y_prob_arr[in_bin]))
            emp_acc = float(np.mean(y_true_arr[in_bin]))
            gap = float(abs(emp_acc - mean_conf))

            weighted_error_sum += (bin_count / n_samples) * gap
            max_error = max(max_error, gap)
        else:
            mean_conf = (lower + upper) / 2.0
            emp_acc = 0.0
            gap = 0.0

        bins_data.append(
            ReliabilityBin(
                bin_index=i + 1,
                bin_lower=round(lower, 3),
                bin_upper=round(upper, 3),
                bin_center=round((lower + upper) / 2.0, 3),
                sample_count=bin_count,
                mean_confidence=round(mean_conf, 4),
                empirical_accuracy=round(emp_acc, 4),
                calibration_gap=round(gap, 4),
            )
        )

    ece = float(weighted_error_sum)
    mce = float(max_error)

    return round(ece, 5), round(mce, 5), bins_data


def evaluate_model_calibration(
    y_true: np.ndarray | list[int],
    y_prob: np.ndarray | list[float],
    n_bins: int = 10,
) -> ReliabilityReport:
    """
    Run full probabilistic calibration and reliability analysis.

    Parameters
    ----------
    y_true : np.ndarray or list of int
        True binary labels (0 = Benign, 1 = Malignant).
    y_prob : np.ndarray or list of float
        Predicted probabilities of class 1.
    n_bins : int, default=10
        Number of calibration bins.

    Returns
    -------
    ReliabilityReport
        Populated reliability metrics.
    """
    y_true_arr = np.asarray(y_true, dtype=int)
    y_prob_arr = np.clip(np.asarray(y_prob, dtype=float), 0.0, 1.0)

    bs = float(brier_score_loss(y_true_arr, y_prob_arr))
    ece, mce, bins = compute_expected_calibration_error(y_true_arr, y_prob_arr, n_bins=n_bins)

    conf_scores = np.maximum(y_prob_arr, 1.0 - y_prob_arr)
    high_conf = float(np.mean(conf_scores >= 0.85))
    low_conf = float(np.mean((y_prob_arr >= 0.35) & (y_prob_arr <= 0.65)))

    return ReliabilityReport(
        brier_score=round(bs, 5),
        ece=ece,
        mce=mce,
        n_bins=n_bins,
        sample_count=len(y_true_arr),
        bins=bins,
        mean_confidence=round(float(np.mean(conf_scores)), 4),
        observed_prevalence=round(float(np.mean(y_true_arr)), 4),
        high_confidence_ratio=round(high_conf, 4),
        low_confidence_ratio=round(low_conf, 4),
    )


def assess_prediction_confidence(
    prob_malignant: float,
    threshold: float = 0.37,
) -> ConfidenceAssessment:
    """
    Assess prediction confidence and flag borderline ambiguous samples.

    Parameters
    ----------
    prob_malignant : float
        Predicted probability of malignancy [0.0, 1.0].
    threshold : float, default=0.37
        Calibrated decision threshold.

    Returns
    -------
    ConfidenceAssessment
        Detailed reliability and confidence assessment.
    """
    p_mal = float(np.clip(prob_malignant, 0.0, 1.0))
    p_ben = 1.0 - p_mal
    confidence = max(p_mal, p_ben)
    dist_threshold = abs(p_mal - threshold)

    pred_label = "Malignant" if p_mal >= threshold else "Benign"

    # Define confidence tiers based on distance to threshold and absolute probability
    is_borderline = (dist_threshold < 0.10) or (0.30 <= p_mal <= 0.45)

    if is_borderline:
        tier = "Borderline / Ambiguous"
        warning = (
            f"Prediction is close to the clinical decision cutoff (τ = {threshold:.2f}). "
            f"Distance to threshold is {dist_threshold:.3f}. Clinical corroboration strongly advised."
        )
        guidance = (
            "Caution: Borderline predictions indicate cytological measurements near the model's "
            "separating hyperplane. Histopathological review recommended."
        )
    elif confidence >= 0.85:
        tier = "High Confidence"
        warning = None
        guidance = (
            "Model exhibits high statistical margin on this sample. "
            "Note: High statistical confidence does not guarantee biological certainty."
        )
    else:
        tier = "Moderate Confidence"
        warning = (
            "Prediction has moderate statistical confidence. Review top SHAP feature contributors."
        )
        guidance = (
            "Intermediate probability observed. Consider checking feature measurements for subtle atypia."
        )

    return ConfidenceAssessment(
        predicted_label=pred_label,
        prob_malignant=round(p_mal, 4),
        prob_benign=round(p_ben, 4),
        confidence_score=round(confidence, 4),
        operating_threshold=threshold,
        confidence_tier=tier,
        distance_to_threshold=round(dist_threshold, 4),
        is_borderline=is_borderline,
        warning_message=warning,
        calibrated_guidance=guidance,
    )


def plot_reliability_diagram_plotly(report: ReliabilityReport) -> go.Figure:
    """
    Render a Plotly reliability diagram (calibration curve) with bin counts.

    Parameters
    ----------
    report : ReliabilityReport
        Computed calibration report.

    Returns
    -------
    go.Figure
        Styled interactive reliability chart.
    """
    confs = [b.mean_confidence for b in report.bins if b.sample_count > 0]
    accs = [b.empirical_accuracy for b in report.bins if b.sample_count > 0]
    counts = [b.sample_count for b in report.bins if b.sample_count > 0]
    bin_labels = [f"[{b.bin_lower:.2f}, {b.bin_upper:.2f}] (N={b.sample_count})" for b in report.bins if b.sample_count > 0]

    fig = go.Figure()

    # Perfect calibration reference diagonal (y = x)
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            name="Perfect Calibration (y = x)",
            line=dict(color="#94A3B8", dash="dash", width=2),
        )
    )

    # Observed calibration curve
    fig.add_trace(
        go.Scatter(
            x=confs,
            y=accs,
            mode="lines+markers",
            name=f"Model Calibration (ECE = {report.ece:.4f})",
            line=dict(color="#3B82F6", width=2.5),
            marker=dict(size=8, color="#60A5FA", symbol="circle"),
            text=bin_labels,
            hoverinfo="x+y+text+name",
        )
    )

    # Bin gap bars (visualizing discrepancy |acc - conf|)
    for c, a, lbl in zip(confs, accs, bin_labels):
        fig.add_trace(
            go.Scatter(
                x=[c, c],
                y=[c, a],
                mode="lines",
                showlegend=False,
                line=dict(color="rgba(239, 68, 68, 0.5)", width=2),
                hoverinfo="skip",
            )
        )

    fig.update_layout(
        title=f"<b>Reliability Diagram (ECE = {report.ece:.4f}, Brier = {report.brier_score:.4f})</b>",
        xaxis=dict(title="Mean Predicted Probability (Confidence)", range=[-0.02, 1.02], gridcolor="#2D3748"),
        yaxis=dict(title="Empirical Accuracy (Fraction of Positives)", range=[-0.02, 1.02], gridcolor="#2D3748"),
        paper_bgcolor="#161B26",
        plot_bgcolor="#161B26",
        font=dict(color="#E2E8F0", family="Inter"),
        height=380,
        margin=dict(l=40, r=20, t=50, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def plot_confidence_distribution_plotly(y_prob: np.ndarray | list[float]) -> go.Figure:
    """
    Render confidence score distribution histogram showing separation toward 0 and 1.

    Parameters
    ----------
    y_prob : np.ndarray or list of float
        Predicted probabilities.

    Returns
    -------
    go.Figure
        Styled histogram.
    """
    probs = np.asarray(y_prob, dtype=float)
    fig = go.Figure()

    fig.add_trace(
        go.Histogram(
            x=probs,
            nbinsx=25,
            marker=dict(color="#3B82F6", line=dict(color="#1E293B", width=1)),
            opacity=0.8,
            name="Predictions",
        )
    )

    # Highlight operating threshold 0.37
    fig.add_vline(
        x=0.37,
        line_dash="dot",
        line_color="#FBBF24",
        line_width=2,
        annotation_text="Decision Cutoff (τ = 0.37)",
        annotation_position="top left",
        annotation_font=dict(color="#FBBF24", size=11),
    )

    fig.update_layout(
        title="<b>Predicted Probability Distribution (Target: Malignant)</b>",
        xaxis=dict(title="Probability P(Malignant)", range=[-0.02, 1.02], gridcolor="#2D3748"),
        yaxis=dict(title="Biopsy Count", gridcolor="#2D3748"),
        paper_bgcolor="#161B26",
        plot_bgcolor="#161B26",
        font=dict(color="#E2E8F0", family="Inter"),
        height=320,
        margin=dict(l=40, r=20, t=50, b=40),
    )
    return fig

