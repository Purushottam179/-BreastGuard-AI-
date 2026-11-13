"""
BreastGuard AI — Feature-Based Subgroup & Fairness Limitations Module
================================================================================
Evaluates model performance across cytological and morphometric subgroups.

CRITICAL RESPONSIBLE AI DISCLOSURE:
  The Wisconsin Diagnostic Breast Cancer (WDBC) dataset does NOT contain demographic
  attributes (patient age, race, ethnicity, menopausal status, genetic markers, or
  socioeconomic factors).
  
  THEREFORE:
    1. We DO NOT invent or synthesize fictional demographic groups.
    2. We DO NOT infer demographic characteristics from nuclear morphology.
    3. We DO NOT claim that BreastGuard AI is demographically unbiased.
    4. Subgroup analyses presented here are strictly FEATURE-BASED CYTOLOGICAL STRATA
       (e.g., nuclear radius / tumor dimension cohorts), designed to assess model
       sensitivity across early-stage vs advanced morphological presentations.
================================================================================
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go

logger = logging.getLogger(__name__)


@dataclass
class SubgroupMetrics:
    """Performance metrics for an individual feature-based subgroup."""
    subgroup_name: str
    feature_criteria: str
    sample_count: int
    malignant_count: int
    benign_count: int
    prevalence: float
    accuracy: float
    sensitivity: float  # Recall on Malignant
    specificity: float  # Recall on Benign
    precision: float
    false_negative_rate: float  # FNR = 1 - Sensitivity
    false_positive_rate: float  # FPR = 1 - Specificity
    tp: int
    fn: int
    fp: int
    tn: int


@dataclass
class SubgroupAnalysisReport:
    """Complete collection of feature-based subgroup evaluations."""
    stratum_name: str
    subgroups: list[SubgroupMetrics]
    operating_threshold: float
    overall_sample_count: int
    demographic_data_present: bool = False
    fairness_disclaimer: str = (
        "NOTICE ON DEMOGRAPHIC FAIRNESS: WDBC contains no patient demographic records "
        "(age, race, ethnicity, or socioeconomic status). The strata analyzed represent "
        "cytological tumor size and texture categories. True demographic equity requires "
        "prospective multi-center demographic data collection."
    )


def compute_binary_metrics_for_mask(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    mask: np.ndarray,
    subgroup_name: str,
    criteria_str: str,
) -> SubgroupMetrics | None:
    """Compute standard clinical classification metrics for a boolean subset mask."""
    sub_true = y_true[mask]
    sub_pred = y_pred[mask]
    n = len(sub_true)

    if n == 0:
        return None

    mal_count = int(np.sum(sub_true == 1))
    ben_count = int(np.sum(sub_true == 0))
    prev = mal_count / n if n > 0 else 0.0

    tp = int(np.sum((sub_true == 1) & (sub_pred == 1)))
    fn = int(np.sum((sub_true == 1) & (sub_pred == 0)))
    fp = int(np.sum((sub_true == 0) & (sub_pred == 1)))
    tn = int(np.sum((sub_true == 0) & (sub_pred == 0)))

    acc = (tp + tn) / n if n > 0 else 0.0
    sens = tp / (tp + fn) if (tp + fn) > 0 else 1.0  # If no positive cases, sensitivity not penalized
    spec = tn / (tn + fp) if (tn + fp) > 0 else 1.0
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    fnr = 1.0 - sens
    fpr = 1.0 - spec

    return SubgroupMetrics(
        subgroup_name=subgroup_name,
        feature_criteria=criteria_str,
        sample_count=n,
        malignant_count=mal_count,
        benign_count=ben_count,
        prevalence=round(prev, 4),
        accuracy=round(acc, 4),
        sensitivity=round(sens, 4),
        specificity=round(spec, 4),
        precision=round(prec, 4),
        false_negative_rate=round(fnr, 4),
        false_positive_rate=round(fpr, 4),
        tp=tp,
        fn=fn,
        fp=fp,
        tn=tn,
    )


def evaluate_tumor_size_subgroups(
    df: pd.DataFrame,
    y_true: np.ndarray | list[int],
    y_prob: np.ndarray | list[float],
    threshold: float = 0.37,
) -> SubgroupAnalysisReport:
    """
    Evaluate model performance stratified across clinical tumor size tiers based on mean_radius.

    Strata:
      - Small / Early Biopsies: mean_radius <= 13.0 mm (predominantly benign, early screening)
      - Intermediate Biopsies: 13.0 < mean_radius <= 17.5 mm (borderline/transitional zone)
      - Large / Advanced Biopsies: mean_radius > 17.5 mm (high malignant prevalence)
    """
    y_true_arr = np.asarray(y_true, dtype=int)
    y_prob_arr = np.asarray(y_prob, dtype=float)
    y_pred_arr = (y_prob_arr >= threshold).astype(int)

    radius = df["mean_radius"].values

    mask_small = (radius <= 13.0)
    mask_inter = (radius > 13.0) & (radius <= 17.5)
    mask_large = (radius > 17.5)

    subgroups: list[SubgroupMetrics] = []
    
    m1 = compute_binary_metrics_for_mask(
        y_true_arr, y_pred_arr, mask_small,
        subgroup_name="Small Nuclei / Early Biopsies",
        criteria_str="mean_radius <= 13.0 mm",
    )
    if m1:
        subgroups.append(m1)

    m2 = compute_binary_metrics_for_mask(
        y_true_arr, y_pred_arr, mask_inter,
        subgroup_name="Intermediate Biopsies",
        criteria_str="13.0 < mean_radius <= 17.5 mm",
    )
    if m2:
        subgroups.append(m2)

    m3 = compute_binary_metrics_for_mask(
        y_true_arr, y_pred_arr, mask_large,
        subgroup_name="Large / Advanced Biopsies",
        criteria_str="mean_radius > 17.5 mm",
    )
    if m3:
        subgroups.append(m3)

    return SubgroupAnalysisReport(
        stratum_name="Nuclear Dimension (mean_radius) Strata",
        subgroups=subgroups,
        operating_threshold=threshold,
        overall_sample_count=len(y_true_arr),
    )


def evaluate_nuclear_texture_subgroups(
    df: pd.DataFrame,
    y_true: np.ndarray | list[int],
    y_prob: np.ndarray | list[float],
    threshold: float = 0.37,
) -> SubgroupAnalysisReport:
    """
    Evaluate model performance stratified across nuclear chromatin texture tiers.

    Strata:
      - Low / Homogeneous Texture: worst_texture <= 22.0
      - Moderate Texture: 22.0 < worst_texture <= 30.0
      - High / Heterogeneous Texture: worst_texture > 30.0
    """
    y_true_arr = np.asarray(y_true, dtype=int)
    y_prob_arr = np.asarray(y_prob, dtype=float)
    y_pred_arr = (y_prob_arr >= threshold).astype(int)

    texture = df["worst_texture"].values

    mask_low = (texture <= 22.0)
    mask_mod = (texture > 22.0) & (texture <= 30.0)
    mask_high = (texture > 30.0)

    subgroups: list[SubgroupMetrics] = []
    
    m1 = compute_binary_metrics_for_mask(
        y_true_arr, y_pred_arr, mask_low,
        subgroup_name="Low / Homogeneous Texture",
        criteria_str="worst_texture <= 22.0",
    )
    if m1:
        subgroups.append(m1)

    m2 = compute_binary_metrics_for_mask(
        y_true_arr, y_pred_arr, mask_mod,
        subgroup_name="Moderate Texture",
        criteria_str="22.0 < worst_texture <= 30.0",
    )
    if m2:
        subgroups.append(m2)

    m3 = compute_binary_metrics_for_mask(
        y_true_arr, y_pred_arr, mask_high,
        subgroup_name="High / Heterogeneous Texture",
        criteria_str="worst_texture > 30.0",
    )
    if m3:
        subgroups.append(m3)

    return SubgroupAnalysisReport(
        stratum_name="Chromatin Texture (worst_texture) Strata",
        subgroups=subgroups,
        operating_threshold=threshold,
        overall_sample_count=len(y_true_arr),
    )


def plot_subgroup_metrics_plotly(report: SubgroupAnalysisReport) -> go.Figure:
    """
    Render grouped bar chart comparing Sensitivity and Specificity across subgroups.
    """
    names = [s.subgroup_name for s in report.subgroups]
    sens = [s.sensitivity * 100 for s in report.subgroups]
    specs = [s.specificity * 100 for s in report.subgroups]
    fnrs = [s.false_negative_rate * 100 for s in report.subgroups]
    counts = [f"N={s.sample_count} (Mal={s.malignant_count})" for s in report.subgroups]

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            name="Sensitivity (Malignant Recall)",
            x=names,
            y=sens,
            marker_color="#EF4444",
            text=[f"{v:.1f}%" for v in sens],
            textposition="auto",
        )
    )

    fig.add_trace(
        go.Bar(
            name="Specificity (Benign Recall)",
            x=names,
            y=specs,
            marker_color="#10B981",
            text=[f"{v:.1f}%" for v in specs],
            textposition="auto",
        )
    )

    fig.add_trace(
        go.Bar(
            name="False Negative Rate (Missed Cancers)",
            x=names,
            y=fnrs,
            marker_color="#F59E0B",
            text=[f"{v:.1f}%" for v in fnrs],
            textposition="auto",
        )
    )

    fig.update_layout(
        title=f"<b>Performance Across {report.stratum_name}</b>",
        barmode="group",
        yaxis=dict(title="Metric (%)", range=[0, 108], gridcolor="#2D3748"),
        paper_bgcolor="#161B26",
        plot_bgcolor="#161B26",
        font=dict(color="#E2E8F0", family="Inter"),
        height=360,
        margin=dict(l=40, r=20, t=50, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig

