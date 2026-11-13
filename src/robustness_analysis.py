"""
BreastGuard AI — Model Robustness & Input Perturbation Analysis Module
================================================================================
Evaluates model stability against small numerical perturbations in cytological measurements:
  - Measurement noise simulation (Gaussian perturbation scaled to feature std)
  - Prediction stability rate & class flip rate
  - Probability dispersion (mean, std, max swing)
  - Single-feature sensitivity ranking (elasticity analysis)
  - Stability comparison between clear-cut vs borderline samples

IMPORTANT CLINICAL VALIDATION NOTICE:
  Perturbation experiments evaluate mathematical stability under synthetic sensor noise.
  They DO NOT constitute clinical validation, nor do they simulate living tissue biology.
================================================================================
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from config.config import RANDOM_SEED

logger = logging.getLogger(__name__)


@dataclass
class PerturbationTrialResult:
    """Summary of a multi-trial perturbation experiment on a single sample."""
    sample_id: str
    baseline_prob: float
    baseline_label: str
    noise_scale_pct: float       # e.g., 5.0 for 5% feature noise
    n_trials: int
    stability_rate: float        # Fraction of trials maintaining baseline classification
    class_flip_rate: float       # Fraction of trials that flipped class
    mean_perturbed_prob: float
    prob_std: float
    max_prob_swing: float
    is_robust: bool              # stability_rate >= 0.90
    perturbed_probs: list[float] = field(default_factory=list)


@dataclass
class FeatureSensitivity:
    """Impact of perturbing an individual feature by a fixed percentage."""
    feature_name: str
    baseline_value: float
    baseline_prob: float
    prob_plus_10pct: float
    prob_minus_10pct: float
    max_delta: float
    sensitivity_rank: int


class ModelRobustnessAnalyzer:
    """
    Evaluates classifier sensitivity and prediction stability.
    """

    def __init__(
        self,
        pipeline: Any,
        feature_names: list[str],
        feature_stds: dict[str, float] | np.ndarray,
        operating_threshold: float = 0.37,
        random_seed: int = RANDOM_SEED,
    ) -> None:
        self.pipeline = pipeline
        self.feature_names = list(feature_names)
        self.operating_threshold = operating_threshold
        self.random_seed = random_seed

        if isinstance(feature_stds, dict):
            self.stds = np.array([feature_stds.get(f, 1.0) for f in self.feature_names])
        else:
            self.stds = np.asarray(feature_stds, dtype=float)

        self.stds[self.stds == 0] = 1.0

    def run_noise_perturbation(
        self,
        feature_values: dict[str, float] | np.ndarray,
        noise_scale: float = 0.05,
        n_trials: int = 100,
        sample_id: str = "sample",
    ) -> PerturbationTrialResult:
        """
        Inject Gaussian noise scaled to feature standard deviations across multiple trials.

        Parameters
        ----------
        feature_values : dict or array-like
            Baseline cytological measurements.
        noise_scale : float, default=0.05
            Noise standard deviation as fraction of feature std (0.05 = 5% noise).
        n_trials : int, default=100
            Number of stochastic Monte Carlo trials.
        sample_id : str
            Identifier for the evaluated biopsy.

        Returns
        -------
        PerturbationTrialResult
            Stability statistics across trials.
        """
        rng = np.random.default_rng(self.random_seed)

        if isinstance(feature_values, dict):
            x_base = np.array([float(feature_values[f]) for f in self.feature_names])
        else:
            x_base = np.asarray(feature_values, dtype=float).flatten()

        # Baseline prediction
        df_base = pd.DataFrame([x_base], columns=self.feature_names)
        p_base = float(self.pipeline.predict_proba(df_base)[0, 1])
        lbl_base = "Malignant" if p_base >= self.operating_threshold else "Benign"

        # Generate noise matrix: (N_trials x 30)
        noise = rng.normal(0, noise_scale * self.stds, size=(n_trials, len(self.feature_names)))
        x_perturbed = np.maximum(0.0, x_base + noise)  # Enforce non-negative cell dimensions

        df_trials = pd.DataFrame(x_perturbed, columns=self.feature_names)
        p_trials = self.pipeline.predict_proba(df_trials)[:, 1]

        lbl_trials = np.where(p_trials >= self.operating_threshold, "Malignant", "Benign")
        stable_count = int(np.sum(lbl_trials == lbl_base))
        stability_rate = stable_count / n_trials
        flip_rate = 1.0 - stability_rate

        mean_p = float(np.mean(p_trials))
        std_p = float(np.std(p_trials))
        max_swing = float(np.max(np.abs(p_trials - p_base)))

        return PerturbationTrialResult(
            sample_id=sample_id,
            baseline_prob=round(p_base, 4),
            baseline_label=lbl_base,
            noise_scale_pct=round(noise_scale * 100.0, 1),
            n_trials=n_trials,
            stability_rate=round(stability_rate, 4),
            class_flip_rate=round(flip_rate, 4),
            mean_perturbed_prob=round(mean_p, 4),
            prob_std=round(std_p, 4),
            max_prob_swing=round(max_swing, 4),
            is_robust=(stability_rate >= 0.90),
            perturbed_probs=[round(float(p), 4) for p in p_trials[:50]],
        )

    def analyze_single_feature_sensitivity(
        self,
        feature_values: dict[str, float] | np.ndarray,
        perturbation_pct: float = 0.10,
    ) -> list[FeatureSensitivity]:
        """
        Measure probability swing when perturbing each feature individually by ±10%.
        """
        if isinstance(feature_values, dict):
            x_base = np.array([float(feature_values[f]) for f in self.feature_names])
        else:
            x_base = np.asarray(feature_values, dtype=float).flatten()

        df_base = pd.DataFrame([x_base], columns=self.feature_names)
        p_base = float(self.pipeline.predict_proba(df_base)[0, 1])

        results: list[FeatureSensitivity] = []

        for i, feat in enumerate(self.feature_names):
            val = x_base[i]
            delta = val * perturbation_pct

            # +10%
            x_plus = x_base.copy()
            x_plus[i] = val + delta
            df_p = pd.DataFrame([x_plus], columns=self.feature_names)
            p_plus = float(self.pipeline.predict_proba(df_p)[0, 1])

            # -10%
            x_minus = x_base.copy()
            x_minus[i] = max(0.0, val - delta)
            df_m = pd.DataFrame([x_minus], columns=self.feature_names)
            p_minus = float(self.pipeline.predict_proba(df_m)[0, 1])

            max_delta = max(abs(p_plus - p_base), abs(p_minus - p_base))

            results.append(
                FeatureSensitivity(
                    feature_name=feat,
                    baseline_value=round(val, 4),
                    baseline_prob=round(p_base, 4),
                    prob_plus_10pct=round(p_plus, 4),
                    prob_minus_10pct=round(p_minus, 4),
                    max_delta=round(max_delta, 4),
                    sensitivity_rank=0,
                )
            )

        # Sort by maximum probability delta
        results.sort(key=lambda r: r.max_delta, reverse=True)
        for rank, item in enumerate(results, start=1):
            item.sensitivity_rank = rank

        return results


def plot_perturbation_distribution_plotly(result: PerturbationTrialResult) -> go.Figure:
    """Render histogram of probability distribution across noise trials."""
    fig = go.Figure()

    fig.add_trace(
        go.Histogram(
            x=result.perturbed_probs,
            nbinsx=20,
            marker=dict(color="#3B82F6", line=dict(color="#1E293B", width=1)),
            opacity=0.85,
            name="Perturbed Trials",
        )
    )

    # Highlight baseline probability
    fig.add_vline(
        x=result.baseline_prob,
        line_color="#10B981",
        line_width=2.5,
        annotation_text=f"Baseline P={result.baseline_prob:.2f}",
        annotation_position="top left",
        annotation_font=dict(color="#10B981", size=11),
    )

    # Operating cutoff
    fig.add_vline(
        x=0.37,
        line_dash="dot",
        line_color="#FBBF24",
        line_width=2,
        annotation_text="Cutoff (τ=0.37)",
        annotation_position="top right",
        annotation_font=dict(color="#FBBF24", size=11),
    )

    fig.update_layout(
        title=f"<b>Robustness Under {result.noise_scale_pct}% Noise (Stability: {result.stability_rate:.1%})</b>",
        xaxis=dict(title="Perturbed Malignancy Probability", range=[-0.02, 1.02], gridcolor="#2D3748"),
        yaxis=dict(title="Trial Count", gridcolor="#2D3748"),
        paper_bgcolor="#161B26",
        plot_bgcolor="#161B26",
        font=dict(color="#E2E8F0", family="Inter"),
        height=320,
        margin=dict(l=40, r=20, t=50, b=40),
    )
    return fig


def plot_feature_sensitivity_ranking_plotly(sensitivities: list[FeatureSensitivity], top_k: int = 10) -> go.Figure:
    """Render horizontal bar chart of top sensitive features under ±10% perturbation."""
    top_items = sensitivities[:top_k]
    feats = [item.feature_name.replace("_", " ").title() for item in reversed(top_items)]
    deltas = [item.max_delta for item in reversed(top_items)]

    fig = go.Figure(
        go.Bar(
            x=deltas,
            y=feats,
            orientation="h",
            marker=dict(color="#8B5CF6", line=dict(color="#2D3748", width=1)),
            text=[f"{d:.4f}" for d in deltas],
            textposition="auto",
        )
    )

    fig.update_layout(
        title=f"<b>Top {top_k} Features by Probability Sensitivity (±10% Perturbation)</b>",
        xaxis=dict(title="Max Absolute Probability Delta (|ΔP|)", gridcolor="#2D3748"),
        paper_bgcolor="#161B26",
        plot_bgcolor="#161B26",
        font=dict(color="#E2E8F0", family="Inter"),
        height=360,
        margin=dict(l=10, r=20, t=50, b=30),
    )
    return fig

