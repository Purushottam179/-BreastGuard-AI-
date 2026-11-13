"""
BreastGuard AI — Explainability UI Components
SHAP waterfall plots, horizontal contribution bars, and feature importance visualizers.
"""
from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import streamlit as st


def render_shap_waterfall_bar(
    feature_names: list[str],
    shap_values: np.ndarray,
    top_k: int = 10,
) -> go.Figure:
    """
    Render a horizontal contribution bar chart of top SHAP values oriented to Malignancy.
    Red indicates features increasing cancer risk; Teal indicates features decreasing cancer risk.
    """
    # Sort by absolute SHAP contribution
    abs_vals = np.abs(shap_values)
    top_idx = np.argsort(abs_vals)[::-1][:top_k]

    feats = [feature_names[i] for i in reversed(top_idx)]
    vals = [float(shap_values[i]) for i in reversed(top_idx)]
    colors = ["#FF6584" if v > 0 else "#00C9A7" for v in vals]

    fig = go.Figure(
        go.Bar(
            x=vals,
            y=feats,
            orientation="h",
            marker=dict(color=colors, line=dict(color="#2D3748", width=1)),
            text=[f"{v:+.3f}" for v in vals],
            textposition="outside",
        )
    )

    fig.update_layout(
        title="<b>Local SHAP Feature Contributions (Target: Malignancy)</b>",
        xaxis_title="SHAP Value (Impact on Malignancy Risk)",
        yaxis_title="",
        paper_bgcolor="#161B26",
        plot_bgcolor="#161B26",
        font=dict(color="#E2E8F0", family="Inter"),
        height=400,
        margin=dict(l=150, r=60, t=50, b=40),
    )
    return fig

