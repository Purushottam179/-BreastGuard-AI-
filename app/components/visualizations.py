"""
BreastGuard AI — Visualization Components
================================================================================
Plotly interactive visualizations styled with clinical palettes and dark theme:
  - Clinical 2x2 confusion matrix with error consequences
  - Interactive class distribution donut chart
  - Cross-validation performance comparison bars
  - Cytological feature distribution histograms and boxplots
  - Multicollinearity correlation heatmap
  - Threshold sweep trade-off curves
================================================================================
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def render_clinical_confusion_matrix(
    tp: int,
    fn: int,
    fp: int,
    tn: int,
) -> go.Figure:
    """Render a 2x2 confusion matrix styled specifically for clinical oncology."""
    z = [[tp, fn], [fp, tn]]
    x = ["Predicted Malignant", "Predicted Benign"]
    y = ["Actual Malignant", "Actual Benign"]

    annotations = [
        f"<b>True Malignant (TP)</b><br><span style='font-size:20px; font-weight:700;'>{tp}</span><br><i>Accurate Cancer Detection</i>",
        f"<b>Missed Malignant (FN)</b><br><span style='font-size:20px; font-weight:700; color:#F87171;'>{fn}</span><br><i>⚠️ Critical Clinical Danger</i>",
        f"<b>False Alarm (FP)</b><br><span style='font-size:20px; font-weight:700; color:#FBBF24;'>{fp}</span><br><i>Unnecessary Biopsy Stress</i>",
        f"<b>True Benign (TN)</b><br><span style='font-size:20px; font-weight:700;'>{tn}</span><br><i>Accurate Normal Clearance</i>",
    ]

    fig = go.Figure(
        data=go.Heatmap(
            z=z,
            x=x,
            y=y,
            colorscale=[[0, "#161B26"], [0.5, "#1E293B"], [1.0, "#3B82F6"]],
            showscale=False,
        )
    )

    idx = 0
    for row_i in range(2):
        for col_j in range(2):
            fig.add_annotation(
                x=x[col_j],
                y=y[row_i],
                text=annotations[idx],
                showarrow=False,
                font=dict(color="#FFFFFF", size=12, family="Inter"),
            )
            idx += 1

    fig.update_layout(
        title="<b>Clinical Confusion Matrix (Held-Out Test Set)</b>",
        paper_bgcolor="#161B26",
        plot_bgcolor="#161B26",
        font=dict(color="#E2E8F0", family="Inter"),
        height=320,
        margin=dict(l=30, r=30, t=50, b=30),
    )
    return fig


def render_class_distribution_donut(n_benign: int = 357, n_malignant: int = 212) -> go.Figure:
    """Render an interactive donut chart of benign vs malignant class balance."""
    labels = ["Benign (Negative)", "Malignant (Positive)"]
    values = [n_benign, n_malignant]
    colors = ["#10B981", "#EF4444"]

    fig = go.Figure(
        data=[
            go.Pie(
                labels=labels,
                values=values,
                hole=0.62,
                marker=dict(colors=colors, line=dict(color="#161B26", width=2)),
                textinfo="label+percent",
                hoverinfo="label+value+percent",
                textfont=dict(size=12, color="#FFFFFF", family="Inter"),
            )
        ]
    )

    fig.update_layout(
        title="<b>Diagnostic Cohort Distribution</b>",
        annotations=[
            dict(
                text=f"<b>{n_benign + n_malignant}</b><br><span style='font-size:12px; color:#94A3B8;'>Samples</span>",
                x=0.5,
                y=0.5,
                font_size=18,
                font_color="#E2E8F0",
                showarrow=False,
            )
        ],
        paper_bgcolor="#161B26",
        plot_bgcolor="#161B26",
        font=dict(color="#E2E8F0", family="Inter"),
        height=300,
        margin=dict(l=20, r=20, t=45, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5),
    )
    return fig


def render_cv_model_comparison_chart(cv_df: pd.DataFrame) -> go.Figure:
    """Render a grouped bar chart comparing ROC-AUC, Recall (Sensitivity), and Accuracy across models."""
    fig = go.Figure()

    models = cv_df["Model"].tolist()
    
    # Accuracy
    fig.add_trace(
        go.Bar(
            name="Accuracy",
            x=models,
            y=cv_df["accuracy_mean"],
            error_y=dict(type="data", array=cv_df["accuracy_std"]),
            marker_color="#3B82F6",
        )
    )

    # Malignant Recall (Sensitivity)
    fig.add_trace(
        go.Bar(
            name="Recall (Sensitivity)",
            x=models,
            y=cv_df["recall_mean"],
            error_y=dict(type="data", array=cv_df["recall_std"]),
            marker_color="#EF4444",
        )
    )

    # ROC-AUC
    fig.add_trace(
        go.Bar(
            name="ROC-AUC",
            x=models,
            y=cv_df["roc_auc_mean"],
            error_y=dict(type="data", array=cv_df["roc_auc_std"]),
            marker_color="#10B981",
        )
    )

    fig.update_layout(
        title="<b>5-Fold Cross-Validation Performance Across 5 Architectures</b>",
        barmode="group",
        yaxis=dict(title="Score", range=[0.85, 1.01], tickformat=".2f", gridcolor="#2D3748"),
        xaxis=dict(title=""),
        paper_bgcolor="#161B26",
        plot_bgcolor="#161B26",
        font=dict(color="#E2E8F0", family="Inter"),
        height=350,
        margin=dict(l=40, r=20, t=50, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def render_feature_distribution_chart(df: pd.DataFrame, feature_name: str) -> go.Figure:
    """Render overlaid histogram and KDE box for a specific cytological feature."""
    fig = px.histogram(
        df,
        x=feature_name,
        color="Diagnosis",
        color_discrete_map={"Malignant": "#EF4444", "Benign": "#10B981"},
        barmode="overlay",
        marginal="box",
        opacity=0.72,
        nbins=40,
        template="plotly_dark",
        title=f"Distribution of <b>{feature_name}</b> by Class",
    )
    fig.update_layout(
        paper_bgcolor="#161B26",
        plot_bgcolor="#161B26",
        font=dict(color="#E2E8F0", family="Inter"),
        margin=dict(l=20, r=20, t=50, b=20),
        height=380,
        xaxis=dict(gridcolor="#2D3748"),
        yaxis=dict(gridcolor="#2D3748"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def render_correlation_heatmap(corr_matrix: pd.DataFrame) -> go.Figure:
    """Render interactive correlation heatmap with diverging palette."""
    fig = px.imshow(
        corr_matrix,
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1,
        labels=dict(color="Pearson r"),
        template="plotly_dark",
        title="<b>Feature Correlation Matrix (30 WDBC Features)</b>",
    )
    fig.update_layout(
        paper_bgcolor="#161B26",
        plot_bgcolor="#161B26",
        font=dict(color="#E2E8F0", family="Inter", size=9),
        height=620,
        margin=dict(l=10, r=10, t=50, b=10),
    )
    return fig


def render_threshold_tradeoff_chart(sweep_df: pd.DataFrame) -> go.Figure:
    """Render Sensitivity vs Specificity curve across decision thresholds."""
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=sweep_df["threshold"],
            y=sweep_df["sensitivity"],
            mode="lines+markers",
            name="Sensitivity (Recall)",
            line=dict(color="#EF4444", width=2.5),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=sweep_df["threshold"],
            y=sweep_df["specificity"],
            mode="lines+markers",
            name="Specificity",
            line=dict(color="#10B981", width=2.5),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=sweep_df["threshold"],
            y=sweep_df["f1"],
            mode="lines",
            name="F1 Score",
            line=dict(color="#3B82F6", width=2, dash="dash"),
        )
    )

    # Highlight operating threshold 0.37
    fig.add_vline(
        x=0.37,
        line_width=2,
        line_dash="dot",
        line_color="#FBBF24",
        annotation_text="Operating Cutoff (τ = 0.37)",
        annotation_position="top left",
        annotation_font=dict(color="#FBBF24", size=11),
    )

    fig.update_layout(
        title="<b>Operating Threshold Trade-Off Analysis (Sensitivity vs Specificity)</b>",
        xaxis=dict(title="Classification Cutoff (τ)", gridcolor="#2D3748"),
        yaxis=dict(title="Score", range=[0.70, 1.02], gridcolor="#2D3748"),
        paper_bgcolor="#161B26",
        plot_bgcolor="#161B26",
        font=dict(color="#E2E8F0", family="Inter"),
        height=360,
        margin=dict(l=40, r=20, t=50, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig
