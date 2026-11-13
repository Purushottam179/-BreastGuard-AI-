"""
BreastGuard AI — Page 02: Model Performance Comparison
Side-by-side comparison of all models with metrics table, ROC curves, confusion matrices, and radar chart.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3]))

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import pandas as pd

from app.pages.00_overview import _load_everything


bundle, split, trained, reports = _load_everything()
PALETTE = ["#6C63FF", "#00C9A7", "#FF6584", "#FFB347"]

st.markdown('<h2 class="section-header">🤖 Model Performance Comparison</h2>', unsafe_allow_html=True)

# ── Metrics Table ────────────────────────────────────────────────────────────
rows = []
for name, r in reports.items():
    rows.append({
        "Model":        name,
        "Accuracy":     round(r.accuracy, 4),
        "Precision":    round(r.precision, 4),
        "Recall (Sens.)": round(r.recall, 4),
        "Specificity":  round(r.specificity, 4),
        "F1-Score":     round(r.f1, 4),
        "ROC-AUC":      round(r.roc_auc, 4),
        "CV AUC (mean)": round(r.cv_auc_mean, 4),
        "CV AUC (±std)": round(r.cv_auc_std, 4),
    })

metric_df = pd.DataFrame(rows).set_index("Model")
best_model = metric_df["ROC-AUC"].idxmax()

st.markdown("#### 📊 Test-Set Metrics (held-out 20%)")

def highlight_best(s):
    is_max = s == s.max()
    return ["background-color:#1A2A1A; color:#00C9A7; font-weight:700;" if v else "" for v in is_max]

styled = metric_df.style.apply(highlight_best).format("{:.4f}")
st.dataframe(styled, use_container_width=True)
st.caption(f"🏆 Best model by ROC-AUC: **{best_model}** — highlighted in teal.")

st.markdown("---")

# ── ROC + Confusion Matrix ────────────────────────────────────────────────────
col_roc, col_cm = st.columns(2)

with col_roc:
    st.markdown("#### 📈 ROC Curves")
    fig_roc = go.Figure()
    fig_roc.add_shape(type="line", x0=0, y0=0, x1=1, y1=1,
                      line=dict(dash="dash", color="#4A5568", width=1.5))
    for i, (name, r) in enumerate(reports.items()):
        fig_roc.add_trace(go.Scatter(
            x=r.fpr, y=r.tpr, mode="lines",
            name=f"{name}<br>AUC={r.roc_auc:.4f}",
            line=dict(color=PALETTE[i % len(PALETTE)], width=2.5),
        ))
    fig_roc.update_layout(
        paper_bgcolor="#1A1E2E", plot_bgcolor="#1A1E2E",
        font=dict(color="#CBD5E0", family="Inter"),
        xaxis=dict(title="FPR", gridcolor="#2D3748"),
        yaxis=dict(title="TPR", gridcolor="#2D3748"),
        legend=dict(bgcolor="#0E1117", bordercolor="#2D3748"),
        margin=dict(l=20, r=20, t=30, b=20), height=380,
    )
    st.plotly_chart(fig_roc, use_container_width=True)

with col_cm:
    st.markdown("#### 🔲 Confusion Matrix")
    selected_model = st.selectbox(
        "Select model to inspect",
        list(reports.keys()),
        key="cm_model",
    )
    rpt = reports[selected_model]
    cm  = rpt.confusion

    fig_cm = go.Figure(go.Heatmap(
        z=cm,
        x=["Pred Malignant", "Pred Benign"],
        y=["True Malignant", "True Benign"],
        colorscale=[[0, "#0E1117"], [1, "#6C63FF"]],
        showscale=False,
        text=[[str(v) for v in row] for row in cm],
        texttemplate="<b>%{text}</b>",
        textfont=dict(size=28, color="white"),
    ))
    fig_cm.update_layout(
        paper_bgcolor="#1A1E2E", plot_bgcolor="#1A1E2E",
        font=dict(color="#CBD5E0", family="Inter"),
        margin=dict(l=20, r=20, t=30, b=20), height=340,
    )
    st.plotly_chart(fig_cm, use_container_width=True)
    tn, fp, fn, tp = rpt.confusion.ravel()
    st.markdown(
        f"<div style='text-align:center; font-size:0.85rem; color:#A0AEC0;'>"
        f"TP={tp} · TN={tn} · FP={fp} · FN={fn}"
        f"</div>",
        unsafe_allow_html=True,
    )

st.markdown("---")

# ── Radar Chart ──────────────────────────────────────────────────────────────
st.markdown("#### 🕸️ Model Profile Radar Chart")
radar_metrics = ["Accuracy", "Precision", "Recall (Sens.)", "Specificity", "F1-Score", "ROC-AUC"]

fig_radar = go.Figure()
for i, (name, r) in enumerate(reports.items()):
    vals = [
        r.accuracy, r.precision, r.recall,
        r.specificity, r.f1, r.roc_auc,
    ]
    fig_radar.add_trace(go.Scatterpolar(
        r=vals + [vals[0]],
        theta=radar_metrics + [radar_metrics[0]],
        fill="toself",
        name=name,
        line=dict(color=PALETTE[i % len(PALETTE)], width=2),
        fillcolor=PALETTE[i % len(PALETTE)],
        opacity=0.15,
    ))
fig_radar.update_layout(
    polar=dict(
        radialaxis=dict(visible=True, range=[0.85, 1.0],
                        gridcolor="#2D3748", tickfont=dict(color="#718096")),
        angularaxis=dict(gridcolor="#2D3748", tickfont=dict(color="#CBD5E0")),
        bgcolor="#1A1E2E",
    ),
    paper_bgcolor="#1A1E2E",
    font=dict(color="#CBD5E0", family="Inter"),
    legend=dict(bgcolor="#0E1117", bordercolor="#2D3748"),
    margin=dict(l=20, r=20, t=30, b=20), height=450,
)
st.plotly_chart(fig_radar, use_container_width=True)

# ── CV Scores Plot ────────────────────────────────────────────────────────────
st.markdown("#### 📉 Cross-Validation AUC (Training Set · 5-Fold Stratified)")
cv_data = [
    {"Model": name, "CV AUC Mean": r.cv_auc_mean, "CV AUC Std": r.cv_auc_std}
    for name, r in reports.items()
]
cv_df = pd.DataFrame(cv_data)

fig_cv = go.Figure()
for i, row in cv_df.iterrows():
    fig_cv.add_trace(go.Bar(
        x=[row["Model"]],
        y=[row["CV AUC Mean"]],
        error_y=dict(type="data", array=[row["CV AUC Std"]], visible=True, color="#E2E8F0"),
        marker_color=PALETTE[i % len(PALETTE)],
        name=row["Model"],
        text=[f"{row['CV AUC Mean']:.4f}"],
        textposition="outside",
        textfont=dict(color="#CBD5E0"),
    ))
fig_cv.update_layout(
    paper_bgcolor="#1A1E2E", plot_bgcolor="#1A1E2E",
    font=dict(color="#CBD5E0", family="Inter"),
    yaxis=dict(range=[0.92, 1.01], title="AUC", gridcolor="#2D3748"),
    xaxis=dict(title="", gridcolor="#2D3748"),
    showlegend=False,
    margin=dict(l=20, r=20, t=30, b=20), height=300,
    bargap=0.35,
)
st.plotly_chart(fig_cv, use_container_width=True)

st.info(
    "**ℹ️ Model Selection Note:** The recommended model is chosen by ROC-AUC, not accuracy alone. "
    "In a clinical context, **recall (sensitivity)** — correctly identifying malignant cases — "
    "is prioritised to minimise false negatives, which carry higher clinical cost."
)
