"""
BreastGuard AI — Page 04: Global Explainability (XAI)
SHAP beeswarm, global feature importance bar chart, and cross-model comparison.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3]))

import numpy as np
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

from app.pages.00_overview import _load_everything
from src.explainability.shap_explainer import compute_shap

bundle, split, trained, reports = _load_everything()
PALETTE = ["#6C63FF", "#00C9A7", "#FF6584", "#FFB347"]


@st.cache_resource(show_spinner="🧠 Computing SHAP values for all models …")
def _get_all_shap():
    results = {}
    for name, tm in trained.models.items():
        results[name] = compute_shap(
            pipeline      = tm.pipeline,
            X_train       = split.X_train,
            X_explain     = split.X_test,
            feature_names = bundle.feature_names,
            model_name    = name,
        )
    return results


shap_results = _get_all_shap()

st.markdown('<h2 class="section-header">🧠 Global Explainability (SHAP)</h2>', unsafe_allow_html=True)
st.markdown(
    "<p style='color:#718096;'>SHAP (SHapley Additive exPlanations) distributes the model's "
    "prediction fairly across all features. Global explanations reveal which features "
    "<em>consistently</em> influence the model across all test samples.</p>",
    unsafe_allow_html=True,
)

model_sel = st.selectbox("Select model", list(shap_results.keys()), key="xai_model")
shap_r    = shap_results[model_sel]
sv        = shap_r.shap_values    # (n_test, 30)
fnames    = shap_r.feature_names

st.markdown("---")
tab_importance, tab_beeswarm, tab_cross = st.tabs(
    ["📊 Feature Importance", "🐝 SHAP Beeswarm", "🔀 Cross-Model Comparison"]
)

# ── Tab 1: Feature Importance Bar ────────────────────────────────────────────
with tab_importance:
    top_n = st.slider("Top N features", 10, 30, 15, key="gi_topn")
    gi    = shap_r.global_importance   # mean |SHAP| per feature
    idx   = np.argsort(gi)[-top_n:]

    fig_bar = go.Figure(go.Bar(
        x=gi[idx],
        y=[fnames[i] for i in idx],
        orientation="h",
        marker=dict(
            color=gi[idx],
            colorscale=[[0, "#1A1E2E"], [0.5, "#6C63FF"], [1, "#00C9A7"]],
            showscale=True,
            colorbar=dict(title="Mean |SHAP|", tickfont=dict(color="#CBD5E0")),
        ),
        text=[f"{v:.4f}" for v in gi[idx]],
        textposition="outside",
        textfont=dict(color="#CBD5E0", size=10),
    ))
    fig_bar.update_layout(
        paper_bgcolor="#1A1E2E", plot_bgcolor="#1A1E2E",
        font=dict(color="#CBD5E0", family="Inter"),
        xaxis=dict(title="Mean |SHAP value|", gridcolor="#2D3748"),
        yaxis=dict(gridcolor="#2D3748"),
        margin=dict(l=20, r=80, t=30, b=20), height=500,
    )
    st.plotly_chart(fig_bar, use_container_width=True)
    st.caption(
        "**Mean |SHAP value|** = average absolute impact of each feature on predictions "
        "across all test samples. Higher = more globally important."
    )

# ── Tab 2: Beeswarm (SHAP dot plot) ──────────────────────────────────────────
with tab_beeswarm:
    top_b = st.slider("Top N features to display", 5, 20, 12, key="bee_topn")

    gi_idx = np.argsort(shap_r.global_importance)[-top_b:][::-1]
    top_sv_matrix = sv[:, gi_idx]   # (n_test, top_b)
    top_fnames    = [fnames[i] for i in gi_idx]

    # Build beeswarm-like scatter (jitter on y-axis per feature)
    rng = np.random.default_rng(42)
    fig_bee = go.Figure()

    # X_test scaled for colour
    scaler = list(trained.models.values())[0].pipeline.named_steps["scaler"]
    X_test_scaled = scaler.transform(split.X_test)

    for fi, (fname, col_idx) in enumerate(zip(top_fnames, gi_idx)):
        shap_col  = sv[:, col_idx]
        feat_col  = X_test_scaled[:, col_idx]
        jitter    = rng.uniform(-0.3, 0.3, size=len(shap_col))

        fig_bee.add_trace(go.Scatter(
            x=shap_col,
            y=[fi + j for j in jitter],
            mode="markers",
            marker=dict(
                size=5,
                color=feat_col,
                colorscale=[[0, "#FF6584"], [0.5, "#718096"], [1, "#00C9A7"]],
                showscale=(fi == 0),
                colorbar=dict(
                    title="Feature value<br>(scaled)",
                    tickvals=[-2, 0, 2],
                    ticktext=["Low", "Mid", "High"],
                    tickfont=dict(color="#CBD5E0"),
                    len=0.5, y=0.8,
                ),
                opacity=0.7,
            ),
            name=fname,
            showlegend=False,
        ))

    fig_bee.add_vline(x=0, line_color="#E2E8F0", line_width=1.5)
    fig_bee.update_layout(
        paper_bgcolor="#1A1E2E", plot_bgcolor="#1A1E2E",
        font=dict(color="#CBD5E0", family="Inter"),
        xaxis=dict(title="SHAP value (impact on model output)", gridcolor="#2D3748"),
        yaxis=dict(
            tickvals=list(range(top_b)),
            ticktext=top_fnames,
            gridcolor="#2D3748",
        ),
        margin=dict(l=20, r=80, t=30, b=20), height=500,
    )
    st.plotly_chart(fig_bee, use_container_width=True)
    st.caption(
        "Each dot = one test sample. **Colour** shows the feature's scaled value. "
        "**Position (x-axis)** shows how much that feature pushed the prediction "
        "toward Benign (+) or Malignant (−)."
    )

# ── Tab 3: Cross-model feature importance comparison ─────────────────────────
with tab_cross:
    st.markdown(
        "Compare which features each model considers most important globally."
    )
    top_c = 10
    all_gi = {name: r.global_importance for name, r in shap_results.items()}

    # Use union of top features from all models
    all_top_idx = set()
    for gi in all_gi.values():
        all_top_idx.update(np.argsort(gi)[-top_c:].tolist())
    all_top_idx = list(all_top_idx)
    top_feat_names = [fnames[i] for i in all_top_idx]

    fig_cross = go.Figure()
    for i, (name, gi) in enumerate(all_gi.items()):
        fig_cross.add_trace(go.Bar(
            name=name,
            x=top_feat_names,
            y=[gi[j] for j in all_top_idx],
            marker_color=PALETTE[i % len(PALETTE)],
        ))
    fig_cross.update_layout(
        barmode="group",
        paper_bgcolor="#1A1E2E", plot_bgcolor="#1A1E2E",
        font=dict(color="#CBD5E0", family="Inter"),
        xaxis=dict(title="Feature", gridcolor="#2D3748", tickangle=-35),
        yaxis=dict(title="Mean |SHAP value|", gridcolor="#2D3748"),
        legend=dict(bgcolor="#0E1117", bordercolor="#2D3748"),
        margin=dict(l=20, r=20, t=30, b=80), height=420,
    )
    st.plotly_chart(fig_cross, use_container_width=True)
    st.caption(
        "Features that are consistently important across all models have higher "
        "trustworthiness as clinical biomarkers in this dataset."
    )
