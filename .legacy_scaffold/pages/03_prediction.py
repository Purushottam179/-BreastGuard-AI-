"""
BreastGuard AI — Page 03: Interactive Prediction
30-feature sliders → real-time probability gauge + SHAP waterfall chart.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3]))

import numpy as np
import streamlit as st
import plotly.graph_objects as go

from app.pages.00_overview import _load_everything
from src.prediction.predictor import PredictionService

bundle, split, trained, reports = _load_everything()

# Precompute feature ranges from the full dataset for slider bounds
feat_min  = bundle.X_df.min()
feat_max  = bundle.X_df.max()
feat_mean = bundle.X_df.mean()

st.markdown('<h2 class="section-header">🔮 Interactive Prediction</h2>', unsafe_allow_html=True)
st.markdown(
    "<p style='color:#718096;'>Adjust the 30 diagnostic feature sliders below and run the model. "
    "The SHAP waterfall chart will explain exactly why the model made its decision.</p>",
    unsafe_allow_html=True,
)

# ── Model selector ────────────────────────────────────────────────────────────
col_ctrl1, col_ctrl2 = st.columns([2, 1])
with col_ctrl1:
    model_choice = st.selectbox(
        "Select model for prediction & explanation",
        list(trained.models.keys()),
        key="pred_model",
    )
with col_ctrl2:
    sample_mode = st.selectbox(
        "Quick-fill sample",
        ["Custom (use sliders)", "Typical Malignant", "Typical Benign"],
        key="sample_mode",
    )

# Quick-fill defaults
if sample_mode == "Typical Malignant":
    defaults = bundle.X_df[bundle.y_series == 0].mean()
elif sample_mode == "Typical Benign":
    defaults = bundle.X_df[bundle.y_series == 1].mean()
else:
    defaults = feat_mean

st.markdown("---")
st.markdown("#### ⚙️ Feature Inputs (30 Diagnostic Measurements)")

# Render sliders in 3 columns of 10
feature_values = {}
groups = [bundle.feature_names[i:i+10] for i in range(0, 30, 10)]
cols = st.columns(3)

for col, group in zip(cols, groups):
    with col:
        for feat in group:
            lo  = float(feat_min[feat])
            hi  = float(feat_max[feat])
            mid = float(defaults.get(feat, feat_mean[feat]))
            mid = max(lo, min(hi, mid))
            feature_values[feat] = st.slider(
                feat,
                min_value=round(lo, 4),
                max_value=round(hi, 4),
                value=round(mid, 4),
                step=round((hi - lo) / 200, 6),
                key=f"slider_{feat}",
            )

st.markdown("---")

# ── Prediction ────────────────────────────────────────────────────────────────
if st.button("🚀 Run Prediction & Explain", key="btn_predict", use_container_width=True):

    with st.spinner("Computing prediction and SHAP values …"):
        svc = PredictionService(
            trained_model = trained.models[model_choice],
            X_train       = split.X_train,
            feature_names = bundle.feature_names,
        )
        result = svc.predict(feature_values)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Result banner ─────────────────────────────────────────────────────────
    if result.label == "Benign":
        cls  = "result-benign"
        icon = "✅"
        colour = "#00C9A7"
    else:
        cls  = "result-malignant"
        icon = "⚠️"
        colour = "#FF6584"

    st.markdown(
        f"""<div class="{cls}">
            <div style="font-size:2.5rem;">{icon}</div>
            <div style="font-family:'Space Grotesk',sans-serif; font-size:2rem;
                        font-weight:700; color:{colour}; margin:0.5rem 0;">
                {result.label}
            </div>
            <div style="font-size:1rem; color:#CBD5E0; margin-bottom:0.5rem;">
                Model: <strong>{result.model_name}</strong>
            </div>
        </div>""",
        unsafe_allow_html=True,
    )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Probability gauge ─────────────────────────────────────────────────────
    col_g1, col_g2 = st.columns(2)
    with col_g1:
        fig_gauge_b = go.Figure(go.Indicator(
            mode="gauge+number",
            value=result.prob_benign * 100,
            title=dict(text="P(Benign) %", font=dict(color="#CBD5E0", size=14)),
            number=dict(suffix="%", font=dict(color="#00C9A7", size=28)),
            gauge=dict(
                axis=dict(range=[0, 100], tickcolor="#CBD5E0"),
                bar=dict(color="#00C9A7"),
                bgcolor="#1A1E2E",
                bordercolor="#2D3748",
                steps=[
                    dict(range=[0, 50],  color="#2D1B1B"),
                    dict(range=[50, 100], color="#0D2B22"),
                ],
                threshold=dict(line=dict(color="#E2E8F0", width=2), thickness=0.75, value=50),
            ),
        ))
        fig_gauge_b.update_layout(
            paper_bgcolor="#1A1E2E",
            font=dict(color="#CBD5E0"), height=230, margin=dict(l=20,r=20,t=30,b=10)
        )
        st.plotly_chart(fig_gauge_b, use_container_width=True)

    with col_g2:
        fig_gauge_m = go.Figure(go.Indicator(
            mode="gauge+number",
            value=result.prob_malignant * 100,
            title=dict(text="P(Malignant) %", font=dict(color="#CBD5E0", size=14)),
            number=dict(suffix="%", font=dict(color="#FF6584", size=28)),
            gauge=dict(
                axis=dict(range=[0, 100], tickcolor="#CBD5E0"),
                bar=dict(color="#FF6584"),
                bgcolor="#1A1E2E",
                bordercolor="#2D3748",
                steps=[
                    dict(range=[0, 50],  color="#0D2B22"),
                    dict(range=[50, 100], color="#2D1B1B"),
                ],
                threshold=dict(line=dict(color="#E2E8F0", width=2), thickness=0.75, value=50),
            ),
        ))
        fig_gauge_m.update_layout(
            paper_bgcolor="#1A1E2E",
            font=dict(color="#CBD5E0"), height=230, margin=dict(l=20,r=20,t=30,b=10)
        )
        st.plotly_chart(fig_gauge_m, use_container_width=True)

    # ── SHAP Waterfall chart ───────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### 🧠 SHAP Explanation — Why did the model decide this?")
    st.markdown(
        "<p style='color:#718096; font-size:0.9rem;'>"
        "Each bar shows how much that feature <strong>pushed</strong> the prediction "
        "toward <span style='color:#00C9A7;'>Benign</span> (+) or "
        "<span style='color:#FF6584;'>Malignant</span> (−).</p>",
        unsafe_allow_html=True,
    )

    shap_r = result.shap_result
    sv     = shap_r.shap_values[0]   # single instance
    fnames = shap_r.feature_names

    # Sort by absolute value, show top 20
    top_idx = np.argsort(np.abs(sv))[-20:][::-1]
    top_sv  = sv[top_idx]
    top_fn  = [fnames[i] for i in top_idx]
    top_val = [feature_values[fn] for fn in top_fn]

    colours = ["#00C9A7" if v >= 0 else "#FF6584" for v in top_sv]

    fig_shap = go.Figure(go.Bar(
        x=top_sv[::-1],
        y=[f"{n}<br><sub>{v:.3f}</sub>" for n, v in zip(top_fn[::-1], top_val[::-1])],
        orientation="h",
        marker=dict(color=colours[::-1], line=dict(width=0)),
        text=[f"{v:+.4f}" for v in top_sv[::-1]],
        textposition="outside",
        textfont=dict(color="#CBD5E0", size=10),
    ))
    fig_shap.add_vline(x=0, line_color="#E2E8F0", line_width=1.5)
    fig_shap.update_layout(
        paper_bgcolor="#1A1E2E", plot_bgcolor="#1A1E2E",
        font=dict(color="#CBD5E0", family="Inter"),
        xaxis=dict(title="SHAP value (impact on model output)",
                   gridcolor="#2D3748", zerolinecolor="#E2E8F0"),
        yaxis=dict(gridcolor="#2D3748"),
        margin=dict(l=20, r=60, t=30, b=20),
        height=600,
    )
    st.plotly_chart(fig_shap, use_container_width=True)

    st.warning(
        "⚕️ **Reminder**: This prediction is from an educational prototype and **must not** "
        "be used for clinical diagnosis or medical decision-making."
    )
