"""
BreastGuard AI — Page 01: Dataset Explorer
Interactive EDA with distributions, correlation heatmap, and violin plots.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3]))

import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

from src.data.loader import load_data
from src.data.validator import validate


@st.cache_resource(show_spinner="Loading dataset …")
def _get_bundle():
    b = load_data(); validate(b); return b


bundle = _get_bundle()
df     = bundle.X_df.copy()
df["Target"] = bundle.y_series.map({0: "Malignant", 1: "Benign"})

st.markdown('<h2 class="section-header">📊 Dataset Explorer</h2>', unsafe_allow_html=True)

tab_dist, tab_corr, tab_violin, tab_raw = st.tabs(
    ["📈 Distributions", "🔥 Correlation Heatmap", "🎻 Violin Plots", "📋 Raw Data"]
)

# ── Tab 1: Distributions ─────────────────────────────────────────────────────
with tab_dist:
    col1, col2 = st.columns([1, 3])
    with col1:
        feature = st.selectbox(
            "Select Feature",
            bundle.feature_names,
            index=0,
            key="dist_feature",
        )
        show_kde = st.checkbox("Show KDE curve", value=True)
    with col2:
        fig = px.histogram(
            df, x=feature, color="Target",
            color_discrete_map={"Malignant": "#FF6584", "Benign": "#00C9A7"},
            barmode="overlay", opacity=0.7,
            nbins=40,
            marginal="rug" if show_kde else None,
            template="plotly_dark",
            title=f"Distribution of '{feature}' by Class",
        )
        fig.update_layout(
            paper_bgcolor="#1A1E2E", plot_bgcolor="#1A1E2E",
            font=dict(color="#CBD5E0", family="Inter"),
            legend=dict(bgcolor="#1A1E2E"),
            margin=dict(l=20, r=20, t=50, b=20), height=380,
        )
        st.plotly_chart(fig, use_container_width=True)

    # Summary stats
    st.markdown("#### 📐 Summary Statistics by Class")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Malignant**")
        st.dataframe(
            df[df["Target"] == "Malignant"][feature].describe().to_frame().T.round(4),
            use_container_width=True,
        )
    with c2:
        st.markdown("**Benign**")
        st.dataframe(
            df[df["Target"] == "Benign"][feature].describe().to_frame().T.round(4),
            use_container_width=True,
        )

# ── Tab 2: Correlation Heatmap ───────────────────────────────────────────────
with tab_corr:
    top_n = st.slider("Number of top features to show", 10, 30, 20, key="corr_top_n")
    numeric_df = bundle.X_df

    # Select top_n features by variance
    top_features = numeric_df.var().nlargest(top_n).index.tolist()
    corr = numeric_df[top_features].corr()

    fig_heat = go.Figure(go.Heatmap(
        z=corr.values,
        x=corr.columns.tolist(),
        y=corr.index.tolist(),
        colorscale=[[0, "#FF6584"], [0.5, "#1A1E2E"], [1, "#00C9A7"]],
        zmid=0,
        text=corr.round(2).values,
        texttemplate="%{text}",
        textfont=dict(size=9),
        colorbar=dict(title="r", tickfont=dict(color="#CBD5E0")),
    ))
    fig_heat.update_layout(
        paper_bgcolor="#1A1E2E", plot_bgcolor="#1A1E2E",
        font=dict(color="#CBD5E0", family="Inter", size=10),
        margin=dict(l=20, r=20, t=30, b=20),
        height=600,
        xaxis=dict(tickangle=-45),
    )
    st.plotly_chart(fig_heat, use_container_width=True)

    st.info(
        "💡 **Reading the heatmap**: Values near +1 (teal) indicate strong positive "
        "correlation; near -1 (rose) strong negative; near 0 (dark) no linear correlation."
    )

# ── Tab 3: Violin Plots ──────────────────────────────────────────────────────
with tab_violin:
    feat_violin = st.selectbox("Select feature", bundle.feature_names, key="violin_feat")
    fig_v = go.Figure()
    for cls, colour in [("Malignant", "#FF6584"), ("Benign", "#00C9A7")]:
        mask = df["Target"] == cls
        fig_v.add_trace(go.Violin(
            x=df["Target"][mask],
            y=df[feat_violin][mask],
            name=cls,
            box_visible=True,
            meanline_visible=True,
            fillcolor=colour,
            opacity=0.7,
            line_color="#E2E8F0",
        ))
    fig_v.update_layout(
        violinmode="group",
        paper_bgcolor="#1A1E2E", plot_bgcolor="#1A1E2E",
        font=dict(color="#CBD5E0", family="Inter"),
        title=f"Violin Plot: {feat_violin}",
        yaxis_title=feat_violin,
        xaxis_title="Class",
        showlegend=True,
        margin=dict(l=20, r=20, t=50, b=20), height=420,
    )
    st.plotly_chart(fig_v, use_container_width=True)

# ── Tab 4: Raw Data ──────────────────────────────────────────────────────────
with tab_raw:
    st.markdown(f"**{len(df)} rows × {len(bundle.feature_names)} features + target**")
    search = st.text_input("Filter feature columns (comma-separated, blank = all):", key="raw_filter")
    if search.strip():
        cols = [c.strip() for c in search.split(",") if c.strip() in bundle.feature_names]
        cols = cols if cols else bundle.feature_names
    else:
        cols = bundle.feature_names
    st.dataframe(
        df[cols + ["Target"]].style.applymap(
            lambda v: "color:#FF6584;font-weight:600;" if v == "Malignant"
                      else ("color:#00C9A7;font-weight:600;" if v == "Benign" else ""),
        ),
        use_container_width=True, height=420,
    )
