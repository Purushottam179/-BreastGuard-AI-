"""
BreastGuard AI — Page 00: Overview Dashboard
Hero page shown first. Displays project summary, key metrics, and architecture.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3]))

import streamlit as st
import plotly.graph_objects as go

from src.data.loader import load_data
from src.data.validator import validate
from src.preprocessing.pipeline import make_split
from src.models.trainer import train_all
from src.models.evaluator import evaluate_all


@st.cache_resource(show_spinner="⚙️ Training models — this takes ~30 seconds the first time …")
def _load_everything():
    bundle = load_data()
    validate(bundle)
    split   = make_split(bundle.X, bundle.y)
    trained = train_all(split)
    reports = evaluate_all(trained, split)
    return bundle, split, trained, reports


bundle, split, trained, reports = _load_everything()

# ── Hero header ──────────────────────────────────────────────────────────────
st.markdown(
    """
    <div style="text-align:center; padding: 2rem 0 1rem;">
        <div style="font-size:3.5rem;">🩺</div>
        <h1 style="font-family:'Space Grotesk',sans-serif; font-size:2.8rem;
                   font-weight:700; color:#E2E8F0; margin:0.5rem 0 0.25rem;">
            BreastGuard <span style="color:#6C63FF;">AI</span>
        </h1>
        <p style="color:#718096; font-size:1.05rem; max-width:600px; margin:0 auto;">
            Explainable Breast Cancer Classification &amp; Trustworthy AI Dashboard<br>
            <em>Wisconsin Diagnostic Breast Cancer Dataset · Educational Research Prototype</em>
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown("---")

# ── Key Metrics ──────────────────────────────────────────────────────────────
best_model_name = max(reports, key=lambda n: reports[n].roc_auc)
best            = reports[best_model_name]

col1, col2, col3, col4, col5 = st.columns(5)
metrics_data = [
    (col1, "569",           "Total Samples",     "#6C63FF"),
    (col2, "30",            "Features",          "#8B5CF6"),
    (col3, f"{best.roc_auc:.3f}", f"Best AUC ({best_model_name.split()[0]})", "#00C9A7"),
    (col4, f"{best.recall:.3f}", "Best Sensitivity", "#FFB347"),
    (col5, str(len(reports)),  "Models Compared",  "#FF6584"),
]
for col, val, label, colour in metrics_data:
    with col:
        st.markdown(
            f"""<div class="metric-card">
                <div class="metric-value" style="color:{colour};">{val}</div>
                <div class="metric-label">{label}</div>
            </div>""",
            unsafe_allow_html=True,
        )

st.markdown("<br>", unsafe_allow_html=True)

# ── Mini ROC chart (all models) ──────────────────────────────────────────────
col_left, col_right = st.columns([3, 2])

with col_left:
    st.markdown('<div class="section-header">📈 ROC Curves — All Models</div>', unsafe_allow_html=True)

    PALETTE = ["#6C63FF", "#00C9A7", "#FF6584", "#FFB347", "#63B3ED"]
    fig_roc = go.Figure()
    fig_roc.add_shape(type="line", x0=0, y0=0, x1=1, y1=1,
                      line=dict(dash="dash", color="#4A5568", width=1))
    for i, (name, rpt) in enumerate(reports.items()):
        fig_roc.add_trace(go.Scatter(
            x=rpt.fpr, y=rpt.tpr,
            mode="lines",
            name=f"{name} (AUC={rpt.roc_auc:.3f})",
            line=dict(color=PALETTE[i % len(PALETTE)], width=2.5),
        ))
    fig_roc.update_layout(
        paper_bgcolor="#1A1E2E", plot_bgcolor="#1A1E2E",
        font=dict(color="#CBD5E0", family="Inter"),
        xaxis=dict(title="False Positive Rate", gridcolor="#2D3748", zerolinecolor="#2D3748"),
        yaxis=dict(title="True Positive Rate",  gridcolor="#2D3748", zerolinecolor="#2D3748"),
        legend=dict(bgcolor="#0E1117", bordercolor="#2D3748", borderwidth=1),
        margin=dict(l=20, r=20, t=30, b=20),
        height=350,
    )
    st.plotly_chart(fig_roc, use_container_width=True)

with col_right:
    st.markdown('<div class="section-header">🏆 Model Leaderboard</div>', unsafe_allow_html=True)

    sorted_models = sorted(reports.values(), key=lambda r: r.roc_auc, reverse=True)
    for rank, rpt in enumerate(sorted_models, start=1):
        medal = ["🥇", "🥈", "🥉", "🏅"][min(rank - 1, 3)]
        bg    = "#1A1E2E" if rank > 1 else "linear-gradient(135deg,#1A1E2E,#221E3E)"
        st.markdown(
            f"""<div style="background:{bg}; border:1px solid #2D3748;
                border-left:4px solid {'#6C63FF' if rank==1 else '#2D3748'};
                border-radius:8px; padding:0.75rem 1rem; margin-bottom:0.5rem;">
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <span style="font-weight:600; color:#E2E8F0;">{medal} {rpt.name}</span>
                    <span style="color:#6C63FF; font-weight:700; font-family:'Space Grotesk',sans-serif;">
                        AUC {rpt.roc_auc:.4f}
                    </span>
                </div>
                <div style="font-size:0.78rem; color:#718096; margin-top:0.3rem;">
                    Acc {rpt.accuracy:.3f} · F1 {rpt.f1:.3f} · Recall {rpt.recall:.3f}
                </div>
            </div>""",
            unsafe_allow_html=True,
        )

st.markdown("---")

# ── Architecture overview ────────────────────────────────────────────────────
st.markdown('<div class="section-header">🏗️ System Architecture</div>', unsafe_allow_html=True)
arch_cols = st.columns(6)
arch_steps = [
    ("📦", "Data Layer",      "Load & validate WDBC dataset from sklearn"),
    ("🔭", "EDA",             "Statistical analysis & interactive visualisations"),
    ("⚙️", "Preprocessing",  "Leakage-free StandardScaler inside Pipeline"),
    ("🤖", "Model Training",  "4 models · Stratified 5-fold CV"),
    ("📊", "Evaluation",      "7 metrics on held-out test set"),
    ("🧠", "XAI (SHAP)",     "Local waterfall + global beeswarm explanations"),
]
for col, (icon, title, desc) in zip(arch_cols, arch_steps):
    with col:
        st.markdown(
            f"""<div style="background:#1A1E2E; border:1px solid #2D3748; border-radius:10px;
                padding:1rem; text-align:center; height:160px;">
                <div style="font-size:2rem;">{icon}</div>
                <div style="font-weight:600; color:#E2E8F0; margin:0.4rem 0 0.3rem;
                     font-size:0.9rem;">{title}</div>
                <div style="font-size:0.75rem; color:#718096;">{desc}</div>
            </div>""",
            unsafe_allow_html=True,
        )

st.markdown("<br>", unsafe_allow_html=True)

# ── Dataset class distribution mini-chart ───────────────────────────────────
col_a, col_b = st.columns([1, 2])
with col_a:
    st.markdown('<div class="section-header">🔵 Class Balance</div>', unsafe_allow_html=True)
    malignant_n = int((bundle.y == 0).sum())
    benign_n    = int((bundle.y == 1).sum())
    fig_donut = go.Figure(go.Pie(
        labels=["Malignant", "Benign"],
        values=[malignant_n, benign_n],
        hole=0.65,
        marker=dict(colors=["#FF6584", "#00C9A7"],
                    line=dict(color="#0E1117", width=3)),
        textfont=dict(color="#E2E8F0"),
    ))
    fig_donut.add_annotation(text=f"569<br><span style='font-size:10px'>samples</span>",
                              x=0.5, y=0.5, font=dict(size=18, color="#E2E8F0"),
                              showarrow=False)
    fig_donut.update_layout(
        paper_bgcolor="#1A1E2E", plot_bgcolor="#1A1E2E",
        font=dict(color="#CBD5E0"),
        legend=dict(bgcolor="#1A1E2E"),
        margin=dict(l=10, r=10, t=20, b=10), height=280,
    )
    st.plotly_chart(fig_donut, use_container_width=True)

with col_b:
    st.markdown('<div class="section-header">📋 Dataset at a Glance</div>', unsafe_allow_html=True)
    st.markdown(
        f"""
        | Property | Value |
        |---|---|
        | Source | UCI ML Repository (via scikit-learn) |
        | Samples | 569 |
        | Features | 30 real-valued |
        | Malignant (class 0) | {malignant_n} ({malignant_n/569*100:.1f}%) |
        | Benign (class 1) | {benign_n} ({benign_n/569*100:.1f}%) |
        | Train set | {len(split.X_train)} samples |
        | Test set | {len(split.X_test)} samples |
        | CV strategy | Stratified {5}-fold |
        | Random state | 42 (fully reproducible) |
        """,
    )
