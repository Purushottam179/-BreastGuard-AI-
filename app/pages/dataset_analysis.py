"""
BreastGuard AI — Page: Dataset Analysis & Cytological EDA
================================================================================
Interactive exploration of the Wisconsin Diagnostic Breast Cancer (WDBC) cohort:
  - Feature distributions (histogram + boxplot) separated by diagnosis
  - Multicollinearity and 30x30 correlation heatmap
  - Top discriminating cytological biomarkers
  - Interactive sanitized data matrix and summary statistics
================================================================================
"""
from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is available in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import json
from pathlib import Path
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from app.components.dashboard import (
    render_disclaimer_banner,
    render_info_callout,
    render_metric_card,
    render_page_header,
    render_section_header,
)
from app.components.visualizations import (
    render_correlation_heatmap,
    render_feature_distribution_chart,
)
from config.config import PROCESSED_DATA_DIR, REPORTS_DIR


# Clinical feature descriptions for cytological context
FEATURE_DESCRIPTIONS: dict[str, str] = {
    "radius": "Mean distances from cell nucleus center to perimeter contour points.",
    "texture": "Standard deviation of gray-scale pixel intensities across cell nucleus.",
    "perimeter": "Total nuclear contour boundary length measured in micrometers.",
    "area": "Cross-sectional surface area of cell nucleus (πr² approximation).",
    "smoothness": "Local variation in contour radius lengths (sharpness of boundary).",
    "compactness": "Isoperimetric quotient (perimeter² / area - 1.0); measures contour roundness.",
    "concavity": "Severity and depth of concave depressions in the nuclear contour.",
    "concave_points": "Total count of distinct concave indents along the cell perimeter.",
    "symmetry": "Bilateral nuclear symmetry score compared to principal axes.",
    "fractal_dimension": "Contour complexity using Mandelbrot's coastline approximation (higher = more irregular).",
}


@st.cache_data
def load_cleaned_dataset() -> pd.DataFrame:
    """Load and cache the cleaned WDBC dataset."""
    csv_path = PROCESSED_DATA_DIR / "breast_cancer_cleaned.csv"
    if not csv_path.exists():
        # Fallback to data_loader if processed CSV missing
        from src.data_loader import load_dataset
        bundle = load_dataset(save_raw_csv=False)
        df = bundle.X_df.copy()
        df["diagnosis"] = bundle.y_series
        return df

    df = pd.read_csv(csv_path)
    return df


@st.cache_data
def load_eda_summary() -> dict:
    """Load cached EDA summary JSON."""
    eda_path = REPORTS_DIR / "evaluation" / "eda_summary.json"
    if eda_path.exists():
        with open(eda_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def render_page() -> None:
    """Render the exploratory cytological dataset analysis view."""
    render_disclaimer_banner()

    render_page_header(
        title="Dataset Analysis & Cytological EDA",
        subtitle="Exploratory investigation of 569 digitized cell nucleus biopsies across 30 morphometric attributes",
        icon="🔬",
        category="DATASET ACQUISITION · CYTOLOGICAL EXPLORATION",
    )

    df_raw = load_cleaned_dataset()
    eda_summary = load_eda_summary()

    # Create presentation dataframe with mapped diagnosis column
    df = df_raw.copy()
    target_col = "diagnosis" if "diagnosis" in df.columns else "target"
    df["Diagnosis"] = df[target_col].map({0: "Benign", 1: "Malignant"})
    feature_cols = [c for c in df.columns if c not in ["diagnosis", "target", "Diagnosis", "id"]]

    # Top KPI row
    k1, k2, k3, k4, k5 = st.columns(5)
    with k1:
        render_metric_card("Total Observations", str(len(df)), "Biopsy Samples", "#3B82F6", "📋")
    with k2:
        render_metric_card("Cellular Features", str(len(feature_cols)), "Continuous Variables", "#8B5CF6", "📐")
    with k3:
        render_metric_card("Benign Cases", "357", "62.74% Negative", "#10B981", "🟢")
    with k4:
        render_metric_card("Malignant Cases", "212", "37.26% Positive", "#EF4444", "🔴")
    with k5:
        render_metric_card("Missing Values", "0", "100% Complete", "#06B6D4", "✅")

    st.markdown("---")

    # Tab navigation
    tab_dist, tab_corr, tab_biomarkers, tab_data = st.tabs(
        [
            "📈 Cytological Distributions",
            "🔥 Multicollinearity & Correlation",
            "🏆 Top Discriminating Biomarkers",
            "📋 Interactive Data Matrix",
        ]
    )

    # TAB 1: FEATURE DISTRIBUTIONS
    with tab_dist:
        col_ctrl, col_plot = st.columns([1, 2.5])

        with col_ctrl:
            render_section_header("Feature Selection", "Choose cytological measurement to inspect", "🎯")
            
            # Grouping filter
            group_choice = st.radio(
                "Filter Feature Group",
                ["All Features (30)", "Mean Measurements (10)", "Standard Error (10)", "Worst / Extreme (10)"],
                index=0,
            )

            if "Mean" in group_choice:
                filtered_feats = [f for f in feature_cols if f.startswith("mean_")]
            elif "Standard Error" in group_choice:
                filtered_feats = [f for f in feature_cols if f.endswith("_se") or "error" in f]
            elif "Worst" in group_choice:
                filtered_feats = [f for f in feature_cols if f.startswith("worst_")]
            else:
                filtered_feats = feature_cols

            selected_feat = st.selectbox("Select Measurement", filtered_feats, index=0)

            # Determine base attribute for description
            base_key = selected_feat.replace("mean_", "").replace("worst_", "").replace("_se", "")
            desc_text = FEATURE_DESCRIPTIONS.get(base_key, "Cytological measurement of cell nucleus contour.")

            render_info_callout(
                title=f"Measurement: {selected_feat}",
                text=f"<b>Definition:</b> {desc_text}<br><br>"
                     f"In malignant biopsies, cell nuclei exhibit abnormal uncontrolled DNA synthesis, "
                     f"causing increased size, contour roughness, and irregular chromatin distribution.",
                icon="💡",
            )

            # Cohort statistics table for this feature
            mean_b = df[df["Diagnosis"] == "Benign"][selected_feat].mean()
            mean_m = df[df["Diagnosis"] == "Malignant"][selected_feat].mean()
            diff_pct = ((mean_m - mean_b) / mean_b) * 100 if mean_b != 0 else 0

            st.markdown(
                f"""
                <div style="background: #1A2234; border: 1px solid #2D3748; border-radius: 6px;
                            padding: 0.75rem; font-size: 0.82rem; line-height: 1.6;">
                    <div><b>Benign Mean:</b> {mean_b:.4f}</div>
                    <div><b>Malignant Mean:</b> {mean_m:.4f}</div>
                    <div><b>Relative Difference:</b> <span style="color: {'#EF4444' if diff_pct > 0 else '#10B981'}; font-weight:700;">{diff_pct:+.1f}%</span></div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with col_plot:
            st.plotly_chart(render_feature_distribution_chart(df, selected_feat), use_container_width=True)

    # TAB 2: MULTICOLLINEARITY & CORRELATION
    with tab_corr:
        col_c_desc, col_c_info = st.columns([2, 1])
        with col_c_desc:
            st.markdown(
                "#### Full 30-Feature Cytological Multicollinearity Analysis\n"
                "The Pearson correlation heatmap reveals substantial collinear clusters among geometric dimensions "
                "(radius, perimeter, area) and shape complexity metrics (compactness, concavity, concave points)."
            )
        with col_c_info:
            render_info_callout(
                title="Multicollinearity Handling",
                text="<b>70 pairs</b> exhibit $|r| > 0.80$. Our machine learning pipeline incorporates "
                     "L2/L1 regularization and standardized feature scaling, ensuring stable model training without "
                     "instability from high variance.",
                icon="📐",
            )

        corr_matrix = df[feature_cols].corr()
        st.plotly_chart(render_correlation_heatmap(corr_matrix), use_container_width=True)

    # TAB 3: TOP DISCRIMINATING BIOMARKERS
    with tab_biomarkers:
        st.markdown("#### Top Discriminating Cytological Features Between Cohorts")
        st.markdown(
            "Ranking of the strongest individual features distinguishing malignant from benign biopsies "
            "based on separation metrics and Cohen's effect size."
        )

        top_feats = eda_summary.get(
            "top_discriminating_features",
            ["worst_concave_points", "worst_perimeter", "mean_concave_points", "worst_radius", "worst_area", "mean_radius"]
        )[:10]

        # Calculate relative difference for each top feature
        comp_data = []
        for feat in top_feats:
            if feat in df.columns:
                b_m = df[df["Diagnosis"] == "Benign"][feat].mean()
                m_m = df[df["Diagnosis"] == "Malignant"][feat].mean()
                pct_diff = ((m_m - b_m) / b_m) * 100 if b_m != 0 else 0
                comp_data.append({
                    "Cytological Feature": feat.replace("_", " ").title(),
                    "Benign Mean": round(b_m, 4),
                    "Malignant Mean": round(m_m, 4),
                    "Relative Increase": f"+{pct_diff:.1f}%",
                    "Relative_Pct_Float": pct_diff,
                })

        df_top = pd.DataFrame(comp_data)

        col_t_table, col_t_chart = st.columns([1, 1])
        with col_t_table:
            st.dataframe(df_top[["Cytological Feature", "Benign Mean", "Malignant Mean", "Relative Increase"]], use_container_width=True)

        with col_t_chart:
            fig_bar = px.bar(
                df_top,
                x="Relative_Pct_Float",
                y="Cytological Feature",
                orientation="h",
                color="Relative_Pct_Float",
                color_continuous_scale="Reds",
                title="<b>Malignant Cohort % Increase over Benign Baseline</b>",
                labels={"Relative_Pct_Float": "Relative Increase (%)", "Cytological Feature": ""},
                template="plotly_dark",
            )
            fig_bar.update_layout(
                paper_bgcolor="#161B26",
                plot_bgcolor="#161B26",
                font=dict(color="#E2E8F0", family="Inter"),
                height=380,
                coloraxis_showscale=False,
                margin=dict(l=10, r=20, t=50, b=30),
            )
            st.plotly_chart(fig_bar, use_container_width=True)

    # TAB 4: INTERACTIVE DATA MATRIX
    with tab_data:
        st.markdown("#### Sanitized Cytological Biopsy Dataset (No Patient PII)")
        st.caption("569 records with 30 continuous measurements and ground-truth diagnosis.")

        st.dataframe(
            df[["Diagnosis"] + feature_cols],
            use_container_width=True,
            height=380,
        )

        st.markdown("#### Cohort Statistical Summaries")
        col_s1, col_s2 = st.columns(2)
        with col_s1:
            st.markdown("<b>Benign Biopsies (N=357)</b>", unsafe_allow_html=True)
            st.dataframe(df[df["Diagnosis"] == "Benign"][feature_cols[:10]].describe().T[["mean", "std", "min", "50%", "max"]], use_container_width=True)
        with col_s2:
            st.markdown("<b>Malignant Biopsies (N=212)</b>", unsafe_allow_html=True)
            st.dataframe(df[df["Diagnosis"] == "Malignant"][feature_cols[:10]].describe().T[["mean", "std", "min", "50%", "max"]], use_container_width=True)


if __name__ == "__main__":
    render_page()
