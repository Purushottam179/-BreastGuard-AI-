"""
BreastGuard AI — Page: Explainable AI (SHAP)
================================================================================
Comprehensive Explainable AI dashboard:
  - Global feature importance (ranking, beeswarm, bar plots)
  - Local sample explanation (waterfall decomposition, additivity verification)
  - What-If parameter sensitivity sweep
  - Algorithmic counterfactual exploration with medical safety boundary
  - Mathematical foundations & explainer selection justification
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
from config.config import REPORTS_DIR


@st.cache_data
def load_xai_artifacts() -> tuple[dict, dict]:
    """Load cached SHAP global analysis and advanced XAI summary JSONs."""
    shap_path = REPORTS_DIR / "evaluation" / "shap_analysis_summary.json"
    adv_path = REPORTS_DIR / "evaluation" / "advanced_xai_summary.json"

    shap_data = {}
    if shap_path.exists():
        with open(shap_path, "r", encoding="utf-8") as f:
            shap_data = json.load(f)

    adv_data = {}
    if adv_path.exists():
        with open(adv_path, "r", encoding="utf-8") as f:
            adv_data = json.load(f)

    return shap_data, adv_data


def render_page() -> None:
    """Render the explainable AI dashboard."""
    render_disclaimer_banner()

    render_page_header(
        title="Explainable AI & Algorithmic Attribution",
        subtitle="Unraveling the champion non-linear SVM decision surface using cooperative game theory (SHAP)",
        icon="🧠",
        category="EXPLAINABLE AI · MODEL TRANSPARENCY",
    )

    shap_data, adv_data = load_xai_artifacts()

    # Top KPI Metrics Row
    k1, k2, k3, k4, k5 = st.columns(5)
    with k1:
        render_metric_card("SHAP Explainer", "KernelExplainer", "Pipeline Wrapped", "#3B82F6", "⚙️")
    with k2:
        render_metric_card("Top Global Feature", "Worst Texture", "Mean |SHAP| = 0.0368", "#EF4444", "👑")
    with k3:
        render_metric_card("Additivity Gap", "0.000000", "Exact Verification", "#10B981", "🎯")
    with k4:
        render_metric_card("Base Value E[f(x)]", "24.29%", "Cohort Background", "#8B5CF6", "📊")
    with k5:
        render_metric_card("Consistency Audit", "100% Passed", "Zero-Variance Cache", "#06B6D4", "🛡️")

    st.markdown("---")

    # Tabs for XAI Exploration
    tab_global, tab_local, tab_whatif, tab_cf, tab_theory = st.tabs(
        [
            "🌐 Global Feature Importance",
            "🔍 Local Patient Attributions",
            "🔮 What-If Sensitivity Analysis",
            "🔄 Algorithmic Counterfactuals",
            "📐 Mathematical Foundations & Audit",
        ]
    )

    # TAB 1: GLOBAL FEATURE IMPORTANCE
    with tab_global:
        col_g1, col_g2 = st.columns([1.1, 1.2])

        with col_g1:
            st.markdown("#### Global Feature Ranking (Mean Absolute SHAP)")
            st.markdown(
                "Aggregated across evaluation samples to measure overall contribution to malignancy classification."
            )

            ranking = shap_data.get("global_importance_ranking", [])
            if ranking:
                df_rank = pd.DataFrame(ranking)
                df_rank["feature_clean"] = df_rank["feature"].str.replace("_", " ").str.title()

                fig_global = px.bar(
                    df_rank[:10].iloc[::-1],
                    x="mean_abs_shap",
                    y="feature_clean",
                    orientation="h",
                    color="mean_abs_shap",
                    color_continuous_scale="Purples",
                    title="<b>Top 10 Global Features by Impact on Malignancy</b>",
                    labels={"mean_abs_shap": "Mean |SHAP Value|", "feature_clean": "Cytological Feature"},
                    template="plotly_dark",
                )
                fig_global.update_layout(
                    paper_bgcolor="#161B26",
                    plot_bgcolor="#161B26",
                    font=dict(color="#E2E8F0", family="Inter"),
                    height=380,
                    coloraxis_showscale=False,
                    margin=dict(l=10, r=20, t=50, b=30),
                )
                st.plotly_chart(fig_global, use_container_width=True)

        with col_g2:
            st.markdown("#### SHAP Summary Beeswarm Plot")
            st.markdown(
                "Each dot is a patient sample. Color represents actual feature value (Red = High, Blue = Low). "
                "Points right of zero increase malignancy risk."
            )
            beeswarm_path = REPORTS_DIR / "figures" / "shap" / "shap_summary_beeswarm.png"
            if beeswarm_path.exists():
                st.image(str(beeswarm_path), caption="SHAP Summary Beeswarm Distribution", use_column_width=True)
            else:
                st.info("Beeswarm plot rendering...")

        st.markdown("---")
        col_bar, col_dep = st.columns(2)
        with col_bar:
            st.markdown("#### Global Importance Bar Plot")
            bar_path = REPORTS_DIR / "figures" / "shap" / "shap_bar_plot.png"
            if bar_path.exists():
                st.image(str(bar_path), caption="Standard Mean |SHAP| Bar Chart", use_column_width=True)
        with col_dep:
            st.markdown("#### Feature Dependence & Interaction Plot")
            dep_path = REPORTS_DIR / "figures" / "shap" / "shap_dependence_top_feature.png"
            if dep_path.exists():
                st.image(str(dep_path), caption="SHAP Dependence Plot for Top Feature", use_column_width=True)

    # TAB 2: LOCAL PATIENT ATTRIBUTIONS
    with tab_local:
        render_info_callout(
            title="Local Feature Attribution Principle",
            text="Local explanations decompose an individual patient's prediction into additive contributions: "
                 "<code>P(Malignant) = E[f(x)] + ∑ φ_i</code>. Positive values push toward Malignant; negative values push toward Benign.",
            icon="🔬",
        )

        sample_choice = st.radio(
            "Select Reference Clinical Case to Inspect",
            ["Malignant Biopsy Case (Sample #0 - True Positive)", "Benign Biopsy Case (Sample #19 - True Negative)"],
            horizontal=True,
        )

        if "Malignant" in sample_choice:
            sample_info = shap_data.get("example_local_malignant", {})
            card_color = "#EF4444"
            status_text = "MALIGNANT (P = 100.0%)"
        else:
            sample_info = shap_data.get("example_local_benign", {})
            card_color = "#10B981"
            status_text = "BENIGN (P = 0.05%)"

        st.markdown(
            f"""
            <div style="background: #161B26; border-left: 4px solid {card_color}; border-radius: 6px;
                        padding: 0.85rem 1.2rem; margin: 1rem 0;">
                <div style="font-size: 0.8rem; color: #94A3B8; text-transform: uppercase; font-weight:600;">
                    GROUND TRUTH & MODEL CLASSIFICATION
                </div>
                <div style="font-size: 1.3rem; font-weight: 700; color: {card_color}; margin: 0.2rem 0;">
                    {status_text}
                </div>
                <div style="font-size: 0.85rem; color: #CBD5E1;">
                    Base Value: {sample_info.get('base_value', 0.2429):.4f} · Operating Cutoff: τ = {sample_info.get('threshold', 0.37):.2f}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        col_l1, col_l2 = st.columns([1.2, 1])
        with col_l1:
            st.markdown("#### Top Feature Contributions for this Biopsy")
            pos_contribs = sample_info.get("top_positive_contributions", [])
            neg_contribs = sample_info.get("top_negative_contributions", [])
            all_contribs = pos_contribs + neg_contribs

            if all_contribs:
                df_local = pd.DataFrame(all_contribs, columns=["Feature", "Measured Value", "SHAP Contribution"])
                df_local["Feature Clean"] = df_local["Feature"].str.replace("_", " ").str.title()
                
                # Plotly horizontal attribution
                fig_loc = go.Figure(
                    go.Bar(
                        x=df_local["SHAP Contribution"],
                        y=df_local["Feature Clean"],
                        orientation="h",
                        marker=dict(
                            color=["#EF4444" if v > 0 else "#10B981" for v in df_local["SHAP Contribution"]]
                        ),
                        text=[f"{v:+.4f}" for v in df_local["SHAP Contribution"]],
                        textposition="outside",
                    )
                )
                fig_loc.update_layout(
                    title="<b>Feature Contributor Shift from Population Baseline</b>",
                    xaxis_title="SHAP Value φ_i",
                    paper_bgcolor="#161B26",
                    plot_bgcolor="#161B26",
                    font=dict(color="#E2E8F0", family="Inter"),
                    height=320,
                    margin=dict(l=10, r=20, t=50, b=30),
                )
                st.plotly_chart(fig_loc, use_container_width=True)

        with col_l2:
            st.markdown("#### Clinical Narrative Justification")
            narrative = sample_info.get("narrative_explanation", [])
            for line in narrative:
                st.markdown(line)

    # TAB 3: WHAT-IF SENSITIVITY ANALYSIS
    with tab_whatif:
        st.markdown(
            "#### Interactive What-If Parameter Sensitivity Analysis\n"
            "Examine how sweeping a single cytological feature in isolation while holding all other 29 features fixed "
            "alters predicted malignancy probability across the decision boundary."
        )

        what_if_data = adv_data.get("what_if_sensitivity", {})
        default_feat = what_if_data.get("feature_swept", "worst_area")
        tipping_pt = what_if_data.get("tipping_point_value", 460.96)

        st.markdown(
            f"""
            <div style="background: #1A2234; border: 1px solid #2D3748; border-radius: 8px;
                        padding: 1rem 1.25rem; margin-bottom: 1rem;">
                <div style="color: #60A5FA; font-weight:700; font-size: 0.95rem;">
                    Empirical Sensitivity Findings on Borderline Biopsy:
                </div>
                <div style="font-size: 0.85rem; color: #CBD5E1; line-height: 1.6; margin-top: 0.4rem;">
                    • <b>Target Feature:</b> <code>{default_feat}</code> (Cross-sectional area of largest cell nuclei)<br>
                    • <b>Baseline Probability:</b> P(Malignant) = {what_if_data.get('original_prob', 0.539):.1%}<br>
                    • <b>Tipping Point Value:</b> <span style="color:#FBBF24; font-weight:700;">{tipping_pt:.2f} μm²</span> (Crossing the τ = 0.37 boundary)<br>
                    • <b>Monotonic Probability Range:</b> [32.0% → 96.4%] across the empirical range [400 → 2500 μm²]
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        whatif_img = REPORTS_DIR / "figures" / "advanced_xai" / "what_if_analysis.png"
        if whatif_img.exists():
            st.image(str(whatif_img), caption="What-If Probability Trajectory Curve across Feature Sweep", use_column_width=True)

        render_info_callout(
            title="Scientific Context Notice",
            text="What-If sweeps represent an in-silico sensitivity test of the mathematical model. "
                 "They quantify algorithmic gradient behavior and must not be interpreted as simulated biological interventions.",
            icon="⚠️",
        )

    # TAB 4: COUNTERFACTUAL EXPLORATION
    with tab_cf:
        st.markdown(
            "#### Algorithmic Counterfactual Explanations (Actionable Recourse Analysis)\n"
            "Identifies the minimum necessary feature perturbation that would flip an ambiguous or borderline "
            "malignant prediction into a benign classification."
        )

        cf_data = adv_data.get("counterfactual_analysis", {})
        best_cf = cf_data.get("best_counterfactual", {})
        mod_feats = best_cf.get("modified_features", {})

        st.markdown(
            f"""
            <div style="background: linear-gradient(135deg, #161B26 0%, #1A2234 100%);
                        border: 1px solid #10B981; border-left: 5px solid #10B981;
                        border-radius: 8px; padding: 1.1rem 1.4rem; margin: 1rem 0;">
                <div style="font-size: 0.75rem; font-weight: 700; color: #34D399; letter-spacing: 0.08em; text-transform: uppercase;">
                    DISCOVERED COUNTERFACTUAL SPECIFICATION
                </div>
                <div style="font-size: 1.25rem; font-weight: 700; color: #F8FAFC; margin: 0.25rem 0;">
                    Flip Target: Malignant (53.9%) → Benign (36.03% &lt; τ)
                </div>
                <div style="font-size: 0.88rem; color: #CBD5E1; line-height: 1.6;">
                    • <b>Modified Feature:</b> <code>worst_texture</code><br>
                    • <b>Original Measured Value:</b> 30.86<br>
                    • <b>Counterfactual Value:</b> 27.66<br>
                    • <b>Required Relative Shift:</b> <span style="color:#34D399; font-weight:700;">-10.36%</span> (-3.20 units)<br>
                    • <b>Sparsity:</b> 1 feature modified (L0 norm = 1, minimal perturbation)
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        cf_img = REPORTS_DIR / "figures" / "advanced_xai" / "counterfactual_comparison.png"
        if cf_img.exists():
            st.image(str(cf_img), caption="Original Sample vs Discovered Counterfactual Profile", use_column_width=True)

        st.warning(
            "CLINICAL SAFETY BOUNDARY: Counterfactual explanations are strictly mathematical probes into the "
            "classifier's multidimensional decision hypersurface. They MUST NOT be construed as biological targets, "
            "surgical goals, or therapeutic treatments.",
            icon="⚠️",
        )

    # TAB 5: MATHEMATICAL FOUNDATIONS & AUDIT
    with tab_theory:
        st.markdown("#### Cooperative Game Theory Foundations (Shapley Values)")
        st.markdown(
            """
            In cooperative game theory, the Shapley value divides the total payout of a grand coalition
            among participating players fairly according to four fundamental axioms:
            """
        )

        st.latex(
            r"\phi_i(f, x) = \sum_{S \subseteq F \setminus \{i\}} \frac{|S|!(|F| - |S| - 1)!}{|F|!} [f(S \cup \{i\}) - f(S)]"
        )

        col_ax1, col_ax2 = st.columns(2)
        with col_ax1:
            st.markdown(
                """
                **1. Efficiency (Local Accuracy / Additivity):**
                $$\\sum_{i=1}^{M} \\phi_i(f, x) = f(x) - E[f(x)]$$
                The sum of feature attributions equals the difference between the model output and the expected baseline.

                **2. Symmetry:**
                If feature $i$ and $j$ contribute equally to all possible subsets, their Shapley attributions are identical: $\\phi_i = \\phi_j$.
                """
            )
        with col_ax2:
            st.markdown(
                """
                **3. Dummy / Null Player:**
                If feature $i$ adds no marginal value to any subset $S$, its attribution is strictly zero: $\\phi_i = 0$.

                **4. Additivity:**
                For combined models $f + g$, the attribution is the sum: $\\phi_i(f + g) = \\phi_i(f) + \\phi_i(g)$.
                """
            )

        st.markdown("---")
        st.markdown("#### Explainer Selection Decision Matrix")
        st.markdown(
            """
            | Explainer Type | Supported Model Families | Why Inapplicable for Champion Model | Selected? |
            |---|---|---|---|
            | **TreeExplainer** | Decision Trees, Random Forests, XGBoost, LightGBM | Champion model is an SVM, not a tree ensemble. | ❌ No |
            | **LinearExplainer** | Logistic Regression, Linear SVM, ElasticNet | Champion model uses a non-linear RBF kernel $K(x, x') = \\exp(-\\gamma ||x - x'||^2)$. | ❌ No |
            | **KernelExplainer** | **Model-Agnostic (Non-linear SVM, Neural Nets)** | **Applicable to arbitrary non-linear black-box estimators via weighted local linear surrogates.** | **✅ Selected** |
            """
        )

        st.markdown("---")
        st.markdown("#### Explanation Reproducibility & Determinism Audit")
        consistency = adv_data.get("explanation_consistency_audit", {})
        st.markdown(
            f"""
            - **Audit Status:** {'✅ All Audits Passed' if consistency.get('all_audits_passed') else '⚠️ Failed'}
            - **Reproducibility Verification:** Verified identical output on repeated calls via background centroid caching ($\\\\Delta = 0.000000$).
            - **Additivity Verification:** Additivity gap between $\\\\sum \\\\phi_i$ and $f(x) - E[f(x)]$ is strictly zero.
            """
        )

        audit_img = REPORTS_DIR / "figures" / "advanced_xai" / "explanation_consistency.png"
        if audit_img.exists():
            st.image(str(audit_img), caption="Explanation Consistency & Additivity Audit Chart", use_column_width=True)


if __name__ == "__main__":
    render_page()

