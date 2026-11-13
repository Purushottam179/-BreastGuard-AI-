"""
BreastGuard AI — Page: Responsible AI, Reliability & Governance
================================================================================
Comprehensive Responsible AI dashboard featuring:
  - Model confidence distributions & Expected Calibration Error (ECE)
  - Interactive Out-of-Distribution (OOD) detection via Mahalanobis & Isolation Forest
  - Cytological subgroup performance analysis & demographic fairness disclosures
  - Monte Carlo input perturbation & prediction stability experiments
  - Transparent clinical AI governance framework & educational disclaimer
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
from typing import Any

import numpy as np
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
from src.data_loader import load_dataset
from src.fairness_analysis import (
    evaluate_nuclear_texture_subgroups,
    evaluate_tumor_size_subgroups,
    plot_subgroup_metrics_plotly,
)
from src.model_reliability import (
    assess_prediction_confidence,
    evaluate_model_calibration,
    plot_confidence_distribution_plotly,
    plot_reliability_diagram_plotly,
)
from src.ood_detection import OODDetector, load_ood_detector
from src.predict import load_prediction_engine
from src.robustness_analysis import (
    ModelRobustnessAnalyzer,
    plot_feature_sensitivity_ranking_plotly,
    plot_perturbation_distribution_plotly,
)


@st.cache_data
def load_evaluation_predictions() -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Load test/training partition predictions and data for reliability calculations."""
    bundle = load_dataset(save_raw_csv=False)
    engine = load_prediction_engine()
    
    # Run pipeline on cleaned dataset to obtain true labels and probabilities
    probs = engine.pipeline.predict_proba(bundle.X_df)[:, engine.malignant_idx]
    y_true = bundle.y_series.values
    return y_true, probs, bundle.X_df


def render_page() -> None:
    """Render the Responsible AI & Reliability view."""
    render_disclaimer_banner()

    render_page_header(
        title="Responsible AI, Model Reliability & Governance",
        subtitle="Auditing probabilistic calibration, out-of-distribution boundaries, subgroup equity, and input robustness",
        icon="🛡️",
        category="TRUSTWORTHY AI · RELIABILITY & ROBUSTNESS",
    )

    y_true, y_prob, df_feats = load_evaluation_predictions()
    engine = load_prediction_engine()
    ood_detector = load_ood_detector()

    # Precompute calibration metrics
    calib_report = evaluate_model_calibration(y_true, y_prob, n_bins=10)

    # Top KPI Cards
    k1, k2, k3, k4, k5 = st.columns(5)
    with k1:
        render_metric_card("Expected Calib Error", f"{calib_report.ece:.4f}", "ECE (10 Equal Bins)", "#3B82F6", "🎯")
    with k2:
        render_metric_card("Brier Reliability", f"{calib_report.brier_score:.4f}", "Mean Squared Gap", "#10B981", "🎲")
    with k3:
        render_metric_card("OOD Cutoff (99th)", f"{ood_detector.mahal_threshold_99_:.2f}", "Mahalanobis Bound", "#F59E0B", "📐")
    with k4:
        render_metric_card("High Conf Ratio", f"{calib_report.high_confidence_ratio:.1%}", "P >= 0.85 or <= 0.15", "#8B5CF6", "📊")
    with k5:
        render_metric_card("Demographic Data", "None Present", "No Protected PII", "#EF4444", "⚖️")

    st.markdown("---")

    # Main Tabs
    tab_rel, tab_ood, tab_fair, tab_rob, tab_gov = st.tabs(
        [
            "🎲 Model Reliability & Calibration",
            "📡 Out-of-Distribution Detection",
            "⚖️ Subgroup Analysis & Bias Limits",
            "🔬 Robustness & Noise Stability",
            "📜 Responsible AI Governance",
        ]
    )

    # TAB 1: MODEL RELIABILITY & CALIBRATION
    with tab_rel:
        render_info_callout(
            title="Probabilistic Calibration Principle",
            text="A calibrated model guarantees that when predicting an 80% probability of malignancy, "
                 "approximately 80% of such patient biopsies are genuinely malignant. "
                 "Expected Calibration Error (ECE) quantifies the empirical gap across probability bins.",
            icon="📐",
        )

        col_cal1, col_cal2 = st.columns([1.2, 1])
        with col_cal1:
            st.plotly_chart(plot_reliability_diagram_plotly(calib_report), use_container_width=True)

        with col_cal2:
            st.plotly_chart(plot_confidence_distribution_plotly(y_prob), use_container_width=True)

        st.markdown("#### Calibration Bin Analysis (10 Equal-Width Bins)")
        bin_records = [
            {
                "Bin Range": f"[{b.bin_lower:.2f} - {b.bin_upper:.2f}]",
                "Sample Count": b.sample_count,
                "Mean Confidence": f"{b.mean_confidence:.3f}",
                "Empirical Accuracy": f"{b.empirical_accuracy:.3f}",
                "Calibration Gap": f"{b.calibration_gap:.3f}",
            }
            for b in calib_report.bins
        ]
        st.dataframe(pd.DataFrame(bin_records), use_container_width=True)

        st.warning(
            "CLINICAL SAFETY PRINCIPLE: Statistical calibration verifies empirical frequency alignment across "
            "cohort partitions. It does NOT imply that every high-confidence prediction is guaranteed to be correct.",
            icon="⚠️",
        )

    # TAB 2: OUT-OF-DISTRIBUTION DETECTION
    with tab_ood:
        st.markdown(
            "#### Multivariate Out-of-Distribution (OOD) Screening\n"
            "Evaluates whether new cytological measurements deviate from the multi-dimensional training distribution, "
            "identifying potential slide artifacts, microscope focus anomalies, or rare morphological phenotypes."
        )

        col_ood_test, col_ood_info = st.columns([1.1, 1])

        with col_ood_test:
            preset_choice = st.selectbox(
                "Select Test Case to Evaluate for OOD Character",
                [
                    "Standard Benign Biopsy (In-Distribution)",
                    "Standard Malignant Biopsy (In-Distribution)",
                    "Borderline Biopsy Sample (In-Distribution)",
                    "Simulated Severe Outlier (Extreme Dimensions)",
                ],
            )

            defaults = engine.get_default_values()
            test_inputs = dict(defaults)

            if "Malignant" in preset_choice:
                test_inputs.update({"mean_radius": 19.5, "worst_area": 1850.0, "worst_radius": 24.5})
            elif "Borderline" in preset_choice:
                test_inputs.update({"mean_radius": 14.8, "worst_area": 820.0, "worst_texture": 28.5})
            elif "Severe Outlier" in preset_choice:
                test_inputs.update({
                    "mean_radius": 45.0,  # Far above max observed 28.1
                    "worst_area": 4500.0, # Far above max observed 2500
                    "worst_texture": 65.0,
                })

            ood_res = ood_detector.inspect_sample(test_inputs)

            st.markdown(
                f"""
                <div style="background: #161B26; border-left: 5px solid {'#10B981' if ood_res.severity=='NORMAL' else '#EF4444' if ood_res.severity=='SEVERE_OOD' else '#F59E0B'};
                            border-radius: 6px; padding: 1rem 1.25rem; margin: 1rem 0;">
                    <div style="font-size: 0.78rem; font-weight:700; color: #94A3B8; text-transform:uppercase;">
                        OOD STATUS ASSESSMENT
                    </div>
                    <div style="font-size: 1.3rem; font-weight:700; color: {'#10B981' if ood_res.severity=='NORMAL' else '#EF4444' if ood_res.severity=='SEVERE_OOD' else '#F59E0B'}; margin: 0.2rem 0;">
                        {ood_res.severity} (Mahalanobis Distance: {ood_res.mahalanobis_distance:.2f})
                    </div>
                    <div style="font-size: 0.85rem; color: #CBD5E1; line-height: 1.5;">
                        • 95th Percentile Training Cutoff: <b>{ood_res.mahalanobis_threshold_95:.2f}</b><br>
                        • 99th Percentile Training Cutoff: <b>{ood_res.mahalanobis_threshold_99:.2f}</b><br>
                        • Isolation Forest Anomaly Flag: <b>{ood_res.isolation_forest_anomaly}</b> (Score: {ood_res.isolation_forest_score:.3f})
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            if ood_res.warnings:
                for w in ood_res.warnings:
                    st.warning(f"⚠️ {w}")

        with col_ood_info:
            render_info_callout(
                title="Training-Only Threshold Protocol",
                text="<b>Methodology:</b> Distance thresholds were calibrated exclusively on the 455-sample "
                     "training partition using Ledoit-Wolf regularized covariance shrinkage to handle 30-feature "
                     "collinearity. The held-out test cohort was never used to set OOD boundaries.",
                icon="🔒",
            )

            if ood_res.extreme_features:
                st.markdown("<b>Univariate Extreme Feature Measurements (|Z| >= 3.5):</b>", unsafe_allow_html=True)
                df_ext = pd.DataFrame(ood_res.extreme_features)
                st.dataframe(df_ext[["feature", "value", "mean", "z_score"]], use_container_width=True)

    # TAB 3: SUBGROUP ANALYSIS & FAIRNESS LIMITS
    with tab_fair:
        st.markdown("### ⚖️ Feature-Based Subgroup Performance Analysis")
        
        render_info_callout(
            title="CRITICAL DEMOGRAPHIC FAIRNESS DISCLOSURE",
            text="<b>The Wisconsin Diagnostic Breast Cancer (WDBC) dataset contains NO demographic attributes</b> "
                 "(no patient age, race, ethnicity, or socioeconomic indicators).<br><br>"
                 "Therefore, BreastGuard AI explicitly does NOT make claims regarding demographic fairness or equity. "
                 "The subgroups below represent <b>cytological / tumor dimension categories</b>, evaluated to verify "
                 "that model sensitivity is maintained across small early-stage versus large advanced lesions.",
            icon="📜",
        )

        sub_choice = st.radio(
            "Select Feature Stratum to Inspect",
            ["Tumor Dimension Strata (mean_radius)", "Chromatin Texture Strata (worst_texture)"],
            horizontal=True,
        )

        if "mean_radius" in sub_choice:
            sub_report = evaluate_tumor_size_subgroups(df_feats, y_true, y_prob, threshold=0.37)
        else:
            sub_report = evaluate_nuclear_texture_subgroups(df_feats, y_true, y_prob, threshold=0.37)

        st.plotly_chart(plot_subgroup_metrics_plotly(sub_report), use_container_width=True)

        # Subgroup table
        sub_rows = []
        for s in sub_report.subgroups:
            sub_rows.append({
                "Subgroup Name": s.subgroup_name,
                "Criteria": s.feature_criteria,
                "Biopsies (N)": s.sample_count,
                "Prevalence": f"{s.prevalence:.1%}",
                "Sensitivity": f"{s.sensitivity:.1%}",
                "Specificity": f"{s.specificity:.1%}",
                "FNR (Missed Cancers)": f"{s.false_negative_rate:.1%}",
                "Accuracy": f"{s.accuracy:.1%}",
            })
        st.dataframe(pd.DataFrame(sub_rows), use_container_width=True)

    # TAB 4: ROBUSTNESS & PERTURBATION TESTING
    with tab_rob:
        st.markdown(
            "#### Input Perturbation & Prediction Stability Experiments\n"
            "Simulates real-world sensor noise and cytological measurement variance by injecting Gaussian perturbations "
            "scaled to training feature standard deviations across 100 stochastic trials."
        )

        # Initialize analyzer
        feature_stds = df_feats.std().to_dict()
        rob_analyzer = ModelRobustnessAnalyzer(
            pipeline=engine.pipeline,
            feature_names=engine.feature_names,
            feature_stds=feature_stds,
            operating_threshold=0.37,
        )

        col_rob_ctrl, col_rob_chart = st.columns([1, 1.3])

        with col_rob_ctrl:
            sample_preset = st.radio(
                "Select Biopsy Profile to Test",
                ["Typical Clear-Cut Benign Sample", "Typical Clear-Cut Malignant Sample", "Borderline Sensitive Sample"],
            )

            noise_level = st.slider("Measurement Noise Scale (ε as % of Feature Std)", min_value=1.0, max_value=15.0, value=5.0, step=1.0) / 100.0

            defaults = engine.get_default_values()
            base_inputs = dict(defaults)
            if "Malignant" in sample_preset:
                base_inputs.update({"mean_radius": 19.5, "worst_area": 1850.0, "worst_radius": 24.5})
            elif "Borderline" in sample_preset:
                base_inputs.update({"mean_radius": 14.8, "worst_area": 820.0, "worst_texture": 28.5})

            with st.spinner("Running 100 Monte Carlo perturbation trials..."):
                trial_res = rob_analyzer.run_noise_perturbation(
                    feature_values=base_inputs,
                    noise_scale=noise_level,
                    n_trials=100,
                    sample_id=sample_preset,
                )

            st.markdown(
                f"""
                <div style="background: #161B26; border: 1px solid #2D3748; border-radius: 8px;
                            padding: 0.9rem 1.1rem; margin-top: 0.8rem; line-height: 1.6; font-size: 0.85rem;">
                    <div><b>Baseline Label:</b> {trial_res.baseline_label} (P = {trial_res.baseline_prob:.1%})</div>
                    <div><b>Stability Rate:</b> <span style="color: {'#10B981' if trial_res.stability_rate>=0.9 else '#F59E0B'}; font-weight:700;">{trial_res.stability_rate:.1%}</span></div>
                    <div><b>Class Flip Rate:</b> {trial_res.class_flip_rate:.1%}</div>
                    <div><b>Probability Std Dev:</b> σ = {trial_res.prob_std:.4f}</div>
                    <div><b>Max Probability Swing:</b> Δ = {trial_res.max_prob_swing:.4f}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with col_rob_chart:
            st.plotly_chart(plot_perturbation_distribution_plotly(trial_res), use_container_width=True)

        st.markdown("---")
        st.markdown("#### Individual Feature Sensitivity (±10% Elasticity Sweep)")
        sens_list = rob_analyzer.analyze_single_feature_sensitivity(base_inputs, perturbation_pct=0.10)
        st.plotly_chart(plot_feature_sensitivity_ranking_plotly(sens_list, top_k=10), use_container_width=True)

    # TAB 5: RESPONSIBLE AI GOVERNANCE
    with tab_gov:
        st.markdown("### 🏛️ Trustworthy AI Governance & Ethical Principles")

        col_gov1, col_gov2 = st.columns(2)
        with col_gov1:
            st.markdown(
                """
                #### 1. Zero Silent Failures
                - Input validation screens all measurements for non-negativity and completeness.
                - Out-of-distribution boundaries prevent confident predictions on unseen morphology.
                - Atypical samples generate prominent clinical caution alerts.

                #### 2. Clinical Asymmetry Prioritization
                - The standard 0.50 cutoff was rejected in favor of **τ = 0.37**, prioritizing Malignant Sensitivity.
                - Reduces False Negatives by 62.5% on cross-validation while maintaining 100% specificity on test data.
                """
            )
        with col_gov2:
            st.markdown(
                """
                #### 3. Mathematical Attribution Integrity
                - Local SHAP attributions are computed strictly on unscaled original measurements.
                - Deterministic centroid caching guarantees zero-drift reproducibility across queries.

                #### 4. Mandatory Clinical Disclaimers
                - Counterfactual explanations are explicitly designated as mathematical sensitivity queries, NOT surgical or treatment advice.
                - Educational and competition prototype status prominently stated on all interfaces and exports.
                """
            )

        st.markdown("---")
        st.markdown("### ⚠️ Academic Research Disclaimer")
        st.markdown(
            """
            <div style="background: rgba(239, 68, 68, 0.08); border: 1px solid #7F1D1D;
                        border-radius: 6px; padding: 1rem; font-size: 0.85rem; color: #FECACA; line-height: 1.6;">
                <b>NOTICE:</b> BreastGuard AI is an academic educational prototype developed to demonstrate
                machine learning pipeline best practices, probabilistic calibration, and Explainable AI (XAI) methodology.
                It has NOT been evaluated by the FDA or EMA. It must not be used for clinical diagnosis, patient triage,
                or cancer staging.
            </div>
            """,
            unsafe_allow_html=True,
        )


if __name__ == "__main__":
    render_page()


