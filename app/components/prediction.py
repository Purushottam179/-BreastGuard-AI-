"""
BreastGuard AI — Interactive Prediction UI Components
================================================================================
Provides Streamlit UI components for:
  1. Manual clinical measurement input form with sensible defaults and preset cases.
  2. CSV batch file upload and multi-record processing.
  3. Prediction outcome cards with probability metrics and risk tiers.
  4. Local SHAP feature contribution charts.
  5. PDF clinical prediction report generation and download.
================================================================================
"""
from __future__ import annotations

import io
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from config.config import CLASS_LABELS
from src.predict import BatchPredictionResult, PredictionEngine, PredictionResult


def render_risk_gauge(prob_malignant: float, risk_tier: str, risk_color: str) -> go.Figure:
    """Render a semi-circular Plotly risk gauge for malignancy probability."""
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=prob_malignant * 100,
            number={"suffix": "%", "font": {"size": 34, "color": "#E2E8F0"}},
            title={
                "text": f"<b>Malignancy Probability</b><br><span style='color:{risk_color}; font-size:15px;'>{risk_tier}</span>",
                "font": {"size": 17, "color": "#E2E8F0"},
            },
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#718096"},
                "bar": {"color": risk_color, "thickness": 0.28},
                "bgcolor": "#161B26",
                "borderwidth": 1,
                "bordercolor": "#2D3748",
                "steps": [
                    {"range": [0, 20], "color": "rgba(0, 201, 167, 0.2)"},    # Low risk
                    {"range": [20, 50], "color": "rgba(255, 179, 71, 0.2)"},  # Moderate risk
                    {"range": [50, 100], "color": "rgba(255, 101, 132, 0.2)"},# High risk
                ],
                "threshold": {
                    "line": {"color": "#FF4B4B", "width": 3},
                    "thickness": 0.75,
                    "value": 37,  # Calibrated clinical threshold at 37%
                },
            },
        )
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#E2E8F0", "family": "Inter"},
        height=240,
        margin={"l": 20, "r": 20, "t": 40, "b": 10},
    )
    return fig


def render_manual_prediction_form(engine: PredictionEngine) -> dict[str, float] | None:
    """
    Render 30 input controls divided into three tabs: Mean, Standard Error, and Worst.
    Provides preset buttons for 'Typical Benign Sample' and 'Typical Malignant Sample'.
    """
    st.subheader("Manual Biopsy Cytological Measurement Entry")
    st.markdown(
        "Enter fine-needle aspirate (FNA) digitized measurements across all 30 standard "
        "Wisconsin Diagnostic Breast Cancer parameters, or select a reference preset."
    )

    defaults = engine.get_default_values()
    feature_names = engine.get_feature_names()

    # Preset selection
    col_preset1, col_preset2, col_preset3 = st.columns(3)
    preset_chosen = None
    with col_preset1:
        if st.button("Load Typical Benign Preset", use_container_width=True):
            preset_chosen = "benign"
    with col_preset2:
        if st.button("Load Typical Malignant Preset", use_container_width=True):
            preset_chosen = "malignant"
    with col_preset3:
        if st.button("Reset to Population Medians", use_container_width=True):
            preset_chosen = "median"

    # Define preset feature overrides if requested
    preset_values = dict(defaults)
    if preset_chosen == "benign":
        # Representative benign sample
        preset_values.update({
            "mean_radius": 12.0, "mean_texture": 15.0, "mean_perimeter": 78.0,
            "mean_area": 440.0, "mean_smoothness": 0.085, "worst_area": 520.0,
            "worst_radius": 13.0, "worst_texture": 20.0, "mean_concave_points": 0.02,
        })
        st.info("Loaded typical benign biopsy profile.")
    elif preset_chosen == "malignant":
        # Representative malignant sample
        preset_values.update({
            "mean_radius": 19.0, "mean_texture": 24.0, "mean_perimeter": 125.0,
            "mean_area": 1150.0, "mean_smoothness": 0.115, "worst_area": 1800.0,
            "worst_radius": 24.0, "worst_texture": 32.0, "mean_concave_points": 0.12,
        })
        st.warning("Loaded typical malignant biopsy profile.")

    inputs: dict[str, float] = {}

    tab_mean, tab_se, tab_worst = st.tabs(["1. Mean Measurements (10)", "2. Standard Error Measurements (10)", "3. Worst Measurements (10)"])

    with tab_mean:
        mean_feats = [f for f in feature_names if f.startswith("mean_")]
        cols = st.columns(2)
        for i, f in enumerate(mean_feats):
            c = cols[i % 2]
            label = f.replace("mean_", "").replace("_", " ").title()
            val = float(preset_values.get(f, defaults.get(f, 0.0)))
            inputs[f] = c.number_input(f"Mean {label}", value=val, format="%.4f", key=f"inp_{f}")

    with tab_se:
        se_feats = [f for f in feature_names if f.endswith("_se")]
        cols = st.columns(2)
        for i, f in enumerate(se_feats):
            c = cols[i % 2]
            label = f.replace("_se", "").replace("_", " ").title()
            val = float(preset_values.get(f, defaults.get(f, 0.0)))
            inputs[f] = c.number_input(f"{label} SE", value=val, format="%.5f", key=f"inp_{f}")

    with tab_worst:
        worst_feats = [f for f in feature_names if f.startswith("worst_")]
        cols = st.columns(2)
        for i, f in enumerate(worst_feats):
            c = cols[i % 2]
            label = f.replace("worst_", "").replace("_", " ").title()
            val = float(preset_values.get(f, defaults.get(f, 0.0)))
            inputs[f] = c.number_input(f"Worst {label}", value=val, format="%.4f", key=f"inp_{f}")

    if st.button("Run Diagnostic Inference", type="primary", use_container_width=True):
        return inputs
    return None


def render_prediction_result_card(
    result: PredictionResult,
    engine: PredictionEngine,
) -> None:
    """Render structured prediction outcome card, probability metrics, and SHAP attributions."""
    st.markdown("---")
    is_mal = (result.predicted_class == 1)

    if is_mal:
        st.error(f"### Predicted Diagnosis: **MALIGNANT (Clinical Positive)**")
    else:
        st.success(f"### Predicted Diagnosis: **BENIGN (Clinical Negative)**")

    # Metrics row
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Malignant Probability", f"{result.prob_malignant * 100:.1f}%")
    col2.metric("Benign Probability", f"{result.prob_benign * 100:.1f}%")
    col3.metric("Decision Cutoff", f"{result.classification_threshold:.2f}")
    col4.metric("Risk Classification", result.risk_tier)

    # Warnings and Reliability Badges
    wa = getattr(result, "warning_assessment", None)
    if wa is not None and (wa.has_warnings or wa.has_critical_alerts):
        with st.expander("⚠️ Prediction Reliability & Safety Alerts", expanded=True):
            for alert in wa.alerts:
                if alert.level == "CRITICAL":
                    st.error(f"🚨 **{alert.message}**\n\n{alert.detail}")
                elif alert.level == "WARNING":
                    st.warning(f"⚠️ **{alert.message}**\n\n{alert.detail}")
                else:
                    st.info(f"ℹ️ **{alert.message}**\n\n{alert.detail}")

    # Gauge Chart
    st.plotly_chart(render_risk_gauge(result.prob_malignant, result.risk_tier, result.risk_color), use_container_width=True)

    # Local Explainability
    if result.explanation_available and result.top_10_features:
        st.markdown("#### Local Explainability: Top Feature Contributions (SHAP)")
        
        feats = [f[0].replace("_", " ").title() for f in result.top_10_features[:8]][::-1]
        s_vals = [f[2] for f in result.top_10_features[:8]][::-1]
        colors = ["#e63946" if s > 0 else "#2a9d8f" for s in s_vals]

        fig = go.Figure(go.Bar(
            x=s_vals,
            y=feats,
            orientation="h",
            marker=dict(color=colors),
            text=[f"{'+' if s>=0 else ''}{s:.3f}" for s in s_vals],
            textposition="auto",
        ))
        fig.update_layout(
            title="Feature Contributions toward Malignant Classification (SHAP Value)",
            xaxis_title="SHAP Value (Contribution relative to population baseline)",
            yaxis_title="Cytological Feature",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#E2E8F0"),
            height=320,
            margin=dict(l=10, r=10, t=40, b=30),
        )
        st.plotly_chart(fig, use_container_width=True)

    # Narrative Statements
    if result.narrative_explanation:
        with st.expander("Clinical Explanation & Narrative Justification", expanded=True):
            for line in result.narrative_explanation:
                st.markdown(line)

    # PDF Download Button
    st.markdown("#### Clinical Prediction Report")
    pdf_path = engine.generate_pdf_report(result)
    with open(pdf_path, "rb") as f:
        pdf_bytes = f.read()

    st.download_button(
        label="Download Full Clinical Prediction Report (PDF)",
        data=pdf_bytes,
        file_name=f"BreastGuardAI_Report_{result.predicted_label}_{result.timestamp.replace(':', '').replace(' ', '_')}.pdf",
        mime="application/pdf",
        use_container_width=True,
    )

    st.caption(result.disclaimer)


def render_cohort_cancer_distribution_donut(cancer_count: int, benign_count: int) -> go.Figure:
    """Render a donut chart showing cancer vs benign patient breakdown."""
    fig = go.Figure(
        data=[
            go.Pie(
                labels=["Cancer Detected", "No Cancer Detected"],
                values=[cancer_count, benign_count],
                hole=0.58,
                marker=dict(colors=["#FF4B4B", "#10B981"], line=dict(color="#0F172A", width=2)),
                textinfo="label+percent",
                hoverinfo="label+value+percent",
            )
        ]
    )
    fig.update_layout(
        title="<b>Cohort Diagnostic Outcome</b>",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#E2E8F0", family="Inter"),
        height=270,
        margin=dict(l=10, r=10, t=35, b=10),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5),
    )
    return fig


def render_cohort_risk_bar_chart(predictions_df: pd.DataFrame) -> go.Figure:
    """Render a risk stratification breakdown across the uploaded cohort."""
    tiers = [
        ("Low Risk (<20%)", "#10B981"),
        ("Moderate Risk (20-50%)", "#FFB347"),
        ("High Risk (50-80%)", "#FF7675"),
        ("Critical Risk (>80%)", "#FF4B4B"),
    ]
    labels = [t[0] for t in tiers]
    colors = [t[1] for t in tiers]

    tier_series = predictions_df.get("risk_tier", pd.Series(dtype=str))
    counts = [int((tier_series == t[0]).sum()) for t in tiers]

    fig = go.Figure(
        data=[
            go.Bar(
                x=labels,
                y=counts,
                marker=dict(color=colors),
                text=counts,
                textposition="auto",
            )
        ]
    )
    fig.update_layout(
        title="<b>Risk Tier Stratification</b>",
        xaxis_title="Risk Classification Tier",
        yaxis_title="Number of Patients",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#E2E8F0", family="Inter"),
        height=270,
        margin=dict(l=10, r=10, t=35, b=20),
    )
    return fig


def render_batch_prediction_interface(engine: PredictionEngine) -> None:
    """
    Render comprehensive dataset upload interface:
      - Upload CSV or Excel file containing patient biopsy records
      - One-click 'Load Example Patient Dataset' button
      - Automated column name mapping (e.g. radius_mean -> mean_radius)
      - Clear 'CANCER DETECTED' vs 'NO CANCER DETECTED' outcomes for each person
      - Cohort KPI indicators and visual breakdown charts
      - Interactive patient filtering and single-patient drill-down with local SHAP explanation
      - Annotated dataset export (CSV) and executive diagnostic summary report
    """
    st.subheader("📂 Upload Dataset & Detect Cancer (Cohort Screening)")
    st.markdown(
        "Upload a dataset (CSV or Excel) containing fine needle aspirate (FNA) patient measurements. "
        "BreastGuard AI evaluates every individual patient, applies the calibrated clinical cutoff "
        "(**τ = 0.37**), and clearly reports whether **cancer is detected** or **not** for each person."
    )

    # Top action row: Demo Dataset button & Template download
    col_demo, col_template = st.columns([1.2, 1])

    with col_demo:
        if st.button("🚀 Load Example Patient Dataset (15 Biopsies)", use_container_width=True):
            sample_file = PROJECT_ROOT / "data" / "examples" / "sample_patient_cohort.csv"
            if sample_file.exists():
                st.session_state["active_upload_df"] = pd.read_csv(sample_file)
                st.session_state["active_upload_name"] = "sample_patient_cohort.csv (15 Patients)"
                # Clear previous batch results to trigger fresh analysis
                st.session_state.pop("last_batch_result", None)
                st.success("Loaded 15-patient sample cohort. Click 'Analyze Dataset & Predict Cancer Status' below!")
            else:
                st.error("Example dataset file not found.")

    with col_template:
        defaults = engine.get_default_values()
        template_df = pd.DataFrame([defaults, defaults])
        template_df.insert(0, "patient_id", ["PATIENT_001", "PATIENT_002"])
        csv_template = template_df.to_csv(index=False).encode("utf-8")

        st.download_button(
            label="📥 Download CSV Dataset Template",
            data=csv_template,
            file_name="breastguard_patient_dataset_template.csv",
            mime="text/csv",
            use_container_width=True,
        )

    # File uploader (supports both CSV and Excel)
    uploaded_file = st.file_uploader(
        "Upload Patient Biopsy Dataset (.csv, .xlsx, .xls)",
        type=["csv", "xlsx", "xls"],
        key="patient_dataset_file_uploader",
    )

    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith(".csv"):
                df_upload = pd.read_csv(uploaded_file)
            else:
                df_upload = pd.read_excel(uploaded_file)

            st.session_state["active_upload_df"] = df_upload
            st.session_state["active_upload_name"] = uploaded_file.name
        except Exception as e:
            st.error(f"Error reading uploaded file: {e}")

    # Process Active Dataset
    if "active_upload_df" in st.session_state:
        df_active = st.session_state["active_upload_df"]
        ds_name = st.session_state.get("active_upload_name", "Uploaded Dataset")

        st.markdown("---")
        st.markdown(
            f"#### 📄 Active Dataset: **{ds_name}** (`{len(df_active)} Patients`, `{len(df_active.columns)} Columns`)"
        )

        # Check column auto-mapping preview
        df_mapped, col_mapping = engine.standardize_feature_columns(df_active)
        if col_mapping:
            st.info(
                f"✨ **Column Auto-Mapper Active:** Automatically mapped {len(col_mapping)} column variations "
                f"(e.g. `radius_mean` → `mean_radius`, `concave points_mean` → `mean_concave_points`) to model standards."
            )

        # Preview expander
        with st.expander("👁️ Preview Uploaded Patient Records (First 5 Rows)", expanded=False):
            st.dataframe(df_active.head(5), use_container_width=True)

        # Primary Analyze Button
        if st.button("🩺 Analyze Dataset & Predict Cancer Status", type="primary", use_container_width=True):
            with st.spinner(f"Evaluating {len(df_active)} patients through calibrated SVM inference pipeline (τ = 0.37)..."):
                try:
                    batch_res = engine.predict_batch(df_active)
                    st.session_state["last_batch_result"] = batch_res
                    st.session_state["last_batch_mapped_df"] = df_mapped
                except Exception as err:
                    st.error(f"Batch inference failed: {err}")

    # Display Results if available
    if "last_batch_result" in st.session_state:
        batch_res: BatchPredictionResult = st.session_state["last_batch_result"]
        res_df = batch_res.predictions_df

        st.markdown("---")
        st.markdown("### 📊 Cohort Cancer Screening Summary")

        # 4 High-Impact KPI Cards
        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        kpi1.metric("Total Patients Evaluated", f"{batch_res.valid_records} patients")
        kpi2.metric(
            "🚨 Cancer Detected (Malignant)",
            f"{batch_res.cancer_detected_count} patients",
            f"{batch_res.cancer_percentage:.1f}% of cohort",
            delta_color="inverse",
        )
        kpi3.metric(
            "✅ No Cancer (Benign)",
            f"{batch_res.no_cancer_count} patients",
            f"{100.0 - batch_res.cancer_percentage:.1f}% of cohort",
            delta_color="normal",
        )
        kpi4.metric(
            "⚠️ High / Critical Risk",
            f"{batch_res.high_risk_count} patients",
            "Requires Urgent Review",
            delta_color="off",
        )

        # Invalid rows warning if any
        if batch_res.invalid_records > 0:
            st.warning(f"⚠️ {batch_res.invalid_records} invalid records encountered and flagged in results.")
            with st.expander("Review Invalid Row Errors"):
                st.json(batch_res.invalid_rows_summary)

        # Cohort Visual Charts
        col_c1, col_c2 = st.columns([1, 1.2])
        with col_c1:
            st.plotly_chart(
                render_cohort_cancer_distribution_donut(
                    batch_res.cancer_detected_count, batch_res.no_cancer_count
                ),
                use_container_width=True,
            )
        with col_c2:
            st.plotly_chart(
                render_cohort_risk_bar_chart(res_df),
                use_container_width=True,
            )

        # Filter & Patient Results Table
        st.markdown("### 📋 Patient-by-Patient Diagnostic Results")
        
        filter_col, search_col = st.columns([1.5, 1])
        with filter_col:
            filter_choice = st.radio(
                "Filter Patient Roster:",
                [
                    f"All Patients ({batch_res.valid_records})",
                    f"🚨 Cancer Detected ({batch_res.cancer_detected_count})",
                    f"✅ No Cancer Detected ({batch_res.no_cancer_count})",
                    f"⚠️ Borderline Review ({batch_res.borderline_count})",
                ],
                horizontal=True,
            )
        with search_col:
            search_query = st.text_input("🔍 Search by Patient ID:", placeholder="e.g. PATIENT_001")

        # Apply filtering
        filtered_df = res_df.copy()
        if "Cancer Detected (" in filter_choice:
            filtered_df = filtered_df[filtered_df["predicted_class"] == "Malignant"]
        elif "No Cancer Detected (" in filter_choice:
            filtered_df = filtered_df[filtered_df["predicted_class"] == "Benign"]
        elif "Borderline Review (" in filter_choice:
            filtered_df = filtered_df[filtered_df.get("is_borderline", False) == True]

        if search_query.strip():
            filtered_df = filtered_df[
                filtered_df["record_id"].astype(str).str.contains(search_query.strip(), case=False, na=False)
            ]

        # Display Clean Patient Table
        display_cols = [
            "record_id",
            "has_cancer",
            "cancer_probability",
            "confidence",
            "risk_tier",
            "classification_threshold",
            "prediction_status",
        ]
        available_display_cols = [c for c in display_cols if c in filtered_df.columns]

        st.dataframe(
            filtered_df[available_display_cols].rename(
                columns={
                    "record_id": "Patient / Record ID",
                    "has_cancer": "Has Cancer?",
                    "cancer_probability": "Cancer Probability",
                    "confidence": "Certainty",
                    "risk_tier": "Risk Stratification",
                    "classification_threshold": "Cutoff (τ)",
                    "prediction_status": "Status",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )

        st.caption(
            "💡 Note: Decisions apply the clinical decision threshold τ = 0.37 (calibrated via Youden's J) "
            "to prioritize malignant sensitivity and reduce missed carcinomas."
        )

        # Individual Patient Drill-Down Section
        st.markdown("---")
        st.markdown("### 🔍 Individual Patient Deep-Dive & Clinical Explanations")
        st.markdown(
            "Select any patient from your uploaded dataset to inspect their cytological feature profile, "
            "risk gauge, and the local SHAP factors explaining **why** the model flagged cancer or benignity."
        )

        valid_ids = res_df[res_df["prediction_status"] == "SUCCESS"]["record_id"].tolist()
        if valid_ids:
            # Create descriptive dropdown options
            id_to_desc = {}
            for _, r in res_df[res_df["prediction_status"] == "SUCCESS"].iterrows():
                rid = r["record_id"]
                p_mal = r.get("cancer_probability", "N/A")
                stat = "🚨 CANCER" if r.get("predicted_class") == "Malignant" else "✅ BENIGN"
                id_to_desc[rid] = f"{rid} — {stat} (P(Cancer)={p_mal}, Risk: {r.get('risk_tier', '')})"

            selected_id = st.selectbox(
                "Select Patient to Inspect:",
                options=valid_ids,
                format_func=lambda x: id_to_desc.get(x, str(x)),
            )

            # Retrieve patient record from mapped dataframe
            df_mapped = st.session_state.get("last_batch_mapped_df", df_active)
            patient_row = None
            for idx, r in df_mapped.iterrows():
                row_id = r.get("id", r.get("patient_id", r.get("ID", idx + 1)))
                if str(row_id) == str(selected_id):
                    patient_row = r
                    break

            if patient_row is not None:
                p_dict = {f: patient_row[f] for f in engine.get_feature_names() if f in patient_row}
                try:
                    p_res = engine.predict_single(p_dict, generate_explanation=True)
                    render_prediction_result_card(p_res, engine)
                except Exception as p_err:
                    st.error(f"Error computing detailed explanation for {selected_id}: {p_err}")

        # Batch Export Options
        st.markdown("---")
        st.markdown("### 📥 Export Diagnostic Reports")
        col_exp1, col_exp2 = st.columns(2)

        with col_exp1:
            # Merge original features with predictions for full annotated export
            try:
                annotated_df = df_active.copy()
                annotated_df.insert(1, "HAS_CANCER", res_df["has_cancer"].values)
                annotated_df.insert(2, "PREDICTED_DIAGNOSIS", res_df["predicted_class"].values)
                annotated_df.insert(3, "CANCER_PROBABILITY", res_df["cancer_probability"].values)
                annotated_df.insert(4, "RISK_TIER", res_df["risk_tier"].values)
                csv_annotated = annotated_df.to_csv(index=False).encode("utf-8")
            except Exception:
                csv_annotated = res_df.to_csv(index=False).encode("utf-8")

            st.download_button(
                label="📥 Download Full Annotated Dataset (CSV)",
                data=csv_annotated,
                file_name="annotated_patient_cancer_predictions.csv",
                mime="text/csv",
                use_container_width=True,
            )

        with col_exp2:
            # Generate Text/Markdown executive summary
            summary_text = (
                f"# BreastGuard AI — Cohort Cancer Diagnostic Summary\n"
                f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                f"Dataset: {st.session_state.get('active_upload_name', 'Uploaded Cohort')}\n\n"
                f"## Executive Findings:\n"
                f"- Total Patients Processed: {batch_res.valid_records}\n"
                f"- Patients with Cancer Detected: {batch_res.cancer_detected_count} ({batch_res.cancer_percentage:.1f}%)\n"
                f"- Patients with No Cancer Detected: {batch_res.no_cancer_count} ({100.0 - batch_res.cancer_percentage:.1f}%)\n"
                f"- High / Critical Risk Patients: {batch_res.high_risk_count}\n"
                f"- Borderline Cases Requiring Review: {batch_res.borderline_count}\n"
                f"- Calibrated Operating Threshold: τ = 0.37\n\n"
                f"DISCLAIMER:\n{CLINICAL_DISCLAIMER}\n"
            )
            st.download_button(
                label="📋 Download Diagnostic Summary Report (TXT)",
                data=summary_text.encode("utf-8"),
                file_name="cohort_diagnostic_summary.txt",
                mime="text/plain",
                use_container_width=True,
            )
