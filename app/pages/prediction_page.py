"""
BreastGuard AI — Page: Interactive Clinical Prediction
================================================================================
Interactive clinical prediction console:
  - Manual 30-feature entry with preset clinical cases
  - Real-time probabilistic inference with calibrated decision threshold (τ = 0.37)
  - Risk stratification (Low, Moderate, High, Critical)
  - Local SHAP waterfall and horizontal attribution bar chart
  - Narrative clinical justification
  - Downloadable publication-quality PDF clinical report
  - CSV batch upload, automated validation, and batch result export
================================================================================
"""
from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is available in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st

from app.components.dashboard import (
    render_disclaimer_banner,
    render_info_callout,
    render_page_header,
)
from app.components.prediction import (
    render_batch_prediction_interface,
    render_manual_prediction_form,
    render_prediction_result_card,
)
from src.predict import PredictionEngine, load_prediction_engine


@st.cache_resource
def get_engine() -> PredictionEngine:
    """Load and cache the centralized prediction engine singleton."""
    return load_prediction_engine()


def render_page() -> None:
    """Render the clinical prediction console."""
    render_disclaimer_banner()

    render_page_header(
        title="Interactive Clinical Prediction & Triage",
        subtitle="Simulate real-time inference with calibrated decision thresholds (τ = 0.37) and instant SHAP attributions",
        icon="🔮",
        category="CLINICAL INFERENCE · PATIENT TRIAGE",
    )

    try:
        engine = get_engine()
    except Exception as e:
        st.error(f"Failed to initialize Prediction Engine: {e}")
        st.info("Please verify that models/final_breastguard_model.pkl exists and is accessible.")
        return

    # Mode Selector Tabs: Upload Dataset is first and prominent
    tab_upload, tab_single = st.tabs(
        [
            "📁 Upload Dataset & Detect Cancer (Cohort Screening)",
            "🧪 Manual Single-Patient Cytological Diagnosis",
        ]
    )

    # TAB 1: UPLOAD DATASET & DETECT CANCER
    with tab_upload:
        render_batch_prediction_interface(engine)

    # TAB 2: MANUAL SINGLE PATIENT DIAGNOSIS
    with tab_single:
        col_form, col_res = st.columns([1.1, 1])

        with col_form:
            inputs = render_manual_prediction_form(engine)

        with col_res:
            if inputs is not None:
                with st.spinner("Executing non-linear SVM inference & calculating local SHAP attributions..."):
                    try:
                        res = engine.predict_single(inputs, generate_explanation=True)
                        st.session_state["last_prediction_result"] = res
                    except Exception as err:
                        st.error(f"Inference error: {err}")

            # Display cached or last prediction if available
            if "last_prediction_result" in st.session_state:
                render_prediction_result_card(st.session_state["last_prediction_result"], engine)
            else:
                render_info_callout(
                    title="Awaiting Biopsy Input",
                    text="Select a reference preset ('Load Typical Benign' or 'Load Typical Malignant') "
                         "or adjust the 30 sliders on the left, then click <b>Run Diagnostic Inference</b>.",
                    icon="👈",
                )


if __name__ == "__main__":
    render_page()

