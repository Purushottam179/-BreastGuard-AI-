"""
BreastGuard AI — Sidebar Component
================================================================================
Renders project branding, model status badge, navigation selector,
and clinical safety disclaimer.
================================================================================
"""
from __future__ import annotations

import streamlit as st
from config.config import PROJECT_NAME, VERSION

PAGES = {
    "📊 Executive Overview": "overview",
    "🔬 Dataset Analysis": "dataset_analysis",
    "🤖 Model Performance": "model_performance",
    "🔮 Breast Cancer Prediction": "prediction_page",
    "🧠 Explainable AI (SHAP)": "explainable_ai",
    "🛡️ Responsible AI & Reliability": "responsible_ai",
    "⚖️ About Project": "about",
}


def render_sidebar(include_nav: bool = True) -> str:
    """
    Render the standardized sidebar across all application views.
    
    Args:
        include_nav: If True, renders an st.radio navigation menu.
                     If False, only renders branding header, model specs, and disclaimer.
    """
    if "selected_page" not in st.session_state:
        st.session_state["selected_page"] = "📊 Executive Overview"

    with st.sidebar:
        # Branding Header
        st.markdown(
            f"""
            <div style="text-align: center; padding: 0.8rem 0 0.5rem 0;">
                <div style="font-size: 2.8rem; line-height: 1.1;">🩺</div>
                <h2 style="margin: 0.2rem 0 0 0; color: #F8FAFC; font-family: 'Space Grotesk', sans-serif;
                           font-size: 1.45rem; font-weight: 700;">
                    {PROJECT_NAME}
                </h2>
                <div style="font-size: 0.74rem; color: #818CF8; font-weight: 600;
                            letter-spacing: 0.05em; text-transform: uppercase;">
                    v{VERSION} · Trustworthy Clinical AI
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("---")

        # Optional Navigation Menu (used if st.navigation is not handling it)
        if include_nav:
            st.caption("🧭 **NAVIGATION**")
            page_titles = list(PAGES.keys())
            current_idx = 0
            if st.session_state["selected_page"] in page_titles:
                current_idx = page_titles.index(st.session_state["selected_page"])

            selected_page = st.radio(
                "Navigation Menu",
                page_titles,
                index=current_idx,
                label_visibility="collapsed",
                key="sidebar_nav_radio",
            )
            st.session_state["selected_page"] = selected_page
            st.markdown("---")

        # Model Architecture Quick Specs
        st.caption("🏆 **CHAMPION MODEL SPECS**")
        st.markdown(
            """
            <div style="background: #1E2433; border: 1px solid #2D3748; border-radius: 8px;
                        padding: 0.7rem 0.85rem; font-size: 0.8rem; color: #CBD5E1; line-height: 1.6;">
                <div><b>Algorithm:</b> Support Vector Machine</div>
                <div><b>Kernel:</b> Non-linear RBF (γ=0.01)</div>
                <div><b>Regularization:</b> C=5.0 (Balanced)</div>
                <div><b>Operating Cutoff:</b> τ = 0.37 (Youden J)</div>
                <div><b>Test Accuracy:</b> 99.12%</div>
                <div><b>Test Sensitivity:</b> 97.62%</div>
                <div><b>Explainer:</b> SHAP KernelExplainer</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("---")

        # Educational & Medical Disclaimer
        st.caption("⚕️ **MEDICAL SAFETY DISCLAIMER**")
        st.markdown(
            """
            <div style="background: rgba(239, 68, 68, 0.08); border-left: 3px solid #EF4444;
                        border-radius: 4px; padding: 0.55rem 0.75rem; font-size: 0.75rem;
                        color: #FCA5A5; line-height: 1.4;">
                Educational research prototype for competition & academic demonstration only.
                Not validated for clinical patient diagnosis.
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("---")
        st.markdown(
            "<div style='text-align: center; font-size: 0.72rem; color: #64748B;'>"
            "Wisconsin Diagnostic Breast Cancer (WDBC)<br>"
            "University-Level AI/ML Competition Prototype"
            "</div>",
            unsafe_allow_html=True,
        )

    return PAGES[st.session_state["selected_page"]]
