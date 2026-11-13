"""
BreastGuard AI — Page: About & Responsible AI Governance
================================================================================
Comprehensive project documentation, clinical motivation, dataset provenance,
machine learning architecture, regulatory context, limitations, and safety disclosures.
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
    render_section_header,
)
from config.config import DATASET_NAME, DATASET_SOURCE, PROJECT_NAME, VERSION


def render_page() -> None:
    """Render the about, methodology, and responsible AI governance view."""
    render_disclaimer_banner()

    render_page_header(
        title=f"About {PROJECT_NAME} & Governance",
        subtitle="System architecture, clinical methodology, dataset provenance, and responsible AI compliance framework",
        icon="⚖️",
        category="GOVERNANCE · REGULATORY COMPLIANCE",
    )

    tab_overview, tab_methodology, tab_governance, tab_limitations = st.tabs(
        [
            "🩺 Project Overview & Motivation",
            "🔬 Machine Learning & XAI Methodology",
            "📜 Regulatory Framework & Trustworthy AI",
            "⚠️ Limitations & Future Work",
        ]
    )

    # TAB 1: OVERVIEW & MOTIVATION
    with tab_overview:
        col_m1, col_m2 = st.columns([1.2, 1])

        with col_m1:
            st.markdown("### 🎯 Executive Summary & Mission")
            st.markdown(
                f"""
                **{PROJECT_NAME} (v{VERSION})** is an educational, competition-grade clinical decision
                support and explainable AI prototype. It classifies digitized breast cell fine needle aspirate (FNA)
                biopsies into **Benign** or **Malignant** categories while providing comprehensive,
                game-theoretic feature attributions for every algorithmic prediction.

                #### The Clinical Asymmetry Dilemma
                In diagnostic oncology, error consequences are profoundly asymmetric:
                - **False Negative (Missed Cancer):** Leads to delayed intervention, disease progression,
                  upstaging of tumors, and potentially fatal clinical outcomes.
                - **False Positive (False Alarm):** Causes psychological distress and secondary confirmatory biopsies,
                  but does not result in untreated malignant progression.

                Traditional classifiers optimize for generic accuracy or symmetric 0.50 cutoffs. BreastGuard AI
                explicitly recalibrates its operating decision threshold to **τ = 0.37**, maximizing malignant
                sensitivity (recall) while preserving pristine specificity.
                """
            )

        with col_m2:
            st.markdown("### 📚 Dataset Provenance")
            st.markdown(
                f"""
                <div style="background: #161B26; border: 1px solid #2D3748; border-radius: 8px;
                            padding: 1rem; font-size: 0.85rem; line-height: 1.6; color: #CBD5E1;">
                    <div><b>Dataset:</b> {DATASET_NAME}</div>
                    <div><b>Source:</b> {DATASET_SOURCE}</div>
                    <div><b>Originators:</b> Dr. William H. Wolberg, W. Nick Street, Olvi L. Mangasarian (1995)</div>
                    <div><b>Institution:</b> University of Wisconsin Hospitals, Madison</div>
                    <div><b>Specimen Type:</b> Fine Needle Aspirate (FNA) of Breast Mass</div>
                    <div><b>Observations:</b> 569 patient biopsies</div>
                    <div><b>Target Breakdown:</b> 357 Benign (62.7%), 212 Malignant (37.3%)</div>
                    <div><b>Features:</b> 30 continuous real-valued nuclear measurements</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.markdown("### 🛠️ Technology Stack")
            st.markdown(
                """
                - **Language:** Python 3.13
                - **Machine Learning:** Scikit-learn (Pipelines, SVM, Ensembles)
                - **Explainability:** SHAP (KernelExplainer with Background Centroids)
                - **Interactive UI:** Streamlit & Plotly
                - **Verification & QA:** Pytest (78 automated test cases)
                - **Data Structures:** Pandas & NumPy
                - **Reporting:** Matplotlib PDF Vector Graphics
                """
            )

    # TAB 2: METHODOLOGY
    with tab_methodology:
        st.markdown("### 🏗️ End-to-End Machine Learning Pipeline")
        st.markdown(
            """
            ```mermaid
            flowchart LR
                A[UCI WDBC Dataset 569 cases] --> B[Data Cleaning & Validation]
                B --> C[Stratified 80/20 Partition]
                C --> D[Training Partition 455 cases]
                C --> E[Held-Out Test 114 cases]
                D --> F[5-Fold Stratified CV Pipeline]
                F --> G[Hyperparameter Tuning]
                G --> H[Threshold Calibration τ = 0.37]
                H --> I[Final Champion Model: SVM RBF]
                E --> J[Held-Out Benchmark Accuracy: 99.12%]
                I --> K[SHAP KernelExplainer]
                K --> L[Interactive Dashboard & Reports]
            ```
            """
        )

        st.markdown("#### Candidate Architectures Evaluated")
        st.markdown(
            """
            1. **Logistic Regression:** L1-regularized baseline with liblinear solver.
            2. **Random Forest:** Ensemble of 100 balanced decision trees.
            3. **Support Vector Machine (Champion):** Non-linear Radial Basis Function (RBF) kernel with $C=5.0$, $\\gamma=0.01$.
            4. **Gradient Boosting:** Sequential boosted trees with subsampling.
            5. **K-Nearest Neighbors:** Distance-weighted neighborhood classifier ($K=11$).
            """
        )

        st.markdown("#### Explainability Architecture (SHAP)")
        st.markdown(
            """
            - **Model-Agnostic KernelExplainer:** Selected because non-linear RBF SVM decision boundaries
              cannot be explained via TreeExplainer or LinearExplainer.
            - **Background Reference Cohort:** Computed using K-means clustering (40 centroids) across the training partition.
            - **Local Attribution:** Computes exact additive contributions $\\phi_i$ relative to the cohort baseline $E[f(x)] = 24.29\\%$.
            - **Reproducibility Guarantee:** Deterministic explanation cache ensures identical attribution outputs on repeated queries.
            """
        )

    # TAB 3: REGULATORY FRAMEWORK
    with tab_governance:
        st.markdown("### 🏛️ Alignment with Clinical AI Governance Frameworks")

        col_g1, col_g2 = st.columns(2)
        with col_g1:
            st.markdown(
                """
                #### EU Artificial Intelligence Act (High-Risk AI Systems)
                Under the EU AI Act, AI systems utilized for clinical triage or diagnostic assistance
                fall under **High-Risk (Annex III, Medical Device Regulation MDR 2017/745)**:
                - **Article 10 (Data Governance):** Cleaned, validated, leak-free training partitions.
                - **Article 13 (Transparency):** Interpretable predictions with full local and global SHAP attributions.
                - **Article 14 (Human Oversight):** Designed as human-in-the-loop decision support, never autonomous diagnosis.
                - **Article 15 (Accuracy & Robustness):** Verified threshold optimization and 5-fold cross-validation stability.
                """
            )
        with col_g2:
            st.markdown(
                """
                #### FDA Good Machine Learning Practice (GMLP)
                BreastGuard AI adheres to the ten guiding principles established by FDA, Health Canada, and MHRA:
                - **Total Product Life Cycle:** Isolated test partition strictly held out until final model confirmation.
                - **Clinical Intended Use:** Clear definition of primary safety metric (Sensitivity / Recall on Malignant).
                - **Model Explainability:** Algorithmic transparency prioritizing clinician trust and communicable rationale.
                - **Robust Calibration:** Brier score calibration verification preventing deceptive overconfidence.
                """
            )

    # TAB 4: LIMITATIONS & FUTURE WORK
    with tab_limitations:
        render_info_callout(
            title="Scientific Integrity & Honest Reporting",
            text="In competitive AI challenges and clinical evaluation, acknowledging boundary conditions "
                 "and potential failure modes is essential for credibility and patient safety.",
            icon="🛡️",
        )

        col_l1, col_l2 = st.columns(2)
        with col_l1:
            st.markdown("#### Current Model Limitations")
            st.markdown(
                """
                1. **Single-Center Retrospective Data:** The WDBC dataset was acquired at a single academic hospital
                   in the mid-1990s. Slide preparation techniques, staining protocols, and digitization optics
                   may not reflect modern hospital variations.
                2. **Tabular Feature Dependency:** The model relies on pre-extracted numerical cytological features,
                   not raw Whole Slide Images (WSI) or digital pathology scans.
                3. **Sample Size Constraints:** With 569 total cases, deep neural architectures would risk catastrophic
                   overfitting; hence classical regularized estimators (SVM) were prioritized.
                4. **Lack of Prospective Validation:** Evaluated retrospectively on split cohorts without live clinical trials.
                """
            )

        with col_l2:
            st.markdown("#### Future Research Roadmap")
            st.markdown(
                """
                1. **Vision-Language Pathology Models:** Direct classification from high-resolution Whole Slide Images
                   using Vision Transformers (ViT) and pathology foundation models (e.g. UNI, Prov-GigaPath).
                2. **Multi-Center External Cohorts:** Validation against external datasets such as CAMELYON16 and TCGA-BRCA.
                3. **Multimodal EHR Integration:** Combining cytological morphometry with patient age, mammographic density,
                   BRCA1/2 genomic status, and family history.
                4. **Federated Privacy Learning:** Multi-hospital collaborative training without raw patient data transfer.
                """
            )

        st.markdown("---")
        st.markdown("### ⚠️ Academic Research Disclaimer")
        st.markdown(
            """
            <div style="background: rgba(239, 68, 68, 0.08); border: 1px solid #7F1D1D;
                        border-radius: 6px; padding: 1rem; font-size: 0.85rem; color: #FECACA; line-height: 1.6;">
                <b>NOTICE:</b> BreastGuard AI is an academic educational prototype developed to demonstrate
                machine learning pipeline best practices and Explainable AI (XAI) methodology. It has NOT
                been cleared, approved, or evaluated by the United States Food and Drug Administration (FDA),
                the European Medicines Agency (EMA), or any national health authority. It is NOT intended
                for use in clinical patient diagnosis, triage, staging, or treatment planning.
            </div>
            """,
            unsafe_allow_html=True,
        )


if __name__ == "__main__":
    render_page()

