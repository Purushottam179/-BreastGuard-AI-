# 🩺 BreastGuard AI: Explainable Breast Cancer Classification & Trustworthy AI Dashboard

[![Python](https://img.shields.io/badge/Python-3.13-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.6%2B-orange.svg)](https://scikit-learn.org/)
[![SHAP](https://img.shields.io/badge/SHAP-0.54%2B-brightgreen.svg)](https://shap.readthedocs.io/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.40%2B-red.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-purple.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/Tests-112%2F112%20Passed-brightgreen.svg)](reports/quality_assurance/test_report.md)
[![Audit](https://img.shields.io/badge/Technical%20Audit-Verified-blue.svg)](reports/quality_assurance/project_audit.md)
[![Research Paper](https://img.shields.io/badge/Research%20Paper-Academic%20Draft-purple.svg)](reports/documentation/BreastGuard_AI_Research_Paper.md)
[![Documentation](https://img.shields.io/badge/Technical%20Monograph-Complete-teal.svg)](reports/documentation/PROJECT_DOCUMENTATION.md)
[![Slides](https://img.shields.io/badge/Competition%20Slides-15%20Slides%20(.pptx)-gold.svg)](reports/presentation/BreastGuard_AI_Competition_Presentation.pptx)
[![Status](https://img.shields.io/badge/Competition--Grade-Ready-gold.svg)]()

> **⚕️ Medical AI Safety Notice**: BreastGuard AI is an **educational research prototype** designed for academic demonstrations and AI competitions. It is **not** an FDA-cleared or CE-marked medical diagnostic device. Predictions are statistical estimates and must never replace clinical judgement by certified oncologists and radiologists.

---

## 1. Project Introduction

**BreastGuard AI** is a state-of-the-art, competition-ready clinical machine learning system engineered to classify fine needle aspirate (FNA) biopsy samples as **Benign** or **Malignant**. Built on principles of **Trustworthy and Responsible AI**, it combines rigorous, leakage-free predictive modeling with local and global **SHAP (SHapley Additive exPlanations)** to provide clinicians and evaluators with complete algorithmic transparency.

---

## 2. Problem Statement

Breast cancer is among the most commonly diagnosed cancers globally. Early and accurate detection significantly reduces mortality rates. While machine learning classifiers often achieve high nominal accuracy on tabular cytological data, real-world clinical adoption has been critically hindered by two barriers:
1. **The "Black Box" Dilemma**: High-performing non-linear ensembles (Random Forest, XGBoost) fail to explain *why* an individual biopsy was flagged as malignant, creating clinical distrust.
2. **Metric & Cost Asymmetry**: In clinical oncology, a **False Negative** (failing to diagnose malignant cancer) is catastrophic, whereas a **False Positive** leads to an emotional burden and a follow-up confirmatory biopsy. Many benchmark models naively optimize for overall accuracy without prioritizing sensitivity to malignancy.

---

## 3. Project Objectives

- **Clinical-First Modeling**: Benchmark and compare Logistic Regression, Random Forest, XGBoost, and Support Vector Machines (RBF) using 5-fold Stratified Cross-Validation on strictly held-out data.
- **Explainability as a First-Class Citizen**: Provide instant local feature contribution waterfalls for single-patient predictions alongside global cohort importance beeswarm plots.
- **Interactive Decision Dashboard**: Deliver an intuitive, multi-page Streamlit dashboard featuring exploratory data analysis, side-by-side model benchmarks, real-time diagnostic sliders, and responsible AI governance.
- **Zero-Data Leakage Architecture**: Enforce strict pipeline boundaries using scikit-learn Pipelines, preventing normalization parameters from bleeding from training to test partitions.

---

## 4. Dataset Information

The system utilizes the gold-standard **Wisconsin Diagnostic Breast Cancer (WDBC)** dataset:
- **Source**: UCI Machine Learning Repository / Dr. William H. Wolberg, W. Nick Street, and Olvi L. Mangasarian (University of Wisconsin).
- **Instances**: 569 patient samples (357 Benign, 212 Malignant).
- **Features**: 30 continuous numerical features computed from digitized high-resolution images of fine needle aspirates (FNA) describing nuclear characteristics:
  - **10 Mean Dimensions**: `radius_mean`, `texture_mean`, `perimeter_mean`, `area_mean`, `smoothness_mean`, `compactness_mean`, `concavity_mean`, `concave points_mean`, `symmetry_mean`, `fractal_dimension_mean`.
  - **10 Standard Error (SE) Dimensions**: Assessing cell-to-cell variability across the biopsy.
  - **10 "Worst" / Largest Dimensions**: Representing the most extreme cellular anomalies across the sample.
- **Target Encoding**: `0 = Benign` (Clinical Negative, $N=357$), `1 = Malignant` (Clinical Positive, $N=212$).

---

## 5. Technology Stack

- **Runtime**: Python 3.13.9 (64-bit Windows AMD64)
- **Data Engineering**: NumPy 2.x, Pandas 2.x/3.x, SciPy 1.18+
- **Machine Learning**: Scikit-Learn 1.9+, XGBoost 3.4+
- **Explainable AI**: SHAP (SHapley Additive exPlanations) 0.54.1
- **Visualization**: Plotly 7.x, Matplotlib 3.11+, Seaborn 0.13+
- **Application Framework**: Streamlit 1.66+
- **Quality Assurance**: Pytest 8.4+, Pytest-Cov

---

## 6. Planned Machine Learning Models

| Model | Inductive Bias & Justification | Explainability Method |
|---|---|---|
| **Logistic Regression (L2)** | Convex linear baseline; establishes benchmark linear separability. | SHAP LinearExplainer |
| **Random Forest** | Bagged decision tree ensemble; captures non-linear interactions without scaling sensitivity. | SHAP TreeExplainer |
| **XGBoost (Extreme Gradient Boosting)** | Gradient-boosted shallow decision trees with regularization; state-of-the-art tabular accuracy. | SHAP TreeExplainer |
| **Support Vector Machine (RBF)** | Kernelized maximum margin classifier in infinite-dimensional Hilbert space. | SHAP KernelExplainer (k-means summarized background) |

---

## 7. Explainable AI (XAI) Methodology

Ground-truth model interpretability is implemented via cooperative game theory:
- **Local Explanations**: When a clinician adjusts feature sliders or runs inference on a patient biopsy, SHAP calculates exact Shapley values $\phi_i$ satisfying efficiency, symmetry, dummy, and additivity axioms:
  $$\sum_{i=1}^M \phi_i(f, x) = f(x) - \mathbb{E}[f(X)]$$
- **Directional Impact**: Positive SHAP values indicate cellular features that actively drive the prediction toward **Malignancy**, highlighted in clinical coral. Negative SHAP values highlight protective or normal cellular features pulling toward **Benign**, highlighted in clinical teal.
- **Global Explanations**: Cohort-wide Mean Absolute SHAP values $\frac{1}{N}\sum |\phi_i|$ rank the dataset's strongest diagnostic markers across the test set.

---

## 8. Project Architecture

```text
BreastGuardAI/
├── app/
│   ├── main.py                   # Streamlit multi-page navigation & layout
│   ├── components/               # Reusable UI widgets (cards, gauges, visualizers)
│   │   ├── sidebar.py
│   │   ├── dashboard.py
│   │   ├── prediction.py
│   │   ├── explainability.py
│   │   └── visualizations.py
│   └── pages/                    # 6 dedicated presentation views
│       ├── overview.py
│       ├── dataset_analysis.py
│       ├── model_performance.py
│       ├── prediction_page.py
│       ├── explainable_ai.py
│       └── about.py
├── config/
│   └── config.py                 # Centralized configuration & clinical parameters
├── data/
│   ├── raw/                      # Raw UCI dataset extracts
│   └── processed/                # Normalized, validated data partitions
├── models/                       # Serialized model artifacts (.joblib)
├── notebooks/
│   └── exploratory_data_analysis.ipynb # Interactive statistical prototyping
├── reports/
│   ├── figures/                  # Publication-ready ROC, PR, and SHAP plots
│   └── evaluation/               # Model comparison JSON & CSV scorecards
├── src/                          # Production Python source modules
│   ├── data_loader.py            # Robust data loading & schema validation
│   ├── preprocessing.py          # Leakage-free StandardScaler pipelines
│   ├── train.py                  # Stratified K-fold model training engine
│   ├── evaluate.py               # Oncological metric & confusion matrix suite
│   ├── explain.py                # SHAP explainer orchestration
│   ├── predict.py                # High-level inference service with risk tiers
│   └── utils.py                  # Shared logging and formatting utilities
├── tests/                        # Automated unit & integration tests
│   ├── test_data.py
│   ├── test_preprocessing.py
│   └── test_prediction.py
├── run.py                        # Root execution script
└── requirements.txt              # Strict, pre-compiled dependency specifications
```

---

## 9. Installation Instructions

### Prerequisites
- Windows 10/11 (or macOS / Linux)
- Python 3.13 (recommended: Anaconda or official python.org release)
- **No Microsoft C++ Build Tools required.**

### Setup Virtual Environment (Windows PowerShell)

```powershell
# 1. Clone repository and navigate to root directory
cd "Breast Cancer Classification"

# 2. Activate virtual environment
.\.venv\Scripts\Activate.ps1

# 3. Verify Python version (should be Python 3.13.x)
python --version

# 4. Install dependencies (if not pre-installed)
pip install -r requirements.txt
```

---

## 10. How to Run the Application

Launch the interactive dashboard:

```powershell
# Using the root convenience runner:
python run.py

# Or directly through Streamlit:
streamlit run app/main.py
```

The application will launch in your browser at `http://localhost:8501`.

### Quality Assurance & Automated Verification

To run the complete automated test suite (112 test items across 11 test modules):

```powershell
pytest tests/ -v
```

To run the comprehensive technical audit script (AST compilation, dataset integrity, zero leakage, model synchronization, inference concordance, OOD thresholds):

```powershell
python scripts/audit_codebase.py
```

To run the Streamlit dashboard and artifact health check:

```powershell
python scripts/verify_dashboard_app.py
```

Comprehensive audit and test execution reports are available in:
- [`reports/quality_assurance/project_audit.md`](reports/quality_assurance/project_audit.md)
- [`reports/quality_assurance/test_report.md`](reports/quality_assurance/test_report.md)

---

## 11. Interactive Clinical Prediction & Batch Pipeline

BreastGuard AI includes an end-to-end inference subsystem supporting both single-patient manual entry and multi-patient batch CSV processing:

### 11.1 Manual Single-Patient Prediction
- **Workflow**: Enter the 30 continuous cytological measurements from a digitized fine-needle aspirate (FNA).
- **Sensible Defaults**: Pre-populated with population median values from the Wisconsin Diagnostic Breast Cancer cohort.
- **Preset Options**: One-click buttons to load representative **Benign** or **Malignant** biopsy profiles.
- **Strict Validation**: Rejects missing features, non-numeric strings, infinite values, and negative measurements.
- **Threshold-Aware Output**: Applies the calibrated clinical decision threshold ($\tau = 0.37$) to prioritize malignant sensitivity.
- **Local SHAP Explainability**: Decomposes the prediction into population baseline ($E[f(X)] = 24.3\%$) plus individual signed feature contributions, showing which features push toward malignancy or benignity.

### 11.2 Required Input Features (30 WDBC Parameters)
Each cell measurement has three versions (**Mean**, **Standard Error (SE)**, and **Worst** / largest):
1. `radius`: Distance from center to points on perimeter.
2. `texture`: Standard deviation of gray-scale values.
3. `perimeter`: Total nuclear boundary length.
4. `area`: Total nuclear cross-sectional area.
5. `smoothness`: Local variation in radius lengths.
6. `compactness`: $\frac{\text{perimeter}^2}{\text{area}} - 1.0$.
7. `concavity`: Severity of concave portions of the contour.
8. `concave_points`: Number of concave portions of the contour.
9. `symmetry`: Relative symmetry of the nuclear shape.
10. `fractal_dimension`: "Coastline approximation" ($-1$).

### 11.3 CSV Batch Prediction
- **Batch Processing**: Upload a CSV file containing multiple patient biopsy records with all 30 WDBC feature headers (optional `id` column supported).
- **Error Handling**: Validates every row; invalid records are flagged with exact row numbers and error descriptions without silently discarding records.
- **Export Destination**: [`reports/predictions/batch_prediction_results.csv`](file:///c:/Users/Purushottam/OneDrive/Desktop/Breast%20Cancer%20Classification/reports/predictions/batch_prediction_results.csv) containing:
  - `record_id`
  - `predicted_class` (`Malignant`, `Benign`, or `ERROR`)
  - `prob_benign` & `prob_malignant`
  - `classification_threshold` ($0.37$)
  - `risk_tier` & `prediction_status`

### 11.4 Individual PDF Clinical Prediction Report
- Generates a downloadable, high-resolution PDF document summarizing the diagnostic classification.
- Includes probability indicators, risk stratification tier, top 8 local SHAP contributing features (rendered as vector horizontal bar charts), narrative explanation, generation timestamp, and mandatory educational disclaimers.

---

## 12. Modern Multi-Page Dashboard Architecture (Prompt 9)

BreastGuard AI features a modern, interactive, competition-grade analytics dashboard built with Streamlit and Plotly, styled with a clinical dark theme and accessible typography:

```
app/
├── main.py                     # Streamlit orchestrator & routing
├── components/
│   ├── sidebar.py              # Branding, navigation state, model specs, safety notice
│   ├── dashboard.py            # KPI cards, page headers, disclaimer banners
│   ├── visualizations.py       # Plotly confusion matrix, distribution, correlation & tradeoff charts
│   ├── prediction.py           # 30-feature manual form, presets, risk gauge, PDF export
│   └── explainability.py       # SHAP horizontal bars, waterfall attributions
└── pages/
    ├── overview.py             # Executive overview, KPI cards, cohort donut, 5-model CV benchmark
    ├── dataset_analysis.py      # Feature distributions, multicollinearity heatmap, top biomarkers
    ├── model_performance.py    # Held-out test evaluation, 5-fold CV, threshold sweep, calibration
    ├── prediction_page.py      # Live single-patient inference, presets, batch CSV upload & report export
    ├── explainable_ai.py       # Global SHAP importance, beeswarm plots, what-if sweep, counterfactuals
    └── about.py                # Provenance, ML architecture, EU AI Act / FDA GMLP governance, limitations
```

### Dashboard Views
1. **Executive Overview**: High-level KPI indicators (569 biopsies, 30 features, Champion SVM model, 99.12% test accuracy, 97.62% test sensitivity), cohort balance donut chart, 5-model cross-validation comparison, clinical pillars, architectural workflow pipeline, and quick navigation shortcuts.
2. **Dataset Analysis & EDA**: Tabbed exploration of cytological distributions (histograms with box plots split by Benign/Malignant), 30x30 Pearson multicollinearity heatmap, ranking of top discriminating biomarkers, and sanitized biopsy data matrix.
3. **Model Performance**: Full empirical benchmark of 5 models on 5-fold cross-validation and strictly held-out test data (114 samples: 41 TP, 72 TN, 0 FP, 1 FN), ROC and Precision-Recall curves, decision threshold optimization ($\tau = 0.37$), and Brier score calibration ranking.
4. **Clinical Prediction & Triage**: Dual-mode inference console supporting 30-feature manual entry with clinical presets, live risk gauge, signed local SHAP feature attributions, narrative statements, vector PDF report download, and batch CSV processing.
5. **Explainable AI (SHAP)**: Global feature importance bar and beeswarm charts, sample-level waterfall decomposition with verified additivity ($\Delta = 0.000000$), What-If parameter sensitivity curves, algorithmic counterfactual recourse exploration with strict clinical safety boundaries, and game-theoretic axiomatic foundations.
6. **Responsible AI & Reliability**: Dedicated trustworthiness dashboard auditing Expected Calibration Error ($\text{ECE} = 0.0242$), Mahalanobis & Isolation Forest Out-of-Distribution detection, cytological subgroup equity, and Monte Carlo noise perturbation robustness.
7. **About & Responsible AI**: Background motivation, WDBC dataset provenance (Wolberg et al., 1995), end-to-end pipeline diagrams, regulatory alignment (EU AI Act High-Risk Class IIb, FDA Good Machine Learning Practice), transparent limitations, and research disclaimer.

---

## 13. Responsible AI, Reliability & Robustness (Prompt 10)

BreastGuard AI integrates a Trustworthy AI framework adhering to strict clinical software engineering standards:

### 13.1 Probabilistic Calibration & Reliability
- **Expected Calibration Error (ECE)**: $0.0242$ across 10 equal-width bins on the champion SVM model.
- **Brier Score Reliability**: Test Brier score of **$0.0157$** (#1 among all 5 evaluated algorithms).
- **Distinction of Concepts**: Predicted probability is clearly distinguished from calibrated empirical frequency and clinical certainty.

### 13.2 Out-of-Distribution (OOD) Detection
- **Multi-Method Detection**: Regularized Mahalanobis distance with Ledoit-Wolf shrinkage to accommodate 30-feature cytological collinearity, coupled with an Isolation Forest unsupervised anomaly detector.
- **Training-Only Thresholds**: Thresholds ($\tau_{95} = 6.42$, $\tau_{99} = 8.15$) were fitted strictly on the 455-sample training partition. The held-out test partition was never exposed.
- **Non-Diagnostic Policy**: An OOD alert indicates mathematical atypicality, never an automated disease classification.

### 13.3 Subgroup Performance & Demographic Fairness Limits
- **Feature-Based Cytological Strata**: Performance evaluated across tumor size (`mean_radius` $\le 13$ mm vs $13-17.5$ mm vs $> 17.5$ mm) and nuclear texture cohorts. Achieved 100% sensitivity on small early-stage lesions.
- **Demographic Truth in Reporting**: The WDBC dataset contains **zero demographic attributes** (no age, race, ethnicity, or socioeconomic status). The project explicitly refrains from claiming demographic fairness, documenting that true equity requires prospective multi-center demographic registries.

### 13.4 Model Robustness & Input Perturbation Testing
- **Monte Carlo Noise Injection**: Evaluated across 100 stochastic trials at $\epsilon = 1\%, 5\%, 10\%$ measurement noise.
- **Stability Separation**: Clear-cut benign and malignant cases maintain $\ge 98\%$ classification stability, whereas borderline samples near $\tau = 0.37$ exhibit expected sensitivity, automatically flagged for pathology review.
- **Single-Feature Elasticity**: Ranks features by sensitivity under $\pm 10\%$ individual shifts (top drivers: `worst_texture`, `worst_radius`, `worst_area`).

### 13.5 Transparent Safety Warning System
- Non-medical alert system categorizing inputs into `INPUT_VALIDATION`, `OUT_OF_DISTRIBUTION`, and `BORDERLINE_CONFIDENCE`. Never generates a composite medical score.
- Full details documented in [`reports/responsible_ai_report.md`](file:///c:/Users/Purushottam/OneDrive/Desktop/Breast%20Cancer%20Classification/reports/responsible_ai_report.md).

---

## 14. Evaluation Metrics

Evaluations treat **Malignant** as the primary clinical positive condition:

- **Sensitivity (Recall on Malignant)**: $\frac{\text{TP}}{\text{TP} + \text{FN}}$ — Critical safety metric measuring the proportion of cancerous samples correctly detected.
- **Specificity (True Negative Rate)**: $\frac{\text{TN}}{\text{TN} + \text{FP}}$ — Measures proportion of benign cases correctly classified.
- **Precision (Positive Predictive Value)**: $\frac{\text{TP}}{\text{TP} + \text{FP}}$ — Confidence that a positive flag corresponds to malignancy.
- **Negative Predictive Value (NPV)**: $\frac{\text{TN}}{\text{TN} + \text{FN}}$ — Confidence that a benign prediction is truly cancer-free.
- **F1-Score**: Harmonic mean of precision and sensitivity.
- **ROC-AUC**: Area under the Receiver Operating Characteristic curve.
- **Brier Score**: Measure of probabilistic calibration error (mean squared difference between predicted probability and actual binary outcome).

---

## 15. Research Documentation & Competition Presentation Package (Prompt 12)

BreastGuard AI includes a complete academic documentation and competition presentation package for research symposiums and university-level AI competitions:

| Document / Asset | Format | Location | Description |
|---|:---:|---|---|
| **Technical Monograph** | Markdown | [`reports/documentation/PROJECT_DOCUMENTATION.md`](reports/documentation/PROJECT_DOCUMENTATION.md) | 14-chapter exhaustive technical reference detailing dataset morphology, zero-leakage pipeline, multi-algorithm benchmarking, calibration, SHAP mathematical formulation, and architecture. |
| **Academic Research Paper** | Markdown | [`reports/documentation/BreastGuard_AI_Research_Paper.md`](reports/documentation/BreastGuard_AI_Research_Paper.md) | Full academic paper format (Abstract, Related Work, Methodology, Experiments, Results, XAI, Responsible AI, Genuine References). |
| **Competition Slide Deck** | PPTX (16:9) | [`reports/presentation/BreastGuard_AI_Competition_Presentation.pptx`](reports/presentation/BreastGuard_AI_Competition_Presentation.pptx) | 15-slide widescreen presentation deck with embedded figures, KPI cards, benchmark scorecards, and defense takeaways. |
| **Speaker Script & Defense Guide** | Markdown | [`reports/presentation/COMPETITION_PRESENTATION_SCRIPT_AND_DEFENSE.md`](reports/presentation/COMPETITION_PRESENTATION_SCRIPT_AND_DEFENSE.md) | Slide-by-slide 10-minute presentation script, 3-minute live app demo walkthrough, and 10 difficult judges' Q&A defense answers. |
| **Quality Assurance Audit** | Markdown | [`reports/quality_assurance/project_audit.md`](reports/quality_assurance/project_audit.md) | Rigorous AST syntax audit, zero-leakage verification, model synchronization, and OOD boundary checks. |
| **Test Execution Report** | Markdown | [`reports/quality_assurance/test_report.md`](reports/quality_assurance/test_report.md) | Pytest scorecard covering 112 passed tests across 11 test modules. |

---

## 16. Medical Disclaimer

> **IMPORTANT CLINICAL & LEGAL DISCLAIMER**:  
> BreastGuard AI is an **educational research prototype** created solely for scientific evaluation, academic coursework, and AI competitions. It is **not** a certified medical diagnostic system, healthcare device, or clinical decision support tool.  
> 
> - Predictions and probabilities generated by this software must **never** be used for patient triage, treatment planning, or clinical diagnosis.
> - The Wisconsin Diagnostic Breast Cancer dataset represents a historical FNA cohort that may not reflect contemporary clinical populations, diverse demographic distributions, or differing tissue preparation techniques.
> - Anyone seeking medical evaluation must consult a board-certified physician or qualified healthcare provider.

