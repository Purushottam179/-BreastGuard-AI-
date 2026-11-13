# BreastGuard AI — Comprehensive Project Quality Assurance & Technical Audit Report

**Report Version:** 1.0.0  
**Audit Date:** November 2026  
**Auditor:** Automated QA Suite & Senior Machine Learning Engineer  
**Status:** ✅ ALL SUBSYSTEMS VERIFIED — PRODUCTION & COMPETITION READY  
**Test Suite Status:** 103/103 Tests Passing (100% Pass Rate)  

---

## 1. Executive Summary

This document provides a comprehensive technical audit of the **BreastGuard AI** codebase, machine learning pipeline, prediction subsystem, Explainable AI engine, Responsible AI framework, and Streamlit interactive dashboard.

The audit was executed to ensure technical soundness, zero data leakage, probabilistic calibration, mathematical determinism, and robust error handling ahead of university-level competition evaluation.

### Key Audit Findings
1. **Zero Data Leakage Confirmed:** The 20% held-out test partition ($N=114$) was isolated prior to feature scaling or model training. Preprocessing (`StandardScaler`) is encapsulated inside scikit-learn `Pipeline` constructs and fitted strictly within cross-validation folds.
2. **Model & Metadata Synchronization:** The champion model artifact (`models/final_breastguard_model.pkl`) matches the evaluation report metadata across hyperparameters ($C=5.0$, $\gamma=0.01$, balanced weights), test sensitivity ($97.62\%$), test accuracy ($99.12\%$), and decision cutoff ($\tau = 0.37$).
3. **Dynamic Class Order Resolution:** The inference engine dynamically resolves class indices using `clf.classes_` rather than hardcoding column indices, eliminating class transposition risks.
4. **Game-Theoretic Explainability Integrity:** SHAP KernelExplainer attributions exhibit an exact zero additivity gap ($\sum \phi_i = f(x) - E[f(x)]$, $\Delta = 0.000000$) with deterministic background centroid caching.
5. **Responsible AI & Demographic Truth:** Subgroup performance was rigorously verified across tumor size and texture strata. The absence of demographic PII (age, race, ethnicity) in the WDBC dataset is explicitly documented, preventing unsubstantiated fairness claims.

---

## 2. Codebase Inventory & Structural Audit

| Directory / Artifact | File Count | Status | Description |
|---|:---:|:---:|---|
| [`app/`](file:///c:/Users/Purushottam/OneDrive/Desktop/Breast%20Cancer%20Classification/app) | 16 | **Verified** | Streamlit multi-page orchestrator, 7 pages, and 5 reusable UI components. |
| [`src/`](file:///c:/Users/Purushottam/OneDrive/Desktop/Breast%20Cancer%20Classification/src) | 14 | **Verified** | Core machine learning, preprocessing, training, tuning, XAI, OOD, reliability, and inference modules. |
| [`data/`](file:///c:/Users/Purushottam/OneDrive/Desktop/Breast%20Cancer%20Classification/data) | 3 | **Verified** | Raw and processed WDBC datasets (`breast_cancer_cleaned.csv`, `wdbc_cleaned.csv`, `dataset_info.json`). |
| [`models/`](file:///c:/Users/Purushottam/OneDrive/Desktop/Breast%20Cancer%20Classification/models) | 6 | **Verified** | Serialized baseline models (LR, RF, SVM, GB, KNN) and champion model (`final_breastguard_model.pkl`). |
| [`reports/`](file:///c:/Users/Purushottam/OneDrive/Desktop/Breast%20Cancer%20Classification/reports) | 35+ | **Verified** | JSON/CSV evaluation metrics, high-resolution figures, PDF export templates, and Markdown reports. |
| [`tests/`](file:///c:/Users/Purushottam/OneDrive/Desktop/Breast%20Cancer%20Classification/tests) | 9 | **Verified** | 103 automated pytest test cases covering data, pipelines, models, XAI, UI, and reliability. |
| [`config/config.py`](file:///c:/Users/Purushottam/OneDrive/Desktop/Breast%20Cancer%20Classification/config/config.py) | 1 | **Verified** | Centralized paths, random seeds, hyperparameters, color tokens, and target encodings. |
| [`run.py`](file:///c:/Users/Purushottam/OneDrive/Desktop/Breast%20Cancer%20Classification/run.py) | 1 | **Verified** | Convenience CLI entry point for launching Streamlit or executing the test suite. |

---

## 3. Machine Learning Pipeline Verification

### 3.1 Dataset Quality & Schema Validation
- **Source:** Wisconsin Diagnostic Breast Cancer (WDBC) dataset (UCI Repository / scikit-learn).
- **Dimensions:** Exactly 569 patient biopsy records and 30 continuous numerical features.
- **Data Integrity:**
  - Missing Values: **0 (0.0%)**
  - Infinite Values: **0 (0.0%)**
  - Negative Values: **0 (0.0%)** (all physical cellular dimensions are strictly positive).
  - Target Encoding: **Malignant = 1** (Clinical Positive, $N=212$, $37.26\%$), **Benign = 0** (Clinical Negative, $N=357$, $62.74\%$).
  - ID Column: Removed from feature matrices to prevent identity-based memorization.

### 3.2 Partitioning & Leakage Prevention
- **Split Ratio:** Stratified 80% Training ($N=455$) and 20% Held-Out Testing ($N=114$).
- **Partition Overlap:** Evaluated using row-level hash intersections. Overlap is strictly **0 rows**.
- **Preprocessing Placement:** Feature standardization (`StandardScaler`) is enclosed within scikit-learn `Pipeline` objects. In 5-fold cross-validation, scaler parameters ($\mu, \sigma$) are fitted exclusively on the 4 training folds per split, ensuring zero information bleed into validation folds.
- **Held-Out Test Set Isolation:** The 114 test biopsies were isolated before hyperparameter grid/random search, threshold tuning, and OOD threshold calculation.

### 3.3 Model Selection & Threshold Optimization
- **Algorithms Evaluated:** Logistic Regression, Random Forest, Support Vector Machine, Gradient Boosting, K-Nearest Neighbors.
- **Selection Decision:** Support Vector Machine (RBF kernel) was chosen as the champion based on:
  1. Top cross-validation ROC-AUC ($0.9959 \pm 0.0051$) and F1-score ($0.9701 \pm 0.0099$).
  2. Lowest cross-validation variance ($\sigma_{\text{acc}} = 0.0054$), proving stability.
  3. Superior probability calibration (test Brier score $0.0157$).
- **Decision Threshold:** Lowered from the generic $0.50$ default to **$\tau = 0.37$** via Youden's $J$ optimization ($J = 0.9578$), reducing training cross-validation false negatives by **62.5%** while retaining 100% specificity on held-out test data.

---

## 4. Prediction Subsystem Audit

### 4.1 Concordance Testing on Real Cohort Biopsies
Evaluated 10 verified samples (5 Benign, 5 Malignant) from the cleaned dataset through `PredictionEngine.predict_single()`:
- **Benign Samples Concordance:** **5 / 5 (100.0%)**
- **Malignant Samples Concordance:** **5 / 5 (100.0%)**
- **Decision Alignment:** Displayed probability, risk tier, and predicted class perfectly match across UI cards, CSV batch outputs, and exported PDF reports.

### 4.2 Batch Processing & Error Resilience
- Tested batch inference on clean CSV files, missing column inputs, and corrupted records containing strings and negative dimensions.
- The engine rejects invalid rows with row-specific diagnostic reasons without crashing or discarding valid rows.

### 4.3 PDF Clinical Report Generation
- Direct vector PDF generation via `matplotlib.backends.backend_pdf.PdfPages` produces publication-quality clinical summary sheets ($53.8$ KB) containing probability indicators, calibrated cutoff line ($\tau = 0.37$), top SHAP attributions, narrative statements, generation timestamp, and mandatory research disclaimers.

---

## 5. Explainable AI & SHAP Audit

### 5.1 Mathematical Correctness
- **Explainer Selection:** `shap.KernelExplainer` correctly wraps the entire Pipeline (`StandardScaler` + `SVC`), ensuring explanations are calculated directly in unscaled physical measurement units ($\mu\text{m}$, $\mu\text{m}^2$).
- **Additivity Verification:** Verified that $\sum_{j=1}^{30} \phi_j = f(x) - E[f(x)]$ within float tolerance ($\Delta = 0.000000$).
- **Target Class Explained:** Strictly aligned with Class 1 (Malignancy). Positive SHAP values indicate increased cancer likelihood; negative values indicate benignity.
- **Explanation Consistency:** Cached background centroids ($K=40$) ensure deterministic reproducibility with zero drift across repeat queries.

### 5.2 Clinical Safety Framing
- What-If sweeps and counterfactual explanations explicitly include disclaimers stating that feature perturbations represent mathematical sensitivity probes of the decision surface, **not** biological interventions or surgical targets.

---

## 6. Advanced Responsible AI Audit

### 6.1 Calibration & Reliability
- Expected Calibration Error: $\text{ECE} = 0.0242$ across 10 equal bins.
- Brier Score: $0.0157$ on test data (#1 among all models).
- High Confidence Ratio: $91.2\%$ of predictions exhibit high statistical margin. Borderline predictions ($3.5\%$) are flagged for clinical review.

### 6.2 Out-of-Distribution (OOD) Detection
- Regularized Mahalanobis distance with Ledoit-Wolf shrinkage handles cytological collinearity.
- Dual-tier thresholds calibrated strictly on the training partition: $\tau_{95} = 6.42$, $\tau_{99} = 8.15$.
- Verified that synthetic extreme outliers trigger `SEVERE_OOD` alerts while typical biopsies are processed normally.

### 6.3 Subgroup Performance & Demographic Fairness Limits
- Analyzed across tumor dimension tiers (`mean_radius` $\le 13$ mm, $13-17.5$ mm, $> 17.5$ mm).
- Malignant sensitivity on small early-stage lesions is **100.0%** ($0$ missed malignancies out of 11 positive cases).
- Explicitly documented that WDBC lacks demographic data, avoiding deceptive fairness claims.

### 6.4 Input Robustness
- Monte Carlo noise simulation ($100$ trials at $\epsilon = 5\%$ noise) confirms $100\%$ prediction stability on clear-cut cases.
- Borderline cases exhibit expected sensitivity ($76\%$ stability), appropriately generating caution alerts.

---

## 7. Streamlit Dashboard View Integrity

All 7 views render cleanly without syntax errors, missing artifact crashes, or broken session state transitions:
1. **📊 Executive Overview (`app/pages/overview.py`):** Verified KPI cards, interactive cohort donut, 5-model CV comparison, workflow cards, and navigation buttons.
2. **🔬 Dataset Analysis (`app/pages/dataset_analysis.py`):** Verified cytological distributions, 30x30 correlation heatmap, top biomarker rankings, and sanitized data matrix.
3. **🤖 Model Performance (`app/pages/model_performance.py`):** Verified held-out test evaluation, interactive clinical confusion matrix, ROC/PR curves, and threshold sweep.
4. **🔮 Clinical Prediction (`app/pages/prediction_page.py`):** Verified 30-feature manual inputs, presets, risk gauge, local SHAP bars, PDF download, and batch CSV uploader.
5. **🧠 Explainable AI (`app/pages/explainable_ai.py`):** Verified global importance, beeswarm plots, local case decomposition, What-If sweeps, counterfactuals, and game-theoretic foundations.
6. **🛡️ Responsible AI & Reliability (`app/pages/responsible_ai.py`):** Verified calibration curves, live OOD detector, subgroup bars, noise injection slider, and governance framework.
7. **⚖️ About Project (`app/pages/about.py`):** Verified clinical background, WDBC provenance, pipeline diagram, EU AI Act / FDA GMLP alignment, limitations, and academic disclaimer.

---

## 8. Issues Identified & Resolutions Applied

| Identified Issue | Severity | Root Cause | Resolution Applied | Status |
|---|---|---|---|---|
| Inverted Diagnosis Mapping in early EDA template | High | Mapping was `{0: "Malignant", 1: "Benign"}` in an early stub. | Corrected to standard `{0: "Benign", 1: "Malignant"}` matching UCI ground truth. | **Fixed** |
| LaTeX Escape Sequences in f-strings | Low | Unescaped backslashes (`\%`, `\Delta`) emitted Python 3.12+ syntax warnings. | Converted to raw strings and proper KaTeX escapes in `src/tune.py` and `app/pages/`. | **Fixed** |
| `load_dataset` signature inconsistency | Medium | Different modules called `load_dataset()` with or without keyword arguments. | Implemented flexible `load_dataset(*args, **kwargs)` with `AcquisitionBundle` properties (`X_df`, `y_series`). | **Fixed** |
| Pipeline named step indexing in audit script | Low | Audit script looked for `'classifier'` instead of `'model'`. | Updated lookup to support `pipe.named_steps['model']`. | **Fixed** |
| Legacy scaffold folder parsing | Low | Unused `.legacy_scaffold` files had old numeric script prefixes. | Filtered out hidden directories from active compilation suite. | **Fixed** |

---

## 9. Conclusion & Competition Readiness

BreastGuard AI has passed all verification gates:
- **103 / 103 automated tests passing.**
- **Zero data leakage.**
- **High malignant sensitivity (97.62% on test, 100% on small lesions).**
- **Calibrated probabilities (Brier score 0.0157).**
- **OOD boundaries and non-medical warning system active.**
- **Full SHAP mathematical explainability verified.**
- **Publication-grade PDF reports and interactive dashboard fully operational.**

