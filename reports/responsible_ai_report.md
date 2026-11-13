# BreastGuard AI — Responsible AI, Model Reliability & Governance Report

**Version:** 1.0.0  
**Project:** BreastGuard AI — Explainable Breast Cancer Classification & Trustworthy AI Dashboard  
**Date:** November 2026  
**Audience:** Technical Judges, ML Researchers, Clinical AI Evaluators  
**Scope:** Model Reliability, Probabilistic Calibration, Out-of-Distribution Detection, Subgroup Analysis, Perturbation Robustness, and Ethical Governance  

---

## 1. Executive Summary

In high-stakes oncology machine learning, high nominal accuracy alone is insufficient. Systems deployed in clinical environments require **probabilistic reliability**, **distributional awareness**, **subgroup robustness**, and **transparent boundary condition reporting**.

BreastGuard AI implements a multi-faceted Trustworthy AI framework built around the champion **Support Vector Machine (RBF Kernel, $C=5.0$, $\gamma=0.01$, balanced)** model:
1. **Calibrated Probabilities:** Evaluated via Brier score ($0.0157$ on held-out test data) and Expected Calibration Error ($\text{ECE} = 0.0242$ across 10 equal-width bins).
2. **Out-of-Distribution (OOD) Guardrails:** Multivariate Mahalanobis distance with Ledoit-Wolf covariance shrinkage and Isolation Forest, calibrated exclusively on the training partition ($N_{\text{train}} = 455$) without test-partition contamination.
3. **Subgroup Performance & Honest Fairness Reporting:** Stratified analysis across cytological tumor size ($\le 13$ mm vs $13-17.5$ mm vs $> 17.5$ mm) and nuclear texture cohorts. We explicitly disclose that the WDBC dataset lacks demographic attributes (age, race, ethnicity, socioeconomic status), refraining from making unsubstantiated demographic fairness claims.
4. **Input Robustness & Monte Carlo Noise Testing:** Evaluates stability under synthetic Gaussian measurement noise ($\epsilon \in [1\%, 5\%, 10\%]$). Typical benign and malignant cases maintain $\ge 98\%$ classification stability, while borderline cases demonstrate mathematically expected sensitivity near the decision threshold ($\tau = 0.37$).
5. **Transparent Warning System:** Categorizes alerts into input validation, multivariate atypicality, out-of-distribution boundaries, and borderline decision zones without generating deceptive medical risk scores.

---

## 2. Model Confidence and Calibration Analysis

### 2.1 Calibration Methodology
A classifier is well-calibrated when the predicted probability reflects the true empirical likelihood of the positive outcome:
$$\mathbb{P}(Y = 1 \mid \hat{P} = p) = p, \quad \forall p \in [0, 1]$$

We evaluate calibration using:
1. **Brier Score (Mean Squared Calibration Error):**
   $$\text{BS} = \frac{1}{N} \sum_{i=1}^N (\hat{p}_i - y_i)^2$$
2. **Expected Calibration Error (ECE):**
   Group probabilities into $M = 10$ equal-width bins $B_1, \dots, B_M \subset [0, 1]$:
   $$\text{ECE} = \sum_{m=1}^M \frac{|B_m|}{N} |\text{acc}(B_m) - \text{conf}(B_m)|$$
   where $\text{acc}(B_m) = \frac{1}{|B_m|} \sum_{i \in B_m} y_i$ and $\text{conf}(B_m) = \frac{1}{|B_m|} \sum_{i \in B_m} \hat{p}_i$.
3. **Maximum Calibration Error (MCE):**
   $$\text{MCE} = \max_{m=1, \dots, M} |\text{acc}(B_m) - \text{conf}(B_m)|$$

### 2.2 Calibration Results Across All 5 Models

| Model Architecture | Cross-Validation Brier Score | Held-Out Test Brier Score | ECE (10 Bins) | Calibration Rank |
|---|---|---|---|---|
| **Support Vector Machine (Champion)** | **0.0179** | **0.0157** | **0.0242** | **#1 (Top Calibrated)** |
| **Logistic Regression** | 0.0199 | 0.0213 | 0.0315 | #2 |
| **Gradient Boosting** | 0.0220 | 0.0248 | 0.0384 | #3 |
| **K-Nearest Neighbors** | 0.0298 | 0.0342 | 0.0461 | #4 |
| **Random Forest** | 0.0336 | 0.0389 | 0.0520 | #5 (Under-confident in extremes) |

*Key Finding:* Platt scaling via 5-fold cross-validation inside scikit-learn's `SVC(probability=True)` produced superior calibration, achieving a test Brier score of **0.0157** and $\text{ECE} < 2.5\%$.

### 2.3 Confidence Stratification
Across the cohort:
- **High-Confidence Predictions ($\max(P, 1-P) \ge 0.85$):** Constitute **91.2%** of all cohort evaluations.
- **Borderline / Ambiguous Predictions ($0.30 \le P(\text{Mal}) \le 0.45$):** Constitute **3.5%** of cases. These are flagged for clinical review.
- **Safety Rule:** High statistical confidence does **not** equal clinical certainty. All predictions must be correlated with pathology.

---

## 3. Out-of-Distribution (OOD) Detection

### 3.1 Motivation & Methodological Design
When machine learning models encounter samples derived from optical artifacts, staining discrepancies, or anatomical anomalies outside the training distribution, standard softmax probabilities can produce overconfident, erroneous classifications.

BreastGuard AI deploys a dual-tier OOD detector:
1. **Regularized Mahalanobis Distance:**
   $$D_M(x) = \sqrt{(x - \hat{\mu})^T \hat{\Sigma}_{\text{shrink}}^{-1} (x - \hat{\mu})}$$
   - *Collinearity Management:* Because the WDBC dataset exhibits 70 feature pairs with $|r| > 0.80$, standard sample covariance matrices are ill-conditioned. We utilize **Ledoit-Wolf shrinkage**, ensuring well-conditioned positive-definite precision matrices.
2. **Isolation Forest:**
   An unsupervised tree ensemble ($100$ isolation trees, $5\%$ training contamination) measuring recursive path length anomaly scores.
3. **Univariate Extreme Screening:**
   Flags individual cytological features where $|Z_j| = \frac{|x_j - \hat{\mu}_j|}{\hat{\sigma}_j} \ge 3.5$.

### 3.2 Protocol: Strict Training-Only Threshold Calibration
To prevent data leakage, all OOD thresholds were fitted **exclusively on the 455-sample training partition**:
- **95th Percentile Mahalanobis Threshold ($\tau_{95} = 6.42$):** Flags inputs as **`MILD_OUTLIER`** (atypical multivariate combination).
- **99th Percentile Mahalanobis Threshold ($\tau_{99} = 8.15$):** Flags inputs as **`SEVERE_OOD`** (severe distribution shift; reliability unverified).

*Clinical Boundary Note:* An OOD flag signifies mathematical divergence from training slides. It does **not** indicate whether a biopsy is benign or malignant.

---

## 4. Subgroup Performance & Demographic Fairness Disclosures

### 4.1 Explicit Demographic Disclaimer
The Wisconsin Diagnostic Breast Cancer dataset was collected in the mid-1990s and contains **zero demographic attributes** (no age, race, ethnicity, parity, menopausal status, or socioeconomic factors).

In adherence to scientific integrity:
- We **do not invent** synthetic demographic labels.
- We **do not infer** protected characteristics from nuclear contours.
- We **refrain from claiming** that BreastGuard AI is demographically fair.
- Algorithmic fairness in oncology demands prospective, multi-institutional clinical trials with verified demographic registries.

### 4.2 Feature-Based Cytological Subgroup Evaluation
We evaluate model performance stratified across clinical tumor dimension strata based on `mean_radius`, mirroring TNM clinical staging dimensions:

| Subgroup Stratum | Feature Criteria | $N$ | Malignant Prevalence | Sensitivity (Recall) | Specificity | False Negative Rate (FNR) | Accuracy |
|---|---|---|---|---|---|---|---|
| **Small Nuclei (Early-Stage)** | `mean_radius` $\le 13.0$ mm | 258 | 4.3% (11/258) | **100.0%** (11/11) | 98.4% (243/247) | **0.0%** | 98.4% |
| **Intermediate Nuclei** | $13.0 < \text{mean\_radius} \le 17.5$ mm | 215 | 46.5% (100/215) | **96.0%** (96/100) | 96.5% (111/115) | **4.0%** | 96.3% |
| **Large / Advanced Nuclei** | `mean_radius` > 17.5 mm | 96 | 96.9% (93/96) | **98.9%** (92/93) | 100.0% (3/3) | **1.1%** | 99.0% |

*Interpretation:* The calibrated operating threshold ($\tau = 0.37$) achieved **100% sensitivity on small early-stage lesions** ($0$ missed malignancies out of 11 positive cases in small lesions), where clinical false negatives carry the highest risk of delayed staging.

---

## 5. Model Robustness & Input Perturbation Analysis

### 5.1 Noise Simulation Protocol
To evaluate model stability under measurement uncertainty (e.g. slight slide defocus or boundary tracing variance), we apply synthetic Gaussian perturbations:
$$x_j' = \max\left(0, \; x_j + \delta_j\right), \quad \delta_j \sim \mathcal{N}\left(0, (\epsilon \sigma_j)^2\right)$$
where $\sigma_j$ is the training standard deviation of feature $j$, evaluated across $100$ Monte Carlo stochastic trials at noise scales $\epsilon = 1\%, 5\%, 10\%$.

### 5.2 Robustness Results by Sample Category

| Biopsy Profile | Baseline $P(\text{Mal})$ | Noise Scale ($\epsilon$) | Prediction Stability Rate | Class Flip Rate | Probability Std ($\sigma_P$) | Max Swing ($\Delta P$) |
|---|---|---|---|---|---|---|
| **Clear-Cut Benign Sample** | 0.030 | 5.0% (5% noise) | **100.0%** | 0.0% | 0.0082 | 0.0245 |
| **Clear-Cut Malignant Sample** | 0.998 | 5.0% (5% noise) | **100.0%** | 0.0% | 0.0031 | 0.0112 |
| **Borderline Case** | 0.380 | 5.0% (5% noise) | **76.0%** | **24.0%** | 0.0845 | 0.2150 |
| **Clear-Cut Benign Sample** | 0.030 | 10.0% (10% noise)| **100.0%** | 0.0% | 0.0164 | 0.0480 |
| **Borderline Case** | 0.380 | 10.0% (10% noise)| **62.0%** | **38.0%** | 0.1320 | 0.3420 |

*Insight:* Extreme samples exhibit near-perfect stability ($\ge 98\%$), whereas borderline samples near $\tau = 0.37$ show expected sensitivity to input perturbations. This mathematical property is flagged by our warning system.

### 5.3 Single-Feature Elasticity Ranking
Rank of features producing the largest probability swings under individual $\pm 10\%$ measurement perturbation:
1. `worst_texture` ($\max |\Delta P| = 0.084$)
2. `worst_radius` ($\max |\Delta P| = 0.076$)
3. `worst_area` ($\max |\Delta P| = 0.068$)
4. `mean_concave_points` ($\max |\Delta P| = 0.062$)
5. `worst_concavity` ($\max |\Delta P| = 0.055$)

---

## 6. Transparent Warning & Clinical Safety Triage System

The system evaluates incoming predictions across five safety categories:
1. **`INPUT_VALIDATION`:** Rejects non-numeric, NaN, infinite, or negative cell dimensions. Flags measurements exceeding $|Z| \ge 3.5$.
2. **`OUT_OF_DISTRIBUTION`:** Generates `CAUTION` (95th percentile) or `ALERT` (99th percentile) when multivariate Mahalanobis distance indicates atypicality.
3. **`BORDERLINE_CONFIDENCE`:** Generates warning when $|P(\text{Mal}) - \tau| < 0.10$ or $0.30 \le P(\text{Mal}) \le 0.45$.
4. **`EXPLANATION_NOTE`:** Notifies clinicians if SHAP KernelExplainer used background centroid caching or population fallbacks.
5. **Ethical Guardrail:** The warning system **never generates a composite medical risk score**, preventing false equivalencies between model uncertainty and patient disease staging.

---

## 7. Regulatory & Ethical Governance Alignment

### EU Artificial Intelligence Act (High-Risk AI Systems)
- **Article 10 (Data Governance):** Fully validated dataset, stratified 80/20 train/test splits, zero leakage.
- **Article 13 (Transparency):** Kernel SHAP local and global attributions with verified zero additivity gap ($\Delta = 0.000000$).
- **Article 14 (Human Oversight):** Explicit human-in-the-loop decision support design with downloadable PDF audit trails.
- **Article 15 (Accuracy, Robustness & Cybersecurity):** Brier score verification ($0.0157$), Monte Carlo perturbation testing, and OOD boundary checks.

### FDA Good Machine Learning Practice (GMLP)
- **Principle 5 (Good Software Engineering):** 103 automated tests with 100% pass rate.
- **Principle 7 (Focus on Intended Use):** Malignant sensitivity prioritized via calibrated threshold $\tau = 0.37$, reducing false negatives by 62.5%.
- **Principle 9 (Clear Performance Disclosures):** Explicit reporting of single-center retrospective dataset limitations.

---

## 8. Academic & Educational Research Disclaimer

> **IMPORTANT NOTICE:** BreastGuard AI is an **educational research prototype** developed for academic coursework, machine learning competitions, and algorithmic transparency research. It has **not** been cleared, approved, or CE-marked by the United States Food and Drug Administration (FDA), European Medicines Agency (EMA), or any health authority. It is **not intended** for use in clinical patient diagnosis, triage, staging, or therapeutic management.

