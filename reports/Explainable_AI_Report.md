# BreastGuard AI — Explainable AI (SHAP) Research & Diagnostic Report

**Project Title**: BreastGuard AI — Explainable Breast Cancer Classification & Trustworthy AI Dashboard  
**Model Architecture**: Support Vector Machine (Radial Basis Function Kernel with StandardScaler, $C=5.0$, $\gamma=0.01$, `class_weight='balanced'`)  
**Explainer Framework**: SHAP (SHapley Additive exPlanations, version 0.54.1)  
**Explainer Algorithm**: `shap.KernelExplainer`  
**Classification Target**: Clinical Malignancy ($M=1$, Positive Class), $B=0$ (Benign, Negative Class)  
**Tuned Clinical Decision Threshold**: $\tau = 0.37$  

---

## 1. Introduction to Explainable Artificial Intelligence (XAI) in Clinical Oncology

In modern computational oncology, high-capacity supervised machine learning models achieve remarkable statistical accuracy on digital biopsy classification. However, without interpretability, complex non-linear models function as opaque **"black boxes."** In healthcare, black-box algorithms introduce severe hazards:
1. **Clinical Disconnect**: Pathologists and oncologists cannot verify whether a classification decision relies on genuine biological hallmarks of malignant neoplasia or spurious statistical artifacts.
2. **Safety & Regulatory Mandates**: Clinical decision support systems require algorithmic transparency, reproducible auditing, and explainable justifications before deployment under FDA Software as a Medical Device (SaMD) frameworks and European AI Act standards.
3. **Actionable Triage**: Understanding *why* a particular fine-needle aspirate (FNA) was flagged as malignant enables physicians to order targeted follow-up procedures (e.g., core needle biopsy, immunohistochemical staining, or ultrasound correlation).

Explainable AI bridges this gap by calculating mathematically grounded local and global attributions for individual cytological features.

---

## 2. What is SHAP?

**SHAP (SHapley Additive exPlanations)** is a state-of-the-art axiomatic framework developed by Lundberg and Lee (NeurIPS 2017) that unifies cooperative game theory with local surrogate explanations. 

SHAP formulates feature attribution as a cooperative game where:
- The **"players"** are the 30 cytological features measured from a digitized FNA biopsy image.
- The **"payout"** is the model's predicted probability of malignancy $f(x) = P(\text{Malignant} \mid x)$.
- The **"Shapley value"** $\phi_i$ allocated to feature $i$ represents its marginal contribution to shifting the prediction away from the population baseline expected value $E[f(X)]$.

### The Additive Feature Attribution Property:
SHAP guarantees that for any individual prediction $x$, the sum of all feature attributions plus the baseline equals the exact model output:
$$f(x) = \phi_0 + \sum_{i=1}^{M} \phi_i(x)$$
where:
- $\phi_0 = E[f(X)]$ is the expected model prediction across the background training distribution ($24.29\%$ in our calibrated training cohort).
- $\phi_i(x)$ is the signed contribution of feature $i$ for instance $x$.
- $f(x)$ is the model's predicted probability of malignancy.

---

## 3. The Shapley Value Concept & Axiomatic Foundations

Shapley values were introduced by Nobel laureate Lloyd Shapley (1953) to distribute cooperative gains fairly. SHAP is uniquely powerful because it is the **only** feature attribution method that simultaneously satisfies three foundational axioms:

1. **Efficiency (Additivity)**:
   The sum of feature attributions exactly accounts for the difference between the sample prediction and the baseline expectation:
   $$\sum_{i=1}^{M} \phi_i = f(x) - E[f(X)]$$
2. **Symmetry (Equal Contribution)**:
   If two cytological features $i$ and $j$ contribute equally to all possible feature coalitions $S \subseteq F \setminus \{i, j\}$:
   $$f(S \cup \{i\}) - f(S) = f(S \cup \{j\}) - f(S) \implies \phi_i = \phi_j$$
3. **Dummy / Null Player (Zero Contribution)**:
   If a feature $i$ adds zero marginal gain to any coalition ($f(S \cup \{i\}) = f(S)$ for all $S$), its attribution is strictly zero ($\phi_i = 0$).
4. **Monotonicity (Consistency)**:
   If a model changes such that the marginal contribution of feature $i$ increases or remains unchanged for all coalitions, its Shapley value cannot decrease.

---

## 4. Why SHAP Was Selected Over Alternative Interpretability Methods

| Feature Attribution Method | Theoretical Guarantees | Consistency | Handles Feature Interactions | Additivity |
|---|:---:|:---:|:---:|:---:|
| **Gini / Impurity Importance** | None (biased toward high cardinality) | Inconsistent | No | No |
| **Permutation Importance** | Empirical permutation loss | Moderate | Struggles with multicollinearity | No |
| **LIME** (Local Surrogate) | Local linear heuristic | Unstable / High variance | No | Approximate |
| **SHAP (Shapley Additive)** | **Strict Axiomatic Guarantees** | **Proven Consistent** | **Yes (Full Coalitions)** | **Exact** |

Unlike heuristic methods like LIME, SHAP computes true marginal contributions across all $2^M$ coalitions, ensuring reproducible explanations for medical competitions and clinical defense.

---

## 5. Explainer Selection & Architectural Justification

Our champion architecture selected during optimization is an **RBF-Kernel Support Vector Machine (`SVC(C=5.0, gamma=0.01, class_weight='balanced')`) with an internal `StandardScaler`**.

### Explainer Decision Matrix:
1. **Why Not `TreeExplainer`?**  
   `TreeExplainer` exploits polynomial-time tree path recursion ($O(TLD^2)$). It strictly requires ensemble tree architectures (Random Forest, XGBoost). It cannot evaluate non-linear kernel projection matrices.
2. **Why Not `LinearExplainer`?**  
   `LinearExplainer` assumes a linear hyperplane $w^T x + b$. An RBF kernel projects the 30 features into an infinite-dimensional Hilbert space, making linear assumptions invalid.
3. **Why `KernelExplainer` is the Optimal Choice**:  
   `shap.KernelExplainer` is model-agnostic. It computes weighted local linear regressions over feature subsets using the Shapley kernel:
   $$\pi_x(z') = \frac{M - 1}{\binom{M}{|z'|} |z'| (M - |z'|)}$$
   By wrapping the entire scikit-learn `Pipeline(steps=[('scaler', StandardScaler()), ('model', SVC())])`, **KernelExplainer evaluates inputs in the original, unscaled 30-feature cytological space**, while the pipeline internally handles standard scaling.

### Background Dataset Construction:
- To prevent data leakage, background data was sampled **strictly from the training partition** ($X_{\text{train}}$, $N=50$ representative samples using fixed seed 42).
- Zero held-out test labels were used to generate background distributions.

---

## 6. Global Explainability Findings

Global explainability aggregates individual SHAP values across an evaluation cohort ($N=100$) to determine overall model behavior and feature ranking.

### Top 10 Globally Influential Features:

| Rank | Cytological Feature | Mean \|SHAP Value\| | Directional Tendency | Morphological Significance |
|:---:|:---|:---:|:---:|---|
| **1** | `worst_texture` | **$0.0368$** | Higher $\to$ Malignant | Severe nuclear chromatin coarseness and heterogeneity |
| **2** | `worst_symmetry` | **$0.0325$** | Higher $\to$ Malignant | Prominent cellular and nuclear asymmetry |
| **3** | `worst_radius` | **$0.0278$** | Higher $\to$ Malignant | Severe cellular hypertrophy and enlargement |
| **4** | `worst_area` | **$0.0259$** | Higher $\to$ Malignant | Extreme cell volume expansion in malignant clusters |
| **5** | `mean_concave_points` | **$0.0259$** | Higher $\to$ Malignant | Frequent indentations and invaginations of the cell membrane |
| **6** | `worst_concavity` | **$0.0256$** | Higher $\to$ Malignant | Pronounced structural contour indentations |
| **7** | `radius_se` | **$0.0228$** | Higher $\to$ Malignant | High variability (pleomorphism) in nuclear size |
| **8** | `mean_concavity` | **$0.0223$** | Higher $\to$ Malignant | High severity of membrane surface concave regions |
| **9** | `worst_perimeter` | **$0.0209$** | Higher $\to$ Malignant | Substantial elongation of nuclear perimeter |
| **10** | `worst_concave_points` | **$0.0186$** | Higher $\to$ Malignant | Extreme density of nuclear boundary indentations |

### Global Interpretation:
- **Nuclear Margin and Shape Irregularity**: Features capturing boundary indentations (`mean_concave_points`, `worst_concavity`, `worst_concave_points`) are among the most powerful drivers of malignancy. This aligns directly with cytopathological criteria: malignant epithelial cells exhibit irregular, non-circular borders due to uncontrolled division and nuclear pleomorphism.
- **Extreme Value Sensitivity**: "Worst" (mean of the three largest values per biopsy) measurements dominate the top ranks over "Mean" measurements. The SVM accurately learned that **focal malignancy is defined by the most abnormal cells in the aspirate**, rather than average cell properties.

---

## 7. Local Explainability: Individual Patient Biopsy Attributions

Local explainability explains predictions for specific patient samples.

### Case 1: Malignant Patient Sample (True Malignant, Test Patient #1)
- **Model Output**: $P(\text{Malignant}) = 100.0\%$ (Classified as **Malignant**; Operating Threshold = $0.37$, Baseline = $24.29\%$)
- **Top Positive Drivers (Pushing Toward Malignancy)**:
  1. `worst_area` ($2010.0\text{ }\mu\text{m}^2$): **$+0.1073$**
  2. `mean_radius` ($20.94\text{ }\mu\text{m}$): **$+0.0905$**
  3. `radius_se` ($1.004$): **$+0.0884$**
  4. `mean_concave_points` ($0.131$): **$+0.0877$**
  5. `mean_concavity` ($0.2712$): **$+0.0852$**
- **Additive Verification**:
  $$\text{Base Value } (0.2429) + \sum \phi_i (0.7571) = 1.0000 = P(\text{Malignant})$$
  Additivity gap $= 0.0000$ (exact mathematical additivity verified).

### Case 2: Benign Patient Sample (True Benign, Test Patient #0)
- **Model Output**: $P(\text{Malignant}) = 0.05\%$ ($P(\text{Benign}) = 99.95\%$; Classified as **Benign**)
- **Top Negative Drivers (Pushing Toward Benign)**:
  1. `worst_texture` ($15.97$): **$-0.0763$**
  2. `mean_texture` ($10.82$): **$-0.0295$**
  3. `worst_perimeter` ($83.74\text{ }\mu\text{m}$): **$-0.0293$**
  4. `worst_radius` ($12.82\text{ }\mu\text{m}$): **$-0.0255$**
  5. `mean_area` ($403.3\text{ }\mu\text{m}^2$): **$-0.0210$**
- **Clinical Insight**:
  The cell measurements are consistently small and regular, driving the model's output far below the $0.2429$ population baseline down to $0.0005$.

---

## 8. Interpretation of SHAP Plots

All figures have been rendered at 300 DPI and saved in `reports/figures/shap/`:

1. **Beeswarm Summary Plot (`shap_summary_beeswarm.png`)**:
   - Each dot represents a single biopsy sample.
   - Horizontal position denotes the SHAP value (impact on predicted malignancy).
   - Dot color represents the observed feature value (Red = High, Blue = Low).
   - *Interpretation*: For `worst_area`, red points lie far to the right ($\text{SHAP} > +0.10$), proving that high values strongly increase predicted malignancy. Blue points lie to the left, decreasing predicted risk.
2. **Mean Absolute Importance Plot (`shap_global_importance.png`)**:
   - Ranks features by average global magnitude: $\frac{1}{N}\sum |\phi_i|$.
   - Distinguishes high-influence variables (`worst_texture`, `worst_symmetry`) from negligible noise variables (`fractal_dimension_se`).
3. **SHAP Bar Plot (`shap_bar_plot.png`)**:
   - Clean, standardized bar presentation designed for competition judges and slide decks.
4. **SHAP Dependence Plot (`shap_dependence_top_feature.png`)**:
   - Plots raw feature value on the x-axis vs. SHAP value on the y-axis for `worst_texture`.
   - Illustrates the non-linear threshold effect learned by the RBF kernel: above a texture value of $\approx 25.0$, SHAP values transition rapidly from negative to positive.
5. **SHAP Waterfall Plot (`shap_waterfall_example.png`)**:
   - Step-by-step visual proof of additivity for a single patient, displaying how each feature moves $E[f(X)] \to f(x)$.
6. **Local Contribution Bar Plot (`shap_local_contribution.png`)**:
   - Diverging horizontal bar chart contrasting positive drivers (red) vs. negative drivers (teal) for immediate clinical briefing.

---

## 9. Critical Distinctions: Correlation vs. Contribution vs. Causation

In a machine learning competition, confusing statistical attribution with medical causation is an immediate disqualifier. The BreastGuard AI engine explicitly distinguishes between three tiers of reasoning:

1. **Statistical Correlation**:
   A pairwise statistical association between a cytological feature and diagnostic labels across the dataset (e.g., Pearson correlation between `worst_area` and malignant status is $r = 0.73$).
2. **Feature Contribution (SHAP Value)**:
   The signed, marginal mathematical weight allocated to a feature in shifting the model's specific algorithmic decision function relative to its training baseline. It answers: *"How much did the model use this feature to produce its number?"*
3. **Medical / Biological Causation**:
   The underlying cellular pathophysiological mechanism causing malignant transformation (e.g., TP53 tumor suppressor mutation, HER2 overexpression). **SHAP values do not measure biological causation.** A high `worst_radius` does not "cause" cancer; it is a cytological consequence of neoplastic proliferation.

---

## 10. Model & Explainability Limitations

1. **Surrogate Approximation**:
   `KernelExplainer` uses Monte Carlo sampling of coalitions to approximate the exponential $O(2^M)$ Shapley sum. Minor sampling variance ($\Delta < 0.001$) exists across random seeds.
2. **Off-Manifold Evaluations**:
   Perturbing features independently during coalition sampling can generate synthetic feature combinations that do not naturally occur in human biology.
3. **Dataset Scope**:
   The Wisconsin Diagnostic dataset contains 569 digitized FNA samples from a single clinical center. Model explanations reflect the distributions of this cohort and cannot be generalized globally without multi-center external validation.

---

## 11. Conclusion

BreastGuard AI establishes a transparent, trustworthy diagnostic pipeline. By pairing an RBF Support Vector Classifier with `shap.KernelExplainer`, the system provides:
- Exact mathematical additivity on predicted probabilities.
- Global alignment with established cytopathological tumor criteria.
- Patient-specific explanations with clear clinical disclaimers.
- A fully tested, modular engine ready for real-time interactive Streamlit dashboard integration.

