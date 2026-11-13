# 🩺 BreastGuard AI

**Explainable Breast Cancer Classification & Trustworthy AI Dashboard**

> ⚕️ **Educational Research Prototype — NOT a Medical Device.**  
> Predictions are not confirmed diagnoses. Always consult a qualified medical professional.

---

## Overview

BreastGuard AI is a competition-level machine learning project that:

- Classifies breast cell samples as **benign** or **malignant** using the Wisconsin Diagnostic Breast Cancer (WDBC) dataset
- Explains individual predictions using **SHAP** (SHapley Additive exPlanations)
- Presents results in a polished **Streamlit** interactive dashboard with 6 pages

**Dataset**: 569 samples · 30 real-valued features · 2 classes (212 malignant, 357 benign)  
**Source**: UCI ML Repository via `sklearn.datasets.load_breast_cancer()`

---

## Quick Start

### Prerequisites
- Python 3.13 (Anaconda recommended — available as `py -3.13`)
- Windows 10/11 (tested); no Microsoft C++ Build Tools required

### 1. Create Virtual Environment

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 2. Install Dependencies

```powershell
pip install -r requirements.txt
```

### 3. Run the Application

```powershell
streamlit run app/main.py
```

The app opens at `http://localhost:8501` in your browser.

### 4. Run Tests

```powershell
pip install pytest pytest-cov
pytest tests/ -v --tb=short
```

---

## Project Structure

```
Breast Cancer Classification/
├── src/
│   ├── data/              # Dataset loading & validation
│   ├── eda/               # Exploratory data analysis
│   ├── preprocessing/     # Leakage-free sklearn pipelines
│   ├── models/            # Registry, trainer, evaluator
│   ├── explainability/    # SHAP explainers (Tree/Linear/Kernel)
│   ├── prediction/        # High-level prediction service
│   └── utils/             # Config, logger
├── app/
│   ├── main.py            # Streamlit entry point
│   └── pages/
│       ├── 00_overview.py          # Hero dashboard
│       ├── 01_dataset.py           # EDA explorer
│       ├── 02_model_comparison.py  # Performance comparison
│       ├── 03_prediction.py        # Interactive prediction + SHAP
│       ├── 04_global_xai.py        # Global SHAP explanations
│       └── 05_responsible_ai.py    # Ethics & limitations
├── tests/                 # Pytest suite
├── docs/                  # Documentation
├── requirements.txt
└── README.md
```

---

## Models Compared

| Model | Type | Explainer |
|---|---|---|
| Logistic Regression | Linear | SHAP LinearExplainer |
| Random Forest | Ensemble (trees) | SHAP TreeExplainer |
| XGBoost | Gradient boosting | SHAP TreeExplainer |
| SVM (RBF kernel) | Kernel | SHAP KernelExplainer |

---

## Evaluation Metrics

All models are evaluated on a **held-out 20% test set** (never seen during training or CV):

- Accuracy, Precision, Recall (Sensitivity), Specificity
- F1-Score, ROC-AUC
- Confusion Matrix, Full ROC Curve

Model selection is based on **ROC-AUC**, not accuracy alone.  
**Stratified 5-fold cross-validation** is used during training.

---

## Technology Stack

| Library | Version | Purpose |
|---|---|---|
| Python | 3.13.x | Runtime |
| scikit-learn | ≥1.6 | ML pipelines, models, metrics |
| SHAP | 0.54.x | Explainability (prebuilt wheel, no MSVC) |
| XGBoost | ≥2.1 | Gradient boosting model |
| Streamlit | ≥1.51 | Interactive web dashboard |
| Plotly | ≥5.20 | Interactive visualisations |
| pandas / numpy | 2.x / 2.x | Data manipulation |
| pytest | ≥8.0 | Testing |

---

## Reproducibility

- **Random state**: `42` (set globally in `src/utils/config.py`)
- **Split**: Stratified 80/20 with `random_state=42`
- **CV**: `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`
- **No data leakage**: `StandardScaler` is placed inside the sklearn `Pipeline`

---

## Medical Disclaimer

BreastGuard AI is an **educational research prototype** built for academic demonstration and AI/ML competition purposes.

- ❌ Not a medical diagnostic device
- ❌ Predictions are not confirmed diagnoses
- ❌ Probabilities are not clinically validated
- ✅ Always consult a qualified medical professional for health decisions

---

## Dataset Citation

> W.N. Street, W.H. Wolberg, O.L. Mangasarian.  
> *Nuclear feature extraction for breast tumor diagnosis.*  
> IS&T/SPIE 1993 International Symposium on Electronic Imaging, vol. 1905, pp. 861–870.

---

## License

MIT License — Educational use only.
