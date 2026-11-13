"""
BreastGuard AI — Shared Configuration & Clinical Constants
All project-wide constants, clinical thresholds, color tokens, and model keys.
"""
from __future__ import annotations

import pathlib

# ── Paths ──────────────────────────────────────────────────────────────────
PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
ASSETS_DIR   = PROJECT_ROOT / "assets"
DOCS_DIR     = PROJECT_ROOT / "docs"

# ── Reproducibility ────────────────────────────────────────────────────────
RANDOM_STATE: int = 42
TEST_SIZE:    float = 0.20   # 80 / 20 stratified train/test split
CV_FOLDS:     int = 5        # 5-fold Stratified K-Fold

# ── Clinical Target Mapping ────────────────────────────────────────────────
# In sklearn's WDBC dataset:
# 0 = Malignant (Clinical Positive Condition: pathology identified cancer)
# 1 = Benign    (Clinical Negative Condition: normal / non-cancerous lesion)
LABEL_MALIGNANT: int = 0
LABEL_BENIGN:    int = 1
CLASS_NAMES: dict[int, str] = {0: "Malignant", 1: "Benign"}
TARGET_NAMES: list[str] = ["Malignant", "Benign"]
POSITIVE_LABEL: int = LABEL_MALIGNANT   # In oncology, test positive = malignant

# ── Risk Assessment Thresholds (P(Malignancy)) ─────────────────────────────
RISK_THRESHOLD_LOW:      float = 0.20
RISK_THRESHOLD_MODERATE: float = 0.50
RISK_THRESHOLD_HIGH:     float = 0.80

# ── Model Registry Keys ────────────────────────────────────────────────────
MODEL_LR:  str = "Logistic Regression"
MODEL_RF:  str = "Random Forest"
MODEL_XGB: str = "XGBoost"
MODEL_SVM: str = "SVM (RBF)"

ALL_MODELS: list[str] = [MODEL_LR, MODEL_RF, MODEL_XGB, MODEL_SVM]

# ── Explainability Settings ────────────────────────────────────────────────
SHAP_BACKGROUND_SAMPLES: int = 40   # Background sample size for KernelExplainer / background
SHAP_KERNEL_NSAMPLES:    int = 80   # Iterations for KernelExplainer to balance speed & precision

# ── UI Design Tokens & Clinical Palette ────────────────────────────────────
COLOR_BG_DARK:   str = "#0E1117"   # Near-black canvas
COLOR_SURFACE:   str = "#161B26"   # Surface/card background
COLOR_BORDER:    str = "#2D3748"   # Border stroke
COLOR_PRIMARY:   str = "#6C63FF"   # Indigo violet primary accent
COLOR_SECONDARY: str = "#8B5CF6"   # Soft purple
COLOR_BENIGN:    str = "#00C9A7"   # Vibrant medical teal (Benign)
COLOR_MALIGNANT: str = "#FF6584"   # High-visibility coral rose (Malignant)
COLOR_WARNING:   str = "#FFB347"   # Amber warning
COLOR_TEXT_MAIN: str = "#E2E8F0"   # Light gray text
COLOR_TEXT_MUTED:str = "#718096"   # Slate muted text
