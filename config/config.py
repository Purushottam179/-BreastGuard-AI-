"""
BreastGuard AI — Centralized Project Configuration
All project-wide settings, paths, random seeds, and hyperparameters live here.
Paths are dynamically resolved using pathlib to avoid hardcoded absolute paths.
"""
from __future__ import annotations

import os
from pathlib import Path

# ==============================================================================
# PROJECT METADATA
# ==============================================================================
PROJECT_NAME: str = "BreastGuard AI"
PROJECT_TAGLINE: str = (
    "Explainable Breast Cancer Classification & Trustworthy AI Dashboard"
)
VERSION: str = "1.0.0"
DATASET_NAME: str = "Wisconsin Diagnostic Breast Cancer (WDBC)"
DATASET_SOURCE: str = "UCI Machine Learning Repository / scikit-learn"

# ==============================================================================
# DYNAMIC DIRECTORY PATHS
# ==============================================================================
CONFIG_DIR: Path = Path(__file__).resolve().parent
PROJECT_ROOT: Path = CONFIG_DIR.parent

APP_DIR: Path = PROJECT_ROOT / "app"
COMPONENTS_DIR: Path = APP_DIR / "components"
PAGES_DIR: Path = APP_DIR / "pages"

DATA_DIR: Path = PROJECT_ROOT / "data"
RAW_DATA_DIR: Path = DATA_DIR / "raw"
PROCESSED_DATA_DIR: Path = DATA_DIR / "processed"

MODELS_DIR: Path = PROJECT_ROOT / "models"
NOTEBOOKS_DIR: Path = PROJECT_ROOT / "notebooks"

REPORTS_DIR: Path = PROJECT_ROOT / "reports"
FIGURES_DIR: Path = REPORTS_DIR / "figures"
EVALUATION_DIR: Path = REPORTS_DIR / "evaluation"

ASSETS_DIR: Path = PROJECT_ROOT / "assets"
IMAGES_DIR: Path = ASSETS_DIR / "images"
DOCS_DIR: Path = PROJECT_ROOT / "docs"

# Create core runtime directories automatically if missing
for _dir in [
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    MODELS_DIR,
    FIGURES_DIR,
    EVALUATION_DIR,
    IMAGES_DIR,
]:
    _dir.mkdir(parents=True, exist_ok=True)

# Set writable Matplotlib config dir to prevent Windows temp path warnings
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".matplotlib_cache"))
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

# ==============================================================================
# DATASET & CLINICAL TARGET SPECIFICATIONS
# ==============================================================================
# PROJECT-WIDE TARGET ENCODING CONVENTION (consistent with UCI wdbc.data source):
#   M (Malignant) → 1   (Clinical Positive — cancer present)
#   B (Benign)    → 0   (Clinical Negative — non-malignant lesion)
#
# NOTE: sklearn's load_breast_cancer() internally encodes 0=Malignant, 1=Benign.
# All BreastGuard AI modules use M=1, B=0 as defined here (standard convention).
TARGET_COLUMN: str = "target"
DIAGNOSIS_COLUMN: str = "diagnosis"

LABEL_MALIGNANT: int = 1    # M = 1 (Malignant, clinical positive)
LABEL_BENIGN: int = 0       # B = 0 (Benign, clinical negative)

CLASS_LABELS: dict[int, str] = {
    LABEL_MALIGNANT: "Malignant",
    LABEL_BENIGN: "Benign",
}

CLASS_NAMES: list[str] = ["Benign", "Malignant"]  # index position: 0=Benign, 1=Malignant
POSITIVE_LABEL: int = LABEL_MALIGNANT   # Clinical positive = Malignant = 1

# Feature count: 30 numerical features (10 mean, 10 standard error, 10 worst)
N_FEATURES: int = 30
EXPECTED_SAMPLE_COUNT: int = 569

# ==============================================================================
# MACHINE LEARNING & REPRODUCIBILITY HYPERPARAMETERS
# ==============================================================================
RANDOM_SEED: int = 42
TEST_SIZE: float = 0.20  # 80% train, 20% held-out test split
CV_FOLDS: int = 5        # 5-fold Stratified Cross-Validation
SCORING_METRIC: str = "roc_auc"

# Model Family Identifiers
MODEL_LOGISTIC_REGRESSION: str = "Logistic Regression"
MODEL_RANDOM_FOREST: str = "Random Forest"
MODEL_SVM: str = "Support Vector Machine"
MODEL_GRADIENT_BOOSTING: str = "Gradient Boosting"
MODEL_KNN: str = "K-Nearest Neighbors"

SUPPORTED_MODELS: list[str] = [
    MODEL_LOGISTIC_REGRESSION,
    MODEL_RANDOM_FOREST,
    MODEL_SVM,
    MODEL_GRADIENT_BOOSTING,
    MODEL_KNN,
]

MODEL_FILENAMES: dict[str, str] = {
    MODEL_LOGISTIC_REGRESSION: "logistic_regression.pkl",
    MODEL_RANDOM_FOREST: "random_forest.pkl",
    MODEL_SVM: "svm.pkl",
    MODEL_GRADIENT_BOOSTING: "gradient_boosting.pkl",
    MODEL_KNN: "knn.pkl",
}

# ==============================================================================
# EXPLAINABLE AI (SHAP) SETTINGS
# ==============================================================================
SHAP_BACKGROUND_SAMPLES: int = 40  # Samples used for KernelExplainer reference
SHAP_KERNEL_NSAMPLES: int = 80     # Evaluations for KernelExplainer
TOP_K_EXPLAIN_FEATURES: int = 10   # Default top features displayed in waterfall charts

# ==============================================================================
# CLINICAL RISK STRATIFICATION THRESHOLDS
# ==============================================================================
RISK_THRESHOLD_LOW: float = 0.20       # < 20%: Low probability of malignancy
RISK_THRESHOLD_MODERATE: float = 0.50  # 20% - 50%: Intermediate / borderline
RISK_THRESHOLD_HIGH: float = 0.80      # > 50%: High probability of malignancy

# ==============================================================================
# UI DESIGN SYSTEM & COLOR TOKENS
# ==============================================================================
THEME_DARK = {
    "bg": "#0E1117",
    "surface": "#161B26",
    "surface_card": "#1E2433",
    "border": "#2D3748",
    "primary": "#6C63FF",
    "primary_hover": "#5B52E0",
    "malignant": "#FF6584",  # Coral rose for Malignant
    "benign": "#00C9A7",     # Clinical teal for Benign
    "warning": "#FFB347",    # Amber warning
    "text_main": "#E2E8F0",
    "text_muted": "#718096",
}

