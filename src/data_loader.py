"""
BreastGuard AI — Dataset Acquisition, Cleaning & Validation Module
================================================================================
PRIMARY SOURCE:
  UCI Machine Learning Repository — Breast Cancer Wisconsin (Diagnostic) Dataset
  URL: https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic
  File: wdbc.data  |  569 instances, 32 columns (ID + diagnosis + 30 features)

DOCUMENTED FALLBACK:
  sklearn.datasets.load_breast_cancer() — Identical dataset, distributed with
  scikit-learn. Used when UCI is not reachable (offline / sandboxed environment).
  The two sources contain the same 569 samples with the same values.

IMPORTANT TARGET ENCODING (used consistently project-wide):
  Per UCI original encoding:
    M = Malignant  →  label 1  (clinical positive)
    B = Benign     →  label 0  (clinical negative)

  NOTE: This is REVERSED from sklearn's internal encoding
  (sklearn uses 0=Malignant, 1=Benign internally). This module applies
  the clinically standard M=1, B=0 convention throughout the saved files
  and returns it for all downstream consumers.

FEATURE GROUPS (10 measurements × 3 statistics = 30 features):
  Group 1 — MEAN:          mean_radius to mean_fractal_dimension
  Group 2 — STANDARD ERROR (SE): radius_se to fractal_dimension_se
  Group 3 — WORST:         worst_radius to worst_fractal_dimension

TEN BASE MEASUREMENTS:
  1.  radius           — Mean distances from center to perimeter points
  2.  texture          — Standard deviation of gray-scale values
  3.  perimeter        — Nucleus perimeter length
  4.  area             — Nucleus cross-sectional area
  5.  smoothness       — Local variation in radius lengths
  6.  compactness      — perimeter² / area − 1.0
  7.  concavity        — Severity of concave portions of contour
  8.  concave_points   — Number of concave portions of contour
  9.  symmetry         — Nucleus symmetry
  10. fractal_dimension — Coastline approximation (higher = more irregular)
================================================================================
"""
from __future__ import annotations

import json
import logging
import urllib.error
import urllib.request
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import pandas as pd

from config.config import (
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
)

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Official UCI column layout (32 columns in wdbc.data)
# ─────────────────────────────────────────────────────────────────────────────
_UCI_COLUMNS = [
    "id",
    "diagnosis",
    "mean_radius",
    "mean_texture",
    "mean_perimeter",
    "mean_area",
    "mean_smoothness",
    "mean_compactness",
    "mean_concavity",
    "mean_concave_points",
    "mean_symmetry",
    "mean_fractal_dimension",
    "radius_se",
    "texture_se",
    "perimeter_se",
    "area_se",
    "smoothness_se",
    "compactness_se",
    "concavity_se",
    "concave_points_se",
    "symmetry_se",
    "fractal_dimension_se",
    "worst_radius",
    "worst_texture",
    "worst_perimeter",
    "worst_area",
    "worst_smoothness",
    "worst_compactness",
    "worst_concavity",
    "worst_concave_points",
    "worst_symmetry",
    "worst_fractal_dimension",
]

FEATURE_COLUMNS: list[str] = _UCI_COLUMNS[2:]  # 30 feature names (no id, no diagnosis)
FEATURE_GROUPS = {
    "mean":  [c for c in FEATURE_COLUMNS if c.startswith("mean_")],
    "se":    [c for c in FEATURE_COLUMNS if c.endswith("_se")],
    "worst": [c for c in FEATURE_COLUMNS if c.startswith("worst_")],
}
BASE_MEASUREMENTS = [
    "radius", "texture", "perimeter", "area", "smoothness",
    "compactness", "concavity", "concave_points", "symmetry", "fractal_dimension",
]

# UCI URLs
_UCI_DATA_URL  = "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data"
_UCI_NAMES_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.names"


# ─────────────────────────────────────────────────────────────────────────────
# Dataclasses
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class DatasetInfo:
    """Metadata describing the acquired dataset."""
    source: str               # "UCI" or "sklearn_fallback"
    n_samples: int
    n_features: int
    n_malignant: int          # M = 1
    n_benign: int             # B = 0
    n_missing: int
    n_duplicates: int
    class_balance_ratio: float  # malignant / total
    raw_csv_path: str
    cleaned_csv_path: str


@dataclass
class AcquisitionBundle:
    """Full output of dataset acquisition."""
    df_raw: pd.DataFrame       # Original, unmodified data (with id)
    df_cleaned: pd.DataFrame   # Cleaned features + diagnosis (M=1, B=0, no id)
    feature_names: list[str]   # 30 feature column names
    feature_groups: dict[str, list[str]]
    info: DatasetInfo

    @property
    def X_df(self) -> pd.DataFrame:
        cols = [c for c in self.feature_names if c in self.df_cleaned.columns]
        return self.df_cleaned[cols]

    @property
    def y_series(self) -> pd.Series:
        col = "diagnosis" if "diagnosis" in self.df_cleaned.columns else "target"
        return self.df_cleaned[col]

    @property
    def target_names(self) -> list[str]:
        return ["Benign", "Malignant"]


class DataValidationError(ValueError):
    """Raised when dataset fails schema or integrity validation."""


# ─────────────────────────────────────────────────────────────────────────────
# PART 1 — Acquisition
# ─────────────────────────────────────────────────────────────────────────────
def _try_uci_download() -> pd.DataFrame | None:
    """
    Attempt to download wdbc.data from the official UCI repository.
    Returns raw DataFrame with correct column names if successful, else None.
    """
    logger.info("Attempting UCI download from: %s", _UCI_DATA_URL)
    try:
        with urllib.request.urlopen(_UCI_DATA_URL, timeout=8) as resp:
            content = resp.read().decode("utf-8")
        lines = [ln.strip() for ln in content.splitlines() if ln.strip()]
        rows = [ln.split(",") for ln in lines]
        df = pd.DataFrame(rows, columns=_UCI_COLUMNS)
        for col in FEATURE_COLUMNS:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        logger.info("UCI download successful. %d rows received.", len(df))
        return df
    except (urllib.error.URLError, urllib.error.HTTPError, Exception) as exc:
        logger.warning("UCI download failed (%s: %s). Switching to documented fallback.", type(exc).__name__, exc)
        return None


def _try_uci_names_download() -> str:
    """Download the wdbc.names file (dataset documentation). Returns text or empty string."""
    try:
        with urllib.request.urlopen(_UCI_NAMES_URL, timeout=8) as resp:
            return resp.read().decode("utf-8")
    except Exception as exc:
        logger.warning("wdbc.names download failed: %s", exc)
        return ""


def _load_from_sklearn() -> pd.DataFrame:
    """
    Fallback: load from sklearn.datasets.load_breast_cancer().
    Reconstructs the full UCI-format DataFrame including synthetic ID column
    and alphabetic 'M'/'B' diagnosis labels for authentic format consistency.

    sklearn encoding: 0=malignant, 1=benign (reversed from UCI M/B convention)
    This function converts back to UCI format: M / B labels.
    """
    logger.info("Loading dataset using sklearn.datasets.load_breast_cancer() (documented fallback)")
    from sklearn.datasets import load_breast_cancer

    raw = load_breast_cancer(as_frame=True)

    # Reconstruct the UCI-format DataFrame
    df = pd.DataFrame(index=range(len(raw.data)))
    # Synthetic sequential IDs matching original UCI id format style
    df["id"] = range(842302, 842302 + len(raw.data))
    # sklearn: 0=malignant → UCI 'M', sklearn: 1=benign → UCI 'B'
    df["diagnosis"] = raw.target.map({0: "M", 1: "B"})

    # sklearn feature names use spaces; remap to underscore UCI names
    sklearn_to_uci = {
        sk: uci for sk, uci in zip(
            list(raw.feature_names),
            FEATURE_COLUMNS,
        )
    }
    for sk_name, uci_name in sklearn_to_uci.items():
        df[uci_name] = raw.data[sk_name].values

    logger.info(
        "sklearn fallback loaded: %d samples | M=%d | B=%d",
        len(df),
        (df["diagnosis"] == "M").sum(),
        (df["diagnosis"] == "B").sum(),
    )
    return df


def acquire_dataset() -> tuple[pd.DataFrame, str]:
    """
    Acquire the WDBC dataset from the best available source.

    Returns
    -------
    (df_raw, source_name)
        df_raw   — Raw DataFrame with all UCI columns including 'id' and 'diagnosis' as M/B.
        source   — 'UCI' or 'sklearn_fallback'
    """
    df = _try_uci_download()
    source = "UCI"

    if df is None:
        df = _load_from_sklearn()
        source = "sklearn_fallback"

    return df, source


# ─────────────────────────────────────────────────────────────────────────────
# PART 2 — Understand the Dataset
# ─────────────────────────────────────────────────────────────────────────────
def understand_dataset(df_raw: pd.DataFrame, source: str) -> dict:
    """
    Produce a comprehensive dataset understanding report.

    Returns
    -------
    dict with keys: dimensions, feature_names, data_types, target_info,
                    missing_values, duplicates, feature_groups, ...
    """
    n_rows, n_cols = df_raw.shape
    n_features = len(FEATURE_COLUMNS)
    n_malignant = int((df_raw["diagnosis"] == "M").sum())
    n_benign = int((df_raw["diagnosis"] == "B").sum())
    n_missing = int(df_raw.isnull().sum().sum())
    n_dups = int(df_raw.duplicated().sum())

    understanding = {
        "source": source,
        "dimensions": {
            "rows": n_rows,
            "total_columns": n_cols,
            "feature_columns": n_features,
        },
        "feature_names": FEATURE_COLUMNS,
        "feature_groups": FEATURE_GROUPS,
        "base_measurements": BASE_MEASUREMENTS,
        "data_types": df_raw.dtypes.astype(str).to_dict(),
        "target_info": {
            "column": "diagnosis",
            "unique_values": sorted(df_raw["diagnosis"].unique().tolist()),
            "encoding": {"M": 1, "B": 0},
            "n_malignant": n_malignant,
            "n_benign": n_benign,
            "class_imbalance_ratio": round(n_malignant / n_rows, 4),
        },
        "missing_values": {
            "total": n_missing,
            "per_column": df_raw.isnull().sum().to_dict(),
            "any_missing": n_missing > 0,
        },
        "duplicates": {
            "count": n_dups,
            "any_duplicates": n_dups > 0,
        },
        "numeric_summary": df_raw[FEATURE_COLUMNS].describe().round(6).to_dict(),
    }

    return understanding


# ─────────────────────────────────────────────────────────────────────────────
# PART 3 — Data Cleaning
# ─────────────────────────────────────────────────────────────────────────────
def clean_dataset(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Perform documented data cleaning steps on the raw WDBC DataFrame:

    1. Drop 'id' column — not a predictive feature (UCI documentation states this).
    2. Drop any unnamed/extra columns.
    3. Validate all 30 feature columns are present.
    4. Coerce all feature columns to float64.
    5. Check for and report missing values.
    6. Check for and drop duplicate records (full row deduplication).
    7. Encode diagnosis: M → 1 (Malignant), B → 0 (Benign).

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame with 30 feature columns + 'diagnosis' (int 0/1).
    """
    logger.info("Starting data cleaning pipeline ...")
    df = df_raw.copy()

    # Step 1: Drop 'id' column
    if "id" in df.columns:
        df = df.drop(columns=["id"])
        logger.info("Step 1: Dropped 'id' column (not a predictive feature).")

    # Step 2: Drop any unnamed/extra columns (artifact columns from CSV readers)
    extra_cols = [c for c in df.columns if c.startswith("Unnamed") or c.strip() == ""]
    if extra_cols:
        df = df.drop(columns=extra_cols)
        logger.info("Step 2: Dropped %d unnamed/extra columns: %s", len(extra_cols), extra_cols)
    else:
        logger.info("Step 2: No unnamed columns found.")

    # Step 3: Validate 30 feature columns are present
    missing_feats = [f for f in FEATURE_COLUMNS if f not in df.columns]
    if missing_feats:
        raise DataValidationError(
            f"Missing expected feature columns after cleaning: {missing_feats[:5]}"
        )
    logger.info("Step 3: All 30 feature columns confirmed present.")

    # Step 4: Coerce feature columns to float64
    for col in FEATURE_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    logger.info("Step 4: All 30 feature columns coerced to float64.")

    # Step 5: Check and handle missing values
    missing_counts = df[FEATURE_COLUMNS].isnull().sum()
    total_missing = int(missing_counts.sum())
    if total_missing > 0:
        cols_with_missing = missing_counts[missing_counts > 0].to_dict()
        logger.warning(
            "Step 5: %d missing values detected: %s — Imputing with column median.",
            total_missing, cols_with_missing,
        )
        for col, n_miss in cols_with_missing.items():
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
            logger.info("  Imputed %d NaN in '%s' with median %.6f", n_miss, col, median_val)
    else:
        logger.info("Step 5: No missing values detected — dataset is complete.")

    # Step 6: Check and remove duplicate rows
    n_dups = int(df.duplicated().sum())
    if n_dups > 0:
        df = df.drop_duplicates()
        logger.warning("Step 6: Removed %d exact duplicate rows.", n_dups)
    else:
        logger.info("Step 6: No duplicate rows found.")

    # Step 7: Encode diagnosis labels M → 1, B → 0
    original_vals = set(df["diagnosis"].unique())
    df["diagnosis"] = df["diagnosis"].map({"M": 1, "B": 0})
    if df["diagnosis"].isnull().any():
        raise DataValidationError(
            f"Unexpected diagnosis values found. Original values: {original_vals}"
        )
    df["diagnosis"] = df["diagnosis"].astype(int)
    logger.info(
        "Step 7: Encoded diagnosis — M→1 (Malignant), B→0 (Benign). "
        "Distribution: M=%d, B=%d",
        int((df["diagnosis"] == 1).sum()),
        int((df["diagnosis"] == 0).sum()),
    )

    # Reorder columns: features first, then target
    df = df[FEATURE_COLUMNS + ["diagnosis"]]

    logger.info(
        "Cleaning complete. Final shape: %s | dtype check OK: %s",
        df.shape,
        df[FEATURE_COLUMNS].dtypes.eq("float64").all(),
    )
    return df


# ─────────────────────────────────────────────────────────────────────────────
# PART 4 — Validation Against Known Schema
# ─────────────────────────────────────────────────────────────────────────────
def validate_cleaned_dataset(df_cleaned: pd.DataFrame) -> None:
    """
    Final schema and integrity validation on the cleaned dataset.

    Raises DataValidationError on any failure.
    """
    logger.info("Running final validation on cleaned dataset ...")

    # Shape
    if df_cleaned.shape[1] != 31:  # 30 features + 1 target
        raise DataValidationError(
            f"Expected 31 columns (30 features + diagnosis), got {df_cleaned.shape[1]}"
        )
    if df_cleaned.shape[0] < 560:  # Allow for duplicate removal
        raise DataValidationError(
            f"Too few rows after cleaning: {df_cleaned.shape[0]}"
        )

    # Missing values
    if df_cleaned.isnull().any().any():
        raise DataValidationError("Cleaned dataset still contains missing values.")

    # Target encoding
    unique_targets = set(df_cleaned["diagnosis"].unique())
    if unique_targets != {0, 1}:
        raise DataValidationError(
            f"Diagnosis column must contain only 0 and 1, found: {unique_targets}"
        )

    # All feature dtypes
    bad_dtypes = [
        col for col in FEATURE_COLUMNS
        if not pd.api.types.is_float_dtype(df_cleaned[col])
    ]
    if bad_dtypes:
        raise DataValidationError(
            f"Non-float feature columns detected: {bad_dtypes[:5]}"
        )

    # Non-negative features (all WDBC measurements are physically non-negative)
    for col in FEATURE_COLUMNS:
        if (df_cleaned[col] < 0).any():
            raise DataValidationError(
                f"Feature '{col}' contains negative values, which is not expected for WDBC."
            )

    n_mal = int((df_cleaned["diagnosis"] == 1).sum())
    n_ben = int((df_cleaned["diagnosis"] == 0).sum())
    logger.info(
        "Validation PASSED — Shape: %s | Malignant (M=1): %d | Benign (B=0): %d | Missing: 0",
        df_cleaned.shape, n_mal, n_ben,
    )


# ─────────────────────────────────────────────────────────────────────────────
# ORCHESTRATOR — Run the full acquisition pipeline
# ─────────────────────────────────────────────────────────────────────────────
def run_acquisition() -> AcquisitionBundle:
    """
    Execute the complete acquisition pipeline:
    1. Acquire dataset (UCI or sklearn fallback).
    2. Save raw data to data/raw/.
    3. Clean the dataset.
    4. Validate cleaned dataset.
    5. Save cleaned dataset to data/processed/.
    6. Save dataset documentation & understanding report.

    Returns
    -------
    AcquisitionBundle
    """
    # ── Step 1: Acquire ──────────────────────────────────────────────────────
    df_raw, source = acquire_dataset()

    # ── Step 2: Save raw data ─────────────────────────────────────────────────
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    raw_csv_path = RAW_DATA_DIR / "wdbc_original.csv"
    df_raw.to_csv(raw_csv_path, index=False)
    logger.info("Raw dataset saved to: %s (%d rows)", raw_csv_path, len(df_raw))

    # Save acquisition source metadata
    source_meta = {
        "source": source,
        "uci_url": _UCI_DATA_URL if source == "UCI" else "N/A (offline)",
        "sklearn_fallback_used": source == "sklearn_fallback",
        "note": (
            "sklearn.datasets.load_breast_cancer() is a documented reproduction "
            "of the UCI WDBC dataset. The two sources contain identical observations. "
            "UCI was unreachable (network unavailable in current environment)."
        ) if source == "sklearn_fallback" else "UCI download successful.",
    }
    with open(RAW_DATA_DIR / "acquisition_source.json", "w") as f:
        json.dump(source_meta, f, indent=2)
    logger.info("Acquisition metadata saved.")

    # ── Step 3: Understand ───────────────────────────────────────────────────
    understanding = understand_dataset(df_raw, source)
    with open(RAW_DATA_DIR / "dataset_understanding.json", "w") as f:
        json.dump(understanding, f, indent=2, default=str)
    logger.info("Dataset understanding report saved.")

    # ── Step 4: Clean ────────────────────────────────────────────────────────
    df_cleaned = clean_dataset(df_raw)

    # ── Step 5: Validate ─────────────────────────────────────────────────────
    validate_cleaned_dataset(df_cleaned)

    # ── Step 6: Save cleaned data ─────────────────────────────────────────────
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    cleaned_csv_path = PROCESSED_DATA_DIR / "wdbc_cleaned.csv"
    df_cleaned.to_csv(cleaned_csv_path, index=False)
    # Also save as breast_cancer_cleaned.csv for standardized accessibility
    alt_cleaned_path = PROCESSED_DATA_DIR / "breast_cancer_cleaned.csv"
    df_cleaned.to_csv(alt_cleaned_path, index=False)
    logger.info("Cleaned dataset saved to: %s and %s", cleaned_csv_path, alt_cleaned_path)

    # ── Step 7: Build DatasetInfo ─────────────────────────────────────────────
    n_mal = int((df_cleaned["diagnosis"] == 1).sum())
    n_ben = int((df_cleaned["diagnosis"] == 0).sum())
    info = DatasetInfo(
        source=source,
        n_samples=len(df_cleaned),
        n_features=len(FEATURE_COLUMNS),
        n_malignant=n_mal,
        n_benign=n_ben,
        n_missing=understanding["missing_values"]["total"],
        n_duplicates=understanding["duplicates"]["count"],
        class_balance_ratio=round(n_mal / len(df_cleaned), 4),
        raw_csv_path=str(raw_csv_path),
        cleaned_csv_path=str(cleaned_csv_path),
    )
    with open(PROCESSED_DATA_DIR / "dataset_info.json", "w") as f:
        json.dump(asdict(info), f, indent=2)

    logger.info(
        "Acquisition pipeline complete. Source=%s | Samples=%d | M=%d | B=%d",
        source, info.n_samples, info.n_malignant, info.n_benign,
    )

    return AcquisitionBundle(
        df_raw=df_raw,
        df_cleaned=df_cleaned,
        feature_names=FEATURE_COLUMNS,
        feature_groups=FEATURE_GROUPS,
        info=info,
    )


def load_cleaned_dataset() -> tuple[pd.DataFrame, list[str]]:
    """
    Load the pre-cleaned WDBC dataset from data/processed/wdbc_cleaned.csv.
    If it does not exist, runs the full acquisition pipeline first.

    Returns
    -------
    (df_cleaned, feature_names)
    """
    cleaned_path = PROCESSED_DATA_DIR / "wdbc_cleaned.csv"
    if not cleaned_path.exists():
        logger.info("Cleaned dataset not found. Running acquisition pipeline ...")
        bundle = run_acquisition()
        return bundle.df_cleaned, bundle.feature_names

    df = pd.read_csv(cleaned_path)
    logger.info("Loaded cleaned dataset from disk: %s (%d rows)", cleaned_path, len(df))
    return df, FEATURE_COLUMNS


def load_and_validate_processed_data(
    filepath: Path | str | None = None,
) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """
    Load and rigorously validate the processed breast cancer dataset.
    Fulfills all validation constraints:
      1. Target column exists.
      2. Target mapping: 0 = Benign (357), 1 = Malignant (212).
      3. Exactly 30 expected numerical features exist.
      4. No missing (NaN) or infinite (Inf) values.
      5. No ID column present.
      6. No target information included in feature set.

    Parameters
    ----------
    filepath : Path or str, optional
        Path to cleaned CSV. Defaults to data/processed/breast_cancer_cleaned.csv,
        falling back to data/processed/wdbc_cleaned.csv.

    Returns
    -------
    X : pd.DataFrame
        Input features (shape: 569, 30).
    y : pd.Series
        Target labels (shape: 569,), where 0=Benign, 1=Malignant.
    feature_names : list[str]
        List of 30 feature names.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    DataValidationError
        If any validation requirement is violated.
    """
    if filepath is None:
        p1 = PROCESSED_DATA_DIR / "breast_cancer_cleaned.csv"
        p2 = PROCESSED_DATA_DIR / "wdbc_cleaned.csv"
        target_path = p1 if p1.exists() else p2
    else:
        target_path = Path(filepath)

    if not target_path.exists():
        raise FileNotFoundError(
            f"Processed dataset not found at '{target_path}'. "
            "Please run the data acquisition pipeline first."
        )

    logger.info("Loading processed data from: %s", target_path)
    df = pd.read_csv(target_path)

    # 1. Verify target column exists
    target_col = None
    for cand in ["diagnosis", "target"]:
        if cand in df.columns:
            target_col = cand
            break

    if target_col is None:
        raise DataValidationError(
            f"Target column ('diagnosis' or 'target') not found in {list(df.columns)}"
        )

    # 2. Confirm target mapping: 0 = Benign, 1 = Malignant
    unique_targets = set(df[target_col].unique())
    if unique_targets != {0, 1}:
        raise DataValidationError(
            f"Target column must contain exactly binary values {{0, 1}}, found: {unique_targets}"
        )

    n_benign = int((df[target_col] == 0).sum())
    n_malignant = int((df[target_col] == 1).sum())
    if n_benign != 357 or n_malignant != 212:
        raise DataValidationError(
            f"Target class distribution mismatch: expected Benign(0)=357, Malignant(1)=212. "
            f"Found Benign(0)={n_benign}, Malignant(1)={n_malignant}"
        )

    # 3. Check for ID column presence
    id_cols = [c for c in df.columns if c.lower() in ("id", "patient_id", "sample_id", "unnamed: 0")]
    if id_cols:
        raise DataValidationError(
            f"Prohibited ID or index column(s) detected in dataset: {id_cols}"
        )

    # 4. Separate X and y
    y = df[target_col].copy()
    X = df.drop(columns=[target_col]).copy()

    # 5. Confirm exactly 30 expected numerical features
    if X.shape[1] != 30:
        raise DataValidationError(
            f"Expected exactly 30 numerical input features, found {X.shape[1]}: {list(X.columns)}"
        )

    for expected_feat in FEATURE_COLUMNS:
        if expected_feat not in X.columns:
            raise DataValidationError(
                f"Expected feature '{expected_feat}' is missing from feature matrix."
            )

    # 6. Check for missing (NaN) and infinite (Inf) values
    if X.isnull().any().any():
        null_counts = X.isnull().sum()[lambda x: x > 0].to_dict()
        raise DataValidationError(
            f"Missing values (NaN) detected in feature matrix: {null_counts}"
        )

    if np.isinf(X.to_numpy()).any():
        raise DataValidationError(
            "Infinite values (Inf or -Inf) detected in feature matrix."
        )

    if y.isnull().any():
        raise DataValidationError("Target vector contains missing values.")

    # 7. Verify no target information in input features (leakage check)
    for col in X.columns:
        if col.lower() in ("diagnosis", "target", "label", "class", "malignant", "benign"):
            raise DataValidationError(
                f"Potential target leakage detected: feature '{col}' matches target keywords."
            )

    logger.info(
        "Data validation successful: 569 samples, 30 features, Benign(0)=%d, Malignant(1)=%d. "
        "No NaNs, no Infs, no ID columns.",
        n_benign, n_malignant,
    )
    return X, y, list(X.columns)


def load_dataset(*args: Any, **kwargs: Any) -> AcquisitionBundle:
    """Convenience helper to load the WDBC dataset bundle."""
    return run_acquisition()


