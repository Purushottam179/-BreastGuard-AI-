"""
BreastGuard AI — Preprocessing & Pipeline Module
Enforces strict data leakage prevention using Scikit-learn Pipelines.
Models requiring scaling (Logistic Regression, SVM, KNN) are wrapped with StandardScaler.
Models not requiring scaling (Random Forest, Gradient Boosting) use a dedicated pipeline
without unnecessary scaling transformations.
The scaling parameters (mean, std) are strictly learned on the training split/folds only.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from config.config import (
    EVALUATION_DIR,
    PROCESSED_DATA_DIR,
    RANDOM_SEED,
    TEST_SIZE,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SplitData:
    """Immutable container holding train/test partitions."""
    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    feature_names: list[str]

    @property
    def train_size(self) -> int:
        return len(self.X_train)

    @property
    def test_size(self) -> int:
        return len(self.X_test)

    @property
    def n_features(self) -> int:
        return self.X_train.shape[1]


def create_stratified_split(
    X: np.ndarray | pd.DataFrame,
    y: np.ndarray | pd.Series,
    test_size: float = TEST_SIZE,
    random_seed: int = RANDOM_SEED,
    feature_names: list[str] | None = None,
    save_metadata: bool = True,
) -> SplitData:
    """
    Split the dataset into training and test partitions with stratification.
    Saves split metadata for complete clinical reproducibility.

    Parameters
    ----------
    X : array-like or DataFrame, shape (n_samples, n_features)
    y : array-like or Series, shape (n_samples,)
    test_size : float, default=0.20
    random_seed : int, default=42
    feature_names : list[str], optional
    save_metadata : bool, default=True

    Returns
    -------
    SplitData
        Container with X_train, X_test, y_train, y_test, feature_names.
    """
    # Extract feature names if DataFrame
    if isinstance(X, pd.DataFrame):
        if feature_names is None:
            feature_names = list(X.columns)
        X_arr = X.to_numpy(dtype=np.float64)
    else:
        X_arr = np.asarray(X, dtype=np.float64)
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(X_arr.shape[1])]

    if isinstance(y, pd.Series):
        y_arr = y.to_numpy(dtype=np.int64)
    else:
        y_arr = np.asarray(y, dtype=np.int64)

    X_train, X_test, y_train, y_test = train_test_split(
        X_arr,
        y_arr,
        test_size=test_size,
        stratify=y_arr,
        random_state=random_seed,
    )

    n_train_mal = int((y_train == 1).sum())
    n_train_ben = int((y_train == 0).sum())
    n_test_mal = int((y_test == 1).sum())
    n_test_ben = int((y_test == 0).sum())

    logger.info(
        "Stratified split created: Train=%d (Malignant=%d, Benign=%d) | "
        "Test=%d (Malignant=%d, Benign=%d)",
        len(X_train), n_train_mal, n_train_ben,
        len(X_test), n_test_mal, n_test_ben,
    )

    split = SplitData(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        feature_names=feature_names,
    )

    if save_metadata:
        metadata = {
            "random_seed": random_seed,
            "test_size_ratio": test_size,
            "train_size": len(X_train),
            "test_size": len(X_test),
            "train_class_counts": {"Benign_0": n_train_ben, "Malignant_1": n_train_mal},
            "test_class_counts": {"Benign_0": n_test_ben, "Malignant_1": n_test_mal},
            "train_malignant_ratio": round(n_train_mal / len(X_train), 4),
            "test_malignant_ratio": round(n_test_mal / len(X_test), 4),
            "n_features": len(feature_names),
            "feature_names": feature_names,
        }

        EVALUATION_DIR.mkdir(parents=True, exist_ok=True)
        meta_path = EVALUATION_DIR / "split_metadata.json"
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2)
        logger.info("Split metadata saved to: %s", meta_path)

        # Also store in data/processed for convenience
        PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
        with open(PROCESSED_DATA_DIR / "split_metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

    return split


def build_pipeline(
    estimator: Any,
    use_scaler: bool = True,
    feature_names: list[str] | None = None,
) -> Pipeline:
    """
    Build a leak-free Scikit-learn Pipeline for a given estimator.

    Parameters
    ----------
    estimator : BaseEstimator
        The classification model.
    use_scaler : bool, default=True
        Whether to include a StandardScaler. Set to True for scale-sensitive models
        (Logistic Regression, SVM, KNN) and False for tree-based models (Random Forest,
        Gradient Boosting).
    feature_names : list[str], optional
        Optional list of feature names stored for downstream explainability (SHAP).

    Returns
    -------
    Pipeline
        Configured Scikit-learn Pipeline.
    """
    steps = []
    if use_scaler:
        steps.append(("scaler", StandardScaler()))
    else:
        # Separate pipeline without scaling for tree-based algorithms
        steps.append(("passthrough", "passthrough"))

    steps.append(("model", estimator))
    pipe = Pipeline(steps)

    # Attach feature names attribute to pipeline object for transparent SHAP tracking
    if feature_names is not None:
        pipe.feature_names_ = feature_names

    return pipe


def export_processed_splits(
    split: SplitData,
    output_dir: Path = PROCESSED_DATA_DIR,
) -> None:
    """Export train and test splits to disk for full analytical transparency."""
    output_dir.mkdir(parents=True, exist_ok=True)

    train_df = pd.DataFrame(split.X_train, columns=split.feature_names)
    train_df["diagnosis"] = split.y_train
    train_df.to_csv(output_dir / "train_split.csv", index=False)

    test_df = pd.DataFrame(split.X_test, columns=split.feature_names)
    test_df["diagnosis"] = split.y_test
    test_df.to_csv(output_dir / "test_split.csv", index=False)

    logger.info("Exported train_split.csv and test_split.csv to: %s", output_dir)
