"""
BreastGuard AI — Preprocessing Pipeline
Builds a leakage-free sklearn Pipeline with StandardScaler.
The scaler is fitted ONLY on training data.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.utils.config import RANDOM_STATE, TEST_SIZE

logger = logging.getLogger(__name__)


@dataclass
class SplitData:
    """Train/test split container."""
    X_train: np.ndarray
    X_test:  np.ndarray
    y_train: np.ndarray
    y_test:  np.ndarray


def make_split(X: np.ndarray, y: np.ndarray) -> SplitData:
    """
    Stratified 80/20 train-test split.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
    y : array-like, shape (n_samples,)

    Returns
    -------
    SplitData
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE,
    )
    logger.info(
        "Split → train: %d | test: %d | malignant(train): %d | benign(train): %d",
        len(X_train), len(X_test),
        (y_train == 0).sum(), (y_train == 1).sum(),
    )
    return SplitData(X_train, X_test, y_train, y_test)


def build_pipeline(estimator) -> Pipeline:
    """
    Wrap an estimator in a StandardScaler→Estimator pipeline.
    The scaler learns only from training data (no leakage).

    Parameters
    ----------
    estimator : sklearn-compatible estimator

    Returns
    -------
    sklearn.pipeline.Pipeline
    """
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model",  estimator),
    ])
