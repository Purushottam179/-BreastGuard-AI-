"""
BreastGuard AI — Data Loader
Wraps sklearn's load_breast_cancer() into a clean, typed DataBundle.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer

logger = logging.getLogger(__name__)


@dataclass
class DataBundle:
    """Immutable container for the raw breast-cancer dataset."""
    X: np.ndarray          # shape (569, 30)
    y: np.ndarray          # shape (569,)  0=malignant, 1=benign
    X_df: pd.DataFrame     # feature matrix as DataFrame
    y_series: pd.Series    # target as Series
    feature_names: list[str]
    target_names: list[str]   # ['malignant', 'benign']
    description: str          # original UCI description


def load_data() -> DataBundle:
    """
    Load the Wisconsin Diagnostic Breast Cancer dataset.

    Returns
    -------
    DataBundle
        A fully populated DataBundle; never returns None.

    Notes
    -----
    Source: UCI ML Repository / scikit-learn 1.x built-in dataset.
    The dataset has 569 samples and 30 real-valued features computed
    from digitised images of fine needle aspirates (FNA) of breast masses.
    Target: 0 = malignant (212 samples), 1 = benign (357 samples).
    """
    logger.info("Loading Wisconsin Diagnostic Breast Cancer dataset …")
    raw = load_breast_cancer(as_frame=True)

    X_df: pd.DataFrame = raw.frame.drop(columns=["target"])
    y_series: pd.Series = raw.frame["target"]

    bundle = DataBundle(
        X=raw.data.values,
        y=raw.target.values,
        X_df=X_df,
        y_series=y_series,
        feature_names=list(raw.feature_names),
        target_names=list(raw.target_names),
        description=raw.DESCR,
    )
    logger.info(
        "Dataset loaded: %d samples, %d features, classes=%s",
        bundle.X.shape[0],
        bundle.X.shape[1],
        bundle.target_names,
    )
    return bundle
