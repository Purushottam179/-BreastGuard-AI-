"""
BreastGuard AI — Data Validator
Asserts dataset integrity before any ML pipeline runs.
"""
from __future__ import annotations

import logging

import numpy as np

from src.data.loader import DataBundle

logger = logging.getLogger(__name__)

EXPECTED_SHAPE    = (569, 30)
EXPECTED_CLASSES  = {0, 1}          # 0=malignant, 1=benign
EXPECTED_FEATURES = 30


class DataValidationError(ValueError):
    """Raised when the dataset fails a structural integrity check."""


def validate(bundle: DataBundle) -> None:
    """
    Validate a DataBundle against expected WDBC schema.

    Raises
    ------
    DataValidationError
        If any check fails.
    """
    logger.info("Validating dataset …")

    # Shape
    actual_shape = bundle.X.shape
    if actual_shape != EXPECTED_SHAPE:
        raise DataValidationError(
            f"Expected shape {EXPECTED_SHAPE}, got {actual_shape}"
        )

    # No missing values
    if np.isnan(bundle.X).any():
        raise DataValidationError("Feature matrix contains NaN values.")

    if bundle.y_series.isna().any():
        raise DataValidationError("Target vector contains NaN values.")

    # Binary target
    actual_classes = set(np.unique(bundle.y))
    if actual_classes != EXPECTED_CLASSES:
        raise DataValidationError(
            f"Expected target classes {EXPECTED_CLASSES}, got {actual_classes}"
        )

    # Feature count matches names
    if len(bundle.feature_names) != EXPECTED_FEATURES:
        raise DataValidationError(
            f"Expected {EXPECTED_FEATURES} feature names, "
            f"got {len(bundle.feature_names)}"
        )

    # DataFrame column count matches
    if bundle.X_df.shape[1] != EXPECTED_FEATURES:
        raise DataValidationError(
            f"DataFrame has {bundle.X_df.shape[1]} columns, expected {EXPECTED_FEATURES}"
        )

    logger.info("Dataset validation passed ✓")
