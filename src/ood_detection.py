"""
BreastGuard AI — Out-of-Distribution (OOD) Detection Module
================================================================================
Detects cytological measurement inputs that deviate significantly from the
training distribution, indicating potential sensor artifacts, slide staining anomalies,
or biological edge cases.

Detection Methods:
  1. Standardized Mahalanobis Distance: Accounts for multivariate covariance structures.
     Uses regularized covariance (Ledoit-Wolf shrinkage) to handle cytological collinearity.
  2. Isolation Forest: Unsupervised recursive partitioning anomaly detection.
  3. Univariate Z-Score Feature Outlier Screening: Highlights specific measurements exceeding ±3.5σ.

PROTOCOL COMPLIANCE:
  - FITTED STRICTLY ON TRAINING DATA ONLY (455-sample training partition).
  - Held-out test set is NEVER used to calibrate OOD thresholds.
  - An OOD classification indicates mathematical atypicality, NOT malignancy or benignity.
  - OOD detection does NOT guarantee identification of all erroneous predictions.
================================================================================
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf
from sklearn.ensemble import IsolationForest

from config.config import MODELS_DIR, RANDOM_SEED

logger = logging.getLogger(__name__)


@dataclass
class OODResult:
    """Out-of-Distribution evaluation for a single patient sample."""
    is_ood: bool
    severity: str                     # 'NORMAL', 'MILD_OUTLIER', 'SEVERE_OOD'
    mahalanobis_distance: float
    mahalanobis_threshold_95: float
    mahalanobis_threshold_99: float
    isolation_forest_score: float
    isolation_forest_anomaly: bool
    extreme_features: list[dict[str, Any]]  # [{feature, value, mean, std, z_score}]
    warnings: list[str]
    disclaimer: str = (
        "Out-of-distribution detection flags mathematical divergence from the training cohort. "
        "It does not assess disease progression or replace clinical evaluation."
    )


class OODDetector:
    """
    Multivariate and univariate Out-of-Distribution detector.
    Fitted exclusively on the training partition.
    """

    def __init__(
        self,
        feature_names: list[str],
        random_state: int = RANDOM_SEED,
    ) -> None:
        self.feature_names = list(feature_names)
        self.n_features = len(feature_names)
        self.random_state = random_state

        # Parameters fitted on training data
        self.means_: np.ndarray | None = None
        self.stds_: np.ndarray | None = None
        self.cov_estimator_: LedoitWolf | None = None
        self.inv_cov_: np.ndarray | None = None
        self.iso_forest_: IsolationForest | None = None

        # Calibrated training distance thresholds
        self.train_mahal_distances_: np.ndarray | None = None
        self.mahal_threshold_95_: float = 0.0
        self.mahal_threshold_99_: float = 0.0
        self.is_fitted: bool = False

    def fit(self, X_train: np.ndarray | pd.DataFrame) -> OODDetector:
        """
        Fit OOD parameters strictly on the training partition.

        Parameters
        ----------
        X_train : np.ndarray or pd.DataFrame
            Training feature matrix (N_train x 30).
        """
        if isinstance(X_train, pd.DataFrame):
            X_mat = X_train[self.feature_names].values.astype(float)
        else:
            X_mat = np.asarray(X_train, dtype=float)

        if X_mat.shape[1] != self.n_features:
            raise ValueError(
                f"Feature dimension mismatch: expected {self.n_features}, got {X_mat.shape[1]}"
            )

        # 1. Univariate parameters
        self.means_ = np.mean(X_mat, axis=0)
        self.stds_ = np.std(X_mat, axis=0)
        # Avoid division by zero
        self.stds_[self.stds_ == 0] = 1.0

        # Standardize training data for stable covariance estimation
        X_scaled = (X_mat - self.means_) / self.stds_

        # 2. Regularized Multivariate Covariance (Ledoit-Wolf shrinkage)
        self.cov_estimator_ = LedoitWolf(assume_centered=False)
        self.cov_estimator_.fit(X_scaled)
        
        # Precision matrix (inverse covariance)
        self.inv_cov_ = self.cov_estimator_.precision_

        # 3. Isolation Forest fitted on training partition
        self.iso_forest_ = IsolationForest(
            n_estimators=100,
            contamination=0.05,
            random_state=self.random_state,
        )
        self.iso_forest_.fit(X_scaled)

        # 4. Compute training Mahalanobis distances to establish empirical percentiles
        diffs = X_scaled
        # D_M^2 = sum_j (diff * inv_cov)_j * diff_j
        left = np.dot(diffs, self.inv_cov_)
        mahal_sq = np.sum(left * diffs, axis=1)
        self.train_mahal_distances_ = np.sqrt(np.maximum(mahal_sq, 0.0))

        self.mahal_threshold_95_ = float(np.percentile(self.train_mahal_distances_, 95))
        self.mahal_threshold_99_ = float(np.percentile(self.train_mahal_distances_, 99))

        self.is_fitted = True
        logger.info(
            f"OODDetector fitted on {len(X_mat)} samples. "
            f"Mahalanobis 95th={self.mahal_threshold_95_:.2f}, 99th={self.mahal_threshold_99_:.2f}"
        )
        return self

    def score_mahalanobis(self, x: np.ndarray) -> float:
        """Compute Mahalanobis distance for a single sample."""
        if not self.is_fitted:
            raise RuntimeError("OODDetector must be fitted before scoring.")
        x_scaled = (x - self.means_) / self.stds_
        diff = x_scaled.reshape(1, -1)
        left = np.dot(diff, self.inv_cov_)
        dist_sq = float(np.dot(left, diff.T)[0, 0])
        return float(np.sqrt(max(dist_sq, 0.0)))

    def inspect_sample(
        self,
        feature_values: dict[str, float] | pd.Series | np.ndarray,
        z_threshold: float = 3.5,
    ) -> OODResult:
        """
        Evaluate a single patient sample for out-of-distribution characteristics.

        Parameters
        ----------
        feature_values : dict, Series, or 1D array
            Input measurements across 30 WDBC features.
        z_threshold : float, default=3.5
            Univariate Z-score threshold for extreme individual measurements.

        Returns
        -------
        OODResult
            Structured OOD assessment with distance metrics and warning messages.
        """
        if not self.is_fitted:
            raise RuntimeError("OODDetector must be fitted before evaluating samples.")

        # Convert input to ordered array
        if isinstance(feature_values, dict):
            x_arr = np.array([float(feature_values[f]) for f in self.feature_names])
        elif isinstance(feature_values, pd.Series):
            x_arr = np.array([float(feature_values[f]) for f in self.feature_names])
        else:
            x_arr = np.asarray(feature_values, dtype=float).flatten()

        # 1. Mahalanobis distance
        mahal_dist = self.score_mahalanobis(x_arr)

        # 2. Isolation Forest score
        x_scaled = ((x_arr - self.means_) / self.stds_).reshape(1, -1)
        iso_score = float(self.iso_forest_.decision_function(x_scaled)[0])
        iso_anomaly = bool(self.iso_forest_.predict(x_scaled)[0] == -1)

        # 3. Univariate extreme feature detection
        z_scores = (x_arr - self.means_) / self.stds_
        extreme_feats: list[dict[str, Any]] = []
        for i, feat in enumerate(self.feature_names):
            z = z_scores[i]
            if abs(z) >= z_threshold:
                extreme_feats.append({
                    "feature": feat,
                    "value": round(float(x_arr[i]), 4),
                    "mean": round(float(self.means_[i]), 4),
                    "std": round(float(self.stds_[i]), 4),
                    "z_score": round(float(z), 2),
                })

        # 4. Severity classification
        warnings: list[str] = []
        if mahal_dist > self.mahal_threshold_99_:
            severity = "SEVERE_OOD"
            is_ood = True
            warnings.append(
                f"Severe Out-of-Distribution: Mahalanobis distance ({mahal_dist:.2f}) exceeds the 99th "
                f"percentile training threshold ({self.mahal_threshold_99_:.2f}). "
                "Prediction reliability is unverified."
            )
        elif mahal_dist > self.mahal_threshold_95_ or iso_anomaly or len(extreme_feats) >= 3:
            severity = "MILD_OUTLIER"
            is_ood = True
            warnings.append(
                f"Atypical Input: Mahalanobis distance ({mahal_dist:.2f}) exceeds the 95th percentile "
                f"({self.mahal_threshold_95_:.2f}) or exhibits unusual multi-feature combinations."
            )
        else:
            severity = "NORMAL"
            is_ood = False

        if extreme_feats:
            feat_strs = [f"{ef['feature']} (Z={ef['z_score']:+.1f})" for ef in extreme_feats[:3]]
            warnings.append(
                f"Extremely high/low individual measurements detected: {', '.join(feat_strs)}."
            )

        return OODResult(
            is_ood=is_ood,
            severity=severity,
            mahalanobis_distance=round(mahal_dist, 3),
            mahalanobis_threshold_95=round(self.mahal_threshold_95_, 3),
            mahalanobis_threshold_99=round(self.mahal_threshold_99_, 3),
            isolation_forest_score=round(iso_score, 4),
            isolation_forest_anomaly=iso_anomaly,
            extreme_features=extreme_feats,
            warnings=warnings,
        )


_CACHED_OOD_DETECTOR: OODDetector | None = None


def load_ood_detector() -> OODDetector:
    """
    Factory function returning a singleton OODDetector fitted strictly on the training partition.
    """
    global _CACHED_OOD_DETECTOR
    if _CACHED_OOD_DETECTOR is not None:
        return _CACHED_OOD_DETECTOR

    from src.data_loader import load_dataset
    from src.preprocessing import create_stratified_split

    bundle = load_dataset(save_raw_csv=False)
    split_data = create_stratified_split(
        bundle.X_df,
        bundle.y_series,
        test_size=0.20,
        random_seed=RANDOM_SEED,
    )

    detector = OODDetector(feature_names=bundle.feature_names, random_state=RANDOM_SEED)
    detector.fit(split_data.X_train)

    _CACHED_OOD_DETECTOR = detector
    return detector

