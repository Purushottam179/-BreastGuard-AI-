"""
BreastGuard AI — Counterfactual Explanations Module
================================================================================
Implements research-oriented counterfactual explanations for breast cancer diagnosis:
  1. Identifies minimal, plausible perturbations to cytological input features
     that flip the model's classification across the clinical decision threshold (0.37).
  2. Enforces empirical feature bounds based on the observed training distribution.
  3. Prioritizes sparse changes restricted to top influential features.
  4. Ranks candidate counterfactuals by normalized L1/L2 distance from the original sample.
  5. Implements strict medical safety framing: hypothetical sensitivity analysis ONLY.
================================================================================
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from config.config import CLASS_LABELS, LABEL_BENIGN, LABEL_MALIGNANT

logger = logging.getLogger(__name__)

MEDICAL_SAFETY_DISCLAIMER: str = (
    "CLINICAL SAFETY NOTICE: Counterfactual explanations represent a strictly mathematical, "
    "in-silico exploration of the model's learned decision boundary. They quantify algorithmic "
    "sensitivity to input perturbations. They MUST NEVER be interpreted as medical advice, clinical "
    "prognosis, surgical targets, or biological treatment strategies."
)


@dataclass
class CounterfactualCandidate:
    """A single candidate counterfactual feature set that flips the model classification."""
    counterfactual_vector: np.ndarray
    counterfactual_dict: dict[str, float]
    predicted_class: int
    predicted_label: str
    prob_malignant: float
    prob_benign: float
    modified_features: dict[str, dict[str, float]]  # {feat: {'original': v0, 'counterfactual': v1, 'delta': d, 'pct_change': pct}}
    normalized_distance: float
    sparsity: int  # number of modified features
    strategy: str  # 'single_feature_sweep', 'top_k_coordinated', 'exemplar_interpolation'


@dataclass
class CounterfactualResult:
    """Complete structured counterfactual analysis for an individual patient sample."""
    sample_id: int | str | None
    original_vector: np.ndarray
    original_dict: dict[str, float]
    original_class: int
    original_label: str
    original_prob_malignant: float
    original_prob_benign: float
    threshold_applied: float
    target_class: int
    target_label: str
    counterfactual_found: bool
    best_counterfactual: CounterfactualCandidate | None
    all_candidates: list[CounterfactualCandidate]
    feature_bounds_used: dict[str, tuple[float, float]]
    safety_disclaimer: str = MEDICAL_SAFETY_DISCLAIMER


class CounterfactualExplainer:
    """
    Generates sparse, bounded counterfactual explanations for the champion model pipeline.
    """

    def __init__(
        self,
        pipeline: Pipeline,
        feature_names: list[str],
        X_train: pd.DataFrame | np.ndarray,
        threshold: float = 0.37,
        percentile_bounds: tuple[float, float] = (1.0, 99.0),
    ):
        self.pipeline = pipeline
        self.feature_names = list(feature_names)
        self.threshold = float(threshold)
        
        # Calculate empirical feature bounds and ranges from training data
        X_arr = np.asarray(X_train, dtype=float)
        self.low_bounds = np.percentile(X_arr, percentile_bounds[0], axis=0)
        self.high_bounds = np.percentile(X_arr, percentile_bounds[1], axis=0)
        self.medians = np.median(X_arr, axis=0)
        
        # Feature ranges (max - min) to normalize distances
        ranges = self.high_bounds - self.low_bounds
        ranges[ranges <= 1e-9] = 1.0
        self.ranges = ranges

        self.bounds_dict = {
            self.feature_names[i]: (float(self.low_bounds[i]), float(self.high_bounds[i]))
            for i in range(len(self.feature_names))
        }

    def _predict_prob_mal(self, x_2d: np.ndarray) -> float:
        """Returns P(Malignant) for input sample(s)."""
        probs = self.pipeline.predict_proba(x_2d)
        return float(probs[0, 1])

    def _compute_normalized_distance(self, orig: np.ndarray, cf: np.ndarray) -> float:
        """Normalized L1 Manhattan distance divided by observed feature range."""
        diff = np.abs(orig - cf) / self.ranges
        return float(np.sum(diff))

    def generate_counterfactual(
        self,
        sample: pd.Series | np.ndarray | dict[str, float],
        sample_id: int | str | None = None,
        influential_features: list[str] | None = None,
        max_search_features: int = 6,
        n_steps: int = 50,
    ) -> CounterfactualResult:
        """
        Generate candidate counterfactuals by exploring minimal perturbations
        along influential feature dimensions within domain-plausible bounds.
        """
        if isinstance(sample, dict):
            x_orig = np.array([float(sample[f]) for f in self.feature_names], dtype=float)
            orig_dict = {f: float(sample[f]) for f in self.feature_names}
        elif isinstance(sample, pd.Series):
            x_orig = np.array([float(sample[f]) for f in self.feature_names], dtype=float)
            orig_dict = {f: float(sample[f]) for f in self.feature_names}
        else:
            x_orig = np.asarray(sample, dtype=float).ravel()
            orig_dict = {self.feature_names[i]: float(x_orig[i]) for i in range(len(self.feature_names))}

        orig_prob_mal = self._predict_prob_mal(x_orig.reshape(1, -1))
        orig_class = int(orig_prob_mal >= self.threshold)
        orig_label = CLASS_LABELS[orig_class]

        # Target class is opposite of original prediction
        target_class = 1 - orig_class
        target_label = CLASS_LABELS[target_class]

        candidates: list[CounterfactualCandidate] = []

        # If user did not provide influential features, default to major known drivers
        if influential_features is None:
            # High-variance morphometric drivers
            priority_features = [
                "worst_area", "worst_radius", "worst_texture",
                "mean_concave_points", "worst_concavity", "worst_perimeter",
                "worst_symmetry", "mean_radius", "radius_se"
            ]
            influential_features = [f for f in priority_features if f in self.feature_names]
        else:
            influential_features = [f for f in influential_features if f in self.feature_names]

        target_features = influential_features[:max_search_features]

        # ─────────────────────────────────────────────────────────────────────
        # STRATEGY 1: Single-Feature Sweeps (Maximum Sparsity: k=1)
        # ─────────────────────────────────────────────────────────────────────
        for feat in target_features:
            feat_idx = self.feature_names.index(feat)
            low_val = self.low_bounds[feat_idx]
            high_val = self.high_bounds[feat_idx]
            orig_val = x_orig[feat_idx]

            # If moving from Malignant to Benign, test decreasing the feature value
            # If moving from Benign to Malignant, test increasing the feature value
            if orig_class == 1:
                test_values = np.linspace(orig_val, low_val, n_steps)
            else:
                test_values = np.linspace(orig_val, high_val, n_steps)

            for val in test_values:
                x_test = x_orig.copy()
                x_test[feat_idx] = val
                prob_mal = self._predict_prob_mal(x_test.reshape(1, -1))
                new_class = int(prob_mal >= self.threshold)

                if new_class == target_class:
                    dist = self._compute_normalized_distance(x_orig, x_test)
                    pct = ((val - orig_val) / orig_val * 100.0) if abs(orig_val) > 1e-6 else 0.0
                    mod_info = {
                        feat: {
                            "original": round(float(orig_val), 4),
                            "counterfactual": round(float(val), 4),
                            "delta": round(float(val - orig_val), 4),
                            "pct_change": round(float(pct), 2),
                        }
                    }
                    candidates.append(
                        CounterfactualCandidate(
                            counterfactual_vector=x_test,
                            counterfactual_dict={self.feature_names[i]: float(x_test[i]) for i in range(len(x_test))},
                            predicted_class=new_class,
                            predicted_label=CLASS_LABELS[new_class],
                            prob_malignant=round(prob_mal, 4),
                            prob_benign=round(1.0 - prob_mal, 4),
                            modified_features=mod_info,
                            normalized_distance=round(dist, 4),
                            sparsity=1,
                            strategy=f"single_feature_sweep ({feat})",
                        )
                    )
                    break  # Stop at minimal boundary crossing for this feature

        # ─────────────────────────────────────────────────────────────────────
        # STRATEGY 2: Coordinated Two-Feature Perturbations (k=2)
        # ─────────────────────────────────────────────────────────────────────
        for i in range(len(target_features)):
            for j in range(i + 1, min(i + 3, len(target_features))):
                f1 = target_features[i]
                f2 = target_features[j]
                idx1 = self.feature_names.index(f1)
                idx2 = self.feature_names.index(f2)

                # Search coordinated fractional changes: 10%, 20%, ..., 90% toward target median
                for alpha in np.linspace(0.1, 0.9, 9):
                    x_test = x_orig.copy()
                    target_v1 = self.medians[idx1] if orig_class == 1 else self.high_bounds[idx1]
                    target_v2 = self.medians[idx2] if orig_class == 1 else self.high_bounds[idx2]

                    val1 = x_orig[idx1] + alpha * (target_v1 - x_orig[idx1])
                    val2 = x_orig[idx2] + alpha * (target_v2 - x_orig[idx2])

                    x_test[idx1] = np.clip(val1, self.low_bounds[idx1], self.high_bounds[idx1])
                    x_test[idx2] = np.clip(val2, self.low_bounds[idx2], self.high_bounds[idx2])

                    prob_mal = self._predict_prob_mal(x_test.reshape(1, -1))
                    new_class = int(prob_mal >= self.threshold)

                    if new_class == target_class:
                        dist = self._compute_normalized_distance(x_orig, x_test)
                        mod_info = {
                            f1: {
                                "original": round(float(x_orig[idx1]), 4),
                                "counterfactual": round(float(x_test[idx1]), 4),
                                "delta": round(float(x_test[idx1] - x_orig[idx1]), 4),
                                "pct_change": round(float((x_test[idx1] - x_orig[idx1]) / x_orig[idx1] * 100.0), 2),
                            },
                            f2: {
                                "original": round(float(x_orig[idx2]), 4),
                                "counterfactual": round(float(x_test[idx2]), 4),
                                "delta": round(float(x_test[idx2] - x_orig[idx2]), 4),
                                "pct_change": round(float((x_test[idx2] - x_orig[idx2]) / x_orig[idx2] * 100.0), 2),
                            },
                        }
                        candidates.append(
                            CounterfactualCandidate(
                                counterfactual_vector=x_test,
                                counterfactual_dict={self.feature_names[k]: float(x_test[k]) for k in range(len(x_test))},
                                predicted_class=new_class,
                                predicted_label=CLASS_LABELS[new_class],
                                prob_malignant=round(prob_mal, 4),
                                prob_benign=round(1.0 - prob_mal, 4),
                                modified_features=mod_info,
                                normalized_distance=round(dist, 4),
                                sparsity=2,
                                strategy=f"coordinated_pair ({f1} + {f2})",
                            )
                        )
                        break

        # ─────────────────────────────────────────────────────────────────────
        # STRATEGY 3: Full Target Cohort Centroid Interpolation (Fallback)
        # ─────────────────────────────────────────────────────────────────────
        if not candidates:
            # Interpolate toward training median
            for alpha in np.linspace(0.05, 1.0, 20):
                x_test = x_orig + alpha * (self.medians - x_orig)
                prob_mal = self._predict_prob_mal(x_test.reshape(1, -1))
                new_class = int(prob_mal >= self.threshold)
                if new_class == target_class:
                    dist = self._compute_normalized_distance(x_orig, x_test)
                    diffs = np.abs(x_test - x_orig)
                    top_mod_idx = np.argsort(diffs)[::-1][:3]
                    mod_info = {
                        self.feature_names[k]: {
                            "original": round(float(x_orig[k]), 4),
                            "counterfactual": round(float(x_test[k]), 4),
                            "delta": round(float(x_test[k] - x_orig[k]), 4),
                            "pct_change": round(float((x_test[k] - x_orig[k]) / (x_orig[k] + 1e-9) * 100.0), 2),
                        }
                        for k in top_mod_idx
                    }
                    candidates.append(
                        CounterfactualCandidate(
                            counterfactual_vector=x_test,
                            counterfactual_dict={self.feature_names[k]: float(x_test[k]) for k in range(len(x_test))},
                            predicted_class=new_class,
                            predicted_label=CLASS_LABELS[new_class],
                            prob_malignant=round(prob_mal, 4),
                            prob_benign=round(1.0 - prob_mal, 4),
                            modified_features=mod_info,
                            normalized_distance=round(dist, 4),
                            sparsity=int(np.sum(np.abs(x_test - x_orig) > 1e-4)),
                            strategy="centroid_interpolation",
                        )
                    )
                    break

        # Rank candidates by normalized distance (lowest distance first)
        candidates.sort(key=lambda c: (c.normalized_distance, c.sparsity))
        best_cf = candidates[0] if candidates else None

        return CounterfactualResult(
            sample_id=sample_id,
            original_vector=x_orig,
            original_dict=orig_dict,
            original_class=orig_class,
            original_label=orig_label,
            original_prob_malignant=round(orig_prob_mal, 4),
            original_prob_benign=round(1.0 - orig_prob_mal, 4),
            threshold_applied=self.threshold,
            target_class=target_class,
            target_label=target_label,
            counterfactual_found=bool(candidates),
            best_counterfactual=best_cf,
            all_candidates=candidates,
            feature_bounds_used=self.bounds_dict,
        )

