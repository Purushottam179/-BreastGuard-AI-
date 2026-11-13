"""
BreastGuard AI — Prediction Service
High-level inference API used by the Streamlit interactive prediction page.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from src.explainability.shap_explainer import ShapResult, explain_single
from src.models.trainer import TrainedModel

logger = logging.getLogger(__name__)


@dataclass
class PredictionResult:
    """Result of a single-instance prediction."""
    model_name:    str
    label:         str          # "Benign" or "Malignant"
    label_index:   int          # 0=Malignant, 1=Benign
    prob_benign:   float        # P(benign)
    prob_malignant: float       # P(malignant)
    shap_result:   ShapResult   # SHAP explanation for this instance


class PredictionService:
    """
    Wraps a trained model pipeline and provides a clean predict() interface.

    Parameters
    ----------
    trained_model : TrainedModel
        The fitted model produced by trainer.train_all()
    X_train : np.ndarray
        Raw (unscaled) training data — used as SHAP background.
    feature_names : list[str]
    """

    def __init__(
        self,
        trained_model: TrainedModel,
        X_train: np.ndarray,
        feature_names: list[str],
    ):
        self._model       = trained_model
        self._X_train     = X_train
        self._feature_names = feature_names

    def predict(self, feature_values: dict[str, float]) -> PredictionResult:
        """
        Run inference on a single sample provided as a feature dict.

        Parameters
        ----------
        feature_values : dict  {feature_name: value}
            Must contain all 30 WDBC features.

        Returns
        -------
        PredictionResult
        """
        # Build ordered array from dict
        x = np.array(
            [feature_values[fn] for fn in self._feature_names],
            dtype=np.float64,
        ).reshape(1, -1)

        pipeline = self._model.pipeline
        proba    = pipeline.predict_proba(x)[0]   # [P(malignant), P(benign)]
        pred_idx = int(pipeline.predict(x)[0])    # 0 or 1

        label = "Benign" if pred_idx == 1 else "Malignant"
        logger.info(
            "Prediction: %s | P(benign)=%.3f | P(malignant)=%.3f",
            label, proba[1], proba[0],
        )

        shap_result = explain_single(
            pipeline      = pipeline,
            X_train       = self._X_train,
            x_instance    = x,
            feature_names = self._feature_names,
            model_name    = self._model.name,
        )

        return PredictionResult(
            model_name    = self._model.name,
            label         = label,
            label_index   = pred_idx,
            prob_benign   = float(proba[1]),
            prob_malignant = float(proba[0]),
            shap_result   = shap_result,
        )
