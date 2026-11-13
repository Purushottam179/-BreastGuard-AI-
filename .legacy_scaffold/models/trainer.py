"""
BreastGuard AI — Model Trainer
Trains all registered models inside leakage-free pipelines and
runs stratified cross-validation.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline

from src.models.registry import get_models
from src.preprocessing.pipeline import SplitData, build_pipeline
from src.utils.config import CV_FOLDS, RANDOM_STATE

logger = logging.getLogger(__name__)

SCORING = ["accuracy", "precision", "recall", "f1", "roc_auc"]


@dataclass
class TrainedModel:
    """Holds a fitted pipeline and its cross-validation scores."""
    name:       str
    pipeline:   Pipeline
    cv_scores:  dict[str, np.ndarray]   # keys like "test_accuracy"
    cv_mean:    dict[str, float]
    cv_std:     dict[str, float]


@dataclass
class TrainingResults:
    """Collection of all trained models."""
    models: dict[str, TrainedModel] = field(default_factory=dict)


def train_all(split: SplitData) -> TrainingResults:
    """
    Train every registered model on the training partition.
    Each model is wrapped in a StandardScaler→Model pipeline to prevent
    data leakage. Cross-validation is run on the training split only.

    Parameters
    ----------
    split : SplitData
        Output of preprocessing.pipeline.make_split()

    Returns
    -------
    TrainingResults
    """
    results = TrainingResults()
    skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    for name, estimator in get_models().items():
        logger.info("Training: %s …", name)
        pipe = build_pipeline(estimator)

        # Cross-validation on training set only
        cv = cross_validate(
            pipe,
            split.X_train, split.y_train,
            cv=skf,
            scoring=SCORING,
            return_train_score=False,
            n_jobs=-1,
        )

        cv_mean = {k: float(v.mean()) for k, v in cv.items()}
        cv_std  = {k: float(v.std())  for k, v in cv.items()}

        # Final fit on full training set
        pipe.fit(split.X_train, split.y_train)

        results.models[name] = TrainedModel(
            name=name,
            pipeline=pipe,
            cv_scores=cv,
            cv_mean=cv_mean,
            cv_std=cv_std,
        )
        logger.info(
            "%s CV AUC: %.4f ± %.4f",
            name,
            cv_mean["test_roc_auc"],
            cv_std["test_roc_auc"],
        )

    return results
