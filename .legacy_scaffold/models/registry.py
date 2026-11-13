"""
BreastGuard AI — Model Registry
Central definition of all candidate models and their hyperparameters.
"""
from __future__ import annotations

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

from src.utils.config import RANDOM_STATE, MODEL_LR, MODEL_RF, MODEL_XGB, MODEL_SVM

try:
    from xgboost import XGBClassifier
    _XGB_AVAILABLE = True
except ImportError:
    _XGB_AVAILABLE = False


def get_models() -> dict:
    """
    Return a dict of model_name → unfitted estimator.
    All estimators are sklearn-compatible (have fit/predict/predict_proba).
    """
    models = {
        MODEL_LR: LogisticRegression(
            max_iter=10_000,
            random_state=RANDOM_STATE,
            solver="lbfgs",
            C=1.0,
        ),
        MODEL_RF: RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            min_samples_split=2,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        MODEL_SVM: SVC(
            kernel="rbf",
            C=1.0,
            gamma="scale",
            probability=True,   # needed for predict_proba and SHAP
            random_state=RANDOM_STATE,
        ),
    }

    if _XGB_AVAILABLE:
        models[MODEL_XGB] = XGBClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.8,
            use_label_encoder=False,
            eval_metric="logloss",
            random_state=RANDOM_STATE,
            verbosity=0,
        )

    return models
