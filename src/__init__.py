"""
BreastGuard AI — Core Source Package
"""
from src.data_loader import (
    FEATURE_COLUMNS,
    FEATURE_GROUPS,
    BASE_MEASUREMENTS,
    DataValidationError,
    AcquisitionBundle,
    DatasetInfo,
    run_acquisition,
    load_cleaned_dataset,
)
from src.preprocessing import SplitData, build_pipeline, create_stratified_split
from src.train import TrainedModel, TrainingSummary, train_models
from src.evaluate import ClinicalConfusionMatrix, EvaluationReport, evaluate_model, evaluate_all_models
from src.explain import ShapExplanation, compute_shap_explanations, explain_single_sample
from src.predict import PredictionResult, Predictor
from src.eda import EDAReport, FeatureProfile, run_eda, print_eda_summary

__all__ = [
    "FEATURE_COLUMNS", "FEATURE_GROUPS", "BASE_MEASUREMENTS",
    "DataValidationError", "AcquisitionBundle", "DatasetInfo",
    "run_acquisition", "load_cleaned_dataset",
    "SplitData", "build_pipeline", "create_stratified_split",
    "TrainedModel", "TrainingSummary", "train_models",
    "ClinicalConfusionMatrix", "EvaluationReport", "evaluate_model", "evaluate_all_models",
    "ShapExplanation", "compute_shap_explanations", "explain_single_sample",
    "PredictionResult", "Predictor",
    "EDAReport", "FeatureProfile", "run_eda", "print_eda_summary",
]
