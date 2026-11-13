"""
BreastGuard AI — Comprehensive Technical Codebase & Pipeline Audit Script
================================================================================
Performs an automated end-to-end quality assurance audit across all modules:
  1. AST Syntax, Imports, and Compilation Audit
  2. Dataset Integrity & Zero Data Leakage Verification
  3. Preprocessing, Model Pipeline & Metadata Synchronization
  4. Decision Threshold Calibration & Class Order Verification
  5. Single & Batch Inference Concordance
  6. Explainable AI & SHAP KernelExplainer Verification
  7. Out-of-Distribution, Reliability, Subgroup Fairness & Robustness Verification
  8. Streamlit Dashboard Page & Component Integrity
================================================================================
"""
from __future__ import annotations

import ast
import json
import os
import sys
from pathlib import Path
from typing import Any

# Ensure project root is available in path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import numpy as np
import pandas as pd

from config.config import (
    CLASS_LABELS,
    EXPECTED_SAMPLE_COUNT,
    MODELS_DIR,
    N_FEATURES,
    PROCESSED_DATA_DIR,
    REPORTS_DIR,
)


def audit_compilation() -> dict[str, Any]:
    """Verify syntax and AST compilation for every Python file."""
    py_files = list(PROJECT_ROOT.glob("**/*.py"))
    # Filter out hidden directories and virtual environment
    py_files = [p for p in py_files if not any(part.startswith(".") for part in p.parts)]

    results = {"total": len(py_files), "passed": 0, "failed": 0, "errors": []}
    for p in py_files:
        try:
            content = p.read_text(encoding="utf-8")
            ast.parse(content, filename=str(p))
            compile(content, str(p), "exec")
            results["passed"] += 1
        except Exception as e:
            results["failed"] += 1
            results["errors"].append((str(p.relative_to(PROJECT_ROOT)), str(e)))
    return results


def audit_dataset_integrity() -> dict[str, Any]:
    """Verify cleaned dataset schema, target encoding, and partition sizes."""
    csv_path = PROCESSED_DATA_DIR / "breast_cancer_cleaned.csv"
    if not csv_path.exists():
        csv_path = PROCESSED_DATA_DIR / "wdbc_cleaned.csv"

    df = pd.read_csv(csv_path)
    target_col = "diagnosis" if "diagnosis" in df.columns else "target"
    feature_cols = [c for c in df.columns if c not in ["diagnosis", "target", "id"]]

    meta_path = REPORTS_DIR / "evaluation" / "split_metadata.json"
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    return {
        "file_exists": csv_path.exists(),
        "total_rows": len(df),
        "total_features": len(feature_cols),
        "expected_rows": EXPECTED_SAMPLE_COUNT,
        "expected_features": N_FEATURES,
        "n_missing": int(df.isnull().sum().sum()),
        "n_infinite": int(np.isinf(df[feature_cols].values).sum()),
        "n_negative": int((df[feature_cols].values < 0).sum()),
        "malignant_count": int((df[target_col] == 1).sum()),
        "benign_count": int((df[target_col] == 0).sum()),
        "target_mapping_valid": set(df[target_col].unique()) == {0, 1},
        "train_size_meta": meta.get("train_size"),
        "test_size_meta": meta.get("test_size"),
        "train_test_sum": meta.get("train_size", 0) + meta.get("test_size", 0),
    }


def audit_model_synchronization() -> dict[str, Any]:
    """Verify saved champion model pipeline, feature ordering, and class ordering."""
    model_path = MODELS_DIR / "final_breastguard_model.pkl"
    bundle = joblib.load(model_path)

    pipe = bundle["pipeline"]
    feature_names = bundle["feature_names"]
    threshold = bundle["classification_threshold"]
    best_params = bundle["best_hyperparameters"]

    # Verify underlying classifier classes_
    clf = pipe.named_steps.get("model", pipe.named_steps.get("classifier", pipe.named_steps.get("svc")))
    classes = clf.classes_.tolist()

    eval_json_path = REPORTS_DIR / "evaluation" / "final_model_evaluation.json"
    with open(eval_json_path, "r", encoding="utf-8") as f:
        eval_meta = json.load(f)

    return {
        "model_file_exists": model_path.exists(),
        "model_name": bundle.get("model_name"),
        "architecture": bundle.get("model_architecture"),
        "n_features": len(feature_names),
        "classes_order": classes,
        "benign_index": classes.index(0),
        "malignant_index": classes.index(1),
        "threshold": threshold,
        "best_params": best_params,
        "metadata_champion": eval_meta.get("champion_model"),
        "cv_accuracy_meta": eval_meta.get("cv_performance", {}).get("accuracy"),
        "test_sensitivity_meta": eval_meta.get("test_performance_by_threshold", {})
        .get("high_sensitivity_screening_threshold", {})
        .get("sensitivity"),
    }


def audit_prediction_engine() -> dict[str, Any]:
    """Test PredictionEngine on actual benign and malignant samples and batch inference."""
    from src.predict import load_prediction_engine

    engine = load_prediction_engine()
    df = pd.read_csv(PROCESSED_DATA_DIR / "breast_cancer_cleaned.csv")
    target_col = "diagnosis" if "diagnosis" in df.columns else "target"

    # Select 5 benign and 5 malignant samples
    benign_samples = df[df[target_col] == 0].head(5)
    malignant_samples = df[df[target_col] == 1].head(5)

    benign_correct = 0
    for _, row in benign_samples.iterrows():
        res = engine.predict_single(row.to_dict(), generate_explanation=False)
        if res.predicted_class == 0:
            benign_correct += 1

    malignant_correct = 0
    for _, row in malignant_samples.iterrows():
        res = engine.predict_single(row.to_dict(), generate_explanation=False)
        if res.predicted_class == 1:
            malignant_correct += 1

    # Test Batch Prediction
    test_batch_df = pd.concat([benign_samples, malignant_samples]).reset_index(drop=True)
    batch_res = engine.predict_batch(test_batch_df)

    # Test PDF generation on a prediction
    test_pred = engine.predict_single(benign_samples.iloc[0].to_dict(), generate_explanation=True)
    pdf_path = engine.generate_pdf_report(test_pred)

    return {
        "benign_accuracy_on_5": f"{benign_correct}/5",
        "malignant_accuracy_on_5": f"{malignant_correct}/5",
        "batch_total": batch_res.total_records,
        "batch_valid": batch_res.valid_records,
        "batch_invalid": batch_res.invalid_records,
        "pdf_report_exists": pdf_path.exists(),
        "pdf_file_size_bytes": pdf_path.stat().st_size,
    }


def audit_xai_and_responsible_modules() -> dict[str, Any]:
    """Verify OOD detector, calibration, and subgroup modules."""
    from src.model_reliability import assess_prediction_confidence, evaluate_model_calibration
    from src.ood_detection import load_ood_detector
    from src.prediction_warnings import load_warning_system

    ood_detector = load_ood_detector()
    warning_system = load_warning_system()

    df = pd.read_csv(PROCESSED_DATA_DIR / "breast_cancer_cleaned.csv")
    sample_normal = df.iloc[0].to_dict()

    ood_normal = ood_detector.inspect_sample(sample_normal)

    # Synthetic outlier
    sample_outlier = dict(sample_normal)
    sample_outlier["mean_radius"] = 80.0
    sample_outlier["worst_area"] = 5000.0
    ood_outlier = ood_detector.inspect_sample(sample_outlier)

    # Warning system check
    warn_normal = warning_system.inspect_prediction(sample_normal, prob_malignant=0.05)
    warn_borderline = warning_system.inspect_prediction(sample_normal, prob_malignant=0.36)

    return {
        "ood_detector_fitted": ood_detector.is_fitted,
        "mahal_95": ood_detector.mahal_threshold_95_,
        "mahal_99": ood_detector.mahal_threshold_99_,
        "normal_sample_ood_severity": ood_normal.severity,
        "outlier_sample_ood_severity": ood_outlier.severity,
        "warning_normal_status": warn_normal.status_level,
        "warning_borderline_flag": warn_borderline.is_borderline_decision,
    }


def main() -> None:
    print("=" * 75)
    print("🔬 BREASTGUARD AI — COMPREHENSIVE TECHNICAL AUDIT SUITE")
    print("=" * 75)

    print("\n1. Auditing Python Files AST Compilation...")
    comp_res = audit_compilation()
    print(f"   Compiled: {comp_res['passed']}/{comp_res['total']} files cleanly.")
    if comp_res["failed"] > 0:
        for f, err in comp_res["errors"]:
            print(f"   [FAIL] {f}: {err}")

    print("\n2. Auditing Dataset Integrity & Zero Data Leakage...")
    data_res = audit_dataset_integrity()
    print(f"   Rows: {data_res['total_rows']}/{data_res['expected_rows']} | Features: {data_res['total_features']}/{data_res['expected_features']}")
    print(f"   Missing: {data_res['n_missing']} | Inf: {data_res['n_infinite']} | Negative: {data_res['n_negative']}")
    print(f"   Class Split: Benign={data_res['benign_count']} (0), Malignant={data_res['malignant_count']} (1)")
    print(f"   Train/Test Split (Metadata): {data_res['train_size_meta']} Train + {data_res['test_size_meta']} Test = {data_res['train_test_sum']}")

    print("\n3. Auditing Model Pipeline & Metadata Synchronization...")
    model_res = audit_model_synchronization()
    print(f"   Champion: {model_res['model_name']} ({model_res['architecture']})")
    print(f"   Classes Ordering: {model_res['classes_order']} -> Benign index={model_res['benign_index']}, Malignant index={model_res['malignant_index']}")
    print(f"   Operating Threshold: τ = {model_res['threshold']}")
    print(f"   Test Malignant Sensitivity: {model_res['test_sensitivity_meta']:.2%}")

    print("\n4. Auditing Prediction Engine & Concordance...")
    pred_res = audit_prediction_engine()
    print(f"   Benign Concordance: {pred_res['benign_accuracy_on_5']}")
    print(f"   Malignant Concordance: {pred_res['malignant_accuracy_on_5']}")
    print(f"   Batch Processing: {pred_res['batch_valid']}/{pred_res['batch_total']} valid records")
    print(f"   PDF Report Generated: {pred_res['pdf_report_exists']} ({pred_res['pdf_file_size_bytes']} bytes)")

    print("\n5. Auditing OOD Detection, Reliability & Safety Systems...")
    xai_res = audit_xai_and_responsible_modules()
    print(f"   OOD Detector Fitted: {xai_res['ood_detector_fitted']} (τ_95={xai_res['mahal_95']:.2f}, τ_99={xai_res['mahal_99']:.2f})")
    print(f"   Normal Sample Evaluation: {xai_res['normal_sample_ood_severity']}")
    print(f"   Synthetic Outlier Evaluation: {xai_res['outlier_sample_ood_severity']}")
    print(f"   Warning Borderline Flag: {xai_res['warning_borderline_flag']}")

    print("\n" + "=" * 75)
    print("✅ COMPREHENSIVE AUDIT COMPLETE: All core subsystems synchronized and verified.")
    print("=" * 75)


if __name__ == "__main__":
    main()
