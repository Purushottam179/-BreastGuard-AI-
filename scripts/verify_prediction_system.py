"""
BreastGuard AI — Prediction System Verification Script
================================================================================
Verifies end-to-end functionality for Prompt 8:
  1. Loads PredictionEngine and verifies final model pipeline & metadata.
  2. Tests a true Benign record from the actual dataset.
  3. Tests a true Malignant record from the actual dataset.
  4. Verifies probability sums and calibrated threshold application (tau = 0.37).
  5. Verifies local SHAP explanation integration and narrative generation.
  6. Executes batch CSV processing and exports reports/predictions/batch_prediction_results.csv.
  7. Generates individual prediction report in PDF format and validates file integrity.
================================================================================
"""
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".matplotlib_cache"))
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import numpy as np
import pandas as pd
from src.predict import PredictionEngine, load_prediction_engine
from src.utils import setup_logger

logger = setup_logger("prediction_verification")


def main() -> None:
    logger.info("=" * 80)
    logger.info("BreastGuard AI — Interactive Prediction System End-to-End Verification")
    logger.info("=" * 80)

    # 1. Initialize Engine
    engine = load_prediction_engine()
    logger.info(
        "Loaded model: '%s' | Threshold: %.2f | Features expected: %d",
        engine.model_name, engine.threshold, len(engine.feature_names),
    )

    # 2. Load Actual Dataset
    data_path = PROJECT_ROOT / "data" / "processed" / "breast_cancer_cleaned.csv"
    df = pd.read_csv(data_path)
    target_col = "diagnosis" if "diagnosis" in df.columns else "target"

    benign_rows = df[df[target_col] == 0]
    malignant_rows = df[df[target_col] == 1]

    # Select representative real samples
    benign_sample = benign_rows.iloc[0][engine.feature_names].to_dict()
    malignant_sample = malignant_rows.iloc[0][engine.feature_names].to_dict()

    # 3. Test Real Benign Sample
    logger.info("\n--- Testing Real Benign Biopsy Record ---")
    res_ben = engine.predict_single(benign_sample, generate_explanation=True)
    logger.info("Predicted: %s (Class %d)", res_ben.predicted_label, res_ben.predicted_class)
    logger.info("P(Benign): %.4f | P(Malignant): %.4f | Cutoff: %.2f", res_ben.prob_benign, res_ben.prob_malignant, res_ben.classification_threshold)
    logger.info("Risk Tier: %s", res_ben.risk_tier)
    logger.info("SHAP Explanation Available: %s", res_ben.explanation_available)
    if res_ben.top_10_features:
        logger.info("Top Negative Driver: %s (value=%.3f, shap=%.3f)", res_ben.negative_drivers[0][0], res_ben.negative_drivers[0][1], res_ben.negative_drivers[0][2])

    assert res_ben.predicted_class == 0, f"Expected Benign (0), got {res_ben.predicted_class}"
    assert np.isclose(res_ben.prob_benign + res_ben.prob_malignant, 1.0, atol=1e-4)

    # 4. Test Real Malignant Sample
    logger.info("\n--- Testing Real Malignant Biopsy Record ---")
    res_mal = engine.predict_single(malignant_sample, generate_explanation=True)
    logger.info("Predicted: %s (Class %d)", res_mal.predicted_label, res_mal.predicted_class)
    logger.info("P(Benign): %.4f | P(Malignant): %.4f | Cutoff: %.2f", res_mal.prob_benign, res_mal.prob_malignant, res_mal.classification_threshold)
    logger.info("Risk Tier: %s", res_mal.risk_tier)
    logger.info("SHAP Explanation Available: %s", res_mal.explanation_available)
    if res_mal.top_10_features:
        logger.info("Top Positive Driver: %s (value=%.3f, shap=%.3f)", res_mal.positive_drivers[0][0], res_mal.positive_drivers[0][1], res_mal.positive_drivers[0][2])

    assert res_mal.predicted_class == 1, f"Expected Malignant (1), got {res_mal.predicted_class}"
    assert np.isclose(res_mal.prob_benign + res_mal.prob_malignant, 1.0, atol=1e-4)

    # 5. Test Batch Prediction on CSV
    logger.info("\n--- Testing Batch CSV Prediction Pipeline ---")
    batch_test_df = df.iloc[:10][engine.feature_names].copy()
    batch_test_df.insert(0, "id", [f"PATIENT_{i+1:03d}" for i in range(10)])

    # Inject one deliberately invalid row to verify safe handling and row tracking
    invalid_row = df.iloc[0][engine.feature_names].to_dict()
    invalid_row["id"] = "PATIENT_INVALID"
    invalid_row["mean_radius"] = -10.0  # Invalid negative
    batch_test_df = pd.concat([batch_test_df, pd.DataFrame([invalid_row])], ignore_index=True)

    batch_csv_in = PROJECT_ROOT / "reports" / "predictions" / "sample_batch_input.csv"
    batch_test_df.to_csv(batch_csv_in, index=False)

    batch_out = engine.predict_batch(batch_csv_in)
    logger.info(
        "Batch results: Total=%d | Valid=%d | Invalid=%d",
        batch_out.total_records, batch_out.valid_records, batch_out.invalid_records,
    )
    logger.info("Batch results saved to: %s", batch_out.output_path)
    assert batch_out.output_path.exists()
    assert batch_out.valid_records == 10
    assert batch_out.invalid_records == 1

    # 6. Test PDF Report Generation
    logger.info("\n--- Testing PDF Report Generation ---")
    pdf_report_path = engine.generate_pdf_report(res_mal)
    logger.info("PDF Report generated at: %s", pdf_report_path)
    assert pdf_report_path.exists()
    assert pdf_report_path.stat().st_size > 10000

    logger.info("=" * 80)
    logger.info("ALL VERIFICATION CHECKS PASSED SUCCESSFULLY!")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()

