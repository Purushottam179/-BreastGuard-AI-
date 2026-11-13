"""
BreastGuard AI — Dataset Acquisition, Cleaning, and EDA Driver Script

Run this script directly to execute the full data pipeline:
  1. Acquire WDBC dataset (UCI if reachable, sklearn fallback otherwise)
  2. Validate and clean the raw data
  3. Run full Exploratory Data Analysis
  4. Save all outputs to data/ and reports/

Usage:
  python scripts/run_data_pipeline.py
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import setup_logger
from src.data_loader import run_acquisition
from src.eda import run_eda, print_eda_summary

logger = setup_logger("data_pipeline")


def main() -> None:
    logger.info("=" * 60)
    logger.info("BreastGuard AI — Data Acquisition & EDA Pipeline")
    logger.info("=" * 60)

    # Part 1-3: Acquisition, understanding, and cleaning
    bundle = run_acquisition()

    logger.info("")
    logger.info("ACQUISITION COMPLETE:")
    logger.info("  Source            : %s", bundle.info.source)
    logger.info("  Samples           : %d", bundle.info.n_samples)
    logger.info("  Malignant (M=1)   : %d", bundle.info.n_malignant)
    logger.info("  Benign    (B=0)   : %d", bundle.info.n_benign)
    logger.info("  Missing values    : %d", bundle.info.n_missing)
    logger.info("  Duplicate records : %d", bundle.info.n_duplicates)
    logger.info("  Raw CSV           : %s", bundle.info.raw_csv_path)
    logger.info("  Cleaned CSV       : %s", bundle.info.cleaned_csv_path)

    # Part 4: EDA
    eda_report = run_eda(bundle.df_cleaned, save_reports=True)
    print_eda_summary(eda_report)

    logger.info("Pipeline complete. All outputs saved.")


if __name__ == "__main__":
    main()

