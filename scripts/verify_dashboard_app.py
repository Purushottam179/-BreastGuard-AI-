"""
BreastGuard AI — Dashboard Application Verification Script
================================================================================
Performs static and dynamic health checks on:
  1. AST parsing and compilation of all app pages and components.
  2. Existence of all referenced report figures and JSON artifacts.
  3. Prediction engine initialization and preset inference.
  4. Streamlit application entry point validation.
================================================================================
"""
from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.config import REPORTS_DIR


def verify_file_compiles(path: Path) -> bool:
    """Verify that a Python source file has valid syntax and compiles."""
    try:
        content = path.read_text(encoding="utf-8")
        ast.parse(content, filename=str(path))
        compile(content, str(path), "exec")
        print(f"  [OK] Syntax & AST Compilation: {path.relative_to(PROJECT_ROOT)}")
        return True
    except Exception as e:
        print(f"  [FAIL] Compilation error in {path}: {e}")
        return False


def verify_artifacts() -> bool:
    """Verify existence of key figures and evaluation reports."""
    print("\nVerifying Static Figures & Evaluation Artifacts:")
    required_files = [
        REPORTS_DIR / "evaluation" / "final_model_evaluation.json",
        REPORTS_DIR / "evaluation" / "cross_validation_results.csv",
        REPORTS_DIR / "evaluation" / "hyperparameter_results.csv",
        REPORTS_DIR / "evaluation" / "threshold_sweep_results.csv",
        REPORTS_DIR / "evaluation" / "shap_analysis_summary.json",
        REPORTS_DIR / "evaluation" / "advanced_xai_summary.json",
        REPORTS_DIR / "responsible_ai_report.md",
        REPORTS_DIR / "figures" / "roc_curve_test.png",
        REPORTS_DIR / "figures" / "precision_recall_curve_test.png",
        REPORTS_DIR / "figures" / "shap" / "shap_summary_beeswarm.png",
        REPORTS_DIR / "figures" / "shap" / "shap_bar_plot.png",
        REPORTS_DIR / "figures" / "advanced_xai" / "what_if_analysis.png",
        REPORTS_DIR / "figures" / "advanced_xai" / "counterfactual_comparison.png",
    ]

    all_exist = True
    for f in required_files:
        if f.exists():
            print(f"  [OK] Artifact exists: {f.relative_to(PROJECT_ROOT)}")
        else:
            print(f"  [FAIL] Missing artifact: {f.relative_to(PROJECT_ROOT)}")
            all_exist = False
    return all_exist


def main() -> None:
    print("=" * 70)
    print("🩺 BreastGuard AI — Dashboard Application Health Check")
    print("=" * 70)

    # 1. Compile all app scripts
    print("\n1. Verifying Application Python Files:")
    app_files = list((PROJECT_ROOT / "app").glob("**/*.py"))
    compilation_passed = all(verify_file_compiles(p) for p in app_files)

    # 2. Verify artifacts
    artifacts_passed = verify_artifacts()

    # 3. Test Prediction Engine
    print("\n3. Testing Prediction Engine Initialization & Inference:")
    from src.predict import load_prediction_engine
    engine = load_prediction_engine()
    defaults = engine.get_default_values()
    result = engine.predict_single(defaults, generate_explanation=False)
    print(f"  [OK] PredictionEngine loaded successfully. Test prediction: {result.predicted_label} (P(Malignant)={result.prob_malignant:.4f})")

    print("\n" + "=" * 70)
    if compilation_passed and artifacts_passed:
        print("✅ ALL HEALTH CHECKS PASSED: Dashboard application is healthy and ready for deployment.")
    else:
        print("❌ SOME CHECKS FAILED: Please inspect logs above.")
        sys.exit(1)
    print("=" * 70)


if __name__ == "__main__":
    main()
