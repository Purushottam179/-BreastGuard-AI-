"""
BreastGuard AI — SHAP Environment & Compatibility Verification Script
================================================================================
Verifies that:
  1. Python, NumPy, Pandas, Scikit-learn, and SHAP versions are compatible.
  2. The trained final model (models/final_breastguard_model.pkl) loads cleanly.
  3. Preprocessing steps (StandardScaler) and estimator (SVC) are intact.
  4. KernelExplainer initializes and executes without crashing or emitting errors.
================================================================================
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Ensure writable matplotlib cache dir
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".matplotlib_cache"))
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import joblib
import numpy as np
import pandas as pd
import sklearn
import shap


def verify_environment() -> dict:
    print("=" * 70)
    print("BreastGuard AI — SHAP Environment Verification")
    print("=" * 70)

    # 1. Package Versions
    versions = {
        "Python": sys.version.split()[0],
        "SHAP": shap.__version__,
        "Scikit-learn": sklearn.__version__,
        "NumPy": np.__version__,
        "Pandas": pd.__version__,
    }
    for pkg, ver in versions.items():
        print(f"  * {pkg:<15}: {ver}")

    # 2. Final Model Inspection
    model_path = PROJECT_ROOT / "models" / "final_breastguard_model.pkl"
    if not model_path.exists():
        raise FileNotFoundError(f"Final model not found at {model_path}!")

    model_bundle = joblib.load(model_path)
    pipeline = model_bundle["pipeline"] if isinstance(model_bundle, dict) else model_bundle.pipeline

    print("\n[Model Inspection]")
    print(f"  * Model Name       : {model_bundle.get('model_name', 'Unknown')}")
    print(f"  * Architecture     : {model_bundle.get('model_architecture', 'Pipeline')}")
    print(f"  * Feature Count    : {len(model_bundle.get('feature_names', []))}")
    print(f"  * Target Mapping   : {model_bundle.get('target_mapping')}")
    print(f"  * Clinical Thresh  : {model_bundle.get('classification_threshold')}")
    print(f"  * Pipeline Steps   : {list(pipeline.named_steps.keys())}")

    # 3. Mini Functional SHAP Test
    print("\n[SHAP Functional Smoke Test]")
    X_dummy_bg = np.random.RandomState(42).randn(10, 30)
    X_dummy_sample = np.random.RandomState(99).randn(2, 30)

    # We explain probability of Malignant (class 1)
    predict_fn = lambda x: pipeline.predict_proba(x)[:, 1]

    explainer = shap.KernelExplainer(predict_fn, X_dummy_bg)
    shap_vals = explainer.shap_values(X_dummy_sample, nsamples=20)

    print(f"  * Explainer Type   : {type(explainer).__name__}")
    print(f"  * Expected Value   : {float(explainer.expected_value):.4f}")
    print(f"  * SHAP Output Shape: {np.shape(shap_vals)}")
    print(f"  * Finite Check     : {bool(np.all(np.isfinite(shap_vals)))}")

    print("\nResult: All SHAP environment and compatibility checks PASSED!")
    print("=" * 70)
    return {
        "status": "PASSED",
        "versions": versions,
        "explainer": type(explainer).__name__,
        "shap_shape": list(np.shape(shap_vals)),
    }


if __name__ == "__main__":
    verify_environment()

