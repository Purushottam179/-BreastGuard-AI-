"""
BreastGuard AI — Root Convenience Runner
Launches the Streamlit application or test suite with validated environment variables.
"""
from __future__ import annotations

import os
import sys
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

# Ensure project root is in sys.path
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Ensure Matplotlib uses local cache directory
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".matplotlib_cache"))
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)


def run_streamlit() -> None:
    """Launch the Streamlit interactive dashboard."""
    app_entry = PROJECT_ROOT / "app" / "main.py"
    if not app_entry.exists():
        print(f"Error: Application entry point not found at {app_entry}")
        sys.exit(1)

    print("=" * 70)
    print("🩺 Starting BreastGuard AI — Interactive Decision Dashboard")
    print(f"📁 Root: {PROJECT_ROOT}")
    print(f"🐍 Python: {sys.version.split()[0]}")
    print("🌐 Launching Streamlit at http://localhost:8501 ...")
    print("=" * 70)

    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(app_entry),
        "--server.headless",
        "true",
        "--browser.gatherUsageStats",
        "false",
    ]
    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\nApplication stopped by user.")


def run_tests() -> None:
    """Execute the automated test suite."""
    print("=" * 70)
    print("🧪 Running BreastGuard AI Test Suite (pytest) ...")
    print("=" * 70)
    cmd = [sys.executable, "-m", "pytest", "tests/", "-v", "--tb=short"]
    subprocess.run(cmd)


def run_audit() -> None:
    """Execute the comprehensive technical audit."""
    print("=" * 70)
    print("🔬 Running BreastGuard AI Technical Audit ...")
    print("=" * 70)
    cmd = [sys.executable, str(PROJECT_ROOT / "scripts" / "audit_codebase.py")]
    subprocess.run(cmd)


def run_verify() -> None:
    """Execute the Streamlit dashboard and artifact health check."""
    print("=" * 70)
    print("🩺 Running BreastGuard AI Dashboard Health Check ...")
    print("=" * 70)
    cmd = [sys.executable, str(PROJECT_ROOT / "scripts" / "verify_dashboard_app.py")]
    subprocess.run(cmd)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        run_tests()
    elif len(sys.argv) > 1 and sys.argv[1] == "audit":
        run_audit()
    elif len(sys.argv) > 1 and sys.argv[1] == "verify":
        run_verify()
    else:
        run_streamlit()

