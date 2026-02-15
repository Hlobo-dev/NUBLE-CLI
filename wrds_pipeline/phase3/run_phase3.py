"""
PHASE 3 — MASTER RUNNER
========================
Execute all Phase 3 steps in the correct order.
Tier 1 (Data): Steps 0-4
Tier 2 (Model): Steps 5-8
Tier 3 (Production): Steps 9-15

Usage:
  python run_phase3.py          # Run ALL steps
  python run_phase3.py 1        # Run specific step
  python run_phase3.py 1 2 3    # Run specific steps
  python run_phase3.py tier1    # Run Tier 1 only
  python run_phase3.py tier2    # Run Tier 2 only
  python run_phase3.py tier3    # Run Tier 3 only
"""

import subprocess
import sys
import os
import time
from datetime import datetime

PHASE3_DIR = os.path.dirname(os.path.abspath(__file__))
PYTHON = sys.executable

STEPS = {
    0:  ("step0_s3_upload.py",         "S3 Data Lake Upload"),
    1:  ("step1_cz_predictors.py",     "Chen-Zimmermann Predictors"),
    2:  ("step2_welch_goyal.py",       "Welch-Goyal Macro"),
    3:  ("step3_fred_macro.py",        "FRED 35+ Macro Series"),
    4:  ("step4_optionmetrics.py",     "OptionMetrics (Attempt)"),
    5:  ("step5_gkx_panel.py",         "GKX Feature Panel (600-900 features)"),
    6:  ("step6_lightgbm.py",          "LightGBM Walk-Forward"),
    7:  ("step7_backtest.py",          "Backtesting Engine"),
    8:  ("step8_wrds_advisor.py",      "WRDSAdvisor Data Access"),
    9:  ("step9_apex_integration.py",  "APEX Integration"),
    10: ("step10_monthly_refresh.py",  "Monthly Refresh Pipeline"),
    11: ("step11_feature_store.py",    "Feature Store (DuckDB)"),
    12: ("step12_prediction_api.py",   "Prediction API (FastAPI)"),
    13: ("step13_multi_horizon.py",    "Multi-Horizon Ensemble"),
    14: ("step14_alt_data.py",         "Alternative Data Integration"),
    15: ("step15_monitoring.py",       "Model Monitoring"),
}

TIERS = {
    "tier1": [0, 1, 2, 3, 4],
    "tier2": [5, 6, 7, 8],
    "tier3": [9, 10, 11, 12, 13, 14, 15],
}


def run_step(step_num):
    """Run a single step."""
    if step_num not in STEPS:
        print(f"  ❌ Unknown step: {step_num}")
        return False

    script, description = STEPS[step_num]
    script_path = os.path.join(PHASE3_DIR, script)

    if not os.path.exists(script_path):
        print(f"  ❌ Script not found: {script_path}")
        return False

    print(f"\n{'━' * 70}")
    print(f"  STEP {step_num}: {description}")
    print(f"  Script: {script}")
    print(f"  Started: {datetime.now().strftime('%H:%M:%S')}")
    print(f"{'━' * 70}")

    t0 = time.time()
    result = subprocess.run(
        [PYTHON, script_path],
        cwd=os.path.dirname(PHASE3_DIR),
    )
    elapsed = time.time() - t0

    if result.returncode == 0:
        print(f"\n  ✅ Step {step_num} COMPLETE ({elapsed:.0f}s)")
        return True
    else:
        print(f"\n  ❌ Step {step_num} FAILED (exit code {result.returncode}, {elapsed:.0f}s)")
        return False


def main():
    print("=" * 70)
    print("PHASE 3 — MASTER RUNNER")
    print(f"  Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 70)

    # Parse arguments
    if len(sys.argv) > 1:
        arg = sys.argv[1].lower()
        if arg in TIERS:
            steps_to_run = TIERS[arg]
            print(f"  Running {arg.upper()}: Steps {steps_to_run}")
        else:
            steps_to_run = [int(a) for a in sys.argv[1:]]
            print(f"  Running steps: {steps_to_run}")
    else:
        steps_to_run = sorted(STEPS.keys())
        print(f"  Running ALL {len(steps_to_run)} steps")

    # Note: Step 12 (API server) runs as a background service
    # Skip it in batch mode unless explicitly requested
    if len(steps_to_run) > 1 and 12 in steps_to_run:
        print("  ⚠️ Step 12 (API server) skipped in batch mode — run separately")
        steps_to_run = [s for s in steps_to_run if s != 12]

    start = time.time()
    results = {}

    for step_num in steps_to_run:
        success = run_step(step_num)
        results[step_num] = success
        if not success and step_num < 5:
            # Tier 1 failures may prevent Tier 2
            print(f"\n  ⚠️ Step {step_num} failed — continuing with remaining steps...")

    elapsed = time.time() - start

    # Summary
    passed = sum(1 for v in results.values() if v)
    failed = sum(1 for v in results.values() if not v)

    print(f"\n{'=' * 70}")
    print(f"PHASE 3 EXECUTION SUMMARY")
    print(f"{'=' * 70}")
    for step_num, success in results.items():
        _, desc = STEPS[step_num]
        symbol = "✅" if success else "❌"
        print(f"  {symbol} Step {step_num:>2}: {desc}")
    print(f"\n  Passed: {passed}/{len(results)}")
    print(f"  Failed: {failed}/{len(results)}")
    print(f"  Time:   {elapsed/60:.1f} min")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
