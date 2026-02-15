#!/usr/bin/env python3
"""
PHASE 3 — MASTER PIPELINE ORCHESTRATOR
=========================================
Runs the complete institutional ML pipeline in sequence:

  Step 6b: Ensemble (LightGBM Huber + Ridge + ElasticNet)
  Step 6c: Deep Learning (FFN + ResNet) [optional, if torch available]
  Step 6d: Meta-Ensemble (IC-weighted + Stacked)
  Step 7:  Backtest (L/S portfolio, transaction costs, Sharpe)
  Step 8:  WRDS Advisor verification

Usage:
  python run_pipeline.py                  # Full pipeline
  python run_pipeline.py --skip-nn        # Skip neural nets (faster)
  python run_pipeline.py --from-step 7    # Resume from step 7
"""

import os
import sys
import time
import subprocess
import json
import argparse

BASE_DIR = "/Users/humbertolobo/Desktop/NUBLE-CLI"
PIPELINE_DIR = os.path.join(BASE_DIR, "wrds_pipeline/phase3")
DATA_DIR = os.path.join(BASE_DIR, "data/wrds")
RESULTS_DIR = os.path.join(PIPELINE_DIR, "results")
PYTHON = os.path.join(BASE_DIR, ".venv/bin/python")


def run_step(script_name: str, step_label: str) -> bool:
    """Run a pipeline step and return True on success."""
    script_path = os.path.join(PIPELINE_DIR, script_name)
    if not os.path.exists(script_path):
        print(f"  ❌ Script not found: {script_path}")
        return False

    print(f"\n{'━' * 70}")
    print(f"▶ {step_label}")
    print(f"{'━' * 70}")

    t0 = time.time()
    result = subprocess.run(
        [PYTHON, script_path],
        cwd=PIPELINE_DIR,
        env={**os.environ, "PYTHONPATH": PIPELINE_DIR},
    )
    elapsed = time.time() - t0

    if result.returncode != 0:
        print(f"  ❌ {step_label} FAILED (exit code {result.returncode})")
        return False
    else:
        print(f"  ✅ {step_label} completed in {elapsed/60:.1f} min")
        return True


def verify_outputs():
    """Check that all expected output files exist."""
    expected = [
        ("data/wrds/lgb_predictions.parquet", "ML Predictions"),
        ("wrds_pipeline/phase3/results/walk_forward_metrics.csv", "Walk-Forward Metrics"),
        ("wrds_pipeline/phase3/results/feature_importance.csv", "Feature Importance"),
        ("wrds_pipeline/phase3/results/model_summary.json", "Model Summary"),
    ]

    print(f"\n{'━' * 70}")
    print(f"📋 OUTPUT VERIFICATION")
    print(f"{'━' * 70}")

    all_ok = True
    for rel_path, label in expected:
        full_path = os.path.join(BASE_DIR, rel_path)
        if os.path.exists(full_path):
            size_mb = os.path.getsize(full_path) / (1024 * 1024)
            print(f"  ✅ {label}: {size_mb:.1f} MB")
        else:
            print(f"  ❌ {label}: MISSING ({rel_path})")
            all_ok = False

    return all_ok


def main():
    parser = argparse.ArgumentParser(description="Phase 3 Pipeline Orchestrator")
    parser.add_argument("--skip-nn", action="store_true",
                        help="Skip neural network step (faster)")
    parser.add_argument("--from-step", type=str, default="6b",
                        help="Resume from step (6b, 6c, 6d, 7)")
    args = parser.parse_args()

    print("=" * 70)
    print("PHASE 3 — MASTER PIPELINE ORCHESTRATOR")
    print("=" * 70)
    t_start = time.time()

    os.makedirs(RESULTS_DIR, exist_ok=True)

    steps = [
        ("6b", "step6b_ensemble.py", "STEP 6B: Institutional ML Ensemble"),
        ("6c", "step6c_neural.py", "STEP 6C: Deep Learning Predictor"),
        ("6d", "step6d_meta_ensemble.py", "STEP 6D: Meta-Ensemble"),
        ("7", "step7_backtest.py", "STEP 7: Portfolio Backtest"),
    ]

    # Determine starting step
    start_idx = 0
    for i, (step_id, _, _) in enumerate(steps):
        if step_id == args.from_step:
            start_idx = i
            break

    for i, (step_id, script, label) in enumerate(steps):
        if i < start_idx:
            print(f"\n  ⏭ Skipping {label}")
            continue

        if step_id == "6c" and args.skip_nn:
            print(f"\n  ⏭ Skipping {label} (--skip-nn)")
            continue

        success = run_step(script, label)
        if not success:
            if step_id == "6c":
                print("  ⚠️ Neural nets failed (PyTorch issue?) — continuing...")
                continue
            else:
                print(f"\n  ❌ Pipeline halted at {label}")
                return

    # Verify
    verify_outputs()

    # Final summary
    elapsed = time.time() - t_start
    report_path = os.path.join(RESULTS_DIR, "final_report.json")
    if os.path.exists(report_path):
        with open(report_path) as f:
            report = json.load(f)
        h = report.get("headline", {})
        print(f"\n{'═' * 70}")
        print(f"PIPELINE COMPLETE — FINAL RESULTS")
        print(f"{'═' * 70}")
        print(f"  IC:          {h.get('ic', 'N/A')}")
        print(f"  IC IR:       {h.get('ic_ir', 'N/A')}")
        print(f"  L/S Sharpe:  {h.get('ls_sharpe', 'N/A')}")
        print(f"  OOS R²:      {h.get('oos_r2_pct', 'N/A')}%")
        print(f"  Monotonic:   {h.get('monotonic', 'N/A')}")
        print(f"  Total time:  {elapsed / 60:.1f} min")
        print(f"{'═' * 70}")
    else:
        print(f"\n  Total time: {elapsed / 60:.1f} min")

    print("\n✅ Phase 3 ML pipeline complete. Ready for APEX integration.")


if __name__ == "__main__":
    main()
