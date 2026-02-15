"""
PHASE 2 — MASTER RUNNER
========================
Orchestrates all Phase 2 steps in the correct order.

Usage:
    python wrds_pipeline/phase2_run_all.py              # Run everything
    python wrds_pipeline/phase2_run_all.py --step 1     # Run only step 1
    python wrds_pipeline/phase2_run_all.py --from 3     # Start from step 3

Execution order:
  Step 1: CRSP Daily Download (2-4 hours)
  Step 2: Rolling Betas (15-40 min)
  Step 3: Training Panel Build (5-15 min)
  Step 4: Daily Features (30-90 min)
  Step 5: Merge Daily + Winsorize (2-5 min)
  Step 6: Validation (2-5 min)
"""

import sys
import os
import time

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from wrds_pipeline.phase2_step1_crsp_daily import main as step1_crsp_daily
from wrds_pipeline.phase2_step2_rolling_betas import compute_rolling_betas as step2_betas
from wrds_pipeline.phase2_step3_training_panel import build_training_panel as step3_panel
from wrds_pipeline.phase2_step4_daily_features import main as step4_daily
from wrds_pipeline.phase2_step5_merge_daily import merge_daily_features as step5_merge
from wrds_pipeline.phase2_step6_validation import run_validation as step6_validate


STEPS = {
    1: ("CRSP Daily Download", step1_crsp_daily),
    2: ("Rolling Betas", step2_betas),
    3: ("Training Panel Build", step3_panel),
    4: ("Daily Features", step4_daily),
    5: ("Merge Daily + Winsorize", step5_merge),
    6: ("Validation", step6_validate),
}


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Phase 2 Master Runner")
    parser.add_argument("--step", type=int, help="Run only this step (1-6)")
    parser.add_argument("--from", dest="from_step", type=int, default=1,
                       help="Start from this step (default: 1)")
    parser.add_argument("--skip-daily", action="store_true",
                       help="Skip CRSP daily download (steps 1, 4, 5)")
    args = parser.parse_args()

    print("█" * 70)
    print("██  PHASE 2: WRDS DATA WAREHOUSE → ML TRAINING PANEL")
    print("██  Master Runner")
    print("█" * 70)

    overall_start = time.time()

    if args.step:
        steps_to_run = [args.step]
    elif args.skip_daily:
        steps_to_run = [2, 3, 6]
    else:
        steps_to_run = list(range(args.from_step, 7))

    print(f"\nSteps to run: {steps_to_run}")
    print()

    results = {}
    for step_num in steps_to_run:
        if step_num not in STEPS:
            print(f"⚠️  Unknown step {step_num}, skipping")
            continue

        name, func = STEPS[step_num]
        print(f"\n{'█'*70}")
        print(f"██  STEP {step_num}: {name}")
        print(f"{'█'*70}\n")

        step_start = time.time()
        try:
            func()
            step_time = time.time() - step_start
            results[step_num] = ("✅ PASS", step_time)
            print(f"\n  Step {step_num} completed in {step_time/60:.1f} minutes")
        except Exception as e:
            step_time = time.time() - step_start
            results[step_num] = (f"❌ FAIL: {str(e)[:60]}", step_time)
            print(f"\n  ❌ Step {step_num} FAILED after {step_time/60:.1f} minutes: {e}")
            import traceback
            traceback.print_exc()

            # Don't abort — continue with remaining steps if possible
            print(f"\n  Continuing with remaining steps...")

    # Summary
    total_time = time.time() - overall_start

    print(f"\n\n{'█'*70}")
    print(f"██  PHASE 2 EXECUTION SUMMARY")
    print(f"{'█'*70}")

    for step_num in steps_to_run:
        if step_num in results:
            status, elapsed = results[step_num]
            name = STEPS[step_num][0]
            print(f"  Step {step_num}: {name:<30} {status:<20} ({elapsed/60:.1f}m)")

    print(f"\n  Total time: {total_time/60:.1f} minutes ({total_time/3600:.1f} hours)")
    print(f"{'█'*70}")


if __name__ == "__main__":
    main()
