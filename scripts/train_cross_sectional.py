#!/usr/bin/env python3
"""
Train the cross-sectional regression model.

Usage:
  python scripts/train_cross_sectional.py --quick              # 500 stocks, pipeline test
  python scripts/train_cross_sectional.py                       # All stocks, full training
  python scripts/train_cross_sectional.py --horizon 5           # 5-day forward returns
  python scripts/train_cross_sectional.py --horizon 10          # 10-day forward returns

Author: NUBLE ML Pipeline — Phase 6 Model Rebuild
"""

import argparse
import json
import logging
import os
import sys
import time

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(name)s: %(message)s",
)
logging.getLogger("lightgbm").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)


def main():
    parser = argparse.ArgumentParser(
        description="Train NUBLE Cross-Sectional Regression Model (Phase 6)"
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Quick mode: 500 stocks, faster training",
    )
    parser.add_argument(
        "--n-stocks", type=int, default=10000,
        help="Number of stocks (default: all available)",
    )
    parser.add_argument(
        "--horizon", type=int, default=5,
        help="Forward return horizon in trading days (default: 5)",
    )
    parser.add_argument(
        "--model-dir", type=str, default="models/cross_sectional/",
        help="Model save directory",
    )
    args = parser.parse_args()

    if args.quick:
        args.n_stocks = min(args.n_stocks, 500)
        print("🚀 QUICK MODE: 500 stocks")

    print()
    print("=" * 65)
    print("NUBLE CROSS-SECTIONAL REGRESSION MODEL — PHASE 6")
    print("=" * 65)
    print(f"  Stocks:           {args.n_stocks}")
    print(f"  Forward horizon:  {args.horizon} days")
    print(f"  Objective:        Huber regression on excess returns")
    print(f"  Normalization:    Cross-sectional rank per date")
    print(f"  Model dir:        {args.model_dir}")
    print()

    # ── Check data ────────────────────────────────────────────
    from nuble.data.polygon_universe import PolygonUniverseData

    polygon_data = PolygonUniverseData()
    data_summary = polygon_data.data_summary()

    if data_summary.get("status") != "ready":
        print("❌ No universe data found.")
        print("   Run: python scripts/backfill_universe.py --quick")
        sys.exit(1)

    print(f"[Data] {data_summary['dates']} dates, "
          f"{data_summary['date_range']}, {data_summary['size_mb']:.1f} MB")

    # ── Train ─────────────────────────────────────────────────
    from nuble.ml.cross_sectional_model import CrossSectionalModel

    start_time = time.time()

    model = CrossSectionalModel(
        polygon_data=polygon_data,
        model_root=args.model_dir,
    )

    metadata = model.train(
        forward_horizon=args.horizon,
        n_stocks=args.n_stocks,
        verbose=True,
    )

    elapsed = time.time() - start_time

    # ── Test predictions on diverse stocks ────────────────────
    test_symbols = ["AAPL", "MSFT", "AMZN", "TSLA", "JPM",
                    "XOM", "JNJ", "PFE", "AMD", "NVDA",
                    "META", "GOOGL", "V", "UNH", "HD",
                    "WMT", "PG", "KO", "MCD", "DIS"]

    print(f"\n{'='*65}")
    print("PREDICTION TEST — 20 Diverse Stocks")
    print(f"{'='*65}")

    for sym in test_symbols:
        try:
            result = model.predict(sym)
            if result:
                exc_ret = result.get("predicted_excess_return", 0)
                direction = result.get("direction", "?")
                conf = result.get("confidence", 0)
                print(f"  {sym:6s}: {direction:7s} | excess={exc_ret:+.4%} | "
                      f"conf={conf:.2f}")
            else:
                print(f"  {sym:6s}: (no prediction)")
        except Exception as e:
            print(f"  {sym:6s}: error — {e}")

    # ── Compare with old model ────────────────────────────────
    old_results_path = "models/universal/backtest_results.json"
    if os.path.exists(old_results_path):
        with open(old_results_path) as f:
            old = json.load(f)

        print(f"\n{'='*65}")
        print("MODEL COMPARISON — Old vs New")
        print(f"{'='*65}")
        print(f"{'':20s}  {'Old (Classification)':>22s}  {'New (Regression)':>22s}")
        print(f"  {'Mean IC':20s}  {old.get('mean_ic', 0):>22.4f}  "
              f"{metadata.get('mean_ic', 0):>22.4f}")
        print(f"  {'IC IR':20s}  {old.get('ic_ir', 0):>22.2f}  "
              f"{metadata.get('ic_ir', 0):>22.2f}")
        print(f"  {'L/S Sharpe':20s}  {old.get('long_short_sharpe', 0):>22.2f}  "
              f"{metadata.get('long_short_sharpe', 0):>22.2f}")
        print(f"  {'Decile Mono':20s}  {old.get('decile_monotonicity', 0):>22.3f}  "
              f"{metadata.get('decile_monotonicity', 0):>22.3f}")
        print(f"  {'D10-D1 Spread':20s}  "
              f"{'N/A':>22s}  "
              f"{metadata.get('d10_d1_spread', 0):>22.6f}")

    # ── Final summary ─────────────────────────────────────────
    print(f"\n{'='*65}")
    print("SUMMARY")
    print(f"{'='*65}")
    print(f"  Model type:       Cross-sectional regression (Huber)")
    print(f"  Forward horizon:  {args.horizon} days")
    print(f"  Training samples: {metadata.get('n_training_samples', 0):,}")
    print(f"  Features:         {metadata.get('n_features', 0)}")
    print(f"  Mean IC:          {metadata.get('mean_ic', 0):.4f}")
    print(f"  IC IR:            {metadata.get('ic_ir', 0):.2f}")
    print(f"  IC Hit Rate:      {metadata.get('ic_hit_rate', 0):.1%}")
    print(f"  L/S Sharpe:       {metadata.get('long_short_sharpe', 0):.2f}")
    print(f"  Decile Mono:      {metadata.get('decile_monotonicity', 0):.3f}")
    print(f"  D10-D1 Spread:    {metadata.get('d10_d1_spread', 0):.6f}")
    print(f"  Quality Gates:    {'✅ ALL PASSED' if metadata.get('quality_gates_passed') else '❌ NOT ALL PASSED'}")
    print(f"  Total time:       {elapsed:.0f}s ({elapsed/60:.1f} min)")

    # Verdict
    ic = metadata.get("mean_ic", 0)
    sharpe = metadata.get("long_short_sharpe", 0)
    mono = metadata.get("decile_monotonicity", 0)

    print(f"\n  VERDICT:")
    if ic > 0.03 and sharpe > 1.0 and mono > 0.7:
        print("  ✅ DEPLOYABLE — Institutional-quality signal")
    elif ic > 0.02 and sharpe > 0.5 and mono > 0.5:
        print("  ⚠️  PROMISING — Needs position sizing and risk management")
    elif ic > 0.01 and mono > 0.3:
        print("  ⚠️  WEAK SIGNAL — May work with strong portfolio construction")
    else:
        print("  ❌ NOT READY — Model needs further development")
    print("=" * 65)


if __name__ == "__main__":
    main()
