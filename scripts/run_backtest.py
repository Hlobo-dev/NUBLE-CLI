#!/usr/bin/env python3
"""
Run walk-forward backtest and signal analysis on the universal model.

Usage:
  python scripts/run_backtest.py                    # Full backtest
  python scripts/run_backtest.py --quick            # Quick validation (fewer windows)
  python scripts/run_backtest.py --signal-analysis  # Include signal decay + factor analysis
  python scripts/run_backtest.py --full             # Everything (backtest + all analysis)

Output:
  models/universal/backtest_results.json   — raw results
  models/universal/backtest_results.png    — charts
  models/universal/signal_analysis.json    — signal analysis
  Prints comprehensive report to stdout
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
# Quieten noisy loggers
logging.getLogger("lightgbm").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)


def main():
    parser = argparse.ArgumentParser(description="NUBLE Walk-Forward Backtest")
    parser.add_argument(
        "--quick", action="store_true",
        help="Quick mode: fewer retrain windows, fewer stocks",
    )
    parser.add_argument(
        "--signal-analysis", action="store_true",
        help="Run signal decay and factor exposure analysis",
    )
    parser.add_argument(
        "--full", action="store_true",
        help="Run everything (backtest + all signal analysis)",
    )
    parser.add_argument(
        "--model", type=str, default="universal",
        choices=["universal", "cross_sectional"],
        help="Model type: 'universal' (classification) or 'cross_sectional' (regression)",
    )
    parser.add_argument(
        "--n-stocks-train", type=int, default=500,
        help="Stocks to train on per window (default: 500)",
    )
    parser.add_argument(
        "--n-stocks-score", type=int, default=None,
        help="Stocks to score in test (default: all active)",
    )
    parser.add_argument(
        "--start-date", type=str, default=None,
        help="Start date for backtest (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--end-date", type=str, default=None,
        help="End date for backtest (YYYY-MM-DD)",
    )
    args = parser.parse_args()

    if args.full:
        args.signal_analysis = True

    # Configure for mode
    if args.quick:
        retrain_freq = 42
        min_train = 60
        test_window = 21
        if args.n_stocks_train == 500:
            args.n_stocks_train = 300
        if args.n_stocks_score is None:
            args.n_stocks_score = 800
    else:
        retrain_freq = 21
        min_train = 120
        test_window = 21

    # Cross-sectional model uses same universe for train/score
    is_regression = args.model == "cross_sectional"
    if is_regression:
        # Use same universe for both — fix the mismatch
        if args.n_stocks_score is None:
            args.n_stocks_score = args.n_stocks_train
        forward_horizon = 5  # Default regression horizon
    else:
        forward_horizon = 10  # Classification model uses 10-day triple-barrier

    # ── Initialize ────────────────────────────────────────────
    from nuble.data.polygon_universe import PolygonUniverseData
    from nuble.ml.universal_features import UniversalFeatureEngine
    from nuble.ml.backtest.walk_forward import WalkForwardBacktest

    polygon_data = PolygonUniverseData()
    feature_engine = UniversalFeatureEngine()

    # Check data
    data_summary = polygon_data.data_summary()
    print()
    print("=" * 65)
    print("NUBLE WALK-FORWARD BACKTEST")
    print("=" * 65)
    print(f"\nMode:            {'QUICK' if args.quick else 'FULL'}")
    print(f"Model:           {'CROSS-SECTIONAL REGRESSION' if is_regression else 'UNIVERSAL CLASSIFICATION'}")
    print(f"Data:            {data_summary.get('dates', 0)} dates, "
          f"{data_summary.get('size_mb', 0)} MB")
    print(f"Date range:      {data_summary.get('date_range', 'N/A')}")
    print(f"Train stocks:    {args.n_stocks_train}")
    print(f"Score stocks:    {args.n_stocks_score or 'all active'}")
    print(f"Retrain freq:    every {retrain_freq} trading days")
    print(f"Min train days:  {min_train}")
    print(f"Test window:     {test_window} days")
    print(f"Purge gap:       10 days")

    if data_summary.get("dates", 0) < min_train + 10 + test_window:
        print(f"\n❌ Not enough data. Need at least {min_train + 10 + test_window} dates, "
              f"have {data_summary.get('dates', 0)}")
        sys.exit(1)

    # ── Run backtest ──────────────────────────────────────────
    start_time = time.time()

    bt = WalkForwardBacktest(
        polygon_data=polygon_data,
        feature_engine=feature_engine,
        min_train_days=min_train,
        retrain_frequency=retrain_freq,
        purge_gap=10,
        test_window=test_window,
        n_stocks_train=args.n_stocks_train,
        n_stocks_score=args.n_stocks_score,
    )

    if is_regression:
        results = bt.run_regression(
            forward_horizon=forward_horizon,
            start_date=args.start_date,
            end_date=args.end_date,
            verbose=True,
        )
    else:
        results = bt.run(
            start_date=args.start_date,
            end_date=args.end_date,
            verbose=True,
        )

    elapsed = time.time() - start_time
    print(f"\nBacktest runtime: {elapsed:.0f}s ({elapsed/60:.1f} min)")

    # ── Print report ──────────────────────────────────────────
    results.print_report()

    # ── Save results ──────────────────────────────────────────
    results_dir = "models/cross_sectional" if is_regression else "models/universal"
    results_path = f"{results_dir}/backtest_results.json"
    os.makedirs(results_dir, exist_ok=True)
    summary = results.summary()
    summary["backtest_runtime_seconds"] = round(elapsed, 1)
    summary["mode"] = "quick" if args.quick else "full"
    summary["model_type"] = args.model
    summary["n_stocks_train"] = args.n_stocks_train
    summary["n_stocks_score"] = args.n_stocks_score
    summary["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")

    with open(results_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nResults saved: {results_path}")

    # ── Generate charts ───────────────────────────────────────
    chart_path = f"{results_dir}/backtest_results.png"
    try:
        results.plot_results(chart_path)
        print(f"Charts saved: {chart_path}")
    except Exception as e:
        print(f"Chart generation failed: {e}")

    # ── Signal analysis ───────────────────────────────────────
    if args.signal_analysis:
        from nuble.ml.backtest.signal_analysis import SignalAnalysis

        print()
        print("=" * 65)
        print("SIGNAL ANALYSIS")
        print("=" * 65)

        sa = SignalAnalysis()
        preds_df = results.get_predictions_df()
        preloaded = results.get_stock_histories()  # Reuse from backtest
        analysis_results = {}

        # Signal decay
        print("\n[Signal Decay] Computing IC at multiple horizons...")
        try:
            decay = sa.signal_decay(preds_df, polygon_data,
                                     preloaded_histories=preloaded)
            analysis_results["signal_decay"] = decay

            print(f"\nSIGNAL DECAY:")
            for h, ic in sorted(decay["horizon_ic"].items(), key=lambda x: x[0]):
                bar = "█" * int(max(0, ic) * 500)
                print(f"  {h:2d}-day IC: {ic:+.4f} {bar}")
            print(f"  Optimal horizon:  {decay['optimal_horizon']} days")
            print(f"  Half-life:        {decay['half_life_days']} days")
            print(f"  Short-lived:      {'⚠️ YES' if decay['is_short_lived'] else '✅ NO'}")
        except Exception as e:
            print(f"  Signal decay analysis failed: {e}")

        # Factor exposure
        print("\n[Factor Exposure] Decomposing predictions into known factors...")
        try:
            factors = sa.factor_exposure(preds_df, polygon_data,
                                          preloaded_histories=preloaded)
            analysis_results["factor_exposure"] = factors

            print(f"\nFACTOR EXPOSURE:")
            print(f"  Factor R²:         {factors['factor_r_squared']:.1%}")
            print(f"  Raw IC:            {factors['raw_ic']:.4f}")
            print(f"  Factor-neutral IC: {factors['factor_neutral_ic']:.4f}")
            print(f"  Alpha fraction:    {factors['alpha_fraction']:.1%}")
            print(f"  Dominant factor:   {factors['dominant_factor']}")
            print(f"  Interpretation:    {factors['interpretation']}")
            if factors["factor_loadings"]:
                print(f"\n  Factor Loadings:")
                for fn, fl in sorted(factors["factor_loadings"].items(),
                                      key=lambda x: -abs(x[1])):
                    bar = "+" * int(max(0, fl) * 100) + "-" * int(max(0, -fl) * 100)
                    print(f"    {fn:25s}: {fl:+.4f} {bar}")
        except Exception as e:
            print(f"  Factor analysis failed: {e}")

        # Turnover
        print("\n[Turnover] Analyzing portfolio changes...")
        try:
            daily_ports = results.get_daily_portfolios()
            turnover = sa.turnover_analysis(daily_ports)

            # Compute net return
            ls_monthly = summary.get("annualized_long_short_return", 0) / 12
            turnover["net_return_after_costs"] = round(
                ls_monthly - turnover["implied_transaction_cost"], 4
            )
            turnover["is_tradeable"] = turnover["net_return_after_costs"] > 0
            analysis_results["turnover"] = turnover

            print(f"\nTURNOVER:")
            print(f"  Daily turnover:    {turnover['mean_daily_turnover']:.1%}")
            print(f"  Monthly turnover:  {turnover['mean_monthly_turnover']:.1%}")
            print(f"  Implied cost:      {turnover['implied_transaction_cost']:.2%}/month")
            print(f"  Net return:        {turnover['net_return_after_costs']:.2%}/month (after costs)")
            print(f"  Tradeable:         {'✅ YES' if turnover['is_tradeable'] else '❌ NO'}")
        except Exception as e:
            print(f"  Turnover analysis failed: {e}")

        # Regime analysis
        print("\n[Regime] Splitting results by market regime...")
        try:
            regime = sa.regime_analysis(results, polygon_data)
            analysis_results["regime"] = regime

            print(f"\nREGIME ANALYSIS:")
            for r, data in regime.get("ic_by_regime", {}).items():
                ic_val = data.get("mean_ic", 0)
                n = data.get("n_days", 0)
                bar = "█" * int(max(0, ic_val) * 500)
                print(f"  {r:12s}: IC={ic_val:+.4f} ({n:3d} days) {bar}")

            for r, data in regime.get("ic_by_volatility", {}).items():
                ic_val = data.get("mean_ic", 0)
                n = data.get("n_days", 0)
                bar = "█" * int(max(0, ic_val) * 500)
                print(f"  {r:12s}: IC={ic_val:+.4f} ({n:3d} days) {bar}")

            print(f"  Worst regime:      {regime.get('worst_regime', 'unknown')}")
            print(f"  Best regime:       {regime.get('best_regime', 'unknown')}")
            print(f"  Regime robust:     "
                  f"{'✅ YES' if regime.get('is_regime_robust') else '❌ NO'}")
        except Exception as e:
            print(f"  Regime analysis failed: {e}")

        # Save signal analysis
        analysis_path = f"{results_dir}/signal_analysis.json"
        try:
            with open(analysis_path, "w") as f:
                json.dump(analysis_results, f, indent=2, default=str)
            print(f"\nSignal analysis saved: {analysis_path}")
        except Exception as e:
            print(f"Failed to save signal analysis: {e}")

    # ── Final summary ─────────────────────────────────────────
    print()
    print("=" * 65)
    print("SUMMARY")
    print("=" * 65)
    print(f"  Mean IC:          {summary.get('mean_ic', 0):.4f}")
    print(f"  IC IR:            {summary.get('ic_ir', 0):.2f}")
    print(f"  L/S Sharpe:       {summary.get('long_short_sharpe', 0):.2f}")
    print(f"  L/S Cumul Return: {summary.get('cumulative_long_short_return', 0):.2%}")
    print(f"  Max Drawdown:     {summary.get('max_drawdown', 0):.2%}")
    print(f"  Decile Mono:      {summary.get('decile_monotonicity', 0):.3f}")
    print(f"  IC Hit Rate:      {summary.get('ic_hit_rate', 0):.1%}")
    print(f"  Fwd Ret Horizon:  {summary.get('fwd_return_horizon', forward_horizon if is_regression else 10)} days")
    print(f"  Model Type:       {'Regression (Huber)' if is_regression else 'Classification (Multiclass)'}")
    print(f"  Total Runtime:    {elapsed:.0f}s")

    # Honest assessment
    ic = summary.get("mean_ic", 0)
    sharpe = summary.get("long_short_sharpe", 0)
    mono = summary.get("decile_monotonicity", 0)

    print(f"\n  VERDICT:")
    if ic > 0.03 and sharpe > 1.0 and mono > 0.7:
        print("  ✅ DEPLOYABLE — Institutional-quality signal")
    elif ic > 0.02 and sharpe > 0.5 and mono > 0.5:
        print("  ⚠️  PROMISING — Needs position sizing and risk management")
    elif ic > 0.01:
        print("  ⚠️  WEAK SIGNAL — May work with strong portfolio construction")
    else:
        print("  ❌ NOT READY — Model needs further development")

    print("=" * 65)


if __name__ == "__main__":
    main()
