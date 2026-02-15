#!/usr/bin/env python3
"""
Train the Universal Technical Model using Phase 3 Feature Engine (112+ features).

Usage:
  python scripts/train_universal.py                       # Full training (ALL available stocks)
  python scripts/train_universal.py --n-stocks 500        # Medium training set
  python scripts/train_universal.py --n-stocks 200 --quick # Quick test (~2 min)

Process:
1. Load universe data from PolygonUniverseData
2. Build training panel using UniversalFeatureEngine (112 features)
3. Train LightGBM with Phase 3 tuned params + class weights
4. Evaluate with stricter quality gates (7 gates incl. IC consistency)
5. Print feature group importance breakdown
6. Test predictions on 20 diverse stocks
"""

import argparse
import gc
import logging
import os
import sys
import time
from datetime import date, timedelta
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description="Train universal technical model (Phase 3)")
    parser.add_argument("--n-stocks", type=int, default=10000,
                        help="Number of stocks to train on (default: all available)")
    parser.add_argument("--n-days", type=int, default=500,
                        help="Days of history per stock (default: 500)")
    parser.add_argument("--quick", action="store_true",
                        help="Quick mode: 200 stocks, relaxed thresholds")
    parser.add_argument("--model-dir", type=str, default="models/universal/",
                        help="Directory to save model")
    parser.add_argument("--data-dir", type=str, default="~/.nuble/universe_data/",
                        help="Universe data directory")
    parser.add_argument("--min-history", type=int, default=None,
                        help="Minimum bars per stock (default: 252, or 60 in quick mode)")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Enable verbose logging")
    args = parser.parse_args()

    if args.quick:
        args.n_stocks = min(args.n_stocks, 200)
        print("🚀 QUICK MODE: 200 stocks, relaxed thresholds")

    if args.verbose:
        logging.basicConfig(level=logging.INFO, format="%(name)s: %(message)s")
    else:
        logging.basicConfig(level=logging.WARNING)

    min_history = args.min_history
    if min_history is None:
        min_history = 60 if args.quick else 252

    print("=" * 65)
    print("NUBLE UNIVERSAL MODEL — PHASE 3 TRAINING (112+ features)")
    print("=" * 65)
    print(f"  Stocks:         {args.n_stocks}")
    print(f"  Days/stock:     {args.n_days}")
    print(f"  Min history:    {min_history} bars")
    print(f"  Feature engine: UniversalFeatureEngine (112 OHLCV features)")
    print(f"  Model dir:      {args.model_dir}")
    print(f"  Data dir:       {os.path.expanduser(args.data_dir)}")
    print()

    # ── Step 1: Check data availability ───────────────────────
    from nuble.data.polygon_universe import PolygonUniverseData
    pud = PolygonUniverseData(data_dir=args.data_dir)
    summary = pud.data_summary()

    if summary["status"] != "ready":
        print("❌ No universe data found.")
        print("   Run: python scripts/backfill_universe.py --quick")
        sys.exit(1)

    print(f"[Data] {summary['dates']} dates available, "
          f"{summary['date_range']}, {summary['size_mb']:.1f} MB")

    universe = pud.get_active_universe()
    if not universe:
        print("❌ No active stocks in universe data.")
        sys.exit(1)
    print(f"[Data] Active universe: {len(universe)} stocks")

    actual_n_stocks = min(args.n_stocks, len(universe))
    print(f"[Data] Using {actual_n_stocks} stocks for training")
    print()

    # ── Step 2: Build training panel with new feature engine ──
    from nuble.ml.universal_model import UniversalTechnicalModel, compute_label
    from nuble.ml.universal_features import UniversalFeatureEngine

    model = UniversalTechnicalModel(
        polygon_data=pud,
        model_root=args.model_dir,
    )

    engine = UniversalFeatureEngine()

    print("[Building training panel — 112+ features per stock]")
    panel_start = time.time()

    batch_size = 200
    selected = universe[:actual_n_stocks]
    all_features = []
    all_labels = []
    stocks_used = 0
    stocks_skipped = 0

    for batch_start in range(0, len(selected), batch_size):
        batch = selected[batch_start:batch_start + batch_size]
        batch_num = batch_start // batch_size + 1
        total_batches = (len(selected) + batch_size - 1) // batch_size
        print(f"  Processing batch {batch_num}/{total_batches} ({len(batch)} stocks)...")

        histories = pud.get_multi_stock_history(symbols=batch, days=args.n_days)
        local_count = len(histories)

        missing = [t for t in batch if t not in histories or len(histories.get(t, pd.DataFrame())) < min_history]
        if missing:
            api_count = len(missing)
            print(f"    Local: {local_count} stocks | API fetch: {api_count} stocks (~{api_count * 0.15:.0f}s)...")
            api_fetched = 0
            api_failed = 0
            for j, ticker in enumerate(missing):
                end_d = date.today()
                start_d = end_d - timedelta(days=int(args.n_days * 1.5))
                df = pud._fetch_single_ticker(ticker, start_d, end_d)
                if df is not None and len(df) >= min_history:
                    histories[ticker] = df.tail(args.n_days)
                    api_fetched += 1
                else:
                    api_failed += 1
                time.sleep(0.12)
                if (j + 1) % 50 == 0:
                    print(f"      API progress: {j+1}/{api_count} ({api_fetched} OK, {api_failed} failed)")
            print(f"    API complete: {api_fetched} fetched, {api_failed} failed")
        else:
            print(f"    All {local_count} stocks loaded from local data")

        for ticker, df in histories.items():
            if len(df) < min_history:
                stocks_skipped += 1
                continue

            try:
                # Phase 3: Use 112-feature engine
                features = engine.compute_features(df)
                labels = compute_label(df)

                # Drop warmup period
                features = features.iloc[engine.WARMUP_ROWS:]
                labels = labels.iloc[engine.WARMUP_ROWS:]

                common_idx = features.index.intersection(labels.dropna().index)
                if len(common_idx) < 50:
                    stocks_skipped += 1
                    continue

                feat = features.loc[common_idx]
                lab = labels.loc[common_idx].dropna()
                common = feat.index.intersection(lab.index)
                feat = feat.loc[common]
                lab = lab.loc[common]

                if len(feat) < 50:
                    stocks_skipped += 1
                    continue

                feat = feat.copy()
                feat["_ticker"] = ticker
                feat["_date"] = feat.index

                all_features.append(feat)
                all_labels.append(lab.astype(np.int32))
                stocks_used += 1

            except Exception as e:
                stocks_skipped += 1
                if args.verbose:
                    print(f"    Skip {ticker}: {e}")
                continue

        gc.collect()

    if not all_features:
        print("❌ No valid training data generated.")
        sys.exit(1)

    X = pd.concat(all_features, ignore_index=True)
    y = pd.concat(all_labels, ignore_index=True)

    assert len(X) == len(y), f"X/y length mismatch: {len(X)} vs {len(y)}"

    meta_cols = ["_ticker", "_date"]
    tickers_used = X["_ticker"].unique().tolist() if "_ticker" in X.columns else []
    X_dates = X["_date"].values if "_date" in X.columns else None
    X = X.drop(columns=[c for c in meta_cols if c in X.columns], errors="ignore")
    if X_dates is not None:
        X.index = pd.DatetimeIndex(X_dates)
        y.index = X.index

    # Drop bad features
    nan_pct = X.isna().mean()
    bad_feats = nan_pct[nan_pct > 0.30].index.tolist()
    if bad_feats:
        print(f"  ⚠️  Dropping {len(bad_feats)} features with >30% NaN: {bad_feats}")
        X = X.drop(columns=bad_feats)

    stds_check = X.std()
    dead_feats = stds_check[stds_check < 0.001].index.tolist()
    if dead_feats:
        print(f"  ⚠️  Dropping {len(dead_feats)} near-zero variance features: {dead_feats}")
        X = X.drop(columns=dead_feats)

    panel_elapsed = time.time() - panel_start

    label_dist = y.value_counts(normalize=True).sort_index()
    print(f"  Panel built: {len(X):,} samples × {X.shape[1]} features")
    print(f"  Stocks used: {stocks_used} (skipped: {stocks_skipped})")
    label_str = ", ".join([
        f"{'SHORT' if k==0 else 'NEUTRAL' if k==1 else 'LONG'}={v:.0%}"
        for k, v in label_dist.items()
    ])
    print(f"  Label distribution: {label_str}")
    if isinstance(X.index, pd.DatetimeIndex) or (hasattr(X.index, 'dtype') and np.issubdtype(X.index.dtype, np.datetime64)):
        sorted_dates = np.sort(X.index)
        print(f"  Date range: {pd.Timestamp(sorted_dates[0]).date()} → {pd.Timestamp(sorted_dates[-1]).date()}")
    print(f"  Memory usage: {X.memory_usage(deep=True).sum() / (1024*1024):.0f} MB")
    print(f"  Panel build time: {panel_elapsed:.1f}s")
    print()

    # ── Step 3: Train ─────────────────────────────────────────
    print("[Training LightGBM — Phase 3 params (112 features, class weights)]")
    train_start = time.time()

    metadata = model.train(X=X, y=y, build_data=False)

    train_elapsed = time.time() - train_start

    # ── Step 4: Evaluation Report ─────────────────────────────
    print()
    print("=" * 65)
    print("UNIVERSAL MODEL — PHASE 3 EVALUATION REPORT")
    print("=" * 65)
    print()

    gates = metadata.get("quality_gates", {})
    print("QUALITY GATES (Phase 3 — stricter):")
    gate_labels = {
        "ic_above_002": ("Test IC", f"{metadata.get('test_ic_mean', 0):.4f}", "> 0.02"),
        "ic_consistency_60pct": ("IC consistency", f"{metadata.get('test_ic_consistency', 0):.0%} of {metadata.get('test_ic_n_dates', 0)} dates", "> 60%"),
        "accuracy_above_42pct": ("Accuracy", f"{metadata.get('test_accuracy', 0):.1%}", "> 42%"),
        "no_feature_over_20pct": ("Max feature", f"{metadata.get('max_feature_importance_pct', 0):.1%}", "< 20%"),
        "calendar_under_10pct": ("Calendar features", f"{metadata.get('calendar_importance_pct', 0):.1f}%", "< 10%"),
        "top20_diversity_4_groups": ("Top 20 groups", f"{metadata.get('top20_group_diversity', 0)}", "≥ 4"),
        "uses_many_features": ("Features used", f"{metadata.get('n_features', 0)}", "> 50"),
    }
    for gate_name, passed in gates.items():
        icon = "✅" if passed else "❌"
        label_info = gate_labels.get(gate_name, (gate_name, "", ""))
        print(f"  {icon} {label_info[0]}: {label_info[1]} (threshold: {label_info[2]})")

    all_passed = metadata.get("quality_gates_passed", False)
    print()
    if all_passed:
        print("  ✅ ALL QUALITY GATES PASSED — Model saved!")
    else:
        failed = [k for k, v in gates.items() if not v]
        print(f"  ⚠️  Gates not all passed: {failed} — Model saved anyway")

    print()
    print("TEST SET METRICS:")
    print(f"  Mean IC: {metadata.get('test_ic_mean', 0):.4f}")
    print(f"  IC consistency: {metadata.get('test_ic_consistency', 0):.0%} positive ({metadata.get('test_ic_n_dates', 0)} dates)")
    print(f"  Accuracy: {metadata.get('test_accuracy', 0):.1%}")
    per_class = metadata.get("per_class_accuracy", {})
    if per_class:
        for cls_id, acc in sorted(per_class.items()):
            cls_name = {"0": "SHORT", "1": "NEUTRAL", "2": "LONG"}.get(str(cls_id), cls_id)
            print(f"  {cls_name} accuracy: {acc:.1%}")
    test_dist = metadata.get("test_class_distribution", {})
    if test_dist:
        total_test = sum(test_dist.values())
        for cls, cnt in sorted(test_dist.items()):
            cls_name = {"0": "SHORT", "1": "NEUTRAL", "2": "LONG"}.get(str(cls), cls)
            print(f"  {cls_name}: {cnt:,} ({cnt/total_test:.1%})")

    print()
    print("FEATURE GROUP IMPORTANCE:")
    group_imp = metadata.get("feature_group_importance", {})
    for group_name, pct in sorted(group_imp.items(), key=lambda x: x[1], reverse=True):
        bar = "█" * int(pct / 2)
        flag = " ← GOOD" if group_name == "Context" and pct < 10 else ""
        flag = " ← ⚠️ HIGH" if group_name == "Context" and pct >= 10 else flag
        print(f"  {group_name:20s} {pct:5.1f}% {bar}{flag}")

    print()
    print("TOP 15 FEATURES:")
    top_feats = metadata.get("top_20_features", [])[:15]
    total_imp = sum(f["importance"] for f in metadata.get("top_20_features", []))
    for i, feat in enumerate(top_feats, 1):
        pct = feat["importance"] / total_imp * 100 if total_imp > 0 else 0
        group = feat["feature"].split("_")[0]
        print(f"  {i:>2}. {feat['feature']:<35} {pct:5.1f}%  [{group}]")

    print()
    print("TRAINING DETAILS:")
    print(f"  Training samples:   {metadata.get('n_training_samples', 0):,}")
    print(f"  Validation samples: {metadata.get('n_validation_samples', 0):,}")
    print(f"  Test samples:       {metadata.get('n_test_samples', 0):,}")
    print(f"  Best iteration:     {metadata.get('best_iteration', 'N/A')}")
    print(f"  Training time:      {train_elapsed:.0f}s ({train_elapsed/60:.1f} min)")
    cw = metadata.get("class_weights_used", {})
    if cw:
        print(f"  Class weights:      SHORT={cw.get('0','?')}, NEUTRAL={cw.get('1','?')}, LONG={cw.get('2','?')}")
    print()

    # ── Step 5: Prediction diversity test ─────────────────────
    print("PREDICTION DIVERSITY TEST (20 stocks):")
    print("-" * 65)
    test_symbols = [
        "NVDA", "AAPL", "MSFT", "AMZN", "GOOGL",
        "JNJ", "JPM", "XOM", "UNH", "PG",
        "MA", "HD", "COST", "ABBV", "MRK",
        "PEP", "TMO", "LIN", "AVGO", "ORCL",
    ]
    predictions = {}
    direction_counts = {"SHORT": 0, "NEUTRAL": 0, "LONG": 0}
    for symbol in test_symbols:
        pred = model.predict(symbol)
        if pred and pred.get("confidence", 0) > 0:
            direction = pred["direction"]
            conf = pred["confidence"]
            top = pred.get("explanation", {}).get("top_features", [])[:3]
            feat_str = ", ".join([f"{f['feature']}={f['value']:.2f}" for f in top])
            print(f"  {symbol:>5}: {direction:<8} ({conf:.0%} conf) | {feat_str}")
            predictions[symbol] = direction
            direction_counts[direction] = direction_counts.get(direction, 0) + 1
        else:
            print(f"  {symbol:>5}: No prediction")

    print()
    print("PREDICTION DIVERSITY:")
    for d, cnt in sorted(direction_counts.items()):
        print(f"  {d}: {cnt} stocks")
    n_directions = sum(1 for v in direction_counts.values() if v > 0)
    diversity = f"{'✅ DIVERSE' if n_directions >= 2 else '⚠️ UNIFORM'}"
    print(f"  Distinct directions: {n_directions} {diversity}")
    print()

    # ── Summary ───────────────────────────────────────────────
    print("=" * 65)
    model_path = Path(args.model_dir)
    if (model_path / "universal_technical_model.txt").exists():
        model_size = (model_path / "universal_technical_model.txt").stat().st_size
        print(f"Model file:    {model_path / 'universal_technical_model.txt'}")
        print(f"Model size:    {model_size / 1024:.0f} KB")
    print(f"Features:      {metadata.get('n_features', 0)} (was 27 in Phase 2)")
    print(f"Stocks used:   {stocks_used}")
    print(f"Total samples: {len(X):,}")
    print(f"Total time:    {(time.time() - panel_start)/60:.1f} min")
    print(f"Engine:        UniversalFeatureEngine v3")
    print("=" * 65)


if __name__ == "__main__":
    main()
