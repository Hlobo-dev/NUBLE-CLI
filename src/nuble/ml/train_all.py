#!/usr/bin/env python3
"""
PROMPT F4 — train_all.py: Batch Training Script
=================================================

Trains models for one or more symbols using the full F1→F2→F3 pipeline.

Usage::

    # Train defaults (SPY, AAPL, TSLA, AMD, QQQ)
    python -m nuble.ml.train_all

    # Train specific symbols
    python -m nuble.ml.train_all --symbols SPY AAPL

    # Custom parameters
    python -m nuble.ml.train_all --symbols SPY --tp 2.5 --sl 1.5 \\
        --holding 15 --folds 7 --model-dir models/

    # Binary / meta-labeling mode
    python -m nuble.ml.train_all --symbols SPY --binary
"""

import argparse
import logging
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import List

import pandas as pd

# Ensure src/ is on the path
_project_root = Path(__file__).resolve().parents[3]  # src/nuble/ml → project root
_src_dir = _project_root / "src"
if str(_src_dir) not in sys.path:
    sys.path.insert(0, str(_src_dir))

from nuble.ml.trainer_v2 import TrainingPipeline, TrainingResults

logger = logging.getLogger("nuble.ml.train_all")

# ── Defaults ──────────────────────────────────────────────────
DEFAULT_SYMBOLS = ["SPY", "AAPL", "TSLA", "AMD", "QQQ"]
DEFAULT_TP = 2.0
DEFAULT_SL = 2.0
DEFAULT_HOLDING = 10
DEFAULT_FOLDS = 5
DEFAULT_MODEL_DIR = "models/"


# ═══════════════════════════════════════════════════════════════
# Data Fetching
# ═══════════════════════════════════════════════════════════════

def fetch_training_data(
    symbol: str,
    days: int = 756,  # ~3 years
) -> pd.DataFrame:
    """
    Fetch OHLCV data from Polygon for training.

    Falls back to cached data or a synthetic stub if the API is unavailable.
    """
    api_key = os.getenv("POLYGON_API_KEY", "")
    if not api_key:
        logger.warning("POLYGON_API_KEY not set — trying cache / synthetic fallback.")
        return _load_cached_or_synthetic(symbol, days)

    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")

    url = (
        f"https://api.polygon.io/v2/aggs/ticker/{symbol}/range/1/day/"
        f"{start_date}/{end_date}?adjusted=true&sort=asc&limit=5000"
        f"&apiKey={api_key}"
    )

    try:
        import requests

        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        results = data.get("results", [])
        if not results:
            logger.warning("Polygon returned 0 bars for %s — using fallback.", symbol)
            return _load_cached_or_synthetic(symbol, days)

        df = pd.DataFrame(results)
        df["date"] = pd.to_datetime(df["t"], unit="ms")
        df = df.set_index("date").sort_index()
        df = df.rename(columns={"o": "open", "h": "high", "l": "low", "c": "close", "v": "volume"})
        df = df[["open", "high", "low", "close", "volume"]]
        df = df.dropna()

        logger.info("Fetched %d bars for %s (%s → %s)", len(df), symbol, df.index[0].date(), df.index[-1].date())
        return df

    except Exception as exc:
        logger.warning("Polygon fetch failed for %s: %s — using fallback.", symbol, exc)
        return _load_cached_or_synthetic(symbol, days)


def _load_cached_or_synthetic(symbol: str, days: int) -> pd.DataFrame:
    """Load cached JSON or generate synthetic data for offline training."""
    # Check data_cache/
    cache_dir = _project_root / "data_cache"
    if cache_dir.exists():
        for f in sorted(cache_dir.iterdir(), reverse=True):
            if f.name.startswith(symbol) and f.suffix == ".json":
                try:
                    import json

                    with open(f) as fh:
                        data = json.load(fh)
                    results = data if isinstance(data, list) else data.get("results", data.get("data", []))
                    if results:
                        df = pd.DataFrame(results)
                        if "t" in df.columns:
                            df["date"] = pd.to_datetime(df["t"], unit="ms")
                        elif "date" in df.columns:
                            df["date"] = pd.to_datetime(df["date"])
                        df = df.set_index("date").sort_index()
                        rename = {}
                        for src, dst in [("o", "open"), ("h", "high"), ("l", "low"), ("c", "close"), ("v", "volume")]:
                            if src in df.columns:
                                rename[src] = dst
                        if rename:
                            df = df.rename(columns=rename)
                        df = df[["open", "high", "low", "close", "volume"]].dropna()
                        if len(df) >= 100:
                            logger.info("Loaded %d cached bars for %s from %s", len(df), symbol, f.name)
                            return df
                except Exception:
                    continue

    # Synthetic fallback
    logger.warning("Generating synthetic data for %s (offline mode)", symbol)
    import numpy as np

    np.random.seed(hash(symbol) % 2**31)
    n = max(days, 500)
    dates = pd.bdate_range(end=datetime.now(), periods=n)
    actual_n = len(dates)
    close = 100.0 + np.cumsum(np.random.randn(actual_n) * 1.5)
    close = close - close.min() + 50  # keep positive
    df = pd.DataFrame(
        {
            "open": close + np.random.randn(actual_n) * 0.3,
            "high": close + abs(np.random.randn(actual_n) * 1.2),
            "low": close - abs(np.random.randn(actual_n) * 1.2),
            "close": close,
            "volume": (np.random.lognormal(15, 1, actual_n)).astype(int),
        },
        index=dates,
    )
    df.index.name = "date"
    return df


# ═══════════════════════════════════════════════════════════════
# Training Loop
# ═══════════════════════════════════════════════════════════════

def train_symbol(
    symbol: str,
    tp_multiplier: float = DEFAULT_TP,
    sl_multiplier: float = DEFAULT_SL,
    max_holding_period: int = DEFAULT_HOLDING,
    n_cv_splits: int = DEFAULT_FOLDS,
    binary_mode: bool = False,
    model_dir: str = DEFAULT_MODEL_DIR,
    days: int = 756,
) -> TrainingResults:
    """
    Train a model for a single symbol.

    Returns TrainingResults.
    """
    logger.info("=" * 60)
    logger.info("TRAINING: %s", symbol)
    logger.info("=" * 60)

    # 1. Fetch data
    df = fetch_training_data(symbol, days=days)
    if df.empty or len(df) < 200:
        logger.error("Insufficient data for %s (%d rows). Skipping.", symbol, len(df))
        return TrainingResults(symbol)

    logger.info("Data: %d bars, %s → %s", len(df), df.index[0].date(), df.index[-1].date())

    # 2. Create and run pipeline
    pipeline = TrainingPipeline(
        symbol=symbol,
        tp_multiplier=tp_multiplier,
        sl_multiplier=sl_multiplier,
        max_holding_period=max_holding_period,
        n_cv_splits=n_cv_splits,
        binary_mode=binary_mode,
        model_dir=model_dir,
    )

    t0 = time.time()
    results = pipeline.run(df)
    elapsed = time.time() - t0

    # 3. Report
    logger.info("-" * 40)
    logger.info("RESULTS for %s (%.1fs):", symbol, elapsed)
    if results.aggregate_metrics:
        agg = results.aggregate_metrics
        logger.info("  Mean IC:        %.4f", agg.get("mean_ic", 0))
        logger.info("  Mean Hit Rate:  %.1f%%", agg.get("mean_hit_rate", 0) * 100)
        logger.info("  Mean PF:        %.2f", agg.get("mean_profit_factor", 0))
        logger.info("  Folds OK:       %d / %d", agg.get("n_folds", 0), n_cv_splits)
    else:
        logger.warning("  No aggregate results (all folds may have failed).")

    return results


def train_all(
    symbols: List[str],
    tp_multiplier: float = DEFAULT_TP,
    sl_multiplier: float = DEFAULT_SL,
    max_holding_period: int = DEFAULT_HOLDING,
    n_cv_splits: int = DEFAULT_FOLDS,
    binary_mode: bool = False,
    model_dir: str = DEFAULT_MODEL_DIR,
    days: int = 756,
) -> dict:
    """
    Train models for all symbols.

    Returns {symbol: TrainingResults}.
    """
    all_results = {}
    start = time.time()

    for sym in symbols:
        try:
            results = train_symbol(
                sym,
                tp_multiplier=tp_multiplier,
                sl_multiplier=sl_multiplier,
                max_holding_period=max_holding_period,
                n_cv_splits=n_cv_splits,
                binary_mode=binary_mode,
                model_dir=model_dir,
                days=days,
            )
            all_results[sym] = results
        except Exception as exc:
            logger.error("Training failed for %s: %s", sym, exc, exc_info=True)
            all_results[sym] = TrainingResults(sym)

    total = time.time() - start
    logger.info("=" * 60)
    logger.info("BATCH TRAINING COMPLETE — %d symbols in %.1fs", len(symbols), total)
    logger.info("=" * 60)

    # Summary table
    for sym, res in all_results.items():
        agg = res.aggregate_metrics or {}
        ic = agg.get("mean_ic", 0)
        hr = agg.get("mean_hit_rate", 0)
        pf = agg.get("mean_profit_factor", 0)
        folds = agg.get("n_folds", 0)
        status = "✅" if folds > 0 and ic > 0 else "❌"
        logger.info("  %s %s: IC=%.4f HR=%.1f%% PF=%.2f folds=%d", status, sym, ic, hr * 100, pf, folds)

    return all_results


# ═══════════════════════════════════════════════════════════════
# CLI Entry Point
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="NUBLE ML — Batch Model Training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m nuble.ml.train_all
  python -m nuble.ml.train_all --symbols SPY AAPL TSLA
  python -m nuble.ml.train_all --symbols SPY --tp 2.5 --sl 1.5 --folds 7
  python -m nuble.ml.train_all --symbols SPY --binary
        """,
    )
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=DEFAULT_SYMBOLS,
        help=f"Symbols to train (default: {' '.join(DEFAULT_SYMBOLS)})",
    )
    parser.add_argument("--tp", type=float, default=DEFAULT_TP, help="Take-profit multiplier (default: 2.0)")
    parser.add_argument("--sl", type=float, default=DEFAULT_SL, help="Stop-loss multiplier (default: 2.0)")
    parser.add_argument("--holding", type=int, default=DEFAULT_HOLDING, help="Max holding period in days (default: 10)")
    parser.add_argument("--folds", type=int, default=DEFAULT_FOLDS, help="Number of CV folds (default: 5)")
    parser.add_argument("--binary", action="store_true", help="Use binary meta-labeling mode")
    parser.add_argument("--model-dir", default=DEFAULT_MODEL_DIR, help="Model save directory (default: models/)")
    parser.add_argument("--days", type=int, default=756, help="Days of history to fetch (default: 756 ≈ 3 years)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable DEBUG logging")

    args = parser.parse_args()

    # Configure logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    train_all(
        symbols=args.symbols,
        tp_multiplier=args.tp,
        sl_multiplier=args.sl,
        max_holding_period=args.holding,
        n_cv_splits=args.folds,
        binary_mode=args.binary,
        model_dir=args.model_dir,
        days=args.days,
    )


if __name__ == "__main__":
    main()
