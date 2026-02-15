#!/usr/bin/env python3
"""
Backfill daily OHLCV for ALL US stocks from Polygon grouped daily bars.

Uses: GET /v2/aggs/grouped/locale/us/market/stocks/{date}
One API call per date = ALL stocks for that date.
~500 trading days = ~500 API calls = ~30-60 min with rate limiting.

Usage:
  python scripts/backfill_universe.py                      # Full 2-year backfill
  python scripts/backfill_universe.py --days 60            # Last 60 trading days
  python scripts/backfill_universe.py --start 2025-01-01   # From specific date
  python scripts/backfill_universe.py --quick              # Last 30 days (test)

Features:
- Resume capability: skips dates already downloaded
- Progress with ETA
- Validates data quality after completion
- Summary statistics at the end
"""

import argparse
import gc
import os
import sys
import time
from datetime import date, timedelta
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description="Backfill US stock universe data")
    parser.add_argument("--days", type=int, default=None,
                        help="Number of calendar days to backfill (default: 730 = ~2 years)")
    parser.add_argument("--start", type=str, default=None,
                        help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, default=None,
                        help="End date (YYYY-MM-DD, default: yesterday)")
    parser.add_argument("--quick", action="store_true",
                        help="Quick mode: last 30 calendar days")
    parser.add_argument("--data-dir", type=str, default="~/.nuble/universe_data/",
                        help="Data storage directory")
    args = parser.parse_args()

    # Validate API key (use default Premium key if env var not set)
    _DEFAULT_KEY = "JHKwAdyIOeExkYOxh3LwTopmqqVVFeBY"
    api_key = os.getenv("POLYGON_API_KEY", _DEFAULT_KEY)
    if not api_key:
        print("❌ POLYGON_API_KEY not set. Cannot backfill.")
        sys.exit(1)

    # Determine date range
    end_date = args.end or (date.today() - timedelta(days=1)).isoformat()

    if args.quick:
        start_date = (date.today() - timedelta(days=30)).isoformat()
        print("🚀 QUICK MODE: Last 30 calendar days")
    elif args.days:
        start_date = (date.today() - timedelta(days=args.days)).isoformat()
    elif args.start:
        start_date = args.start
    else:
        start_date = (date.today() - timedelta(days=730)).isoformat()

    print("=" * 60)
    print("NUBLE UNIVERSE BACKFILL")
    print("=" * 60)
    print(f"Date range: {start_date} → {end_date}")
    print(f"Data dir:   {os.path.expanduser(args.data_dir)}")
    print(f"API key:    {api_key[:8]}...{api_key[-4:]}")
    print()

    # Initialize
    from nuble.data.polygon_universe import PolygonUniverseData
    pud = PolygonUniverseData(data_dir=args.data_dir, api_key=api_key)

    # Check existing data
    summary_before = pud.data_summary()
    if summary_before["status"] == "ready":
        print(f"Existing data: {summary_before['dates']} dates, "
              f"{summary_before['size_mb']:.1f} MB")
    else:
        print("No existing data — starting fresh")
    print()

    # Track progress
    start_time = time.time()
    last_print = 0

    def progress_callback(date_str: str, current: int, total: int):
        nonlocal last_print
        now = time.time()
        elapsed = now - start_time
        pct = current / total * 100

        # ETA calculation
        if current > 1:
            rate = elapsed / current  # seconds per date
            remaining = (total - current) * rate
            eta_min = remaining / 60
            eta_str = f"ETA: {eta_min:.0f} min" if eta_min > 1 else f"ETA: {remaining:.0f} sec"
        else:
            eta_str = "ETA: calculating..."

        # Print every date (they're ~250ms apart)
        if now - last_print >= 2.0 or current == 1 or current == total:
            print(f"  [{current:>4}/{total}] {date_str} | {pct:5.1f}% | {eta_str}")
            last_print = now

    print("Starting backfill...")
    print("-" * 60)

    result = pud.backfill(
        start_date=start_date,
        end_date=end_date,
        progress_callback=progress_callback,
    )

    elapsed = time.time() - start_time

    # Summary
    print()
    print("=" * 60)
    print("BACKFILL COMPLETE")
    print("=" * 60)
    print(f"  Dates downloaded:    {result['dates_downloaded']}")
    print(f"  Dates skipped:       {result['dates_skipped']}")
    print(f"  Total rows:          {result['total_rows']:,}")
    print(f"  Max stocks/day:      {result['total_stocks']:,}")
    print(f"  Date range:          {result['date_range']}")
    print(f"  Elapsed time:        {elapsed/60:.1f} min")

    # Post-backfill summary
    summary_after = pud.data_summary()
    if summary_after["status"] == "ready":
        print(f"  Total dates stored:  {summary_after['dates']}")
        print(f"  Storage used:        {summary_after['size_mb']:.1f} MB")
        print(f"  Parquet files:       {summary_after['parquet_files']}")

    # Active universe
    universe = pud.get_active_universe()
    if universe:
        print(f"  Active universe:     {len(universe)} stocks (>$5, >$1M vol)")
    print()

    # Spot-check validation
    if result["dates_downloaded"] > 0:
        print("VALIDATION: Spot-checking 5 random stocks...")
        print("-" * 60)
        import random
        if universe and len(universe) >= 5:
            test_tickers = random.sample(universe, 5)
        else:
            test_tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"]

        for ticker in test_tickers:
            df = pud.get_stock_history(ticker, days=500)
            if df is not None and not df.empty:
                print(f"  ✅ {ticker:>5}: {len(df)} bars | "
                      f"{df.index[0].date()} → {df.index[-1].date()} | "
                      f"${df['close'].iloc[-1]:.2f}")
            else:
                # Try single-ticker fallback
                from datetime import date as date_type
                end_d = date_type.today()
                start_d = end_d - timedelta(days=60)
                df2 = pud._fetch_single_ticker(ticker, start_d, end_d)
                if df2 is not None:
                    print(f"  ⚠️  {ticker:>5}: {len(df2)} bars (API fallback, not enough local data)")
                else:
                    print(f"  ❌ {ticker:>5}: No data available")

    print()
    print("=" * 60)
    print("Done! You can now train the universal model:")
    print("  python scripts/train_universal.py --quick")
    print("=" * 60)


if __name__ == "__main__":
    main()
