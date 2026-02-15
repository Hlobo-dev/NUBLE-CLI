"""
Universal Data Fetcher — Polygon Grouped Daily Bars
=====================================================

Fetches and manages daily OHLCV data for the ENTIRE US stock universe
using Polygon's grouped daily bars endpoint: one API call per date
returns ALL stocks.

Storage: ~/.nuble/universe_data/ with monthly Parquet files.
Each file: columns [ticker, date, open, high, low, close, volume, vwap, transactions]
Total size estimate: ~500MB for 2 years (efficient with Parquet compression).

Author: NUBLE ML Pipeline — Phase 1 Institutional Upgrade
"""

from __future__ import annotations

import gc
import json
import logging
import os
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import requests

logger = logging.getLogger(__name__)

# ── US Market Holidays (2023-2026) ────────────────────────────

_FIXED_HOLIDAYS = {
    # 2023
    date(2023, 1, 2), date(2023, 1, 16), date(2023, 2, 20),
    date(2023, 4, 7), date(2023, 5, 29), date(2023, 6, 19),
    date(2023, 7, 4), date(2023, 9, 4), date(2023, 11, 23),
    date(2023, 12, 25),
    # 2024
    date(2024, 1, 1), date(2024, 1, 15), date(2024, 2, 19),
    date(2024, 3, 29), date(2024, 5, 27), date(2024, 6, 19),
    date(2024, 7, 4), date(2024, 9, 2), date(2024, 11, 28),
    date(2024, 12, 25),
    # 2025
    date(2025, 1, 1), date(2025, 1, 20), date(2025, 2, 17),
    date(2025, 4, 18), date(2025, 5, 26), date(2025, 6, 19),
    date(2025, 7, 4), date(2025, 9, 1), date(2025, 11, 27),
    date(2025, 12, 25),
    # 2026
    date(2026, 1, 1), date(2026, 1, 19), date(2026, 2, 16),
    date(2026, 4, 3), date(2026, 5, 25), date(2026, 6, 19),
    date(2026, 7, 3), date(2026, 9, 7), date(2026, 11, 26),
    date(2026, 12, 25),
}


def _is_trading_day(d: date) -> bool:
    """True if *d* is a weekday and not a US market holiday."""
    return d.weekday() < 5 and d not in _FIXED_HOLIDAYS


def _trading_days(start: date, end: date) -> List[date]:
    """Generate all trading days in [start, end]."""
    days = []
    current = start
    while current <= end:
        if _is_trading_day(current):
            days.append(current)
        current += timedelta(days=1)
    return days


# ══════════════════════════════════════════════════════════════
# PolygonUniverseData
# ══════════════════════════════════════════════════════════════

class PolygonUniverseData:
    """
    Fetches and manages daily OHLCV data for the ENTIRE US stock universe.

    KEY INSIGHT: Polygon's grouped daily bars endpoint returns ALL stocks for
    a single date in ONE API call. To build 2 years of history:
    500 trading days × 1 API call each = 500 calls.

    Storage: monthly Parquet files at ``~/.nuble/universe_data/``.
    """

    BASE_URL = "https://api.polygon.io"
    CHECKPOINT_FILE = ".checkpoint"

    def __init__(
        self,
        data_dir: str = "~/.nuble/universe_data/",
        api_key: str | None = None,
    ):
        self.data_dir = Path(os.path.expanduser(data_dir))
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.api_key = api_key or os.getenv("POLYGON_API_KEY", "JHKwAdyIOeExkYOxh3LwTopmqqVVFeBY")
        self._session = requests.Session()
        self._session.headers.update({"Accept": "application/json"})
        # Index of downloaded dates (loaded lazily)
        self._downloaded_dates: set[date] | None = None

    # ── Index management ──────────────────────────────────────

    def _load_index(self) -> set[date]:
        """Scan Parquet files to build an index of all downloaded dates."""
        if self._downloaded_dates is not None:
            return self._downloaded_dates

        dates: set[date] = set()
        for pf in self.data_dir.glob("*.parquet"):
            try:
                tbl = pq.read_table(pf, columns=["date"])
                date_col = tbl.column("date").to_pylist()
                for d in date_col:
                    if isinstance(d, datetime):
                        dates.add(d.date())
                    elif isinstance(d, date):
                        dates.add(d)
            except Exception:
                continue
        self._downloaded_dates = dates
        logger.info("Universe data index: %d dates loaded", len(dates))
        return dates

    def _parquet_path(self, d: date) -> Path:
        """Monthly Parquet file path for a given date."""
        return self.data_dir / f"{d.year}-{d.month:02d}.parquet"

    def _save_checkpoint(self, d: date) -> None:
        """Write last successfully downloaded date."""
        cp = self.data_dir / self.CHECKPOINT_FILE
        cp.write_text(d.isoformat())

    def _load_checkpoint(self) -> date | None:
        """Read last checkpoint date."""
        cp = self.data_dir / self.CHECKPOINT_FILE
        if cp.exists():
            try:
                return date.fromisoformat(cp.read_text().strip())
            except Exception:
                pass
        return None

    # ── API calls ─────────────────────────────────────────────

    def _fetch_grouped_bars(self, d: date, retries: int = 3) -> pd.DataFrame | None:
        """
        Fetch all US stock bars for a single date.

        Uses: GET /v2/aggs/grouped/locale/us/market/stocks/{date}
        """
        url = (
            f"{self.BASE_URL}/v2/aggs/grouped/locale/us/market/stocks/"
            f"{d.isoformat()}?adjusted=true&apiKey={self.api_key}"
        )

        for attempt in range(retries):
            try:
                resp = self._session.get(url, timeout=30)
                if resp.status_code == 429:
                    wait = 2 ** (attempt + 1)
                    logger.warning("Rate limited, waiting %ds", wait)
                    time.sleep(wait)
                    continue

                data = resp.json()
                results = data.get("results", [])
                if not results:
                    # Likely a holiday or future date — skip silently
                    return None

                df = pd.DataFrame(results)
                # Rename Polygon columns to standard names
                rename_map = {
                    "T": "ticker", "o": "open", "h": "high", "l": "low",
                    "c": "close", "v": "volume", "vw": "vwap", "n": "transactions",
                    "t": "timestamp",
                }
                df = df.rename(columns=rename_map)

                # Keep only needed columns
                keep = ["ticker", "open", "high", "low", "close", "volume",
                        "vwap", "transactions"]
                for col in keep:
                    if col not in df.columns:
                        df[col] = np.nan
                df = df[keep]

                # Add date column
                df["date"] = pd.Timestamp(d)

                # Filter junk
                df = df[
                    (df["close"] >= 1.0) &      # Exclude sub-$1 (pennies)
                    (df["volume"] > 0) &         # Must have traded
                    (df["ticker"].str.len() <= 5) &  # Exclude warrants/rights (long tickers)
                    (~df["ticker"].str.contains(r"[.\-/]", regex=True, na=False))  # Exclude class shares
                ]

                # Downcast to float32 to save memory/disk
                for col in ["open", "high", "low", "close", "volume", "vwap"]:
                    if col in df.columns:
                        df[col] = df[col].astype(np.float32)
                if "transactions" in df.columns:
                    df["transactions"] = df["transactions"].fillna(0).astype(np.int32)

                return df

            except requests.exceptions.RequestException as e:
                if attempt < retries - 1:
                    time.sleep(2 ** (attempt + 1))
                else:
                    logger.error("Failed to fetch grouped bars for %s: %s", d, e)
                    return None

        return None

    def _fetch_single_ticker(
        self, ticker: str, start: date, end: date
    ) -> pd.DataFrame | None:
        """Fallback: fetch OHLCV for a single ticker via aggs endpoint."""
        url = (
            f"{self.BASE_URL}/v2/aggs/ticker/{ticker}/range/1/day/"
            f"{start.isoformat()}/{end.isoformat()}"
            f"?adjusted=true&sort=asc&limit=5000&apiKey={self.api_key}"
        )
        try:
            resp = self._session.get(url, timeout=15)
            data = resp.json()
            results = data.get("results", [])
            if not results:
                return None
            df = pd.DataFrame(results)
            df["date"] = pd.to_datetime(df["t"], unit="ms")
            df = df.rename(columns={
                "o": "open", "h": "high", "l": "low",
                "c": "close", "v": "volume", "vw": "vwap", "n": "transactions",
            })
            df = df.set_index("date").sort_index()
            keep = ["open", "high", "low", "close", "volume"]
            for col in keep:
                if col not in df.columns:
                    df[col] = np.nan
            return df[keep].dropna()
        except Exception as e:
            logger.warning("Single-ticker fetch for %s failed: %s", ticker, e)
            return None

    # ── Append to monthly Parquet ─────────────────────────────

    def _append_to_parquet(self, df: pd.DataFrame, d: date) -> None:
        """Append a day's data to the monthly Parquet file."""
        path = self._parquet_path(d)
        table = pa.Table.from_pandas(df, preserve_index=False)

        if path.exists():
            existing = pq.read_table(path)
            # Unify schemas: cast new table to match existing schema
            try:
                table = table.cast(existing.schema)
            except (pa.ArrowInvalid, pa.ArrowNotImplementedError):
                # If cast fails, unify by casting both to a common schema
                unified = pa.unify_schemas([existing.schema, table.schema])
                existing = existing.cast(unified)
                table = table.cast(unified)
            table = pa.concat_tables([existing, table])

        pq.write_table(table, path, compression="snappy")

    # ── Public API ────────────────────────────────────────────

    def backfill(
        self,
        start_date: str | None = None,
        end_date: str | None = None,
        progress_callback: Callable[[str, int, int], None] | None = None,
    ) -> dict:
        """
        Download historical data for ALL US stocks.

        Default: last 2 years (500 trading days) if no start_date.
        Resumes from checkpoint if one exists.

        Returns: {dates_downloaded, dates_skipped, total_stocks, total_rows}
        """
        if end_date:
            end_d = date.fromisoformat(end_date)
        else:
            end_d = date.today() - timedelta(days=1)

        if start_date:
            start_d = date.fromisoformat(start_date)
        else:
            start_d = end_d - timedelta(days=730)  # ~2 years

        all_days = _trading_days(start_d, end_d)
        existing = self._load_index()

        # Resume from checkpoint
        checkpoint = self._load_checkpoint()
        missing_days = [d for d in all_days if d not in existing]
        if checkpoint:
            missing_days = [d for d in missing_days if d > checkpoint or d not in existing]

        total = len(missing_days)
        downloaded = 0
        skipped = 0
        total_stocks = 0
        total_rows = 0

        logger.info(
            "Backfill: %d missing dates out of %d trading days (%s → %s)",
            total, len(all_days), start_d, end_d,
        )

        for i, d in enumerate(missing_days):
            if progress_callback:
                progress_callback(d.isoformat(), i + 1, total)

            df = self._fetch_grouped_bars(d)
            if df is None or df.empty:
                skipped += 1
                self._save_checkpoint(d)
                continue

            self._append_to_parquet(df, d)
            n_stocks = df["ticker"].nunique()
            total_stocks = max(total_stocks, n_stocks)
            total_rows += len(df)
            downloaded += 1

            # Update index
            if self._downloaded_dates is not None:
                self._downloaded_dates.add(d)
            self._save_checkpoint(d)

            # Rate limiting — 250ms between calls (safe for Premium)
            time.sleep(0.25)

            if (i + 1) % 25 == 0:
                logger.info(
                    "Backfill progress: %d/%d dates, %d rows so far",
                    i + 1, total, total_rows,
                )
                gc.collect()

        summary = {
            "dates_downloaded": downloaded,
            "dates_skipped": skipped,
            "total_stocks": total_stocks,
            "total_rows": total_rows,
            "date_range": f"{start_d} → {end_d}",
        }
        logger.info("Backfill complete: %s", summary)
        return summary

    def get_stock_history(
        self, symbol: str, days: int = 500
    ) -> pd.DataFrame | None:
        """
        Get daily OHLCV for a single stock from the local data store.

        Falls back to single-stock Polygon API if not in local data.
        Returns None if < 252 bars available.
        """
        symbol = symbol.upper()
        frames = []

        # Read from all Parquet files
        for pf in sorted(self.data_dir.glob("*.parquet")):
            try:
                tbl = pq.read_table(
                    pf,
                    filters=[("ticker", "=", symbol)],
                    columns=["date", "open", "high", "low", "close", "volume", "vwap"],
                )
                if tbl.num_rows > 0:
                    frames.append(tbl.to_pandas())
            except Exception:
                continue

        if frames:
            df = pd.concat(frames, ignore_index=True)
            df["date"] = pd.to_datetime(df["date"])
            df = df.drop_duplicates(subset="date").sort_values("date")
            df = df.set_index("date")
            # Take most recent N days
            df = df.tail(days)
            if len(df) >= 252:
                return df[["open", "high", "low", "close", "volume"]]

        # Fallback to single-ticker API
        end_d = date.today()
        start_d = end_d - timedelta(days=int(days * 1.5))
        df = self._fetch_single_ticker(symbol, start_d, end_d)
        if df is not None and len(df) >= 252:
            return df.tail(days)

        return None

    def get_active_universe(
        self,
        target_date: str | None = None,
        min_price: float = 5.0,
        min_dollar_volume: float = 1_000_000,
        exclude_otc: bool = True,
    ) -> List[str]:
        """
        Get list of active, liquid stocks for a given date.

        Filters: price >= $5, dollar vol >= $1M, ticker length <= 4 (listed).
        Returns typically 3,000–5,000 tickers.
        """
        if target_date:
            target_d = date.fromisoformat(target_date)
        else:
            target_d = date.today() - timedelta(days=1)

        # Find the closest available date
        existing = sorted(self._load_index())
        if not existing:
            logger.warning("No universe data available")
            return []

        closest = min(existing, key=lambda d: abs((d - target_d).days))

        # Read that date's data
        path = self._parquet_path(closest)
        if not path.exists():
            return []

        try:
            tbl = pq.read_table(
                path,
                filters=[("date", "=", pd.Timestamp(closest))],
            )
            df = tbl.to_pandas()
        except Exception:
            return []

        if df.empty:
            return []

        # Apply institutional filters
        df["dollar_volume"] = df["close"] * df["volume"]
        filtered = df[
            (df["close"] >= min_price) &
            (df["dollar_volume"] >= min_dollar_volume) &
            (df["volume"] > 0)
        ]

        if exclude_otc:
            # Tickers > 4 chars are usually OTC or warrants
            filtered = filtered[filtered["ticker"].str.len() <= 4]

        tickers = sorted(filtered["ticker"].unique().tolist())
        logger.info(
            "Active universe for %s: %d tickers (filtered from %d)",
            closest, len(tickers), len(df),
        )
        return tickers

    def get_universe_snapshot(self, target_date: str) -> pd.DataFrame:
        """
        Get ALL stocks' data for a single date.
        Returns DataFrame: index=ticker, columns=[open, high, low, close, volume, vwap].
        """
        d = date.fromisoformat(target_date)
        path = self._parquet_path(d)
        if not path.exists():
            return pd.DataFrame()

        try:
            tbl = pq.read_table(
                path,
                filters=[("date", "=", pd.Timestamp(d))],
            )
            df = tbl.to_pandas()
            if df.empty:
                return pd.DataFrame()
            return df.set_index("ticker")[["open", "high", "low", "close", "volume", "vwap"]]
        except Exception:
            return pd.DataFrame()

    def get_multi_stock_history(
        self,
        symbols: List[str] | None = None,
        days: int = 500,
    ) -> Dict[str, pd.DataFrame]:
        """
        Get history for multiple stocks at once.
        If symbols is None, returns ALL stocks.
        Memory efficient: loads from Parquet with column pruning.
        """
        frames: Dict[str, list] = {}

        for pf in sorted(self.data_dir.glob("*.parquet")):
            try:
                filters = None
                if symbols:
                    filters = [("ticker", "in", [s.upper() for s in symbols])]
                tbl = pq.read_table(
                    pf,
                    filters=filters,
                    columns=["ticker", "date", "open", "high", "low", "close", "volume"],
                )
                df = tbl.to_pandas()
                for ticker, group in df.groupby("ticker"):
                    frames.setdefault(ticker, []).append(group)
            except Exception:
                continue

        result = {}
        for ticker, parts in frames.items():
            combined = pd.concat(parts, ignore_index=True)
            combined["date"] = pd.to_datetime(combined["date"])
            combined = combined.drop_duplicates(subset="date").sort_values("date")
            combined = combined.set_index("date").tail(days)
            combined = combined[["open", "high", "low", "close", "volume"]].dropna()
            if len(combined) >= 50:  # Minimum viable history
                result[ticker] = combined

        logger.info("Loaded history for %d stocks", len(result))
        return result

    def incremental_update(self) -> dict:
        """
        Download any missing recent dates (today and gaps).
        Called daily. Idempotent.
        """
        end_d = date.today() - timedelta(days=1)
        # Go back 5 days to catch any gaps
        start_d = end_d - timedelta(days=7)
        return self.backfill(
            start_date=start_d.isoformat(),
            end_date=end_d.isoformat(),
        )

    def data_summary(self) -> dict:
        """Summary statistics of locally stored data."""
        existing = sorted(self._load_index())
        if not existing:
            return {"status": "empty", "dates": 0}

        total_size = sum(
            f.stat().st_size for f in self.data_dir.glob("*.parquet")
        )
        return {
            "status": "ready",
            "dates": len(existing),
            "date_range": f"{existing[0]} → {existing[-1]}",
            "size_mb": round(total_size / (1024 * 1024), 1),
            "parquet_files": len(list(self.data_dir.glob("*.parquet"))),
        }
