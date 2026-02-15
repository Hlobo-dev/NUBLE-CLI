"""
FRED Macro Data Pipeline — Institutional-Grade Macro Intelligence
==================================================================

Provides the 8 Gu-Kelly-Xiu macro variables plus derived regime indicators.

GRACEFUL DEGRADATION: If FRED_API_KEY is not set, all methods return None.
The system continues to work — it just lacks macro data.

Data Sources (all from FRED):
- DGS3MO: 3-Month Treasury Yield
- DGS10: 10-Year Treasury Yield
- BAMLC0A4CBBB: BBB Corporate Bond Yield
- BAMLC0A1CAAA: AAA Corporate Bond Yield
- T10YIE: 10-Year Breakeven Inflation
- INDPRO: Industrial Production Index
- UNRATE: Unemployment Rate
- FEDFUNDS: Effective Federal Funds Rate

Derived Indicators:
- term_spread: 10Y - 3M yield
- credit_spread: BBB - AAA yield
- industrial_production_yoy: 12-month % change
- real_rate: 10Y yield - breakeven inflation

Author: NUBLE ML Pipeline — Phase 1 Institutional Upgrade
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════
# FRED Series Configuration
# ══════════════════════════════════════════════════════════════

FRED_SERIES: Dict[str, Dict] = {
    "DGS3MO":       {"name": "treasury_3m",         "freq": "daily"},
    "DGS10":        {"name": "treasury_10y",         "freq": "daily"},
    "BAMLC0A4CBBB": {"name": "bbb_yield",           "freq": "daily"},
    "BAMLC0A1CAAA": {"name": "aaa_yield",           "freq": "daily"},
    "T10YIE":       {"name": "breakeven_inflation",  "freq": "daily"},
    "INDPRO":       {"name": "industrial_production","freq": "monthly"},
    "UNRATE":       {"name": "unemployment_rate",    "freq": "monthly"},
    "FEDFUNDS":     {"name": "fed_funds_rate",       "freq": "monthly"},
}


# ══════════════════════════════════════════════════════════════
# FREDMacroData
# ══════════════════════════════════════════════════════════════

class FREDMacroData:
    """
    Institutional-grade macro data pipeline from FRED.
    Gracefully degrades if FRED_API_KEY is not set.
    """

    def __init__(self, cache_path: str = "~/.nuble/macro_data.parquet"):
        self.cache_path = Path(os.path.expanduser(cache_path))
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)

        self.api_key = os.getenv("FRED_API_KEY", "")
        self._data: pd.DataFrame | None = None
        self._available = bool(self.api_key)

        if not self._available:
            logger.info(
                "FRED_API_KEY not set — macro data unavailable. "
                "Get a free key at https://fred.stlouisfed.org/docs/api/api_key.html"
            )

        # Load cached data if exists
        if self.cache_path.exists():
            try:
                self._data = pd.read_parquet(self.cache_path)
                logger.info(
                    "Loaded cached macro data: %d rows, %s → %s",
                    len(self._data),
                    self._data.index.min().date() if len(self._data) > 0 else "N/A",
                    self._data.index.max().date() if len(self._data) > 0 else "N/A",
                )
            except Exception as e:
                logger.warning("Failed to load cached macro data: %s", e)

    @property
    def is_available(self) -> bool:
        """Whether FRED data is available (either from cache or API)."""
        return self._data is not None or self._available

    # ── Fetch from FRED ───────────────────────────────────────

    def _fetch_series(self, series_id: str, start: str = "2020-01-01") -> pd.Series | None:
        """Fetch a single FRED series via the API."""
        if not self._available:
            return None

        try:
            from fredapi import Fred
            fred = Fred(api_key=self.api_key)
            data = fred.get_series(series_id, observation_start=start)
            if data is not None and len(data) > 0:
                data.index = pd.to_datetime(data.index)
                data = data.dropna()
                return data
        except ImportError:
            logger.error("fredapi not installed. Run: pip install fredapi")
        except Exception as e:
            logger.warning("Failed to fetch FRED series %s: %s", series_id, e)

        return None

    def refresh(self, start_date: str = "2020-01-01") -> bool:
        """
        Fetch latest data from FRED for all 8 series.
        Compute derived indicators. Save to cache as Parquet.
        Returns True if successful.
        """
        if not self._available:
            logger.warning("Cannot refresh FRED data — no API key")
            return False

        series_data = {}
        for fred_id, config in FRED_SERIES.items():
            s = self._fetch_series(fred_id, start=start_date)
            if s is not None:
                series_data[config["name"]] = s
                logger.info("Fetched %s (%s): %d observations", config["name"], fred_id, len(s))

        if not series_data:
            logger.error("No FRED series fetched successfully")
            return False

        # Combine into daily DataFrame
        df = pd.DataFrame(series_data)
        df.index = pd.to_datetime(df.index)
        df.index.name = "date"

        # Forward-fill to daily frequency (monthly series → daily)
        idx = pd.date_range(df.index.min(), df.index.max(), freq="D")
        df = df.reindex(idx).ffill()

        # ── Derived indicators ────────────────────────────────
        if "treasury_10y" in df.columns and "treasury_3m" in df.columns:
            df["term_spread"] = df["treasury_10y"] - df["treasury_3m"]

        if "bbb_yield" in df.columns and "aaa_yield" in df.columns:
            df["credit_spread"] = df["bbb_yield"] - df["aaa_yield"]

        if "industrial_production" in df.columns:
            df["industrial_production_yoy"] = (
                df["industrial_production"].pct_change(252)  # ~12 months in daily
            )

        if "treasury_10y" in df.columns and "breakeven_inflation" in df.columns:
            df["real_rate"] = df["treasury_10y"] - df["breakeven_inflation"]

        # Save to cache
        try:
            df.to_parquet(self.cache_path, compression="snappy")
            logger.info("Saved macro data: %d rows to %s", len(df), self.cache_path)
        except Exception as e:
            logger.warning("Failed to save macro cache: %s", e)

        self._data = df
        return True

    # ── Public API ────────────────────────────────────────────

    def get_current(self) -> dict | None:
        """
        Latest values for all indicators plus derived regime signals.
        Returns dict with raw values + regime classifications.
        """
        if self._data is None or self._data.empty:
            return None

        latest = self._data.iloc[-1].to_dict()
        result = {"raw": {}, "regimes": {}}

        # Raw values
        for col, val in latest.items():
            if pd.notna(val):
                result["raw"][col] = round(float(val), 4)

        # ── Yield Curve Regime ────────────────────────────────
        ts = latest.get("term_spread")
        if pd.notna(ts):
            if ts < 0:
                regime = "inverted"
            elif ts < 0.5:
                regime = "flat"
            elif ts < 2.0:
                regime = "normal"
            else:
                regime = "steep"
            result["regimes"]["yield_curve"] = {
                "regime": regime,
                "term_spread": round(float(ts), 4),
            }

        # ── Credit Cycle ──────────────────────────────────────
        cs = latest.get("credit_spread")
        if pd.notna(cs) and "credit_spread" in self._data.columns:
            cs_series = self._data["credit_spread"].dropna()
            if len(cs_series) >= 63:
                cs_63d_ago = cs_series.iloc[-63]
                cs_change = float(cs - cs_63d_ago)

                # Z-score vs 2-year history
                cs_2y = cs_series.tail(504)  # ~2 years daily
                cs_mean = cs_2y.mean()
                cs_std = cs_2y.std()
                z_score = float((cs - cs_mean) / cs_std) if cs_std > 0 else 0

                if cs_change > 0.1:
                    cycle = "tightening"
                elif cs_change < -0.1:
                    cycle = "easing"
                else:
                    cycle = "stable"

                result["regimes"]["credit_cycle"] = {
                    "cycle": cycle,
                    "credit_spread": round(float(cs), 4),
                    "change_63d": round(cs_change, 4),
                    "z_score_2y": round(z_score, 2),
                }

        # ── Monetary Policy ───────────────────────────────────
        ff = latest.get("fed_funds_rate")
        rr = latest.get("real_rate")
        if pd.notna(ff) and "fed_funds_rate" in self._data.columns:
            ff_series = self._data["fed_funds_rate"].dropna()
            if len(ff_series) >= 63:
                ff_change = float(ff - ff_series.iloc[-63])
                if ff_change > 0.25 and (rr is None or (pd.notna(rr) and rr > 0)):
                    policy = "hawkish"
                elif ff_change < -0.25 and (rr is None or (pd.notna(rr) and rr < 0)):
                    policy = "dovish"
                else:
                    policy = "neutral"

                result["regimes"]["monetary_policy"] = {
                    "stance": policy,
                    "fed_funds": round(float(ff), 4),
                    "real_rate": round(float(rr), 4) if pd.notna(rr) else None,
                    "change_63d": round(ff_change, 4),
                }

        return result

    def get_history(
        self,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> pd.DataFrame | None:
        """
        Full daily history of all indicators (forward-filled).
        For ML feature engineering and backtesting.
        """
        if self._data is None or self._data.empty:
            return None

        df = self._data.copy()
        if start_date:
            df = df[df.index >= pd.Timestamp(start_date)]
        if end_date:
            df = df[df.index <= pd.Timestamp(end_date)]

        return df if not df.empty else None

    def get_macro_features(self, as_of_date: str | None = None) -> dict | None:
        """
        Get macro features for ML model training (point-in-time).
        Returns a flat dict of macro values as known on as_of_date.
        """
        if self._data is None or self._data.empty:
            return None

        if as_of_date:
            cutoff = pd.Timestamp(as_of_date)
            available = self._data[self._data.index <= cutoff]
        else:
            available = self._data

        if available.empty:
            return None

        latest = available.iloc[-1]
        return {
            col: round(float(val), 6)
            for col, val in latest.items()
            if pd.notna(val)
        }
