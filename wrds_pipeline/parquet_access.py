#!/usr/bin/env python3
"""
Parquet-Based Data Access Layer
=================================
Reads the downloaded WRDS Parquet files and provides the same interface
as the RDS-based WRDSDataAccess, but without any database dependency.

This is the "Option B" architecture:
  WRDS → Parquet files (local + S3) → pandas → ML pipeline

ALL joins, filtering, and point-in-time logic happen in pandas/numpy.
This is actually FASTER than PostgreSQL for our use case because:
  1. Parquet columnar reads = only load columns you need
  2. No network round-trips to RDS
  3. pandas merge/join is highly optimized for this scale

Usage:
    from wrds_pipeline.parquet_access import ParquetDataAccess
    da = ParquetDataAccess()
    panel = da.get_training_panel(start_date='2000-01-01', end_date='2024-12-31')
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger("wrds_parquet_access")

# Default data directory
BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_DATA_DIR = BASE_DIR / "data" / "wrds"


class ParquetDataAccess:
    """
    Unified data access layer reading from Parquet files.
    
    Provides the same interface as WRDSDataAccess (RDS version)
    but reads from local Parquet files instead.
    
    Data is lazy-loaded on first access and cached in memory.
    Falls back to S3 via S3DataManager if local files are missing.
    """

    def __init__(self, data_dir: Path = DEFAULT_DATA_DIR):
        self.data_dir = Path(data_dir)
        self._dm = None  # Lazy-loaded S3DataManager
        
        if not self.data_dir.exists():
            # Try S3DataManager before failing
            self._dm = self._get_data_manager()
            if self._dm is None:
                raise FileNotFoundError(
                    f"Data directory not found: {self.data_dir}\n"
                    f"Run: python wrds_pipeline/download_to_parquet.py\n"
                    f"Or: python -m nuble.data.s3_data_manager pull"
                )
        
        # Lazy-loaded data caches
        self._cache = {}

    @staticmethod
    def _get_data_manager():
        """Lazy import S3DataManager."""
        try:
            from nuble.data.s3_data_manager import get_data_manager
            dm = get_data_manager()
            if dm.s3_available:
                return dm
        except Exception:
            pass
        return None

    def _load(self, name: str, columns: list = None) -> pd.DataFrame:
        """Load a Parquet file (cached). Local-first, S3-fallback."""
        cache_key = f"{name}_{','.join(columns) if columns else 'all'}"
        if cache_key not in self._cache:
            filepath = self.data_dir / f"{name}.parquet"
            
            if filepath.exists():
                if columns:
                    self._cache[cache_key] = pd.read_parquet(filepath, columns=columns)
                else:
                    self._cache[cache_key] = pd.read_parquet(filepath)
            else:
                # S3 fallback
                if self._dm is None:
                    self._dm = self._get_data_manager()
                if self._dm:
                    try:
                        logger.info(f"Loading {name} from S3 (not found locally)")
                        self._cache[cache_key] = self._dm.load_parquet(
                            f"{name}.parquet", columns=columns
                        )
                    except Exception as e:
                        raise FileNotFoundError(
                            f"Missing: {filepath} (also not in S3: {e})"
                        )
                else:
                    raise FileNotFoundError(f"Missing: {filepath}")
            
            logger.debug(f"Loaded {name}: {len(self._cache[cache_key]):,} rows")
        
        return self._cache[cache_key]

    def clear_cache(self):
        """Free all cached data from memory."""
        self._cache.clear()

    # ───────────────────────────────────────────────────────────────
    # Link tables
    # ───────────────────────────────────────────────────────────────

    def get_gvkey(self, permno: int, as_of_date: str) -> Optional[str]:
        """Map PERMNO → GVKEY using crsp_compustat_link (point-in-time)."""
        link = self._load("crsp_compustat_link")
        link = link[link["lpermno"] == permno].copy()
        link["linkdt"] = pd.to_datetime(link["linkdt"])
        link["linkenddt"] = pd.to_datetime(link["linkenddt"])
        
        as_of = pd.Timestamp(as_of_date)
        mask = (link["linkdt"] <= as_of) & (
            (link["linkenddt"] >= as_of) | link["linkenddt"].isna()
        )
        matched = link[mask].sort_values("linkprim")
        
        return matched.iloc[0]["gvkey"] if not matched.empty else None

    def get_ibes_ticker(self, permno: int, as_of_date: str) -> Optional[str]:
        """Map PERMNO → IBES ticker using ibes_crsp_link."""
        link = self._load("ibes_crsp_link")
        link = link[link["permno"] == permno].copy()
        link["sdate"] = pd.to_datetime(link["sdate"])
        link["edate"] = pd.to_datetime(link["edate"])
        
        as_of = pd.Timestamp(as_of_date)
        mask = (link["sdate"] <= as_of) & (
            (link["edate"] >= as_of) | link["edate"].isna()
        )
        matched = link[mask]
        
        return matched.iloc[0]["ticker"] if not matched.empty else None

    # ───────────────────────────────────────────────────────────────
    # Monthly returns panel
    # ───────────────────────────────────────────────────────────────

    def get_monthly_panel(
        self,
        start_date: str = "2000-01-01",
        end_date: str = "2024-12-31",
        min_market_cap: Optional[float] = None,
        min_price: Optional[float] = None,
        exchanges: list = [1, 2, 3],
    ) -> pd.DataFrame:
        """
        Build monthly cross-sectional panel from CRSP.
        
        Returns DataFrame with delisting-adjusted returns.
        """
        crsp = self._load("crsp_monthly")
        crsp = crsp.copy()
        crsp["date"] = pd.to_datetime(crsp["date"])
        
        # Filters
        mask = (
            (crsp["date"] >= start_date) &
            (crsp["date"] <= end_date) &
            (crsp["exchcd"].isin(exchanges)) &
            (crsp["shrcd"].isin([10, 11]))
        )
        
        if min_market_cap:
            mask &= crsp["market_cap"] >= min_market_cap
        if min_price:
            mask &= crsp["prc"].abs() >= min_price
        
        panel = crsp[mask].copy()
        
        # Delisting-adjusted return
        panel["ret_adj"] = np.where(
            panel["dlret"].notna() & panel["ret"].notna(),
            (1 + panel["ret"]) * (1 + panel["dlret"]) - 1,
            np.where(panel["dlret"].notna(), panel["dlret"], panel["ret"]),
        )
        
        panel = panel.sort_values(["permno", "date"])
        
        logger.info(
            f"Monthly panel: {len(panel):,} rows, "
            f"{panel['permno'].nunique():,} stocks, "
            f"{panel['date'].nunique()} months"
        )
        return panel

    # ───────────────────────────────────────────────────────────────
    # Fama-French factors
    # ───────────────────────────────────────────────────────────────

    def get_ff_factors(
        self, start_date: str = "2000-01-01", end_date: str = "2024-12-31",
        frequency: str = "monthly",
    ) -> pd.DataFrame:
        """Get Fama-French 5 factors + momentum."""
        name = f"ff_factors_{frequency}"
        ff = self._load(name).copy()
        ff["date"] = pd.to_datetime(ff["date"])
        return ff[(ff["date"] >= start_date) & (ff["date"] <= end_date)].sort_values("date")

    # ───────────────────────────────────────────────────────────────
    # Fundamentals (point-in-time)
    # ───────────────────────────────────────────────────────────────

    def get_fundamentals(self, gvkey: str, as_of_date: str, n_quarters: int = 8) -> pd.DataFrame:
        """
        Get most recent n_quarters of Compustat data KNOWN by as_of_date.
        Uses rdq (report date) as point-in-time filter.
        """
        fundq = self._load("compustat_quarterly")
        subset = fundq[fundq["gvkey"] == gvkey].copy()
        subset["rdq"] = pd.to_datetime(subset["rdq"])
        subset["datadate"] = pd.to_datetime(subset["datadate"])
        
        # Point-in-time: only filings with rdq <= as_of_date
        known = subset[subset["rdq"].notna() & (subset["rdq"] <= as_of_date)]
        return known.sort_values("datadate", ascending=False).head(n_quarters)

    # ───────────────────────────────────────────────────────────────
    # Earnings surprise (IBES)
    # ───────────────────────────────────────────────────────────────

    def get_earnings_data(self, ibes_ticker: str, as_of_date: str) -> dict:
        """Get IBES actuals and consensus for earnings surprise."""
        # Actuals
        actuals = self._load("ibes_actuals")
        act_sub = actuals[
            (actuals["ticker"] == ibes_ticker) &
            (actuals["measure"] == "EPS")
        ].copy()
        act_sub["anndats"] = pd.to_datetime(act_sub["anndats"])
        act_sub = act_sub[act_sub["anndats"].notna() & (act_sub["anndats"] <= as_of_date)]
        act_sub = act_sub.sort_values("anndats", ascending=False)
        
        # Summary (consensus)
        summary = self._load("ibes_summary")
        sum_sub = summary[
            (summary["ticker"] == ibes_ticker) &
            (summary["measure"] == "EPS")
        ].copy()
        sum_sub["statpers"] = pd.to_datetime(sum_sub["statpers"])
        sum_sub["fpedats"] = pd.to_datetime(sum_sub["fpedats"])
        
        return {"actuals": act_sub, "summary": sum_sub}

    # ───────────────────────────────────────────────────────────────
    # BOOK EQUITY (Fama-French 1993)
    # ───────────────────────────────────────────────────────────────

    def book_equity(self, gvkey: str, as_of_date: str) -> Optional[float]:
        """
        BE = stockholders_equity + deferred_taxes - preferred_stock
        Point-in-time: only filings with rdq <= as_of_date.
        """
        fundq = self.get_fundamentals(gvkey, as_of_date, n_quarters=1)
        if fundq.empty:
            return None
        
        row = fundq.iloc[0]
        
        # Stockholders equity priority: seqq → ceqq + pstkq → atq - ltq
        se = row.get("seqq")
        if pd.isna(se):
            ceq = row.get("ceqq")
            pstk = row.get("pstkq", 0) or 0
            if not pd.isna(ceq):
                se = float(ceq) + float(pstk)
            else:
                at_ = row.get("atq")
                lt_ = row.get("ltq")
                if not pd.isna(at_) and not pd.isna(lt_):
                    se = float(at_) - float(lt_)
                else:
                    return None
        
        txditc = float(row.get("txditcq", 0) or 0)
        pstk = float(row.get("pstkq", 0) or 0)
        be = float(se) + txditc - pstk
        return be if be > 0 else None

    # ───────────────────────────────────────────────────────────────
    # TTM helper
    # ───────────────────────────────────────────────────────────────

    def _ttm(self, fundamentals: pd.DataFrame, col: str) -> Optional[float]:
        """Sum the last 4 quarters of a flow variable."""
        if len(fundamentals) < 4:
            return None
        vals = fundamentals.head(4)[col]
        if vals.isna().all():
            return None
        return float(vals.sum())

    # ───────────────────────────────────────────────────────────────
    # Characteristics (48 Gu-Kelly-Xiu features)
    # ───────────────────────────────────────────────────────────────

    def compute_characteristics(self, permno: int, as_of_date: str) -> dict:
        """
        Compute all Gu-Kelly-Xiu characteristics for one stock.
        Returns dict of feature_name → value.
        """
        chars = {}
        
        gvkey = self.get_gvkey(permno, as_of_date)
        ibes_ticker = self.get_ibes_ticker(permno, as_of_date)
        
        # Get data
        monthly = self._get_monthly_for_stock(permno, as_of_date, n_months=60)
        fundq = self.get_fundamentals(gvkey, as_of_date, n_quarters=8) if gvkey else pd.DataFrame()
        
        # Compute features
        chars.update(self._momentum_features(monthly))
        chars.update(self._size_features(monthly))
        chars.update(self._value_features(fundq, monthly))
        chars.update(self._profitability_features(fundq))
        chars.update(self._investment_features(fundq))
        chars.update(self._leverage_features(fundq))
        chars.update(self._earnings_features(ibes_ticker, fundq, as_of_date))
        chars.update(self._volatility_features(monthly, as_of_date))
        chars.update(self._liquidity_features(monthly))
        
        return chars

    def _get_monthly_for_stock(self, permno: int, as_of_date: str, n_months: int = 60) -> pd.DataFrame:
        """Get monthly returns for a specific stock."""
        crsp = self._load("crsp_monthly")
        sub = crsp[crsp["permno"] == permno].copy()
        sub["date"] = pd.to_datetime(sub["date"])
        sub = sub[sub["date"] <= as_of_date].sort_values("date", ascending=False).head(n_months)
        return sub

    # ── Feature groups ──

    def _momentum_features(self, monthly: pd.DataFrame) -> dict:
        feats = {k: None for k in [
            "mom_1m", "mom_6m", "mom_12m", "mom_12_2", "mom_36m", "mom_1w", "mom_season"
        ]}
        
        if monthly.empty:
            return feats
        
        ms = monthly.sort_values("date")
        rets = ms["ret"].dropna()
        
        if len(rets) >= 1:
            feats["mom_1m"] = float(rets.iloc[-1])
        if len(rets) >= 6:
            feats["mom_6m"] = float(np.prod(1 + rets.tail(6)) - 1)
        if len(rets) >= 12:
            feats["mom_12m"] = float(np.prod(1 + rets.tail(12)) - 1)
            feats["mom_12_2"] = float(np.prod(1 + rets.tail(12).head(11)) - 1)
        if len(rets) >= 36:
            feats["mom_36m"] = float(np.prod(1 + rets.tail(36)) - 1)
        if len(rets) >= 12:
            feats["mom_season"] = float(rets.iloc[-12])
        
        return feats

    def _size_features(self, monthly: pd.DataFrame) -> dict:
        feats = {"log_market_cap": None, "log_price": None}
        if monthly.empty:
            return feats
        
        latest = monthly.sort_values("date").iloc[-1]
        mc = latest.get("market_cap")
        prc = latest.get("prc")
        
        if mc and not pd.isna(mc) and mc > 0:
            feats["log_market_cap"] = float(np.log(abs(mc)))
        if prc and not pd.isna(prc) and abs(prc) > 0:
            feats["log_price"] = float(np.log(abs(prc)))
        
        return feats

    def _value_features(self, fundq: pd.DataFrame, monthly: pd.DataFrame) -> dict:
        feats = {k: None for k in [
            "book_to_market", "earnings_to_price", "cashflow_to_price",
            "sales_to_price", "dividend_yield", "ev_to_ebitda",
        ]}
        
        if fundq.empty or monthly.empty:
            return feats
        
        latest_m = monthly.sort_values("date").iloc[-1]
        mc = latest_m.get("market_cap")
        if pd.isna(mc) or not mc or mc <= 0:
            return feats
        mc = float(mc)
        
        fq = fundq.sort_values("datadate", ascending=False)
        latest = fq.iloc[0]
        
        # Book equity
        se = latest.get("seqq")
        if pd.isna(se):
            ceq = latest.get("ceqq")
            pstk = latest.get("pstkq", 0) or 0
            if not pd.isna(ceq):
                se = float(ceq) + float(pstk)
            else:
                at_ = latest.get("atq")
                lt_ = latest.get("ltq")
                if not pd.isna(at_) and not pd.isna(lt_):
                    se = float(at_) - float(lt_)
        
        if se and not pd.isna(se):
            txditc = float(latest.get("txditcq", 0) or 0)
            pstk_val = float(latest.get("pstkq", 0) or 0)
            be = float(se) + txditc - pstk_val
            if be > 0:
                feats["book_to_market"] = be / mc
        
        # TTM ratios (Compustat in millions, market_cap in dollars)
        ni_ttm = self._ttm(fq, "niq")
        if ni_ttm is not None:
            feats["earnings_to_price"] = ni_ttm * 1_000_000 / mc
        
        cf_ttm = self._ttm(fq, "oancfq")
        if cf_ttm is not None:
            feats["cashflow_to_price"] = cf_ttm * 1_000_000 / mc
        
        rev_ttm = self._ttm(fq, "revtq")
        if rev_ttm is not None:
            feats["sales_to_price"] = rev_ttm * 1_000_000 / mc
        
        div_ttm = self._ttm(fq, "dvq")
        if div_ttm is not None:
            feats["dividend_yield"] = div_ttm * 1_000_000 / mc
        
        # EV/EBITDA
        debt = float(latest.get("dlttq", 0) or 0) + float(latest.get("dlcq", 0) or 0)
        cash = float(latest.get("cheq", 0) or 0)
        ev = mc + (debt - cash) * 1_000_000
        oi_ttm = self._ttm(fq, "oiadpq")
        dp_ttm = self._ttm(fq, "dpq")
        if oi_ttm is not None and dp_ttm is not None:
            ebitda = (oi_ttm + dp_ttm) * 1_000_000
            if abs(ebitda) > 0:
                feats["ev_to_ebitda"] = ev / ebitda
        
        return feats

    def _profitability_features(self, fundq: pd.DataFrame) -> dict:
        feats = {k: None for k in [
            "roe", "roa", "gross_profitability", "operating_profitability",
            "cash_profitability", "profit_margin", "asset_turnover",
        ]}
        
        if fundq.empty or len(fundq) < 4:
            return feats
        
        fq = fundq.sort_values("datadate", ascending=False)
        latest = fq.iloc[0]
        
        ni_ttm = self._ttm(fq, "niq")
        rev_ttm = self._ttm(fq, "revtq")
        cf_ttm = self._ttm(fq, "oancfq")
        oi_ttm = self._ttm(fq, "oiadpq")
        
        at_now = float(latest.get("atq", 0) or 0)
        se_now = float(latest.get("seqq", 0) or 0)
        at_lag = float(fq.iloc[3].get("atq", 0) or 0) if len(fq) >= 4 else at_now
        se_lag = float(fq.iloc[3].get("seqq", 0) or 0) if len(fq) >= 4 else se_now
        avg_at = (at_now + at_lag) / 2 if at_now + at_lag > 0 else None
        avg_se = (se_now + se_lag) / 2 if se_now + se_lag > 0 else None
        
        if ni_ttm is not None and avg_se and avg_se > 0:
            feats["roe"] = ni_ttm / avg_se
        if ni_ttm is not None and avg_at and avg_at > 0:
            feats["roa"] = ni_ttm / avg_at
        
        cogs_ttm = self._ttm(fq, "cogsq")
        if rev_ttm is not None and cogs_ttm is not None and at_now > 0:
            feats["gross_profitability"] = (rev_ttm - cogs_ttm) / at_now
        if oi_ttm is not None and se_now > 0:
            feats["operating_profitability"] = oi_ttm / se_now
        if cf_ttm is not None and at_now > 0:
            feats["cash_profitability"] = cf_ttm / at_now
        if ni_ttm is not None and rev_ttm is not None and abs(rev_ttm) > 0:
            feats["profit_margin"] = ni_ttm / rev_ttm
        if rev_ttm is not None and avg_at and avg_at > 0:
            feats["asset_turnover"] = rev_ttm / avg_at
        
        return feats

    def _investment_features(self, fundq: pd.DataFrame) -> dict:
        feats = {k: None for k in [
            "asset_growth", "investment", "capex_to_assets", "rd_to_assets", "accruals",
        ]}
        
        if fundq.empty or len(fundq) < 5:
            return feats
        
        fq = fundq.sort_values("datadate", ascending=False)
        now = fq.iloc[0]
        ago = fq.iloc[4] if len(fq) >= 5 else fq.iloc[-1]
        
        at_now = float(now.get("atq", 0) or 0)
        at_ago = float(ago.get("atq", 0) or 0)
        
        if at_ago > 0:
            feats["asset_growth"] = (at_now - at_ago) / at_ago
            ppe_now = float(now.get("ppentq", 0) or 0)
            ppe_ago = float(ago.get("ppentq", 0) or 0)
            inv_now = float(now.get("invtq", 0) or 0)
            inv_ago = float(ago.get("invtq", 0) or 0)
            feats["investment"] = ((ppe_now - ppe_ago) + (inv_now - inv_ago)) / at_ago
        
        capx_ttm = self._ttm(fq, "capxq")
        if capx_ttm is not None and at_now > 0:
            feats["capex_to_assets"] = abs(capx_ttm) / at_now
        
        rd_ttm = self._ttm(fq, "xrdq")
        if rd_ttm is not None and at_now > 0:
            feats["rd_to_assets"] = rd_ttm / at_now
        
        # Accruals
        dp_ttm = self._ttm(fq, "dpq")
        avg_at = (at_now + at_ago) / 2 if (at_now + at_ago) > 0 else None
        if avg_at and dp_ttm is not None:
            delta_ca = float(now.get("actq", 0) or 0) - float(ago.get("actq", 0) or 0)
            delta_cash = float(now.get("cheq", 0) or 0) - float(ago.get("cheq", 0) or 0)
            delta_cl = float(now.get("lctq", 0) or 0) - float(ago.get("lctq", 0) or 0)
            delta_std = float(now.get("dlcq", 0) or 0) - float(ago.get("dlcq", 0) or 0)
            feats["accruals"] = (delta_ca - delta_cash - delta_cl + delta_std - dp_ttm) / avg_at
        
        return feats

    def _leverage_features(self, fundq: pd.DataFrame) -> dict:
        feats = {k: None for k in [
            "debt_to_equity", "current_ratio", "interest_coverage", "cash_to_assets",
        ]}
        
        if fundq.empty:
            return feats
        
        latest = fundq.sort_values("datadate", ascending=False).iloc[0]
        se = float(latest.get("seqq", 0) or 0)
        at_ = float(latest.get("atq", 0) or 0)
        act = float(latest.get("actq", 0) or 0)
        lct = float(latest.get("lctq", 0) or 0)
        dltt = float(latest.get("dlttq", 0) or 0)
        dlc = float(latest.get("dlcq", 0) or 0)
        che = float(latest.get("cheq", 0) or 0)
        oi = float(latest.get("oiadpq", 0) or 0)
        xint = float(latest.get("xintq", 0) or 0)
        
        if se > 0:
            feats["debt_to_equity"] = (dltt + dlc) / se
        if lct > 0:
            feats["current_ratio"] = act / lct
        if xint > 0:
            feats["interest_coverage"] = oi / xint
        if at_ > 0:
            feats["cash_to_assets"] = che / at_
        
        return feats

    def _earnings_features(self, ibes_ticker: Optional[str], fundq: pd.DataFrame, as_of_date: str) -> dict:
        feats = {k: None for k in [
            "earnings_surprise", "earnings_surprise_std",
            "analyst_dispersion", "analyst_revision", "num_analysts",
        ]}
        
        if not ibes_ticker:
            return feats
        
        try:
            data = self.get_earnings_data(ibes_ticker, as_of_date)
            actuals = data["actuals"]
            summary = data["summary"]
            
            if actuals.empty:
                return feats
            
            latest = actuals.iloc[0]
            actual_eps = latest["value"]
            ann_date = latest["anndats"]
            period_end = latest["pends"]
            
            # Get consensus before announcement
            consensus = summary[
                (summary["fpi"] == "6") &
                (summary["fpedats"] == period_end) &
                (summary["statpers"] < ann_date)
            ].sort_values("statpers", ascending=False)
            
            if not consensus.empty:
                est = consensus.iloc[0]
                mean_est = est["meanest"]
                stdev_est = est["stdev"]
                n_analysts = est["numest"]
                
                if mean_est is not None and actual_eps is not None and abs(float(mean_est)) > 0.01:
                    feats["earnings_surprise"] = float(actual_eps - mean_est) / abs(float(mean_est))
                if stdev_est is not None and mean_est is not None and abs(float(mean_est)) > 0.01:
                    feats["analyst_dispersion"] = float(stdev_est) / abs(float(mean_est))
                feats["num_analysts"] = int(n_analysts) if n_analysts is not None else None
            
            # Revision
            recent_consensus = summary[
                (summary["fpi"] == "6") &
                (summary["statpers"] <= as_of_date)
            ].sort_values("statpers", ascending=False).head(4)
            
            if len(recent_consensus) >= 2:
                newest = recent_consensus.iloc[0]["meanest"]
                oldest = recent_consensus.iloc[-1]["meanest"]
                if newest is not None and oldest is not None and abs(float(oldest)) > 0.01:
                    feats["analyst_revision"] = float(newest - oldest) / abs(float(oldest))
        
        except Exception:
            pass
        
        return feats

    def _volatility_features(self, monthly: pd.DataFrame, as_of_date: str) -> dict:
        feats = {k: None for k in [
            "realized_vol_1m", "realized_vol_3m", "realized_vol_12m",
            "idiosyncratic_vol", "beta", "downside_beta",
            "max_daily_return", "skewness",
        ]}
        
        if monthly.empty or len(monthly) < 12:
            return feats
        
        ms = monthly.sort_values("date")
        rets = ms["ret"].dropna().values
        sqrt12 = np.sqrt(12)
        
        if len(rets) >= 3:
            feats["realized_vol_1m"] = float(np.std(rets[-3:]) * sqrt12)  # ~3 months ≈ daily 1m
        if len(rets) >= 12:
            feats["realized_vol_3m"] = float(np.std(rets[-12:]) * sqrt12)
        if len(rets) >= 36:
            feats["realized_vol_12m"] = float(np.std(rets[-36:]) * sqrt12)
        
        if len(rets) >= 3:
            feats["max_daily_return"] = float(np.max(np.abs(rets[-12:]))) if len(rets) >= 12 else float(np.max(np.abs(rets)))
        
        if len(rets) >= 24:
            from scipy.stats import skew
            feats["skewness"] = float(skew(rets[-36:])) if len(rets) >= 36 else float(skew(rets[-24:]))
        
        # Beta from FF factors
        try:
            ff = self.get_ff_factors(
                start_date=str(ms["date"].min()),
                end_date=as_of_date,
                frequency="monthly",
            )
            
            if not ff.empty and len(ff) >= 12:
                ms_copy = ms.copy()
                ms_copy["date"] = pd.to_datetime(ms_copy["date"])
                ff["date"] = pd.to_datetime(ff["date"])
                
                merged = pd.merge_asof(
                    ms_copy.sort_values("date"),
                    ff.sort_values("date"),
                    on="date", direction="nearest",
                    tolerance=pd.Timedelta("45D"),
                )
                
                valid = merged.dropna(subset=["ret", "mktrf", "rf"])
                if len(valid) >= 24:
                    excess = valid["ret"].values - valid["rf"].values
                    mkt = valid["mktrf"].values
                    
                    if np.var(mkt) > 0:
                        feats["beta"] = float(np.cov(excess, mkt)[0, 1] / np.var(mkt))
                        residuals = excess - feats["beta"] * mkt
                        feats["idiosyncratic_vol"] = float(np.std(residuals) * sqrt12)
                    
                    neg_mask = mkt < 0
                    if neg_mask.sum() >= 10:
                        feats["downside_beta"] = float(
                            np.cov(excess[neg_mask], mkt[neg_mask])[0, 1] / np.var(mkt[neg_mask])
                        )
        except Exception:
            pass
        
        return feats

    def _liquidity_features(self, monthly: pd.DataFrame) -> dict:
        feats = {k: None for k in [
            "turnover", "amihud_illiquidity", "dollar_volume", "bid_ask_spread",
        ]}
        
        if monthly.empty or len(monthly) < 3:
            return feats
        
        ms = monthly.sort_values("date").tail(12)
        
        vol = ms["vol"].dropna().values
        prc = ms["prc"].dropna().values
        shrout = ms["shrout"].dropna().values
        rets = ms["ret"].dropna().values
        
        # Turnover
        if len(vol) > 0 and len(shrout) > 0:
            n = min(len(vol), len(shrout))
            shrout_adj = shrout[:n] * 1000
            valid = shrout_adj > 0
            if valid.any():
                feats["turnover"] = float(np.mean(vol[:n][valid] / shrout_adj[valid]))
        
        # Dollar volume
        if len(prc) > 0 and len(vol) > 0:
            n = min(len(prc), len(vol))
            dv = np.abs(prc[:n]) * vol[:n]
            avg_dv = np.mean(dv)
            if avg_dv > 0:
                feats["dollar_volume"] = float(np.log(avg_dv))
        
        # Amihud
        if len(rets) > 0 and len(prc) > 0 and len(vol) > 0:
            n = min(len(rets), len(prc), len(vol))
            dv = np.abs(prc[:n]) * vol[:n]
            valid = dv > 0
            if valid.any():
                feats["amihud_illiquidity"] = float(np.mean(np.abs(rets[:n][valid]) / dv[valid]))
        
        return feats

    # ───────────────────────────────────────────────────────────────
    # TRAINING PANEL (the main ML function)
    # ───────────────────────────────────────────────────────────────

    def get_training_panel(
        self,
        start_date: str = "2000-01-01",
        end_date: str = "2024-12-31",
        min_market_cap: float = 10_000_000,
        min_price: float = 5.0,
        exchanges: list = [1, 2, 3],
        max_stocks_per_month: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        THE main function for ML training.
        
        Returns panel: (permno × date) × features
        
        Steps:
        1. Get CRSP monthly returns
        2. Filter to liquid, investable stocks
        3. Compute all characteristics (point-in-time)
        4. Add forward return labels
        5. Add Fama-French factors
        """
        logger.info(f"Building training panel: {start_date} → {end_date}")
        
        # Step 1: Monthly returns
        monthly = self.get_monthly_panel(
            start_date=start_date, end_date=end_date,
            min_market_cap=min_market_cap, min_price=min_price,
            exchanges=exchanges,
        )
        
        if monthly.empty:
            logger.error("No monthly data!")
            return pd.DataFrame()
        
        # Step 2: Forward returns
        monthly = monthly.sort_values(["permno", "date"])
        monthly["fwd_ret_1m"] = monthly.groupby("permno")["ret_adj"].shift(-1)
        
        # Step 3: Characteristics
        dates = sorted(monthly["date"].unique())
        logger.info(f"  {len(dates)} months, computing characteristics...")
        
        all_chars = []
        for i, dt in enumerate(dates):
            dt_str = str(dt)[:10]
            month_stocks = monthly[monthly["date"] == dt]
            
            if max_stocks_per_month and len(month_stocks) > max_stocks_per_month:
                month_stocks = month_stocks.nlargest(max_stocks_per_month, "market_cap")
            
            for permno in month_stocks["permno"].unique():
                try:
                    chars = self.compute_characteristics(int(permno), dt_str)
                    chars["permno"] = permno
                    chars["date"] = dt
                    all_chars.append(chars)
                except Exception:
                    continue
            
            if (i + 1) % 12 == 0:
                logger.info(f"    Month {i+1}/{len(dates)}: {len(all_chars):,} obs")
        
        if not all_chars:
            logger.error("No characteristics computed!")
            return pd.DataFrame()
        
        chars_df = pd.DataFrame(all_chars)
        chars_df["date"] = pd.to_datetime(chars_df["date"])
        monthly["date"] = pd.to_datetime(monthly["date"])
        
        # Step 4: Merge
        panel = monthly.merge(chars_df, on=["permno", "date"], how="inner")
        
        # Step 5: FF factors
        ff = self.get_ff_factors(start_date, end_date, frequency="monthly")
        if not ff.empty:
            ff["date"] = pd.to_datetime(ff["date"])
            panel = panel.merge(ff, on="date", how="left")
        
        feature_names = self.feature_names()
        available = [c for c in feature_names if c in panel.columns]
        
        logger.info(f"\n  ═══ TRAINING PANEL COMPLETE ═══")
        logger.info(f"  Observations:     {len(panel):>12,}")
        logger.info(f"  Stocks/month:     {panel.groupby('date')['permno'].nunique().mean():>12.0f}")
        logger.info(f"  Features:         {len(available):>12}")
        logger.info(f"  Date range:       {panel['date'].min()} → {panel['date'].max()}")
        
        return panel

    @staticmethod
    def feature_names() -> list:
        """All 48 Gu-Kelly-Xiu feature names."""
        return [
            "mom_1m", "mom_6m", "mom_12m", "mom_12_2", "mom_36m", "mom_1w", "mom_season",
            "log_market_cap", "log_price",
            "book_to_market", "earnings_to_price", "cashflow_to_price",
            "sales_to_price", "dividend_yield", "ev_to_ebitda",
            "roe", "roa", "gross_profitability", "operating_profitability",
            "cash_profitability", "profit_margin", "asset_turnover",
            "asset_growth", "investment", "capex_to_assets", "rd_to_assets", "accruals",
            "debt_to_equity", "current_ratio", "interest_coverage", "cash_to_assets",
            "earnings_surprise", "earnings_surprise_std",
            "analyst_dispersion", "analyst_revision", "num_analysts",
            "realized_vol_1m", "realized_vol_3m", "realized_vol_12m",
            "idiosyncratic_vol", "beta", "downside_beta",
            "max_daily_return", "skewness",
            "turnover", "amihud_illiquidity", "dollar_volume", "bid_ask_spread",
        ]

    # ───────────────────────────────────────────────────────────────
    # Summary
    # ───────────────────────────────────────────────────────────────

    def summary(self):
        """Print summary of all Parquet data."""
        print("\n" + "═" * 55)
        print("  WRDS PARQUET DATA SUMMARY")
        print("═" * 55)
        
        total_rows = 0
        total_size = 0
        
        for filepath in sorted(self.data_dir.glob("*.parquet")):
            try:
                df = pd.read_parquet(filepath)
                n = len(df)
                size = filepath.stat().st_size / (1024 * 1024)
                total_rows += n
                total_size += size
                print(f"  {filepath.stem:35s}: {n:>12,} rows  {size:>8.1f} MB")
                del df
            except Exception as e:
                print(f"  {filepath.stem:35s}: ERROR — {e}")
        
        print("─" * 55)
        print(f"  {'TOTAL':35s}: {total_rows:>12,} rows  {total_size:>8.1f} MB")
        print("═" * 55)


# ═══════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Parquet Data Access (Option B)")
    parser.add_argument("--summary", action="store_true")
    parser.add_argument("--test", action="store_true", help="Quick feature test")
    parser.add_argument("--panel", action="store_true", help="Build training panel (slow)")
    parser.add_argument("--start", default="2020-01-01")
    parser.add_argument("--end", default="2024-12-31")
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    
    da = ParquetDataAccess()
    
    if args.summary:
        da.summary()
    elif args.test:
        da.summary()
        print("\n=== SAMPLE CHARACTERISTICS (PERMNO=14593, AAPL) ===")
        chars = da.compute_characteristics(permno=14593, as_of_date="2024-06-30")
        print(f"  Features computed: {len([v for v in chars.values() if v is not None])}/{len(chars)}")
        for k, v in sorted(chars.items()):
            if v is not None:
                print(f"    {k:35s}: {v:>12.4f}")
    elif args.panel:
        panel = da.get_training_panel(
            start_date=args.start, end_date=args.end,
            max_stocks_per_month=200,
        )
        print(f"\nPanel shape: {panel.shape}")
        if not panel.empty:
            # Save for immediate use
            panel.to_parquet(da.data_dir / "training_panel.parquet", index=False)
            print(f"Saved to: {da.data_dir / 'training_panel.parquet'}")
