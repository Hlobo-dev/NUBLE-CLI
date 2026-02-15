#!/usr/bin/env python3
"""
Gu-Kelly-Xiu (2020) Stock Characteristics
============================================
Computes ~94 firm characteristics used in the cross-sectional ML framework.

All features use POINT-IN-TIME data only (no lookahead):
    - Compustat: filtered by rdq (report date) <= as_of_date
    - IBES: filtered by statpers (consensus date) <= as_of_date
    - CRSP: filtered by date <= as_of_date

Feature groups:
    1. Momentum       (7 features)   — past return signals
    2. Size           (2 features)   — market cap, price
    3. Value          (6 features)   — B/M, E/P, CF/P, S/P, D/Y, EV/EBITDA
    4. Profitability  (7 features)   — ROE, ROA, GP/A, margins
    5. Investment     (5 features)   — asset growth, capex, R&D, accruals
    6. Leverage       (4 features)   — debt, current ratio, coverage, cash
    7. Earnings       (5 features)   — SUE, dispersion, revision, analysts
    8. Volatility     (8 features)   — realized vol, idio vol, beta, skew
    9. Liquidity      (4 features)   — turnover, Amihud, dollar volume, spread
    10. Seasonality   (2 features)   — same-month lag returns

Total: ~50 characteristics implemented here (the most impactful ones).
The full 94 would require additional data sources (short interest, options, etc.)
"""

import math
import logging
from typing import Optional

import numpy as np
import pandas as pd
import psycopg2

from wrds_pipeline.config import RDS_CONFIG

logger = logging.getLogger("wrds_characteristics")


class CharacteristicsEngine:
    """
    Computes cross-sectional stock characteristics from RDS warehouse data.

    Usage:
        eng = CharacteristicsEngine()
        chars = eng.compute(permno=14593, as_of_date='2024-06-30')
        # Returns dict of ~50 named features

    Batch usage:
        panel = eng.compute_panel(
            permnos=[14593, 10107, 93436],
            as_of_date='2024-06-30'
        )
        # Returns DataFrame: (permno) × features
    """

    def __init__(self, conn=None):
        if conn is not None:
            self.conn = conn
            self._own_conn = False
        else:
            self.conn = psycopg2.connect(**RDS_CONFIG)
            self.conn.autocommit = True
            self._own_conn = True

    def close(self):
        if self._own_conn:
            self.conn.close()

    def _q(self, sql: str, params=None) -> pd.DataFrame:
        """Execute query and return DataFrame."""
        return pd.read_sql_query(sql, self.conn, params=params)

    # ═══════════════════════════════════════════════════════════════
    # Helper: get GVKEY for a PERMNO (via link table)
    # ═══════════════════════════════════════════════════════════════

    def _get_gvkey(self, permno: int, as_of_date: str) -> Optional[str]:
        """Map PERMNO → GVKEY using crsp_compustat_link (point-in-time)."""
        df = self._q("""
            SELECT gvkey FROM crsp_compustat_link
            WHERE lpermno = %s
              AND linkdt <= %s
              AND (linkenddt >= %s OR linkenddt IS NULL)
              AND linkprim IN ('P','C')
            ORDER BY linkprim ASC
            LIMIT 1
        """, (permno, as_of_date, as_of_date))
        return df.iloc[0]["gvkey"] if not df.empty else None

    def _get_ibes_ticker(self, permno: int, as_of_date: str) -> Optional[str]:
        """Map PERMNO → IBES ticker using ibes_crsp_link."""
        df = self._q("""
            SELECT ticker FROM ibes_crsp_link
            WHERE permno = %s
              AND sdate <= %s
              AND (edate >= %s OR edate IS NULL)
            LIMIT 1
        """, (permno, as_of_date, as_of_date))
        return df.iloc[0]["ticker"] if not df.empty else None

    # ═══════════════════════════════════════════════════════════════
    # Helper: get recent Compustat quarterly data (point-in-time)
    # ═══════════════════════════════════════════════════════════════

    def _get_fundamentals(self, gvkey: str, as_of_date: str, n_quarters: int = 8) -> pd.DataFrame:
        """
        Get the most recent n_quarters of Compustat data KNOWN by as_of_date.
        Uses rdq (report date) as point-in-time filter.
        """
        return self._q("""
            SELECT * FROM compustat_quarterly
            WHERE gvkey = %s
              AND rdq IS NOT NULL
              AND rdq <= %s
            ORDER BY datadate DESC
            LIMIT %s
        """, (gvkey, as_of_date, n_quarters))

    # ═══════════════════════════════════════════════════════════════
    # Helper: get daily returns
    # ═══════════════════════════════════════════════════════════════

    def _get_daily_returns(self, permno: int, as_of_date: str, n_days: int = 504) -> pd.DataFrame:
        """Get up to n_days of daily returns ending at as_of_date."""
        return self._q("""
            SELECT date, ret, volume, close, high, low, shrout
            FROM stock_prices
            WHERE permno = %s AND date <= %s AND ret IS NOT NULL
            ORDER BY date DESC
            LIMIT %s
        """, (permno, as_of_date, n_days))

    def _get_monthly_returns(self, permno: int, as_of_date: str, n_months: int = 60) -> pd.DataFrame:
        """Get monthly returns for rolling calculations."""
        return self._q("""
            SELECT date, ret, retx, market_cap, prc, shrout, vol
            FROM crsp_monthly
            WHERE permno = %s AND date <= %s
            ORDER BY date DESC
            LIMIT %s
        """, (permno, as_of_date, n_months))

    # ═══════════════════════════════════════════════════════════════
    # BOOK EQUITY (Fama-French 1993 methodology)
    # ═══════════════════════════════════════════════════════════════

    def book_equity(self, gvkey: str, as_of_date: str) -> Optional[float]:
        """
        BE = stockholders_equity + deferred_taxes - preferred_stock

        Stockholders equity priority: seqq → ceqq + pstkq → atq - ltq
        Preferred stock: pstkq → 0
        Point-in-time: only filings with rdq <= as_of_date
        """
        df = self._q("""
            SELECT seqq, ceqq, pstkq, atq, ltq, txditcq
            FROM compustat_quarterly
            WHERE gvkey = %s AND rdq IS NOT NULL AND rdq <= %s
            ORDER BY datadate DESC
            LIMIT 1
        """, (gvkey, as_of_date))

        if df.empty:
            return None

        row = df.iloc[0]

        # Stockholders equity
        se = row.get("seqq")
        if pd.isna(se):
            ceq = row.get("ceqq")
            pstk = row.get("pstkq", 0) or 0
            if not pd.isna(ceq):
                se = ceq + pstk
            else:
                at_ = row.get("atq")
                lt_ = row.get("ltq")
                if not pd.isna(at_) and not pd.isna(lt_):
                    se = at_ - lt_
                else:
                    return None

        # Deferred taxes
        txditc = row.get("txditcq", 0) or 0

        # Preferred stock
        pstk = row.get("pstkq", 0) or 0

        be = float(se) + float(txditc) - float(pstk)
        return be if be > 0 else None

    # ═══════════════════════════════════════════════════════════════
    # TTM (trailing twelve months) helper
    # ═══════════════════════════════════════════════════════════════

    def _ttm(self, fundamentals: pd.DataFrame, col: str) -> Optional[float]:
        """Sum the last 4 quarters of a flow variable (point-in-time)."""
        if len(fundamentals) < 4:
            return None
        vals = fundamentals.head(4)[col]
        if vals.isna().all():
            return None
        return float(vals.sum())

    # ═══════════════════════════════════════════════════════════════
    # COMPUTE ALL CHARACTERISTICS
    # ═══════════════════════════════════════════════════════════════

    def compute(self, permno: int, as_of_date: str) -> dict:
        """
        Compute all characteristics for a single stock as of a date.
        Returns dict of feature_name → value (None if unavailable).
        """
        chars = {}

        # ── Get supporting data ──
        gvkey = self._get_gvkey(permno, as_of_date)
        ibes_ticker = self._get_ibes_ticker(permno, as_of_date)
        daily = self._get_daily_returns(permno, as_of_date, n_days=504)
        monthly = self._get_monthly_returns(permno, as_of_date, n_months=60)
        fundq = self._get_fundamentals(gvkey, as_of_date, n_quarters=8) if gvkey else pd.DataFrame()

        # ── 1. MOMENTUM ──
        chars.update(self._momentum_features(daily, monthly))

        # ── 2. SIZE ──
        chars.update(self._size_features(monthly))

        # ── 3. VALUE ──
        chars.update(self._value_features(fundq, monthly))

        # ── 4. PROFITABILITY ──
        chars.update(self._profitability_features(fundq))

        # ── 5. INVESTMENT ──
        chars.update(self._investment_features(fundq))

        # ── 6. LEVERAGE ──
        chars.update(self._leverage_features(fundq))

        # ── 7. EARNINGS QUALITY ──
        chars.update(self._earnings_features(ibes_ticker, fundq, as_of_date))

        # ── 8. VOLATILITY ──
        chars.update(self._volatility_features(daily, monthly, as_of_date))

        # ── 9. LIQUIDITY ──
        chars.update(self._liquidity_features(daily))

        return chars

    # ───────────────────────────────────────────────────────────────
    # 1. Momentum features
    # ───────────────────────────────────────────────────────────────

    def _momentum_features(self, daily: pd.DataFrame, monthly: pd.DataFrame) -> dict:
        feats = {}

        if not monthly.empty:
            monthly_sorted = monthly.sort_values("date")
            rets = monthly_sorted["ret"].dropna()

            # mom_1m: 1-month return (short-term reversal)
            feats["mom_1m"] = float(rets.iloc[-1]) if len(rets) >= 1 else None

            # mom_6m: cumulative 6-month return
            if len(rets) >= 6:
                feats["mom_6m"] = float(np.prod(1 + rets.tail(6)) - 1)
            else:
                feats["mom_6m"] = None

            # mom_12m: cumulative 12-month return
            if len(rets) >= 12:
                feats["mom_12m"] = float(np.prod(1 + rets.tail(12)) - 1)
            else:
                feats["mom_12m"] = None

            # mom_12_2: 12-month return skipping most recent month (Jegadeesh-Titman)
            if len(rets) >= 12:
                feats["mom_12_2"] = float(np.prod(1 + rets.tail(12).head(11)) - 1)
            else:
                feats["mom_12_2"] = None

            # mom_36m: 36-month return (long-term reversal)
            if len(rets) >= 36:
                feats["mom_36m"] = float(np.prod(1 + rets.tail(36)) - 1)
            else:
                feats["mom_36m"] = None
        else:
            for k in ["mom_1m", "mom_6m", "mom_12m", "mom_12_2", "mom_36m"]:
                feats[k] = None

        # Short-term reversal from daily data
        if not daily.empty and len(daily) >= 5:
            daily_sorted = daily.sort_values("date")
            feats["mom_1w"] = float(np.prod(1 + daily_sorted["ret"].tail(5)) - 1)
        else:
            feats["mom_1w"] = None

        # Seasonal momentum: same calendar month last year
        if not monthly.empty and len(monthly) >= 12:
            monthly_sorted = monthly.sort_values("date")
            feats["mom_season"] = float(monthly_sorted["ret"].iloc[-12]) if len(monthly_sorted) >= 12 else None
        else:
            feats["mom_season"] = None

        return feats

    # ───────────────────────────────────────────────────────────────
    # 2. Size features
    # ───────────────────────────────────────────────────────────────

    def _size_features(self, monthly: pd.DataFrame) -> dict:
        feats = {}
        if not monthly.empty:
            latest = monthly.sort_values("date").iloc[-1]
            mc = latest.get("market_cap")
            prc = latest.get("prc")
            feats["log_market_cap"] = float(np.log(abs(mc))) if mc and not pd.isna(mc) and mc > 0 else None
            feats["log_price"] = float(np.log(abs(prc))) if prc and not pd.isna(prc) and abs(prc) > 0 else None
        else:
            feats["log_market_cap"] = None
            feats["log_price"] = None
        return feats

    # ───────────────────────────────────────────────────────────────
    # 3. Value features
    # ───────────────────────────────────────────────────────────────

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

        # Book equity
        latest_q = fundq.sort_values("datadate", ascending=False).iloc[0]
        se = latest_q.get("seqq")
        if pd.isna(se):
            ceq = latest_q.get("ceqq")
            pstk = latest_q.get("pstkq", 0) or 0
            if not pd.isna(ceq):
                se = float(ceq) + float(pstk)
            else:
                at_ = latest_q.get("atq")
                lt_ = latest_q.get("ltq")
                if not pd.isna(at_) and not pd.isna(lt_):
                    se = float(at_) - float(lt_)
        if se and not pd.isna(se):
            txditc = float(latest_q.get("txditcq", 0) or 0)
            pstk_val = float(latest_q.get("pstkq", 0) or 0)
            be = float(se) + txditc - pstk_val
            if be > 0:
                feats["book_to_market"] = be / mc

        # TTM ratios
        ni_ttm = self._ttm(fundq.sort_values("datadate", ascending=False), "niq")
        if ni_ttm is not None:
            feats["earnings_to_price"] = ni_ttm * 1_000_000 / mc  # Compustat in millions

        cf_ttm = self._ttm(fundq.sort_values("datadate", ascending=False), "oancfq")
        if cf_ttm is not None:
            feats["cashflow_to_price"] = cf_ttm * 1_000_000 / mc

        rev_ttm = self._ttm(fundq.sort_values("datadate", ascending=False), "revtq")
        if rev_ttm is not None:
            feats["sales_to_price"] = rev_ttm * 1_000_000 / mc

        div_ttm = self._ttm(fundq.sort_values("datadate", ascending=False), "dvq")
        if div_ttm is not None:
            feats["dividend_yield"] = div_ttm * 1_000_000 / mc

        # EV/EBITDA — simplified
        # EV = market_cap + total_debt - cash
        debt = float(latest_q.get("dlttq", 0) or 0) + float(latest_q.get("dlcq", 0) or 0)
        cash = float(latest_q.get("cheq", 0) or 0)
        ev = mc + (debt - cash) * 1_000_000  # Compustat in millions

        # EBITDA TTM = operating_income + depreciation (quarterly summed)
        oi_ttm = self._ttm(fundq.sort_values("datadate", ascending=False), "oiadpq")
        dp_ttm = self._ttm(fundq.sort_values("datadate", ascending=False), "dpq")
        if oi_ttm is not None and dp_ttm is not None:
            ebitda = (oi_ttm + dp_ttm) * 1_000_000
            if abs(ebitda) > 0:
                feats["ev_to_ebitda"] = ev / ebitda

        return feats

    # ───────────────────────────────────────────────────────────────
    # 4. Profitability features
    # ───────────────────────────────────────────────────────────────

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
        oi_ttm = self._ttm(fq, "oiadpq")
        cf_ttm = self._ttm(fq, "oancfq")

        at_now = float(latest.get("atq", 0) or 0)
        se_now = float(latest.get("seqq", 0) or 0)

        # Use average of current and 4-quarter-ago values
        at_lag = float(fq.iloc[3].get("atq", 0) or 0) if len(fq) >= 4 else at_now
        se_lag = float(fq.iloc[3].get("seqq", 0) or 0) if len(fq) >= 4 else se_now
        avg_at = (at_now + at_lag) / 2 if at_now + at_lag > 0 else None
        avg_se = (se_now + se_lag) / 2 if se_now + se_lag > 0 else None

        if ni_ttm is not None and avg_se and avg_se > 0:
            feats["roe"] = ni_ttm / avg_se

        if ni_ttm is not None and avg_at and avg_at > 0:
            feats["roa"] = ni_ttm / avg_at

        # Gross profitability = (revenue - COGS) / assets  (Novy-Marx 2013)
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

    # ───────────────────────────────────────────────────────────────
    # 5. Investment features
    # ───────────────────────────────────────────────────────────────

    def _investment_features(self, fundq: pd.DataFrame) -> dict:
        feats = {k: None for k in [
            "asset_growth", "investment", "capex_to_assets",
            "rd_to_assets", "accruals",
        ]}

        if fundq.empty or len(fundq) < 5:
            return feats

        fq = fundq.sort_values("datadate", ascending=False)
        now = fq.iloc[0]
        ago = fq.iloc[4] if len(fq) >= 5 else fq.iloc[-1]  # 4 quarters ago

        at_now = float(now.get("atq", 0) or 0)
        at_ago = float(ago.get("atq", 0) or 0)

        # Asset growth
        if at_ago > 0:
            feats["asset_growth"] = (at_now - at_ago) / at_ago

        # Investment = (delta PPE + delta inventory) / lag assets
        ppe_now = float(now.get("ppentq", 0) or 0)
        ppe_ago = float(ago.get("ppentq", 0) or 0)
        inv_now = float(now.get("invtq", 0) or 0)
        inv_ago = float(ago.get("invtq", 0) or 0)
        if at_ago > 0:
            feats["investment"] = ((ppe_now - ppe_ago) + (inv_now - inv_ago)) / at_ago

        # CapEx / Assets
        capx_ttm = self._ttm(fq, "capxq")
        if capx_ttm is not None and at_now > 0:
            feats["capex_to_assets"] = abs(capx_ttm) / at_now

        # R&D / Assets
        rd_ttm = self._ttm(fq, "xrdq")
        if rd_ttm is not None and at_now > 0:
            feats["rd_to_assets"] = rd_ttm / at_now

        # Accruals = (delta_CA - delta_cash - delta_CL + delta_STD - dep) / avg_assets
        act_now = float(now.get("actq", 0) or 0)
        act_ago = float(ago.get("actq", 0) or 0)
        che_now = float(now.get("cheq", 0) or 0)
        che_ago = float(ago.get("cheq", 0) or 0)
        lct_now = float(now.get("lctq", 0) or 0)
        lct_ago = float(ago.get("lctq", 0) or 0)
        dlc_now = float(now.get("dlcq", 0) or 0)
        dlc_ago = float(ago.get("dlcq", 0) or 0)
        dp_ttm = self._ttm(fq, "dpq")
        avg_at = (at_now + at_ago) / 2 if (at_now + at_ago) > 0 else None

        if avg_at and dp_ttm is not None:
            delta_ca = act_now - act_ago
            delta_cash = che_now - che_ago
            delta_cl = lct_now - lct_ago
            delta_std = dlc_now - dlc_ago
            feats["accruals"] = (delta_ca - delta_cash - delta_cl + delta_std - dp_ttm) / avg_at

        return feats

    # ───────────────────────────────────────────────────────────────
    # 6. Leverage features
    # ───────────────────────────────────────────────────────────────

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

        total_debt = dltt + dlc
        if se > 0:
            feats["debt_to_equity"] = total_debt / se
        if lct > 0:
            feats["current_ratio"] = act / lct
        if xint > 0:
            feats["interest_coverage"] = oi / xint
        if at_ > 0:
            feats["cash_to_assets"] = che / at_

        return feats

    # ───────────────────────────────────────────────────────────────
    # 7. Earnings quality features
    # ───────────────────────────────────────────────────────────────

    def _earnings_features(self, ibes_ticker: Optional[str],
                           fundq: pd.DataFrame, as_of_date: str) -> dict:
        feats = {k: None for k in [
            "earnings_surprise", "earnings_surprise_std",
            "analyst_dispersion", "analyst_revision", "num_analysts",
        ]}

        if not ibes_ticker:
            return feats

        # Get the most recent actual EPS announcement BEFORE as_of_date
        actuals = self._q("""
            SELECT pends, anndats, value
            FROM ibes_actuals
            WHERE ticker = %s AND measure = 'EPS'
              AND anndats IS NOT NULL AND anndats <= %s
            ORDER BY anndats DESC
            LIMIT 8
        """, (ibes_ticker, as_of_date))

        if actuals.empty:
            return feats

        latest_actual = actuals.iloc[0]
        actual_eps = latest_actual["value"]
        ann_date = latest_actual["anndats"]
        period_end = latest_actual["pends"]

        # Get the LAST consensus estimate BEFORE the announcement
        consensus = self._q("""
            SELECT meanest, medest, stdev, numest, statpers
            FROM ibes_summary
            WHERE ticker = %s AND measure = 'EPS'
              AND fpi = '6'
              AND fpedats = %s
              AND statpers < %s
            ORDER BY statpers DESC
            LIMIT 1
        """, (ibes_ticker, period_end, ann_date))

        if not consensus.empty:
            est = consensus.iloc[0]
            mean_est = est["meanest"]
            stdev_est = est["stdev"]
            n_analysts = est["numest"]

            # SUE = (actual - estimate) / |estimate|
            if mean_est is not None and actual_eps is not None and abs(float(mean_est)) > 0.01:
                feats["earnings_surprise"] = float(actual_eps - mean_est) / abs(float(mean_est))

            # Analyst dispersion
            if stdev_est is not None and mean_est is not None and abs(float(mean_est)) > 0.01:
                feats["analyst_dispersion"] = float(stdev_est) / abs(float(mean_est))

            feats["num_analysts"] = int(n_analysts) if n_analysts is not None else None

        # Earnings surprise std over last 8 quarters
        if len(actuals) >= 4:
            surprises = []
            for _, act_row in actuals.iterrows():
                p_end = act_row["pends"]
                a_date = act_row["anndats"]
                a_val = act_row["value"]
                if a_date is None or a_val is None:
                    continue
                cons = self._q("""
                    SELECT meanest FROM ibes_summary
                    WHERE ticker = %s AND measure = 'EPS' AND fpi = '6'
                      AND fpedats = %s AND statpers < %s
                    ORDER BY statpers DESC LIMIT 1
                """, (ibes_ticker, p_end, a_date))
                if not cons.empty and cons.iloc[0]["meanest"] is not None:
                    m = float(cons.iloc[0]["meanest"])
                    if abs(m) > 0.01:
                        surprises.append(float(a_val - m) / abs(m))
            if len(surprises) >= 3:
                feats["earnings_surprise_std"] = float(np.std(surprises))

        # Analyst revision = change in consensus over last 3 months
        revision = self._q("""
            SELECT meanest, statpers FROM ibes_summary
            WHERE ticker = %s AND measure = 'EPS' AND fpi = '6'
              AND statpers <= %s
            ORDER BY statpers DESC
            LIMIT 4
        """, (ibes_ticker, as_of_date))

        if len(revision) >= 2:
            newest = revision.iloc[0]["meanest"]
            oldest = revision.iloc[-1]["meanest"]
            if newest is not None and oldest is not None and abs(float(oldest)) > 0.01:
                feats["analyst_revision"] = float(newest - oldest) / abs(float(oldest))

        return feats

    # ───────────────────────────────────────────────────────────────
    # 8. Volatility features
    # ───────────────────────────────────────────────────────────────

    def _volatility_features(self, daily: pd.DataFrame, monthly: pd.DataFrame,
                             as_of_date: str) -> dict:
        feats = {k: None for k in [
            "realized_vol_1m", "realized_vol_3m", "realized_vol_12m",
            "idiosyncratic_vol", "beta", "downside_beta",
            "max_daily_return", "skewness",
        ]}

        if daily.empty:
            return feats

        daily_sorted = daily.sort_values("date")
        rets = daily_sorted["ret"].dropna().values

        sqrt252 = np.sqrt(252)

        if len(rets) >= 21:
            feats["realized_vol_1m"] = float(np.std(rets[-21:]) * sqrt252)
        if len(rets) >= 63:
            feats["realized_vol_3m"] = float(np.std(rets[-63:]) * sqrt252)
        if len(rets) >= 252:
            feats["realized_vol_12m"] = float(np.std(rets[-252:]) * sqrt252)

        # Max daily return in past month
        if len(rets) >= 21:
            feats["max_daily_return"] = float(np.max(np.abs(rets[-21:])))

        # Skewness (12-month)
        if len(rets) >= 252:
            from scipy.stats import skew
            feats["skewness"] = float(skew(rets[-252:]))
        elif len(rets) >= 63:
            from scipy.stats import skew
            feats["skewness"] = float(skew(rets[-63:]))

        # Beta and idiosyncratic vol from FF factors
        if not monthly.empty and len(monthly) >= 24:
            monthly_sorted = monthly.sort_values("date")
            dates = monthly_sorted["date"].tolist()
            stock_rets = monthly_sorted["ret"].values

            # Get market factor for matching dates
            min_date = str(dates[0])
            max_date = str(dates[-1])
            ff = self._q("""
                SELECT date, mktrf, smb, hml, rf
                FROM ff_factors_monthly
                WHERE date >= %s AND date <= %s
                ORDER BY date
            """, (min_date, max_date))

            if not ff.empty and len(ff) >= 12:
                # Merge on nearest month
                monthly_sorted = monthly_sorted.copy()
                monthly_sorted["date"] = pd.to_datetime(monthly_sorted["date"])
                ff["date"] = pd.to_datetime(ff["date"])

                merged = pd.merge_asof(
                    monthly_sorted.sort_values("date"),
                    ff.sort_values("date"),
                    on="date", direction="nearest", tolerance=pd.Timedelta("45D"),
                )

                valid = merged.dropna(subset=["ret", "mktrf", "rf"])
                if len(valid) >= 24:
                    excess_ret = valid["ret"].values - valid["rf"].values
                    mkt = valid["mktrf"].values

                    # Beta = cov(stock, market) / var(market)
                    if np.var(mkt) > 0:
                        feats["beta"] = float(np.cov(excess_ret, mkt)[0, 1] / np.var(mkt))

                        # Idiosyncratic vol = std(residuals)
                        beta_val = feats["beta"]
                        residuals = excess_ret - beta_val * mkt
                        feats["idiosyncratic_vol"] = float(np.std(residuals) * np.sqrt(12))

                    # Downside beta: using only negative market return days
                    neg_mask = mkt < 0
                    if neg_mask.sum() >= 10:
                        neg_ret = excess_ret[neg_mask]
                        neg_mkt = mkt[neg_mask]
                        if np.var(neg_mkt) > 0:
                            feats["downside_beta"] = float(
                                np.cov(neg_ret, neg_mkt)[0, 1] / np.var(neg_mkt)
                            )

        return feats

    # ───────────────────────────────────────────────────────────────
    # 9. Liquidity features
    # ───────────────────────────────────────────────────────────────

    def _liquidity_features(self, daily: pd.DataFrame) -> dict:
        feats = {k: None for k in [
            "turnover", "amihud_illiquidity", "dollar_volume", "bid_ask_spread",
        ]}

        if daily.empty or len(daily) < 21:
            return feats

        daily_sorted = daily.sort_values("date").tail(21)

        vol = daily_sorted["volume"].dropna().values
        close_px = daily_sorted["close"].dropna().values
        shrout = daily_sorted["shrout"].dropna().values
        rets = daily_sorted["ret"].dropna().values
        highs = daily_sorted["high"].dropna().values
        lows = daily_sorted["low"].dropna().values

        # Turnover = avg(volume / shares_outstanding)
        if len(vol) > 0 and len(shrout) > 0:
            shrout_thousands = shrout * 1000  # shrout is in thousands in stock_prices
            valid = shrout_thousands > 0
            if valid.any():
                t = vol[valid[:len(vol)]] / shrout_thousands[valid[:len(vol)]]
                feats["turnover"] = float(np.mean(t))

        # Dollar volume = log(mean(price * volume))
        if len(close_px) > 0 and len(vol) > 0:
            n = min(len(close_px), len(vol))
            dv = np.abs(close_px[:n]) * vol[:n]
            avg_dv = np.mean(dv)
            if avg_dv > 0:
                feats["dollar_volume"] = float(np.log(avg_dv))

        # Amihud illiquidity = mean(|ret| / dollar_volume)
        if len(rets) > 0 and len(close_px) > 0 and len(vol) > 0:
            n = min(len(rets), len(close_px), len(vol))
            dv = np.abs(close_px[:n]) * vol[:n]
            valid = dv > 0
            if valid.any():
                amihud = np.abs(rets[:n][valid]) / dv[valid]
                feats["amihud_illiquidity"] = float(np.mean(amihud))

        # Bid-ask spread (Corwin-Schultz high-low estimator)
        if len(highs) >= 2 and len(lows) >= 2:
            # sigma^2 = (1/(2*ln2)) * sum(ln(H/L)^2) for consecutive days
            n = min(len(highs), len(lows))
            log_hl = np.log(highs[:n] / np.maximum(lows[:n], 0.01))
            beta_cs = np.mean(log_hl[:-1] ** 2 + log_hl[1:] ** 2) if n >= 2 else 0
            gamma_cs = np.mean(
                np.log(
                    np.maximum(highs[:n-1], highs[1:n]) /
                    np.minimum(np.maximum(lows[:n-1], 0.01), np.maximum(lows[1:n], 0.01))
                ) ** 2
            ) if n >= 2 else 0

            alpha = (np.sqrt(2 * beta_cs) - np.sqrt(beta_cs)) / (3 - 2 * np.sqrt(2)) - np.sqrt(gamma_cs / (3 - 2 * np.sqrt(2)))
            spread = 2 * (np.exp(alpha) - 1) / (1 + np.exp(alpha)) if alpha > 0 else 0
            feats["bid_ask_spread"] = float(max(spread, 0))

        return feats

    # ───────────────────────────────────────────────────────────────
    # Batch computation
    # ───────────────────────────────────────────────────────────────

    def compute_panel(self, permnos: list, as_of_date: str) -> pd.DataFrame:
        """Compute characteristics for multiple stocks. Returns DataFrame."""
        rows = []
        for permno in permnos:
            try:
                chars = self.compute(permno, as_of_date)
                chars["permno"] = permno
                rows.append(chars)
            except Exception as e:
                logger.warning(f"  PERMNO {permno}: {e}")
                continue

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows)
        df = df.set_index("permno")
        return df

    # ───────────────────────────────────────────────────────────────
    # Feature list
    # ───────────────────────────────────────────────────────────────

    @staticmethod
    def feature_names() -> list:
        """Return the list of all feature names computed by this engine."""
        return [
            # Momentum
            "mom_1m", "mom_6m", "mom_12m", "mom_12_2", "mom_36m", "mom_1w", "mom_season",
            # Size
            "log_market_cap", "log_price",
            # Value
            "book_to_market", "earnings_to_price", "cashflow_to_price",
            "sales_to_price", "dividend_yield", "ev_to_ebitda",
            # Profitability
            "roe", "roa", "gross_profitability", "operating_profitability",
            "cash_profitability", "profit_margin", "asset_turnover",
            # Investment
            "asset_growth", "investment", "capex_to_assets", "rd_to_assets", "accruals",
            # Leverage
            "debt_to_equity", "current_ratio", "interest_coverage", "cash_to_assets",
            # Earnings
            "earnings_surprise", "earnings_surprise_std",
            "analyst_dispersion", "analyst_revision", "num_analysts",
            # Volatility
            "realized_vol_1m", "realized_vol_3m", "realized_vol_12m",
            "idiosyncratic_vol", "beta", "downside_beta",
            "max_daily_return", "skewness",
            # Liquidity
            "turnover", "amihud_illiquidity", "dollar_volume", "bid_ask_spread",
        ]
