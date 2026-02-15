#!/usr/bin/env python3
"""
Unified Data Access Layer for ML Pipeline
============================================
Joins CRSP prices + Compustat fundamentals + IBES estimates + FF factors.

All the tricky join logic is encapsulated here so downstream ML code
just calls:

    from wrds_pipeline.data_access import WRDSDataAccess
    da = WRDSDataAccess()
    panel = da.get_training_panel(start_date='2000-01-01', end_date='2024-12-31')

KEY DESIGN DECISIONS:

1. POINT-IN-TIME DISCIPLINE:
   - Compustat: uses rdq (report date) NOT datadate
   - IBES: uses statpers (when consensus measured) NOT fpedats
   - Prevents 5-15% backtest inflation from lookahead bias

2. PERMNO AS PRIMARY KEY:
   Everything maps through CRSP PERMNO:
   - CRSP prices → PERMNO (native)
   - Compustat → GVKEY → PERMNO (via crsp_compustat_link)
   - IBES → IBES ticker → PERMNO (via ibes_crsp_link)

3. SURVIVORSHIP-BIAS FREE:
   Includes delisted stocks. Uses delisting returns from CRSP.
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd
import psycopg2

from wrds_pipeline.config import RDS_CONFIG
from wrds_pipeline.characteristics import CharacteristicsEngine

logger = logging.getLogger("wrds_data_access")


class WRDSDataAccess:
    """
    Provides clean, joined datasets from the RDS warehouse.

    Main methods:
        get_training_panel()   — full ML training dataset
        get_monthly_panel()    — CRSP monthly returns + FF factors
        get_characteristics()  — 50+ Gu-Kelly-Xiu features for one stock
        get_earnings_surprise()— SUE for one stock
        get_book_equity()      — Fama-French book equity
        execute()              — raw SQL
        query_df()             — raw SQL → DataFrame
    """

    def __init__(self, conn=None):
        if conn is not None:
            self.conn = conn
            self._own_conn = False
        else:
            self.conn = psycopg2.connect(**RDS_CONFIG)
            self.conn.autocommit = True
            self._own_conn = True

        self._chars_engine = CharacteristicsEngine(conn=self.conn)

    def close(self):
        self._chars_engine.close()
        if self._own_conn:
            self.conn.close()

    # ───────────────────────────────────────────────────────────────
    # Low-level query helpers
    # ───────────────────────────────────────────────────────────────

    def execute(self, sql: str, params=None) -> list:
        """Execute SQL and return list of tuples."""
        with self.conn.cursor() as cur:
            cur.execute(sql, params)
            return cur.fetchall()

    def query_df(self, sql: str, params=None) -> pd.DataFrame:
        """Execute SQL and return DataFrame."""
        return pd.read_sql_query(sql, self.conn, params=params)

    # ───────────────────────────────────────────────────────────────
    # Monthly panel (CRSP returns + delisting adjustment)
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
        Build the MONTHLY cross-sectional panel from CRSP.

        Returns DataFrame with columns:
            permno, date, ticker, ret, retx, prc, shrout, vol, market_cap,
            exchcd, shrcd, siccd, dlret, dlstcd, ret_adj

        ret_adj = delisting-adjusted return:
            if dlret exists: (1+ret)*(1+dlret) - 1
            else: ret
        """
        exchcd_str = ",".join(str(e) for e in exchanges)

        query = f"""
        SELECT cm.permno, cm.date, cm.ticker,
               cm.ret, cm.retx, cm.prc, cm.shrout, cm.vol, cm.market_cap,
               cm.exchcd, cm.shrcd, cm.siccd,
               cm.dlret, cm.dlstcd,
               CASE
                   WHEN cm.dlret IS NOT NULL AND cm.ret IS NOT NULL
                   THEN (1 + cm.ret) * (1 + cm.dlret) - 1
                   WHEN cm.dlret IS NOT NULL THEN cm.dlret
                   ELSE cm.ret
               END AS ret_adj
        FROM crsp_monthly cm
        WHERE cm.date >= %(start)s
          AND cm.date <= %(end)s
          AND cm.exchcd IN ({exchcd_str})
          AND cm.shrcd IN (10, 11)
        """

        filters = []
        params = {"start": start_date, "end": end_date}

        if min_market_cap:
            filters.append("AND cm.market_cap >= %(min_mc)s")
            params["min_mc"] = min_market_cap

        if min_price:
            filters.append("AND ABS(cm.prc) >= %(min_px)s")
            params["min_px"] = min_price

        query += "\n".join(filters)
        query += "\nORDER BY cm.permno, cm.date"

        df = pd.read_sql_query(query, self.conn, params=params)
        logger.info(
            f"Monthly panel: {len(df):,} rows, "
            f"{df['permno'].nunique():,} stocks, "
            f"{df['date'].nunique()} months"
        )
        return df

    # ───────────────────────────────────────────────────────────────
    # Fama-French factors
    # ───────────────────────────────────────────────────────────────

    def get_ff_factors(
        self, start_date: str = "2000-01-01", end_date: str = "2024-12-31",
        frequency: str = "monthly",
    ) -> pd.DataFrame:
        """Get Fama-French 5 factors + momentum."""
        table = "ff_factors_monthly" if frequency == "monthly" else "ff_factors_daily"
        return self.query_df(
            f"SELECT * FROM {table} WHERE date >= %s AND date <= %s ORDER BY date",
            (start_date, end_date),
        )

    # ───────────────────────────────────────────────────────────────
    # Stock characteristics (delegates to CharacteristicsEngine)
    # ───────────────────────────────────────────────────────────────

    def get_characteristics(self, permno: int, as_of_date: str) -> dict:
        """Compute all Gu-Kelly-Xiu characteristics for a stock."""
        return self._chars_engine.compute(permno, as_of_date)

    # ───────────────────────────────────────────────────────────────
    # Earnings surprise
    # ───────────────────────────────────────────────────────────────

    def get_earnings_surprise(self, permno: int, as_of_date: str) -> dict:
        """
        Compute standardized unexpected earnings (SUE).

        Method 1: Analyst-based (IBES)
            SUE = (actual_EPS - consensus) / |consensus|
            Uses LAST consensus before announcement date.

        Method 2: Time-series (if no analyst coverage)
            SUE = (EPS_q - EPS_q-4) / std(EPS_q - EPS_q-4)
            Seasonal random walk model.
        """
        # Get IBES ticker
        ibes_ticker = self._chars_engine._get_ibes_ticker(permno, as_of_date)

        result = {
            "sue_analyst": None,
            "sue_timeseries": None,
            "method": None,
        }

        # Method 1: IBES analyst-based
        if ibes_ticker:
            actuals = self.query_df("""
                SELECT pends, anndats, value FROM ibes_actuals
                WHERE ticker = %s AND measure = 'EPS'
                  AND anndats IS NOT NULL AND anndats <= %s
                ORDER BY anndats DESC LIMIT 1
            """, (ibes_ticker, as_of_date))

            if not actuals.empty:
                actual = actuals.iloc[0]
                actual_eps = actual["value"]
                ann_date = actual["anndats"]
                period_end = actual["pends"]

                consensus = self.query_df("""
                    SELECT meanest FROM ibes_summary
                    WHERE ticker = %s AND measure = 'EPS' AND fpi = '6'
                      AND fpedats = %s AND statpers < %s
                    ORDER BY statpers DESC LIMIT 1
                """, (ibes_ticker, period_end, ann_date))

                if not consensus.empty:
                    mean_est = consensus.iloc[0]["meanest"]
                    if mean_est is not None and actual_eps is not None and abs(float(mean_est)) > 0.01:
                        result["sue_analyst"] = float(actual_eps - mean_est) / abs(float(mean_est))
                        result["method"] = "analyst"

        # Method 2: Time-series (Compustat)
        gvkey = self._chars_engine._get_gvkey(permno, as_of_date)
        if gvkey:
            eps_q = self.query_df("""
                SELECT datadate, epspxq FROM compustat_quarterly
                WHERE gvkey = %s AND rdq IS NOT NULL AND rdq <= %s
                  AND epspxq IS NOT NULL
                ORDER BY datadate DESC LIMIT 8
            """, (gvkey, as_of_date))

            if len(eps_q) >= 5:
                eps_q = eps_q.sort_values("datadate")
                vals = eps_q["epspxq"].values
                # Seasonal difference: EPS_q - EPS_q-4
                diffs = []
                for i in range(4, len(vals)):
                    diffs.append(vals[i] - vals[i - 4])
                if len(diffs) >= 2:
                    std_diff = float(np.std(diffs))
                    if std_diff > 0.01:
                        result["sue_timeseries"] = float(diffs[-1]) / std_diff
                        if result["method"] is None:
                            result["method"] = "timeseries"

        return result

    # ───────────────────────────────────────────────────────────────
    # Book equity (Fama-French 1993)
    # ───────────────────────────────────────────────────────────────

    def get_book_equity(self, gvkey: str, as_of_date: str) -> Optional[float]:
        """
        BE = stockholders_equity + deferred_taxes - preferred_stock

        Stockholders equity priority: seq → ceq + pstk → at - lt
        Point-in-time: only filings with rdq <= as_of_date
        """
        return self._chars_engine.book_equity(gvkey, as_of_date)

    # ───────────────────────────────────────────────────────────────
    # TRAINING PANEL (the main function for ML)
    # ───────────────────────────────────────────────────────────────

    def get_training_panel(
        self,
        start_date: str = "2000-01-01",
        end_date: str = "2024-12-31",
        frequency: str = "monthly",
        min_market_cap: float = 10_000_000,
        min_price: float = 5.0,
        exchanges: list = [1, 2, 3],
        max_stocks_per_month: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        THE main function for ML training.

        Returns a panel: (permno × date) × features

        Steps:
            1. Get CRSP monthly returns for the date range
            2. Filter to liquid, investable stocks
            3. For each (permno, date), compute all characteristics
            4. Add forward return labels (1-month, 3-month)
            5. Add Fama-French factor returns
            6. Return clean panel ready for LightGBM

        Expected output:
            ~3,000-5,000 stocks per month
            ~300 months (2000-2024)
            = ~1M-1.5M observations
            50+ feature columns + labels + identifiers
        """
        logger.info(f"Building training panel: {start_date} → {end_date}")

        # Step 1: Get monthly returns
        monthly = self.get_monthly_panel(
            start_date=start_date, end_date=end_date,
            min_market_cap=min_market_cap, min_price=min_price,
            exchanges=exchanges,
        )

        if monthly.empty:
            logger.error("No monthly data found!")
            return pd.DataFrame()

        # Step 2: Add forward returns
        monthly = monthly.sort_values(["permno", "date"])
        monthly["fwd_ret_1m"] = monthly.groupby("permno")["ret_adj"].shift(-1)
        monthly["fwd_ret_3m"] = (
            monthly.groupby("permno")["ret_adj"]
            .apply(lambda x: (1 + x).rolling(3).apply(np.prod, raw=True).shift(-3) - 1)
            .reset_index(level=0, drop=True)
        )

        # Step 3: Get unique (permno, date) pairs
        dates = sorted(monthly["date"].unique())
        logger.info(f"  {len(dates)} months, computing characteristics...")

        all_chars = []
        for i, dt in enumerate(dates):
            dt_str = str(dt)
            month_stocks = monthly[monthly["date"] == dt]

            if max_stocks_per_month and len(month_stocks) > max_stocks_per_month:
                # Keep largest by market cap
                month_stocks = month_stocks.nlargest(max_stocks_per_month, "market_cap")

            permnos = month_stocks["permno"].unique().tolist()

            for permno in permnos:
                try:
                    chars = self._chars_engine.compute(permno, dt_str)
                    chars["permno"] = permno
                    chars["date"] = dt
                    all_chars.append(chars)
                except Exception:
                    continue

            if (i + 1) % 12 == 0 or i == len(dates) - 1:
                logger.info(
                    f"    Month {i+1}/{len(dates)}: "
                    f"{len(all_chars):,} obs so far"
                )

        if not all_chars:
            logger.error("No characteristics computed!")
            return pd.DataFrame()

        chars_df = pd.DataFrame(all_chars)
        chars_df["date"] = pd.to_datetime(chars_df["date"])
        monthly["date"] = pd.to_datetime(monthly["date"])

        # Step 4: Merge characteristics with returns
        panel = monthly.merge(
            chars_df,
            on=["permno", "date"],
            how="inner",
        )

        # Step 5: Add FF factors
        ff = self.get_ff_factors(start_date, end_date, frequency="monthly")
        if not ff.empty:
            ff["date"] = pd.to_datetime(ff["date"])
            panel = panel.merge(ff, on="date", how="left")

        # Step 6: Clean up
        feature_cols = CharacteristicsEngine.feature_names()
        available_features = [c for c in feature_cols if c in panel.columns]

        logger.info(f"")
        logger.info(f"  ═══ TRAINING PANEL COMPLETE ═══")
        logger.info(f"  Observations:     {len(panel):>12,}")
        logger.info(f"  Stocks per month: {panel.groupby('date')['permno'].nunique().mean():>12.0f}")
        logger.info(f"  Features:         {len(available_features):>12}")
        logger.info(f"  Date range:       {panel['date'].min()} → {panel['date'].max()}")

        null_rates = panel[available_features].isnull().mean()
        low_null = (null_rates < 0.1).sum()
        high_null = (null_rates > 0.5).sum()
        logger.info(f"  Features <10% null: {low_null}")
        logger.info(f"  Features >50% null: {high_null}")

        return panel

    # ───────────────────────────────────────────────────────────────
    # Quick data summary
    # ───────────────────────────────────────────────────────────────

    def summary(self):
        """Print a quick summary of all data in RDS."""
        tables = [
            "stock_prices", "compustat_quarterly", "compustat_annual",
            "ibes_summary", "ibes_actuals", "ff_factors_daily",
            "ff_factors_monthly", "crsp_compustat_link", "crsp_monthly",
            "ibes_crsp_link", "compustat_security",
        ]
        print("\n" + "═" * 55)
        print("  WRDS RDS DATA SUMMARY")
        print("═" * 55)
        for t in tables:
            try:
                count = self.execute(f"SELECT COUNT(*) FROM {t}")[0][0]
                print(f"  {t:35s}: {count:>12,}")
            except Exception:
                print(f"  {t:35s}: NOT FOUND")
        print("═" * 55)


# ═══════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="WRDS Data Access")
    parser.add_argument("--summary", action="store_true", help="Print data summary")
    parser.add_argument("--test", action="store_true", help="Run quick test")
    parser.add_argument("--panel", action="store_true", help="Build training panel (slow)")
    parser.add_argument("--start", default="2023-01-01")
    parser.add_argument("--end", default="2023-12-31")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    da = WRDSDataAccess()

    if args.summary:
        da.summary()

    elif args.test:
        da.summary()

        print("\n=== SAMPLE CHARACTERISTICS: AAPL (PERMNO=14593) ===")
        chars = da.get_characteristics(permno=14593, as_of_date="2024-06-30")
        print(f"  Features computed: {len([v for v in chars.values() if v is not None])}/{len(chars)}")
        for k, v in sorted(chars.items()):
            if v is not None:
                print(f"    {k:35s}: {v:>12.4f}")

        print("\n=== EARNINGS SURPRISE: AAPL ===")
        sue = da.get_earnings_surprise(permno=14593, as_of_date="2024-06-30")
        for k, v in sue.items():
            print(f"    {k}: {v}")

    elif args.panel:
        panel = da.get_training_panel(
            start_date=args.start, end_date=args.end,
            max_stocks_per_month=200,  # limit for quick test
        )
        print(f"\nPanel shape: {panel.shape}")
        if not panel.empty:
            print(f"Columns: {list(panel.columns)}")
            print(panel.head())

    da.close()
