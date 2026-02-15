"""
PHASE 2 — STEP 7: Data Access Layer (WRDSWarehouse)
=====================================================
Clean Python API for accessing the WRDS training data.

Usage:
    from wrds_pipeline.wrds_data_warehouse import WRDSWarehouse

    wh = WRDSWarehouse()

    # Full training panel
    panel = wh.get_training_panel(start='1970-01-01', end='2024-12-31')

    # Universe for a specific date
    universe = wh.get_universe('2024-06-30', min_mcap=1e9)

    # Features for a stock
    features = wh.get_stock_features(permno=14593, date='2024-06-30')

    # Daily returns
    daily = wh.get_daily_returns(permno=14593, start='2024-01-01', end='2024-12-31')

    # Cross-sectional ranks (features → percentile ranks per month)
    ranked = wh.get_ranked_panel(start='2000-01-01', end='2024-12-31')

    # Train/test split
    train, test = wh.expanding_window_split(gap_months=1, test_months=12)
"""

import pandas as pd
import numpy as np
import os
import glob
from functools import lru_cache
import warnings
warnings.filterwarnings("ignore")


class WRDSWarehouse:
    """Unified data access layer for the WRDS training data warehouse."""

    def __init__(self, data_dir="data/wrds"):
        self.data_dir = data_dir
        self.panel_path = os.path.join(data_dir, "training_panel.parquet")
        self.daily_dir = os.path.join(data_dir, "crsp_daily")
        self._panel_cache = None
        self._daily_cache = {}

        # Feature columns by group (for convenience)
        self.feature_groups = {
            "momentum": ["mom_1m", "mom_3m", "mom_6m", "mom_12m", "mom_12_2",
                         "str_reversal"],
            "volatility": ["vol_3m", "vol_6m", "vol_12m", "realized_vol",
                          "down_vol", "up_vol"],
            "size": ["log_market_cap", "log_price", "market_cap"],
            "liquidity": ["turnover", "turnover_3m", "turnover_6m",
                         "amihud_illiq", "zero_vol_days"],
            "daily_derived": ["max_daily_ret", "min_daily_ret", "realized_vol",
                             "return_skewness", "return_kurtosis", "intraday_range"],
            "betas": ["alpha", "beta_mkt", "beta_smb", "beta_hml",
                     "beta_rmw", "beta_cma", "beta_umd",
                     "idio_vol", "total_vol", "r_squared"],
            "earnings": ["sue", "analyst_dispersion", "analyst_revision",
                        "num_analysts", "rec_score", "num_recs"],
            "institutional": ["total_inst_shares", "num_institutions"],
            "insider": ["insider_buy_ratio", "insider_num_buys", "insider_num_sells"],
            "labels": ["fwd_ret_1m", "fwd_ret_3m", "fwd_ret_6m", "fwd_ret_12m",
                       "excess_ret_1m"],
        }

    def _load_panel(self, force_reload=False):
        """Load (and cache) the training panel."""
        if self._panel_cache is None or force_reload:
            if not os.path.exists(self.panel_path):
                raise FileNotFoundError(
                    f"Training panel not found at {self.panel_path}. "
                    "Run phase2_step3_training_panel.py first."
                )
            self._panel_cache = pd.read_parquet(self.panel_path)
            self._panel_cache["date"] = pd.to_datetime(self._panel_cache["date"])
        return self._panel_cache

    def get_training_panel(self, start=None, end=None, columns=None,
                            min_mcap=None, common_stocks_only=True,
                            major_exchanges_only=True, min_price=None):
        """
        Get the full training panel, optionally filtered.

        Parameters:
            start: Start date (inclusive)
            end: End date (inclusive)
            columns: List of columns to return (None = all)
            min_mcap: Minimum market cap filter
            common_stocks_only: Filter to shrcd IN (10, 11)
            major_exchanges_only: Filter to NYSE/AMEX/NASDAQ (exchcd IN (1,2,3))
            min_price: Minimum absolute price filter
        """
        panel = self._load_panel()

        if start:
            panel = panel[panel["date"] >= pd.Timestamp(start)]
        if end:
            panel = panel[panel["date"] <= pd.Timestamp(end)]

        if common_stocks_only and "shrcd" in panel.columns:
            panel = panel[panel["shrcd"].isin([10, 11])]

        if major_exchanges_only and "exchcd" in panel.columns:
            panel = panel[panel["exchcd"].isin([1, 2, 3])]

        if min_mcap is not None and "market_cap" in panel.columns:
            panel = panel[panel["market_cap"] >= min_mcap]

        if min_price is not None and "log_price" in panel.columns:
            panel = panel[np.exp(panel["log_price"]) >= min_price]

        if columns:
            available = [c for c in columns if c in panel.columns]
            panel = panel[["permno", "date"] + available]

        return panel.reset_index(drop=True)

    def get_universe(self, date, min_mcap=None, common_stocks_only=True,
                     major_exchanges_only=True, min_price=5.0):
        """Get investable universe for a specific date."""
        panel = self._load_panel()
        dt = pd.Timestamp(date) + pd.offsets.MonthEnd(0)

        universe = panel[panel["date"] == dt].copy()

        if common_stocks_only and "shrcd" in universe.columns:
            universe = universe[universe["shrcd"].isin([10, 11])]

        if major_exchanges_only and "exchcd" in universe.columns:
            universe = universe[universe["exchcd"].isin([1, 2, 3])]

        if min_mcap is not None and "market_cap" in universe.columns:
            universe = universe[universe["market_cap"] >= min_mcap]

        if min_price is not None and "log_price" in universe.columns:
            universe = universe[np.exp(universe["log_price"]) >= min_price]

        return universe.reset_index(drop=True)

    def get_stock_features(self, permno, date=None, start=None, end=None):
        """Get features for a specific stock (optionally at a specific date)."""
        panel = self._load_panel()
        stock = panel[panel["permno"] == permno].copy()

        if date:
            dt = pd.Timestamp(date) + pd.offsets.MonthEnd(0)
            stock = stock[stock["date"] == dt]
        else:
            if start:
                stock = stock[stock["date"] >= pd.Timestamp(start)]
            if end:
                stock = stock[stock["date"] <= pd.Timestamp(end)]

        return stock.reset_index(drop=True)

    def get_daily_returns(self, permno=None, start=None, end=None):
        """Get daily CRSP returns from parquet files."""
        files = sorted(glob.glob(os.path.join(self.daily_dir, "crsp_daily_*.parquet")))
        if not files:
            raise FileNotFoundError(
                f"No daily files in {self.daily_dir}. "
                "Run phase2_step1_crsp_daily.py first."
            )

        # Filter files by year range if possible
        if start or end:
            start_year = pd.Timestamp(start).year if start else 1926
            end_year = pd.Timestamp(end).year if end else 2025
            files = [f for f in files
                     if start_year <= int(os.path.basename(f).split("_")[-1].split(".")[0]) <= end_year]

        dfs = []
        for f in files:
            df = pd.read_parquet(f)
            df["date"] = pd.to_datetime(df["date"])

            if permno is not None:
                df = df[df["permno"] == permno]
            if start:
                df = df[df["date"] >= pd.Timestamp(start)]
            if end:
                df = df[df["date"] <= pd.Timestamp(end)]

            if len(df) > 0:
                dfs.append(df)

        if not dfs:
            return pd.DataFrame()

        return pd.concat(dfs, ignore_index=True).sort_values(["permno", "date"])

    def get_ranked_panel(self, start=None, end=None, feature_cols=None,
                          rank_method="percent"):
        """
        Convert features to cross-sectional percentile ranks per month.
        This removes scale differences and makes features directly comparable.

        Parameters:
            rank_method: 'percent' (0-1) or 'normal' (inverse normal transform)
        """
        panel = self.get_training_panel(start=start, end=end)

        if feature_cols is None:
            # Get all available feature columns
            exclude = {"permno", "date", "ret", "market_cap",
                       "fwd_ret_1m", "fwd_ret_3m", "fwd_ret_6m", "fwd_ret_12m",
                       "excess_ret_1m", "rf", "exchcd", "shrcd", "siccd",
                       "sp500_member", "dlstcd", "dlret", "n_months", "num_trading_days"}
            feature_cols = [c for c in panel.columns
                           if c not in exclude and panel[c].dtype in ["float64", "float32"]]

        for col in feature_cols:
            if col in panel.columns:
                if rank_method == "percent":
                    panel[col] = panel.groupby("date")[col].transform(
                        lambda x: x.rank(pct=True)
                    )
                elif rank_method == "normal":
                    from scipy.stats import norm
                    panel[col] = panel.groupby("date")[col].transform(
                        lambda x: pd.Series(
                            norm.ppf(x.rank(pct=True).clip(0.001, 0.999)),
                            index=x.index
                        )
                    )

        return panel

    def expanding_window_split(self, train_end, test_start=None, test_end=None,
                                gap_months=1, test_months=12,
                                min_mcap=None, common_stocks_only=True):
        """
        Create train/test split with expanding window.

        Parameters:
            train_end: Last month of training data
            test_start: First month of test data (default: train_end + gap_months)
            test_end: Last month of test data (default: test_start + test_months)
            gap_months: Gap between train and test to prevent lookahead
        """
        panel = self.get_training_panel(min_mcap=min_mcap,
                                         common_stocks_only=common_stocks_only)

        train_end_dt = pd.Timestamp(train_end) + pd.offsets.MonthEnd(0)

        if test_start is None:
            test_start_dt = train_end_dt + pd.DateOffset(months=gap_months)
            test_start_dt = test_start_dt + pd.offsets.MonthEnd(0)
        else:
            test_start_dt = pd.Timestamp(test_start) + pd.offsets.MonthEnd(0)

        if test_end is None:
            test_end_dt = test_start_dt + pd.DateOffset(months=test_months)
            test_end_dt = test_end_dt + pd.offsets.MonthEnd(0)
        else:
            test_end_dt = pd.Timestamp(test_end) + pd.offsets.MonthEnd(0)

        train = panel[panel["date"] <= train_end_dt]
        test = panel[(panel["date"] >= test_start_dt) & (panel["date"] <= test_end_dt)]

        return train, test

    def rolling_window_splits(self, start_year=2000, end_year=2024,
                               train_years=10, test_months=12, gap_months=1):
        """
        Generate rolling window train/test splits.
        Yields (train_df, test_df, fold_info) tuples.
        """
        for year in range(start_year, end_year - 1):
            train_end = f"{year}-12-31"
            train_start = f"{year - train_years + 1}-01-01"
            test_start = f"{year + 1}-{gap_months + 1:02d}-01"
            test_end = f"{year + 1}-12-31"

            train, test = self.expanding_window_split(
                train_end=train_end,
                test_start=test_start,
                test_end=test_end
            )

            if len(train) > 0 and len(test) > 0:
                fold_info = {
                    "train_period": f"{train_start} to {train_end}",
                    "test_period": f"{test_start} to {test_end}",
                    "train_rows": len(train),
                    "test_rows": len(test),
                }
                yield train, test, fold_info

    def get_feature_names(self, group=None):
        """Get list of feature column names, optionally by group."""
        if group:
            return self.feature_groups.get(group, [])

        all_features = []
        for cols in self.feature_groups.values():
            all_features.extend(cols)
        return list(set(all_features))

    def summary(self):
        """Print a summary of the data warehouse contents."""
        panel = self._load_panel()

        print("=" * 60)
        print("WRDS DATA WAREHOUSE SUMMARY")
        print("=" * 60)
        print(f"  Panel rows:      {len(panel):,}")
        print(f"  Panel columns:   {panel.shape[1]}")
        print(f"  Date range:      {panel['date'].min().date()} to {panel['date'].max().date()}")
        print(f"  Unique stocks:   {panel['permno'].nunique():,}")
        print(f"  Avg stocks/mo:   {len(panel) / panel['date'].nunique():.0f}")

        file_size = os.path.getsize(self.panel_path) / (1024 ** 3)
        print(f"  Panel file size: {file_size:.2f} GB")

        # Daily data
        daily_files = glob.glob(os.path.join(self.daily_dir, "*.parquet"))
        if daily_files:
            total_daily_size = sum(os.path.getsize(f) for f in daily_files) / (1024 ** 3)
            print(f"  Daily files:     {len(daily_files)} years ({total_daily_size:.2f} GB)")

        # Feature coverage
        print(f"\n  Feature coverage:")
        for group, cols in self.feature_groups.items():
            available = [c for c in cols if c in panel.columns]
            if available:
                coverage = panel[available].notna().mean().mean()
                print(f"    {group:<20} {len(available)}/{len(cols)} cols | "
                      f"{coverage:.1%} coverage")

        print("=" * 60)
