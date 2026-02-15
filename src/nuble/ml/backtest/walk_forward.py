"""
Walk-Forward Backtest — Institutional Validation Framework
============================================================

Expanding-window walk-forward backtest for the universal model.
This is the GOLD STANDARD validation at institutional quant funds.

It answers: "If I had trained this model at time T and traded on it
at time T+1, what would my returns have been?"

METHODOLOGY:

1. EXPANDING WINDOW TRAINING:
   - Minimum training window: 120 trading days (~6 months)
   - Retrain every 21 trading days (~monthly)
   - Purge gap: 10 trading days between train and test
   - Each test window: 21 trading days

2. PER-DATE SCORING:
   - For each test date, score ALL stocks in the active universe
   - Rank stocks by predicted LONG probability
   - Form long/short portfolios (decile-based)

3. PORTFOLIO RETURNS:
   - Equal-weight within each decile
   - Long-short = avg(long decile) - avg(short decile)

4. METRICS per window: IC, long-short return, hit rate, decile spread

Author: NUBLE ML Pipeline — Phase 5 Walk-Forward Validation
"""

from __future__ import annotations

import gc
import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

if TYPE_CHECKING:
    from ...data.polygon_universe import PolygonUniverseData
    from ..universal_features import UniversalFeatureEngine

logger = logging.getLogger(__name__)

# Reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)


# ══════════════════════════════════════════════════════════════
# BacktestResults
# ══════════════════════════════════════════════════════════════


class BacktestResults:
    """
    Container for walk-forward backtest results with analysis methods.
    Stores per-date and per-window metrics for comprehensive evaluation.
    """

    def __init__(self):
        self.daily_results: List[Dict[str, Any]] = []
        self.window_results: List[Dict[str, Any]] = []
        self._predictions_records: List[Dict[str, Any]] = []
        self._daily_portfolios: List[Dict[str, Any]] = []
        self._stock_histories: Optional[Dict[str, Any]] = None  # Cache for signal analysis

    def add_daily_result(self, result: Dict[str, Any]) -> None:
        self.daily_results.append(result)

    def add_window_result(self, result: Dict[str, Any]) -> None:
        self.window_results.append(result)

    def add_predictions(self, date: str, preds: Dict[str, float]) -> None:
        """Store predictions for signal analysis. preds = {ticker: long_prob}."""
        for ticker, prob in preds.items():
            self._predictions_records.append(
                {"date": date, "ticker": ticker, "pred_prob": prob}
            )

    def add_daily_portfolio(self, date: str, long_tickers: List[str],
                            short_tickers: List[str]) -> None:
        self._daily_portfolios.append({
            "date": date,
            "long_tickers": long_tickers,
            "short_tickers": short_tickers,
        })

    def get_predictions_df(self) -> pd.DataFrame:
        if not self._predictions_records:
            return pd.DataFrame(columns=["date", "ticker", "pred_prob"])
        return pd.DataFrame(self._predictions_records)

    def get_daily_portfolios(self) -> List[Dict[str, Any]]:
        return self._daily_portfolios

    def set_stock_histories(self, histories: Dict[str, Any]) -> None:
        """Cache stock histories for use by signal analysis (avoid re-loading)."""
        self._stock_histories = histories

    def get_stock_histories(self) -> Optional[Dict[str, Any]]:
        """Get cached stock histories (set during backtest run)."""
        return self._stock_histories

    # ── Summary ───────────────────────────────────────────────

    def summary(self) -> Dict[str, Any]:
        """Aggregate results across all test dates."""
        if not self.daily_results:
            return {"error": "No results — backtest has not been run."}

        ics = [d["ic"] for d in self.daily_results if not np.isnan(d.get("ic", np.nan))]
        ls_rets = [d["long_short_return"] for d in self.daily_results
                    if not np.isnan(d.get("long_short_return", np.nan))]
        long_rets = [d["long_return"] for d in self.daily_results
                      if not np.isnan(d.get("long_return", np.nan))]
        short_rets = [d["short_return"] for d in self.daily_results
                       if not np.isnan(d.get("short_return", np.nan))]
        hit_rates = [d["hit_rate"] for d in self.daily_results
                      if not np.isnan(d.get("hit_rate", np.nan))]

        mean_ic = float(np.mean(ics)) if ics else 0.0
        ic_std = float(np.std(ics)) if ics else 1.0
        ic_ir = mean_ic / ic_std if ic_std > 1e-8 else 0.0
        ic_hit_rate = float(np.mean([1 if ic > 0 else 0 for ic in ics])) if ics else 0.0

        # Cumulative long-short return
        # NOTE: L/S returns are per-observation (one per test date).
        # With 10-day forward returns sampled daily, there is overlap
        # (serial correlation). We sum raw returns and annualize
        # using the number of observations and 252 trading days.
        cum_ls = float(np.sum(ls_rets)) if ls_rets else 0.0
        n_days = len(ls_rets)
        ann_factor = 252 / max(n_days, 1)
        ann_ls = cum_ls * ann_factor

        # Sharpe — uses daily observation frequency.
        # With overlapping 10-day returns, the standard deviation is
        # inflated by serial correlation. This Sharpe is approximate;
        # a Newey-West correction would be more precise.
        if ls_rets and len(ls_rets) > 1:
            ls_sharpe = float(np.mean(ls_rets) / (np.std(ls_rets) + 1e-8) * np.sqrt(252))
        else:
            ls_sharpe = 0.0

        # Max drawdown of L/S equity curve
        if ls_rets:
            cum_curve = np.cumsum(ls_rets)
            running_max = np.maximum.accumulate(cum_curve)
            drawdowns = cum_curve - running_max
            max_dd = float(np.min(drawdowns)) if len(drawdowns) > 0 else 0.0
        else:
            max_dd = 0.0

        ls_hit_rate = float(np.mean([1 if r > 0 else 0 for r in ls_rets])) if ls_rets else 0.0

        # Decile spread analysis
        decile_returns_all = [d.get("decile_returns") for d in self.daily_results
                               if d.get("decile_returns") is not None]
        if decile_returns_all:
            avg_decile = np.nanmean(decile_returns_all, axis=0)
            avg_decile_spread = float(avg_decile[-1] - avg_decile[0]) if len(avg_decile) >= 2 else 0.0
            # Monotonicity: Kendall's tau of average decile returns vs decile rank
            if len(avg_decile) >= 3:
                tau, _ = sp_stats.kendalltau(np.arange(len(avg_decile)), avg_decile)
                decile_mono = float(tau) if not np.isnan(tau) else 0.0
            else:
                decile_mono = 0.0
        else:
            avg_decile = []
            avg_decile_spread = 0.0
            decile_mono = 0.0

        # IC by window
        ic_by_window = []
        for w in self.window_results:
            ic_by_window.append(w.get("mean_ic", 0.0))

        # IC trend
        if len(ic_by_window) >= 3:
            half = len(ic_by_window) // 2
            first_half = np.mean(ic_by_window[:half])
            second_half = np.mean(ic_by_window[half:])
            if second_half > first_half + 0.005:
                ic_trend = "improving"
            elif second_half < first_half - 0.005:
                ic_trend = "degrading"
            else:
                ic_trend = "stable"
        else:
            ic_trend = "insufficient_data"

        # IC autocorrelation (persistence)
        if len(ics) > 10:
            ic_series = pd.Series(ics)
            try:
                ic_autocorr = float(ic_series.autocorr(lag=1))
                if np.isnan(ic_autocorr):
                    ic_autocorr = 0.0
            except Exception:
                ic_autocorr = 0.0
        else:
            ic_autocorr = 0.0

        pct_windows_pos = float(np.mean([1 if ic > 0 else 0 for ic in ic_by_window])) if ic_by_window else 0.0

        # Date range
        all_dates = sorted(set(d["date"] for d in self.daily_results))
        date_range_start = all_dates[0] if all_dates else ""
        date_range_end = all_dates[-1] if all_dates else ""

        return {
            "total_test_days": n_days,
            "total_retrain_windows": len(self.window_results),
            "date_range_start": date_range_start,
            "date_range_end": date_range_end,
            # IC
            "mean_ic": round(mean_ic, 4),
            "ic_std": round(ic_std, 4),
            "ic_ir": round(ic_ir, 2),
            "ic_hit_rate": round(ic_hit_rate, 3),
            "ic_by_window": [round(x, 4) for x in ic_by_window],
            "ic_trend": ic_trend,
            "ic_autocorrelation": round(ic_autocorr, 3),
            "worst_window_ic": round(min(ic_by_window), 4) if ic_by_window else 0.0,
            "best_window_ic": round(max(ic_by_window), 4) if ic_by_window else 0.0,
            "pct_windows_positive_ic": round(pct_windows_pos, 3),
            # Returns
            "cumulative_long_short_return": round(cum_ls, 4),
            "annualized_long_short_return": round(ann_ls, 4),
            "long_short_sharpe": round(ls_sharpe, 2),
            "max_drawdown": round(max_dd, 4),
            "avg_long_return": round(float(np.mean(long_rets)), 6) if long_rets else 0.0,
            "avg_short_return": round(float(np.mean(short_rets)), 6) if short_rets else 0.0,
            "long_short_hit_rate": round(ls_hit_rate, 3),
            # Decile
            "avg_decile_spread": round(avg_decile_spread, 6),
            "avg_decile_returns": [round(float(x), 6) for x in avg_decile] if len(avg_decile) > 0 else [],
            "decile_monotonicity": round(decile_mono, 3),
            "fwd_return_horizon": 10,  # days — must match model training horizon
        }

    # ── Report ────────────────────────────────────────────────

    def print_report(self) -> None:
        """Print comprehensive walk-forward evaluation report."""
        s = self.summary()
        if "error" in s:
            print(s["error"])
            return

        print()
        print("=" * 65)
        print("WALK-FORWARD BACKTEST — RESULTS")
        print("=" * 65)

        # Overview
        dates = sorted(set(d["date"] for d in self.daily_results))
        date_range = f"{dates[0]} → {dates[-1]}" if dates else "N/A"
        avg_scored = np.mean([d.get("n_stocks_scored", 0) for d in self.daily_results])

        print(f"\nOVERVIEW:")
        print(f"  Test period:          {date_range}")
        print(f"  Retrain windows:      {s['total_retrain_windows']}")
        print(f"  Total test days:      {s['total_test_days']}")
        print(f"  Stocks scored/day:    ~{avg_scored:,.0f}")
        print(f"  Fwd return horizon:   {s.get('fwd_return_horizon', 10)} days")

        # IC
        print(f"\nINFORMATION COEFFICIENT:")
        print(f"  Mean IC:              {s['mean_ic']:.4f}")
        print(f"  IC Std:               {s['ic_std']:.4f}")
        print(f"  IC IR:                {s['ic_ir']:.2f}")
        print(f"  IC Hit Rate:          {s['ic_hit_rate']:.1%} (positive IC on {s['ic_hit_rate']:.0%} of days)")
        print(f"  IC Trend:             {s['ic_trend']}")
        print(f"  IC Autocorrelation:   {s['ic_autocorrelation']:.3f}")

        # IC by window
        print(f"\nIC BY WINDOW:")
        for i, w in enumerate(self.window_results):
            train_end = w.get("train_end", "?")
            test_start = w.get("test_start", "?")
            test_end = w.get("test_end", "?")
            wic = w.get("mean_ic", 0)
            bar = "█" * int(max(0, wic) * 500)
            sign = "+" if wic >= 0 else ""
            print(f"  Window {i+1:2d} (train→{train_end}, test {test_start}→{test_end}): "
                  f"{sign}{wic:.4f} {bar}")

        # Long-Short
        print(f"\nLONG-SHORT PORTFOLIO:")
        print(f"  Cumulative return:    {s['cumulative_long_short_return']:.2%}")
        print(f"  Annualized return:    {s['annualized_long_short_return']:.2%}")
        print(f"  Sharpe ratio:         {s['long_short_sharpe']:.2f}")
        print(f"  Max drawdown:         {s['max_drawdown']:.2%}")
        print(f"  Hit rate:             {s['long_short_hit_rate']:.1%} (days with positive L/S return)")
        print(f"  Avg long return/day:  {s['avg_long_return']:.4%}")
        print(f"  Avg short return/day: {s['avg_short_return']:.4%}")

        # Decile
        avg_dec = s.get("avg_decile_returns", [])
        if avg_dec:
            print(f"\nDECILE ANALYSIS:")
            # D1 = lowest predicted LONG probability (SHORT candidates)
            # D10 = highest predicted LONG probability (LONG candidates)
            # For a good model: D10 return > D1 return (monotonically increasing)
            labels = ["D1  (lowest pred — SHORT)", "D2", "D3", "D4", "D5",
                      "D6", "D7", "D8", "D9", "D10 (highest pred — LONG)"]
            n_dec = len(avg_dec)
            for i in range(n_dec):
                lbl = labels[i] if i < len(labels) else f"D{i+1}"
                r = avg_dec[i]
                bar = "█" * int(max(0, r) * 10000)
                nbar = "▒" * int(max(0, -r) * 10000)
                print(f"  {lbl:30s}: {r:+.4%}/day {bar}{nbar}")
            print(f"  Monotonicity (Kendall):  {s['decile_monotonicity']:.3f} "
                  f"{'(strong)' if s['decile_monotonicity'] > 0.8 else '(moderate)' if s['decile_monotonicity'] > 0.5 else '(weak)'}")

        # Interpretation
        print(f"\nINTERPRETATION:")
        if s["mean_ic"] > 0.03 and s["ic_ir"] > 0.7:
            print("  ✅ IC > 0.03 and IC IR > 0.7: competitive with live quant funds")
        elif s["mean_ic"] > 0.02 and s["ic_ir"] > 0.5:
            print("  ✅ IC > 0.02 and IC IR > 0.5: publishable academic result")
        elif s["mean_ic"] > 0.01:
            print("  ⚠️  IC > 0.01: weak but potentially tradeable with position sizing")
        else:
            print("  ❌ IC ≤ 0.01: model predictions are not reliable for trading")

        if s["long_short_sharpe"] > 1.0:
            print("  ✅ L/S Sharpe > 1.0: institutional quality signal")
        elif s["long_short_sharpe"] > 0.5:
            print("  ⚠️  L/S Sharpe 0.5-1.0: needs improvement but has signal")
        else:
            print("  ❌ L/S Sharpe < 0.5: not tradeable as standalone strategy")

        if s["decile_monotonicity"] > 0.8:
            print("  ✅ Decile monotonicity > 0.8: clean separation of return ranks")
        elif s["decile_monotonicity"] > 0.5:
            print("  ⚠️  Decile monotonicity 0.5-0.8: moderate — some noise in rankings")
        else:
            print("  ❌ Decile monotonicity < 0.5: rankings are noisy")

        if s["ic_trend"] == "improving":
            print("  ✅ IC trend improving: model getting better with more data")
        elif s["ic_trend"] == "degrading":
            print("  ⚠️  IC trend degrading: model may be overfitting to old data")
        else:
            print("  ➡️  IC trend stable")

        print(f"\nINSTITUTIONAL BENCHMARKS:")
        print(f"  - IC > 0.02 and IC IR > 0.5: publishable academic result")
        print(f"  - IC > 0.03 and IC IR > 0.7: competitive with live quant funds")
        print(f"  - L/S Sharpe > 1.0: institutional quality")
        print(f"  - Decile monotonicity > 0.8: strong signal")
        print("=" * 65)

    # ── Plotting ──────────────────────────────────────────────

    def plot_results(self, output_path: str = None) -> None:
        """Generate matplotlib charts saved to file."""
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import matplotlib.dates as mdates
        except ImportError:
            logger.warning("matplotlib not available — skipping plot generation")
            return

        if not self.daily_results:
            return

        fig, axes = plt.subplots(3, 2, figsize=(16, 14))
        fig.suptitle("NUBLE Walk-Forward Backtest Results", fontsize=14, fontweight="bold")

        dates = [pd.Timestamp(d["date"]) for d in self.daily_results]
        ics = [d.get("ic", 0) for d in self.daily_results]
        ls_rets = [d.get("long_short_return", 0) for d in self.daily_results]

        # 1. Cumulative L/S return
        ax = axes[0, 0]
        cum_ret = np.cumsum(ls_rets)
        ax.plot(dates, cum_ret, linewidth=1.5, color="#2196F3")
        ax.fill_between(dates, cum_ret, 0, where=np.array(cum_ret) >= 0,
                         alpha=0.15, color="#4CAF50")
        ax.fill_between(dates, cum_ret, 0, where=np.array(cum_ret) < 0,
                         alpha=0.15, color="#F44336")
        ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
        ax.set_title("Cumulative Long-Short Return")
        ax.set_ylabel("Return")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %y"))

        # 2. Rolling IC
        ax = axes[0, 1]
        ic_series = pd.Series(ics, index=dates)
        rolling_ic = ic_series.rolling(21, min_periods=5).mean()
        rolling_std = ic_series.rolling(21, min_periods=5).std()
        ax.plot(dates, rolling_ic, linewidth=1.5, color="#9C27B0", label="21d Rolling IC")
        ax.fill_between(dates,
                         (rolling_ic - rolling_std).values,
                         (rolling_ic + rolling_std).values,
                         alpha=0.15, color="#9C27B0")
        ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
        ax.axhline(y=0.02, color="green", linestyle=":", alpha=0.5, label="IC=0.02 threshold")
        ax.set_title("Rolling 21-Day IC")
        ax.set_ylabel("IC")
        ax.legend(fontsize=8)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %y"))

        # 3. Decile bar chart
        ax = axes[1, 0]
        s = self.summary()
        avg_dec = s.get("avg_decile_returns", [])
        if avg_dec:
            n_dec = len(avg_dec)
            colors = ["#F44336" if r < 0 else "#4CAF50" for r in avg_dec]
            ax.bar(range(1, n_dec + 1), [r * 100 for r in avg_dec], color=colors, alpha=0.8)
            ax.set_xlabel("Decile (1=SHORT/lowest pred, 10=LONG/highest pred)")
            ax.set_ylabel("Avg Daily Return (%)")
            ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
        ax.set_title("Average Return by Prediction Decile")

        # 4. IC histogram
        ax = axes[1, 1]
        valid_ics = [ic for ic in ics if not np.isnan(ic)]
        if valid_ics:
            ax.hist(valid_ics, bins=30, color="#2196F3", alpha=0.7, edgecolor="white")
            ax.axvline(x=0, color="red", linestyle="--", alpha=0.7, label="IC=0")
            ax.axvline(x=np.mean(valid_ics), color="green", linestyle="-",
                       alpha=0.7, label=f"Mean={np.mean(valid_ics):.4f}")
            ax.set_xlabel("IC")
            ax.set_ylabel("Frequency")
            ax.legend(fontsize=8)
        ax.set_title("IC Distribution")

        # 5. Monthly L/S return heatmap
        ax = axes[2, 0]
        try:
            ls_series = pd.Series(ls_rets, index=dates)
            monthly = ls_series.resample("M").sum()
            if len(monthly) > 0:
                # Create a simple bar chart of monthly returns (heatmap alternative)
                month_labels = [d.strftime("%b %y") for d in monthly.index]
                month_colors = ["#4CAF50" if r > 0 else "#F44336" for r in monthly.values]
                ax.bar(range(len(monthly)), [r * 100 for r in monthly.values],
                       color=month_colors, alpha=0.8)
                ax.set_xticks(range(len(monthly)))
                ax.set_xticklabels(month_labels, rotation=45, ha="right", fontsize=7)
                ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
                ax.set_ylabel("Monthly L/S Return (%)")
        except Exception:
            ax.text(0.5, 0.5, "Monthly data unavailable", ha="center",
                    va="center", transform=ax.transAxes)
        ax.set_title("Monthly Long-Short Returns")

        # 6. Window IC comparison
        ax = axes[2, 1]
        if self.window_results:
            window_ics = [w.get("mean_ic", 0) for w in self.window_results]
            window_labels = [f"W{i+1}" for i in range(len(window_ics))]
            colors_bar = ["#4CAF50" if ic > 0 else "#F44336" for ic in window_ics]
            ax.bar(window_labels, window_ics, color=colors_bar, alpha=0.8)
            ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
            ax.axhline(y=0.02, color="green", linestyle=":", alpha=0.5)
            ax.set_ylabel("Mean IC")
        ax.set_title("IC by Retrain Window")

        plt.tight_layout(rect=[0, 0, 1, 0.96])

        if output_path is None:
            output_path = "models/universal/backtest_results.png"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("Charts saved: %s", output_path)


# ══════════════════════════════════════════════════════════════
# WalkForwardBacktest
# ══════════════════════════════════════════════════════════════


class WalkForwardBacktest:
    """
    Expanding-window walk-forward backtest for the universal model.

    Gold-standard institutional validation. Trains on expanding windows,
    tests on each subsequent period, computes IC and portfolio returns
    out-of-sample over the full date range.
    """

    def __init__(
        self,
        polygon_data: "PolygonUniverseData" = None,
        feature_engine: "UniversalFeatureEngine" = None,
        min_train_days: int = 120,
        retrain_frequency: int = 21,
        purge_gap: int = 10,
        test_window: int = 21,
        n_stocks_train: int = 1000,
        n_stocks_score: int = None,
    ):
        if polygon_data is None:
            from ...data.polygon_universe import PolygonUniverseData
            polygon_data = PolygonUniverseData()
        if feature_engine is None:
            from ..universal_features import UniversalFeatureEngine
            feature_engine = UniversalFeatureEngine()

        self.polygon_data = polygon_data
        self.engine = feature_engine
        self.min_train_days = min_train_days
        self.retrain_frequency = retrain_frequency
        self.purge_gap = purge_gap
        self.test_window = test_window
        self.n_stocks_train = n_stocks_train
        self.n_stocks_score = n_stocks_score
        self._holding_period = 10  # Must match compute_label holding_period
        self._fwd_return_horizon = 10  # Evaluate on same horizon model was trained on

        # LightGBM params (identical to training — deterministic)
        self._lgb_params = {
            "objective": "multiclass",
            "num_class": 3,
            "learning_rate": 0.02,
            "num_leaves": 127,
            "max_depth": 8,
            "min_data_in_leaf": 300,
            "feature_fraction": 0.6,
            "bagging_fraction": 0.7,
            "bagging_freq": 1,
            "lambda_l1": 0.05,
            "lambda_l2": 0.5,
            "min_gain_to_split": 0.01,
            "verbose": -1,
            "seed": RANDOM_SEED,
            "num_threads": os.cpu_count() or 4,
        }

    # ── Main Run ──────────────────────────────────────────────

    def run(
        self,
        start_date: str = None,
        end_date: str = None,
        verbose: bool = True,
    ) -> BacktestResults:
        """
        Run the full walk-forward backtest.

        1. Pre-compute features for all stocks (cache to avoid redundant work)
        2. Define expanding training windows
        3. For each window: train model, score test dates, compute returns
        4. Aggregate results

        Returns BacktestResults with all per-date and per-window metrics.
        """
        import lightgbm as lgb
        from ..universal_model import compute_label

        results = BacktestResults()
        total_start = time.time()

        # ── Step 1: Get all available dates ───────────────────
        all_dates = self._get_all_dates()
        if len(all_dates) < self.min_train_days + self.purge_gap + self.test_window:
            raise RuntimeError(
                f"Not enough data for walk-forward: {len(all_dates)} dates, "
                f"need at least {self.min_train_days + self.purge_gap + self.test_window}"
            )

        if start_date:
            all_dates = [d for d in all_dates if d >= start_date]
        if end_date:
            all_dates = [d for d in all_dates if d <= end_date]

        all_dates = sorted(all_dates)
        if verbose:
            print(f"\n[Data] {len(all_dates)} trading dates: {all_dates[0]} → {all_dates[-1]}")

        # ── Step 2: Get liquid universe (for training subset) ─
        liquid_universe = self.polygon_data.get_active_universe(
            min_price=5.0, min_dollar_volume=1_000_000,
        )
        if self.n_stocks_train and len(liquid_universe) > self.n_stocks_train:
            train_universe = liquid_universe[:self.n_stocks_train]
        else:
            train_universe = liquid_universe

        score_universe = liquid_universe if self.n_stocks_score is None \
            else liquid_universe[:self.n_stocks_score]

        if verbose:
            print(f"[Data] Train universe: {len(train_universe)} stocks, "
                  f"Score universe: {len(score_universe)} stocks")

        # ── Step 3: Pre-load all stock histories ──────────────
        if verbose:
            print(f"[Data] Loading stock histories (batched)...")
        stock_histories = self._load_all_histories(score_universe, verbose=verbose)
        if verbose:
            print(f"[Data] Loaded histories for {len(stock_histories)} stocks")

        # Cache histories in results for signal analysis (avoids re-loading)
        results.set_stock_histories(stock_histories)

        # ── Step 4: Pre-compute features for all stocks ───────
        if verbose:
            print(f"[Features] Computing features for {len(stock_histories)} stocks...")
        stock_features = {}
        stock_labels = {}
        computed = 0
        for ticker, df in stock_histories.items():
            if len(df) < max(252, self.engine.MIN_ROWS):
                continue
            try:
                feat = self.engine.compute_features(df)
                if feat.empty or len(feat) < 100:
                    continue
                lab = compute_label(df)
                # Drop warmup
                feat = feat.iloc[self.engine.WARMUP_ROWS:]
                lab = lab.iloc[self.engine.WARMUP_ROWS:]
                common = feat.index.intersection(lab.dropna().index)
                if len(common) < 50:
                    continue
                stock_features[ticker] = feat.loc[common]
                stock_labels[ticker] = lab.loc[common].astype(np.int32)
                computed += 1
            except Exception as e:
                logger.debug("Feature computation failed for %s: %s", ticker, e)
                continue

            if computed % 200 == 0:
                gc.collect()
                if verbose:
                    print(f"  Features computed: {computed}/{len(stock_histories)}")

        if verbose:
            print(f"[Features] Done: {computed} stocks with valid features")

        if computed < 50:
            raise RuntimeError(f"Only {computed} stocks with valid features — need at least 50")

        # ── Step 5: Define walk-forward windows ───────────────
        # Get the superset of dates present in features
        feature_dates = set()
        for ticker, feat in stock_features.items():
            if isinstance(feat.index, pd.DatetimeIndex):
                for d in feat.index:
                    feature_dates.add(d.strftime("%Y-%m-%d"))
            else:
                for d in feat.index:
                    feature_dates.add(str(d))
        feature_dates = sorted(feature_dates)

        if verbose:
            print(f"[Walk-Forward] Feature date range: {feature_dates[0]} → {feature_dates[-1]}")
            print(f"[Walk-Forward] {len(feature_dates)} unique dates with features")

        windows = self._define_windows(feature_dates)
        if verbose:
            print(f"[Walk-Forward] {len(windows)} retrain windows defined")

        # ── Step 6: Walk-forward loop ─────────────────────────
        window_times = []
        for wi, window in enumerate(windows):
            window_start_time = time.time()
            train_dates = window["train_dates"]
            test_dates = window["test_dates"]

            if verbose:
                eta_str = ""
                if window_times:
                    avg_window_time = np.mean(window_times)
                    remaining = (len(windows) - wi) * avg_window_time
                    eta_str = f" | ETA: {remaining/60:.1f} min"
                print(f"\n{'='*65}")
                print(f"Window {wi+1}/{len(windows)}: "
                      f"Train {train_dates[0]}→{train_dates[-1]} ({len(train_dates)}d), "
                      f"Test {test_dates[0]}→{test_dates[-1]} ({len(test_dates)}d){eta_str}")

            # ── Build training panel for this window ──────────
            train_start = time.time()
            X_parts, y_parts = [], []
            tickers_in_train = set()

            # Exclude last holding_period dates from training labels to
            # prevent label leakage (labels look forward holding_period days)
            safe_train_dates = set(train_dates[:-self._holding_period]) \
                if len(train_dates) > self._holding_period else set(train_dates)

            for ticker in train_universe:
                if ticker not in stock_features:
                    continue
                feat = stock_features[ticker]
                lab = stock_labels[ticker]

                # Filter to safe training dates (no label leakage)
                if isinstance(feat.index, pd.DatetimeIndex):
                    mask = feat.index.strftime("%Y-%m-%d").isin(safe_train_dates)
                else:
                    mask = pd.Series(feat.index.astype(str).isin(safe_train_dates), index=feat.index)

                feat_train = feat[mask]
                lab_train = lab.reindex(feat_train.index).dropna()
                common = feat_train.index.intersection(lab_train.index)
                if len(common) < 20:
                    continue

                X_parts.append(feat_train.loc[common])
                y_parts.append(lab_train.loc[common])
                tickers_in_train.add(ticker)

            if not X_parts or len(X_parts) < 10:
                if verbose:
                    print(f"  ⚠️ Only {len(X_parts)} stocks in training — skipping window")
                continue

            X_train_all = pd.concat(X_parts, ignore_index=False)
            y_train_all = pd.concat(y_parts, ignore_index=False)

            # Sort by date for time-series ordering
            if isinstance(X_train_all.index, pd.DatetimeIndex):
                sort_idx = X_train_all.index.argsort()
                X_train_all = X_train_all.iloc[sort_idx]
                y_train_all = y_train_all.iloc[sort_idx]

            X_train_all = X_train_all.reset_index(drop=True)
            y_train_all = y_train_all.reset_index(drop=True)

            # Feature quality: drop >30% NaN and near-zero variance
            nan_pct = X_train_all.isna().mean()
            bad_cols = nan_pct[nan_pct > 0.30].index.tolist()
            if bad_cols:
                X_train_all = X_train_all.drop(columns=bad_cols)
            stds = X_train_all.std()
            dead_cols = stds[stds < 0.001].index.tolist()
            if dead_cols:
                X_train_all = X_train_all.drop(columns=dead_cols)

            feature_cols = X_train_all.select_dtypes(include=[np.number]).columns.tolist()

            # Fill NaN — LightGBM handles NaN natively, but fill for stability
            X_train_all = X_train_all[feature_cols].fillna(0.0)

            # NOTE: No standardization needed. LightGBM is tree-based and
            # invariant to monotone feature transformations. Standardizing
            # can actually hurt by creating distribution mismatch between
            # train and test windows.

            # Class weights
            class_counts = np.bincount(y_train_all.values.astype(int), minlength=3)
            n_samples = len(y_train_all)
            sample_weight = np.array([
                n_samples / (3 * max(class_counts[int(y)], 1))
                for y in y_train_all.values
            ])

            if verbose:
                pcts = [100 * c / n_samples for c in class_counts]
                print(f"  Class distribution: DOWN={pcts[0]:.1f}% NEUTRAL={pcts[1]:.1f}% UP={pcts[2]:.1f}%"
                      f" (n={n_samples:,})")

            # Split: 85% train, 15% val (within the training window)
            n_split = int(len(X_train_all) * 0.85)
            X_t = X_train_all.iloc[:n_split]
            y_t = y_train_all.iloc[:n_split]
            w_t = sample_weight[:n_split]
            X_v = X_train_all.iloc[n_split:]
            y_v = y_train_all.iloc[n_split:]
            w_v = sample_weight[n_split:]

            # Train LightGBM
            train_data = lgb.Dataset(X_t.values, label=y_t.values, weight=w_t,
                                      feature_name=feature_cols, free_raw_data=False)
            val_data = lgb.Dataset(X_v.values, label=y_v.values, weight=w_v,
                                    reference=train_data, free_raw_data=False)

            model = lgb.train(
                self._lgb_params,
                train_data,
                num_boost_round=3000,
                valid_sets=[val_data],
                valid_names=["val"],
                callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
            )

            train_elapsed = time.time() - train_start
            if verbose:
                print(f"  Trained on {len(X_t):,} samples × {len(feature_cols)} features "
                      f"({len(tickers_in_train)} stocks, {train_elapsed:.1f}s, "
                      f"best_iter={model.best_iteration})")

            # ── Score test dates ──────────────────────────────
            window_ics = []
            window_ls_rets = []

            for test_date in test_dates:
                # Get all stocks with features on this date
                date_preds = {}
                date_returns = {}

                for ticker in score_universe:
                    if ticker not in stock_features:
                        continue
                    feat = stock_features[ticker]
                    if isinstance(feat.index, pd.DatetimeIndex):
                        date_mask = feat.index.strftime("%Y-%m-%d") == test_date
                    else:
                        date_mask = pd.Series(feat.index.astype(str) == test_date, index=feat.index)

                    rows = feat[date_mask]
                    if len(rows) == 0:
                        continue

                    # Get forward return — MUST match the model's prediction
                    # horizon. The model predicts 10-day triple-barrier labels,
                    # so we evaluate IC against 10-day forward returns.
                    fwd_ret = self._get_forward_return_from_history(
                        stock_histories.get(ticker), test_date,
                        horizon=self._fwd_return_horizon,
                    )
                    if fwd_ret is None or np.isnan(fwd_ret):
                        continue

                    # Prepare features — no standardization needed (tree-based model)
                    row = rows.iloc[[-1]].copy()
                    for col in feature_cols:
                        if col not in row.columns:
                            row[col] = 0.0
                    row = row.reindex(columns=feature_cols, fill_value=0.0)
                    row = row.fillna(0.0)

                    # Predict
                    try:
                        proba = model.predict(row.values)[0]
                        long_prob = float(proba[2])
                        date_preds[ticker] = long_prob
                        date_returns[ticker] = float(fwd_ret)
                    except Exception:
                        continue

                if len(date_preds) < 20:
                    continue

                # Winsorize extreme returns to prevent outliers dominating IC
                ret_series = pd.Series(date_returns)
                lo_clip = ret_series.quantile(0.01)
                hi_clip = ret_series.quantile(0.99)
                ret_series = ret_series.clip(lower=lo_clip, upper=hi_clip)
                date_returns = ret_series.to_dict()

                # Form decile portfolios
                portfolio = self._form_decile_portfolios(
                    pd.Series(date_preds),
                    pd.Series(date_returns),
                )

                daily_result = {
                    "date": test_date,
                    "ic": portfolio["ic"],
                    "long_return": portfolio["long_return"],
                    "short_return": portfolio["short_return"],
                    "long_short_return": portfolio["long_short_return"],
                    "hit_rate": portfolio["hit_rate"],
                    "n_stocks_scored": portfolio["n_stocks_scored"],
                    "decile_returns": portfolio["decile_returns"],
                    "window_idx": wi,
                }
                results.add_daily_result(daily_result)
                results.add_predictions(test_date, date_preds)
                results.add_daily_portfolio(
                    test_date,
                    portfolio.get("long_tickers", []),
                    portfolio.get("short_tickers", []),
                )

                window_ics.append(portfolio["ic"])
                window_ls_rets.append(portfolio["long_short_return"])

            # Window summary
            window_summary = {
                "window_idx": wi,
                "train_start": train_dates[0],
                "train_end": train_dates[-1],
                "test_start": test_dates[0],
                "test_end": test_dates[-1],
                "n_train_stocks": len(tickers_in_train),
                "n_train_samples": len(X_t),
                "n_test_days": len(test_dates),
                "n_test_days_scored": len(window_ics),
                "mean_ic": float(np.mean(window_ics)) if window_ics else 0.0,
                "mean_ls_return": float(np.mean(window_ls_rets)) if window_ls_rets else 0.0,
                "best_iteration": model.best_iteration,
                "train_time_seconds": round(train_elapsed, 1),
            }
            results.add_window_result(window_summary)

            if verbose:
                avg_ic = float(np.mean(window_ics)) if window_ics else 0.0
                avg_ls = float(np.mean(window_ls_rets)) if window_ls_rets else 0.0
                n_scored = len(window_ics)
                print(f"  Test: {n_scored} dates scored, Mean IC={avg_ic:.4f}, "
                      f"Avg L/S ret={avg_ls:.4%}")

            window_times.append(time.time() - window_start_time)
            gc.collect()

        total_elapsed = time.time() - total_start
        if verbose:
            print(f"\n{'='*65}")
            print(f"Walk-forward complete: {total_elapsed:.0f}s ({total_elapsed/60:.1f} min)")
            print(f"{'='*65}")

        return results

    # ── Regression Run (Cross-Sectional Model) ────────────────

    def run_regression(
        self,
        forward_horizon: int = 5,
        start_date: str = None,
        end_date: str = None,
        verbose: bool = True,
    ) -> BacktestResults:
        """
        Walk-forward backtest for the cross-sectional REGRESSION model.

        Key differences from run() (classification):
        1. Target: continuous excess return (not triple-barrier class)
        2. Features: cross-sectionally RANK NORMALIZED per date
        3. Loss: Huber regression (not multiclass)
        4. Train/score on the SAME universe (no mismatch)
        5. Prediction = predicted excess return (higher = buy)

        Same expanding-window framework, same IC/decile/portfolio metrics.
        """
        import lightgbm as lgb

        results = BacktestResults()
        total_start = time.time()

        # ── Step 1: Get dates ─────────────────────────────────
        all_dates = self._get_all_dates()
        if start_date:
            all_dates = [d for d in all_dates if d >= start_date]
        if end_date:
            all_dates = [d for d in all_dates if d <= end_date]
        all_dates = sorted(all_dates)

        if len(all_dates) < self.min_train_days + self.purge_gap + self.test_window:
            raise RuntimeError(
                f"Not enough data: {len(all_dates)} dates, "
                f"need {self.min_train_days + self.purge_gap + self.test_window}"
            )

        if verbose:
            print(f"\n[Data] {len(all_dates)} trading dates: {all_dates[0]} → {all_dates[-1]}")

        # ── Step 2: Universe — SAME for train and score ───────
        liquid_universe = self.polygon_data.get_active_universe(
            min_price=5.0, min_dollar_volume=1_000_000,
        )
        # Use all stocks for both train and score (fix mismatch)
        n_stocks = self.n_stocks_score or self.n_stocks_train or len(liquid_universe)
        universe = liquid_universe[:n_stocks]

        if verbose:
            print(f"[Data] Universe: {len(universe)} stocks (same for train & score)")

        # ── Step 3: Load histories ────────────────────────────
        if verbose:
            print(f"[Data] Loading stock histories...")
        stock_histories = self._load_all_histories(universe, verbose=verbose)
        if verbose:
            print(f"[Data] Loaded {len(stock_histories)} stocks")
        results.set_stock_histories(stock_histories)

        # ── Step 4: Compute raw features ──────────────────────
        if verbose:
            print(f"[Features] Computing features for {len(stock_histories)} stocks...")
        stock_features: Dict[str, pd.DataFrame] = {}
        computed = 0
        for ticker, df in stock_histories.items():
            if len(df) < max(self.engine.MIN_ROWS, 60):
                continue
            try:
                feat = self.engine.compute_features(df)
                if feat.empty or len(feat) < 50:
                    continue
                feat = feat.iloc[self.engine.WARMUP_ROWS:]
                if len(feat) < 20:
                    continue
                stock_features[ticker] = feat
                computed += 1
            except Exception:
                continue
            if computed % 200 == 0:
                gc.collect()
                if verbose:
                    print(f"  Features: {computed}/{len(stock_histories)}")

        if verbose:
            print(f"[Features] Done: {computed} stocks with valid features")

        if computed < 50:
            raise RuntimeError(f"Only {computed} valid — need at least 50")

        # ── Step 5: Build per-date cross-sectional data ───────
        # For each date, collect raw features + forward returns for all stocks
        feature_dates = set()
        for feat in stock_features.values():
            if isinstance(feat.index, pd.DatetimeIndex):
                for d in feat.index:
                    feature_dates.add(d.strftime("%Y-%m-%d"))
            else:
                for d in feat.index:
                    feature_dates.add(str(d))
        feature_dates = sorted(feature_dates)

        if verbose:
            print(f"[Walk-Forward] {len(feature_dates)} unique dates, "
                  f"{feature_dates[0]} → {feature_dates[-1]}")

        # Pre-build per-date cross-sectional snapshots
        # This avoids recomputing inside the walk-forward loop
        date_snapshots: Dict[str, Dict[str, Any]] = {}  # date -> {ticker: {feats, fwd_ret}}
        for dt_str in feature_dates:
            cs_feats = {}
            cs_rets = {}
            for ticker, feat_df in stock_features.items():
                if isinstance(feat_df.index, pd.DatetimeIndex):
                    mask = feat_df.index.strftime("%Y-%m-%d") == dt_str
                else:
                    mask = pd.Series(feat_df.index.astype(str) == dt_str, index=feat_df.index)
                rows = feat_df[mask]
                if len(rows) == 0:
                    continue
                fwd_ret = self._get_forward_return_from_history(
                    stock_histories.get(ticker), dt_str, horizon=forward_horizon,
                )
                cs_feats[ticker] = rows.iloc[-1]
                if fwd_ret is not None and not np.isnan(fwd_ret):
                    cs_rets[ticker] = fwd_ret

            if len(cs_feats) >= 50:
                date_snapshots[dt_str] = {"features": cs_feats, "returns": cs_rets}

        valid_dates = sorted(date_snapshots.keys())
        if verbose:
            print(f"[Walk-Forward] {len(valid_dates)} dates with ≥50 stocks")

        # ── Step 6: Define windows ────────────────────────────
        windows = self._define_windows(valid_dates)
        if verbose:
            print(f"[Walk-Forward] {len(windows)} retrain windows defined")

        # LightGBM regression params
        lgb_params = {
            "objective": "huber",
            "huber_delta": 1.0,
            "learning_rate": 0.01,
            "num_leaves": 255,
            "max_depth": 8,
            "min_data_in_leaf": 500,
            "feature_fraction": 0.5,
            "bagging_fraction": 0.5,
            "bagging_freq": 1,
            "lambda_l1": 0.1,
            "lambda_l2": 1.0,
            "min_gain_to_split": 0.01,
            "num_threads": os.cpu_count() or 4,
            "verbose": -1,
            "seed": RANDOM_SEED,
        }

        # ── Step 7: Walk-forward loop ─────────────────────────
        window_times = []
        for wi, window in enumerate(windows):
            window_start_time = time.time()
            train_dates = window["train_dates"]
            test_dates = window["test_dates"]

            if verbose:
                eta_str = ""
                if window_times:
                    avg_t = np.mean(window_times)
                    remaining = (len(windows) - wi) * avg_t
                    eta_str = f" | ETA: {remaining/60:.1f} min"
                print(f"\n{'='*65}")
                print(f"Window {wi+1}/{len(windows)}: "
                      f"Train {train_dates[0]}→{train_dates[-1]} ({len(train_dates)}d), "
                      f"Test {test_dates[0]}→{test_dates[-1]} ({len(test_dates)}d){eta_str}")

            # ── Build cross-sectional training panel ──────────
            train_start = time.time()

            # Exclude last forward_horizon dates to prevent label leakage
            safe_train_dates = train_dates[:-forward_horizon] \
                if len(train_dates) > forward_horizon else train_dates

            X_rows = []
            y_vals = []
            feature_cols = None

            for dt_str in safe_train_dates:
                if dt_str not in date_snapshots:
                    continue
                snap = date_snapshots[dt_str]
                cs_feats = snap["features"]
                cs_rets = snap["returns"]

                # Only include stocks that have both features AND returns
                common_tickers = set(cs_feats.keys()) & set(cs_rets.keys())
                if len(common_tickers) < 50:
                    continue

                # Build cross-sectional DataFrame
                cs_df = pd.DataFrame({t: cs_feats[t] for t in common_tickers}).T
                if feature_cols is None:
                    feature_cols = cs_df.columns.tolist()

                # RANK NORMALIZE across stocks on this date
                cs_ranked = cs_df[feature_cols].rank(pct=True).fillna(0.5)

                # Compute excess returns
                rets = pd.Series({t: cs_rets[t] for t in common_tickers})
                median_ret = rets.median()
                excess_rets = rets - median_ret

                for ticker in common_tickers:
                    X_rows.append(cs_ranked.loc[ticker].values)
                    y_vals.append(excess_rets[ticker])

            if not X_rows or len(X_rows) < 500:
                if verbose:
                    print(f"  ⚠️ Only {len(X_rows)} samples — skipping window")
                continue

            X_all = np.array(X_rows, dtype=np.float32)
            y_all = np.array(y_vals, dtype=np.float32)

            # Clean
            valid_mask = np.isfinite(y_all) & np.all(np.isfinite(X_all), axis=1)
            X_all = X_all[valid_mask]
            y_all = y_all[valid_mask]

            if verbose:
                n_samples = len(X_all)
                print(f"  Panel: {n_samples:,} samples × {len(feature_cols)} features, "
                      f"target std={y_all.std():.4f}")

            # Train/val split (85/15 within training window)
            n_split = int(len(X_all) * 0.85)
            X_t, y_t = X_all[:n_split], y_all[:n_split]
            X_v, y_v = X_all[n_split:], y_all[n_split:]

            train_data = lgb.Dataset(X_t, label=y_t,
                                      feature_name=feature_cols, free_raw_data=False)
            val_data = lgb.Dataset(X_v, label=y_v,
                                    reference=train_data, free_raw_data=False)

            model = lgb.train(
                lgb_params,
                train_data,
                num_boost_round=3000,
                valid_sets=[val_data],
                valid_names=["val"],
                callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
            )

            train_elapsed = time.time() - train_start
            if verbose:
                print(f"  Trained: {len(X_t):,} samples, {train_elapsed:.1f}s, "
                      f"best_iter={model.best_iteration}")

            # ── Score test dates ──────────────────────────────
            window_ics = []
            window_ls_rets = []

            for test_date in test_dates:
                if test_date not in date_snapshots:
                    continue
                snap = date_snapshots[test_date]
                cs_feats = snap["features"]
                cs_rets = snap["returns"]

                # Need stocks with both features and returns
                common_tickers = sorted(set(cs_feats.keys()) & set(cs_rets.keys()))
                if len(common_tickers) < 20:
                    continue

                # Build cross-sectional features and RANK NORMALIZE
                cs_df = pd.DataFrame({t: cs_feats[t] for t in common_tickers}).T
                # Ensure same columns as training
                for col in feature_cols:
                    if col not in cs_df.columns:
                        cs_df[col] = 0.0
                cs_df = cs_df[feature_cols]
                cs_ranked = cs_df.rank(pct=True).fillna(0.5)

                # Predict
                preds_arr = model.predict(cs_ranked.values.astype(np.float32))
                date_preds = dict(zip(common_tickers, preds_arr.tolist()))

                # Forward returns (raw, not excess — for fair L/S evaluation)
                date_returns = {t: cs_rets[t] for t in common_tickers}

                # Winsorize returns
                ret_series = pd.Series(date_returns)
                lo = ret_series.quantile(0.01)
                hi = ret_series.quantile(0.99)
                ret_series = ret_series.clip(lower=lo, upper=hi)
                date_returns = ret_series.to_dict()

                # Form decile portfolios
                portfolio = self._form_decile_portfolios(
                    pd.Series(date_preds),
                    pd.Series(date_returns),
                )

                daily_result = {
                    "date": test_date,
                    "ic": portfolio["ic"],
                    "long_return": portfolio["long_return"],
                    "short_return": portfolio["short_return"],
                    "long_short_return": portfolio["long_short_return"],
                    "hit_rate": portfolio["hit_rate"],
                    "n_stocks_scored": portfolio["n_stocks_scored"],
                    "decile_returns": portfolio["decile_returns"],
                    "window_idx": wi,
                }
                results.add_daily_result(daily_result)
                results.add_predictions(test_date, date_preds)
                results.add_daily_portfolio(
                    test_date,
                    portfolio.get("long_tickers", []),
                    portfolio.get("short_tickers", []),
                )

                window_ics.append(portfolio["ic"])
                window_ls_rets.append(portfolio["long_short_return"])

            # Window summary
            window_summary = {
                "window_idx": wi,
                "train_start": train_dates[0],
                "train_end": train_dates[-1],
                "test_start": test_dates[0],
                "test_end": test_dates[-1],
                "n_train_samples": len(X_t),
                "n_test_days": len(test_dates),
                "n_test_days_scored": len(window_ics),
                "mean_ic": float(np.mean(window_ics)) if window_ics else 0.0,
                "mean_ls_return": float(np.mean(window_ls_rets)) if window_ls_rets else 0.0,
                "best_iteration": model.best_iteration,
                "train_time_seconds": round(train_elapsed, 1),
            }
            results.add_window_result(window_summary)

            if verbose:
                avg_ic = float(np.mean(window_ics)) if window_ics else 0.0
                avg_ls = float(np.mean(window_ls_rets)) if window_ls_rets else 0.0
                print(f"  Test: {len(window_ics)} dates, Mean IC={avg_ic:.4f}, "
                      f"Avg L/S ret={avg_ls:.4%}")

            window_times.append(time.time() - window_start_time)
            gc.collect()

        total_elapsed = time.time() - total_start
        if verbose:
            print(f"\n{'='*65}")
            print(f"Walk-forward complete: {total_elapsed:.0f}s ({total_elapsed/60:.1f} min)")
            print(f"{'='*65}")

        return results

    # ── Helper: Get all available dates ───────────────────────

    def _get_all_dates(self) -> List[str]:
        """Get sorted list of all dates in the universe data."""
        index = self.polygon_data._load_index()
        return sorted([d.isoformat() for d in index])

    # ── Helper: Load histories ────────────────────────────────

    def _load_all_histories(
        self, universe: List[str], verbose: bool = False
    ) -> Dict[str, pd.DataFrame]:
        """Load histories for all stocks in batches."""
        all_histories = {}
        batch_size = 300

        for i in range(0, len(universe), batch_size):
            batch = universe[i:i + batch_size]
            histories = self.polygon_data.get_multi_stock_history(
                symbols=batch, days=600
            )
            all_histories.update(histories)

            if verbose and (i + batch_size) % 600 == 0:
                print(f"  Loaded: {len(all_histories)} stocks so far...")

            gc.collect()

        return all_histories

    # ── Helper: Define windows ────────────────────────────────

    def _define_windows(self, all_dates: List[str]) -> List[Dict]:
        """
        Define expanding-window train/test splits.
        Training starts at date 0, expands by retrain_frequency each window.
        """
        windows = []
        n = len(all_dates)

        # First test can start after min_train_days + purge_gap
        first_test_idx = self.min_train_days + self.purge_gap

        test_start_idx = first_test_idx
        while test_start_idx + self.test_window <= n:
            train_end_idx = test_start_idx - self.purge_gap
            test_end_idx = min(test_start_idx + self.test_window, n)

            train_dates = all_dates[:train_end_idx]
            test_dates = all_dates[test_start_idx:test_end_idx]

            if len(train_dates) >= self.min_train_days and len(test_dates) >= 1:
                windows.append({
                    "train_dates": train_dates,
                    "test_dates": test_dates,
                })

            test_start_idx += self.retrain_frequency

        return windows

    # ── Helper: Forward return from history ───────────────────

    def _get_forward_return_from_history(
        self, df: Optional[pd.DataFrame], date_str: str, horizon: int = 1
    ) -> Optional[float]:
        """
        Get actual forward return from a stock's history DataFrame.
        forward_return = log(close_{date+horizon} / close_{date})
        """
        if df is None or df.empty:
            return None

        try:
            if isinstance(df.index, pd.DatetimeIndex):
                idx = df.index.strftime("%Y-%m-%d")
            else:
                idx = df.index.astype(str)

            positions = np.where(idx == date_str)[0]
            if len(positions) == 0:
                return None

            pos = positions[0]
            fwd_pos = pos + horizon
            if fwd_pos >= len(df):
                return None

            close_now = float(df.iloc[pos]["close"])
            close_fwd = float(df.iloc[fwd_pos]["close"])

            if close_now <= 0 or close_fwd <= 0:
                return None

            return float(np.log(close_fwd / close_now))

        except Exception:
            return None

    # ── Helper: Decile portfolios ─────────────────────────────

    def _form_decile_portfolios(
        self,
        predictions: pd.Series,
        returns: pd.Series,
    ) -> Dict[str, Any]:
        """
        Form decile portfolios from predictions and compute returns.

        predictions: Series indexed by ticker, values = predicted LONG probability
        returns: Series indexed by ticker, values = actual forward return
        """
        # Align
        common = predictions.index.intersection(returns.index)
        preds = predictions.loc[common]
        rets = returns.loc[common]
        n = len(common)

        if n < 10:
            return {
                "decile_returns": [0.0] * 10,
                "long_return": 0.0,
                "short_return": 0.0,
                "long_short_return": 0.0,
                "ic": 0.0,
                "hit_rate": 0.0,
                "n_stocks_scored": n,
                "long_tickers": [],
                "short_tickers": [],
            }

        # Rank by prediction
        ranks = preds.rank(pct=True)

        # Decile portfolios (10 bins)
        n_deciles = min(10, n // 2)  # Need at least 2 stocks per decile
        decile_returns = []
        for d in range(n_deciles):
            lo = d / n_deciles
            hi = (d + 1) / n_deciles
            if d == n_deciles - 1:
                hi = 1.01  # Include top
            mask = (ranks >= lo) & (ranks < hi)
            if mask.sum() > 0:
                decile_returns.append(float(rets[mask].mean()))
            else:
                decile_returns.append(0.0)

        # Pad to 10 if fewer deciles
        while len(decile_returns) < 10:
            decile_returns.append(0.0)

        # Long = top decile, Short = bottom decile
        top_pct = max(1.0 / n_deciles, 0.1)
        long_mask = ranks >= (1.0 - top_pct)
        short_mask = ranks <= top_pct

        long_return = float(rets[long_mask].mean()) if long_mask.sum() > 0 else 0.0
        short_return = float(rets[short_mask].mean()) if short_mask.sum() > 0 else 0.0
        long_short_return = long_return - short_return

        # Hit rate: % of long picks with positive returns
        if long_mask.sum() > 0:
            hit_rate = float((rets[long_mask] > 0).mean())
        else:
            hit_rate = 0.0

        # IC: Spearman correlation
        try:
            ic, _ = sp_stats.spearmanr(preds, rets)
            ic = float(ic) if not np.isnan(ic) else 0.0
        except Exception:
            ic = 0.0

        # Tickers for portfolio tracking
        long_tickers = preds[long_mask].index.tolist() if long_mask.sum() > 0 else []
        short_tickers = preds[short_mask].index.tolist() if short_mask.sum() > 0 else []

        return {
            "decile_returns": decile_returns,
            "long_return": long_return,
            "short_return": short_return,
            "long_short_return": long_short_return,
            "ic": ic,
            "hit_rate": hit_rate,
            "n_stocks_scored": n,
            "long_tickers": long_tickers,
            "short_tickers": short_tickers,
        }
