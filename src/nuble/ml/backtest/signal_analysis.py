"""
Signal Analysis — Quality & Characteristics of Model Predictions
==================================================================

Analyzes the quality and characteristics of model predictions to answer
critical questions before deployment:

1. SIGNAL DECAY: How long does a prediction stay useful?
2. FACTOR EXPOSURE: Is the model just rediscovering known factors?
3. TURNOVER: How much does the portfolio change each rebalance?
4. CONCENTRATION: Does the model spread bets or concentrate?
5. REGIME DEPENDENCE: Does the model work in all market conditions?

Author: NUBLE ML Pipeline — Phase 5 Walk-Forward Validation
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

if TYPE_CHECKING:
    from ...data.polygon_universe import PolygonUniverseData
    from .walk_forward import BacktestResults

logger = logging.getLogger(__name__)


class SignalAnalysis:
    """
    Comprehensive signal quality analysis for the universal model.
    Takes predictions from WalkForwardBacktest and runs deep diagnostics.
    """

    # ── Signal Decay ──────────────────────────────────────────

    def signal_decay(
        self,
        predictions: pd.DataFrame,
        polygon_data: "PolygonUniverseData",
        horizons: List[int] = None,
        preloaded_histories: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """
        Compute IC at multiple forward horizons.

        predictions: DataFrame with columns [date, ticker, pred_prob]
        horizons: list of forward horizons in trading days
        preloaded_histories: optional dict of {ticker: DataFrame} to avoid
            redundant loading.

        Returns dict with horizon_ic, optimal_horizon, half_life_days.
        """
        if horizons is None:
            horizons = [1, 2, 3, 5, 10, 21]

        if predictions.empty:
            return {
                "horizon_ic": {h: 0.0 for h in horizons},
                "optimal_horizon": 1,
                "half_life_days": 0,
                "is_short_lived": True,
            }

        # Group predictions by date
        unique_dates = sorted(predictions["date"].unique())

        # We need price histories for computing forward returns at each horizon
        tickers = predictions["ticker"].unique().tolist()

        if preloaded_histories is not None:
            histories = preloaded_histories
            logger.info("Signal decay: using %d preloaded histories", len(histories))
        else:
            logger.info("Signal decay: loading histories for %d tickers...", len(tickers))
            # Load histories in batch
            histories = polygon_data.get_multi_stock_history(
                symbols=tickers[:2000], days=600  # Cap for memory
            )

        horizon_ics: Dict[int, List[float]] = {h: [] for h in horizons}

        for dt in unique_dates:
            dt_preds = predictions[predictions["date"] == dt]
            pred_dict = dict(zip(dt_preds["ticker"], dt_preds["pred_prob"]))

            for h in horizons:
                ics_for_date = []
                preds_list = []
                rets_list = []

                for ticker, prob in pred_dict.items():
                    if ticker not in histories:
                        continue
                    df = histories[ticker]
                    fwd = self._forward_return(df, dt, h)
                    if fwd is not None and not np.isnan(fwd):
                        preds_list.append(prob)
                        rets_list.append(fwd)

                if len(preds_list) >= 20:
                    try:
                        ic, _ = sp_stats.spearmanr(preds_list, rets_list)
                        if not np.isnan(ic):
                            horizon_ics[h].append(float(ic))
                    except Exception:
                        pass

        # Compute mean IC per horizon
        result_ics = {}
        for h in horizons:
            vals = horizon_ics[h]
            result_ics[h] = round(float(np.mean(vals)), 4) if vals else 0.0

        # Optimal horizon: best IC / sqrt(h) (risk-adjusted for holding cost)
        best_h = 1
        best_score = -999
        for h, ic in result_ics.items():
            score = ic / np.sqrt(h) if h > 0 else 0
            if score > best_score:
                best_score = score
                best_h = h

        # Half-life: horizon where IC drops to half of peak IC
        peak_ic = max(result_ics.values()) if result_ics else 0.0
        half_target = peak_ic / 2
        half_life = horizons[-1]  # default to longest
        for h in sorted(horizons):
            if result_ics.get(h, 0) <= half_target and peak_ic > 0:
                half_life = h
                break

        return {
            "horizon_ic": result_ics,
            "optimal_horizon": best_h,
            "half_life_days": half_life,
            "is_short_lived": half_life <= 3,
        }

    # ── Factor Exposure ───────────────────────────────────────

    def factor_exposure(
        self,
        predictions: pd.DataFrame,
        polygon_data: "PolygonUniverseData",
        preloaded_histories: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """
        Decompose predictions into known factor exposures.

        Regresses model predictions on market beta, size, momentum,
        volatility, and short-term reversal. R² tells you how much
        of the prediction is just replicating known factors.
        """
        if predictions.empty:
            return {
                "factor_loadings": {},
                "factor_r_squared": 0.0,
                "raw_ic": 0.0,
                "factor_neutral_ic": 0.0,
                "alpha_fraction": 0.0,
                "dominant_factor": "none",
                "interpretation": "No predictions available",
            }

        unique_dates = sorted(predictions["date"].unique())
        tickers = predictions["ticker"].unique().tolist()

        # Load histories (reuse if preloaded)
        if preloaded_histories is not None:
            histories = preloaded_histories
        else:
            histories = polygon_data.get_multi_stock_history(
                symbols=tickers[:2000], days=600
            )

        # Try to load SPY for market beta computation
        spy_hist = polygon_data.get_stock_history("SPY", days=600)

        all_factor_rows = []
        raw_ics = []
        residual_ics = []

        for dt in unique_dates:
            dt_preds = predictions[predictions["date"] == dt]
            pred_dict = dict(zip(dt_preds["ticker"], dt_preds["pred_prob"]))

            factors_list = []
            preds_list = []
            rets_list = []

            for ticker, prob in pred_dict.items():
                if ticker not in histories:
                    continue
                df = histories[ticker]
                fwd = self._forward_return(df, dt, 1)
                if fwd is None or np.isnan(fwd):
                    continue

                # Compute factor characteristics as of this date
                factor_row = self._compute_factor_chars(df, dt, spy_hist)
                if factor_row is None:
                    continue

                factors_list.append(factor_row)
                preds_list.append(prob)
                rets_list.append(fwd)

            if len(factors_list) < 30:
                continue

            # Build factor matrix
            factor_df = pd.DataFrame(factors_list)
            preds_arr = np.array(preds_list)
            rets_arr = np.array(rets_list)

            # Standardize factors
            for col in factor_df.columns:
                m, s = factor_df[col].mean(), factor_df[col].std()
                if s > 1e-8:
                    factor_df[col] = (factor_df[col] - m) / s

            # Regress predictions on factors
            try:
                from numpy.linalg import lstsq
                X = factor_df.values
                X_with_const = np.column_stack([np.ones(len(X)), X])
                betas, _, _, _ = lstsq(X_with_const, preds_arr, rcond=None)

                # R²
                pred_from_factors = X_with_const @ betas
                ss_res = np.sum((preds_arr - pred_from_factors) ** 2)
                ss_tot = np.sum((preds_arr - preds_arr.mean()) ** 2)
                r2 = 1 - ss_res / (ss_tot + 1e-10)
                r2 = max(0.0, min(1.0, r2))

                # Residual (factor-neutral) predictions
                residual = preds_arr - pred_from_factors

                # IC of raw predictions vs returns
                try:
                    raw_ic, _ = sp_stats.spearmanr(preds_arr, rets_arr)
                    if not np.isnan(raw_ic):
                        raw_ics.append(float(raw_ic))
                except Exception:
                    pass

                # IC of residual predictions vs returns
                try:
                    res_ic, _ = sp_stats.spearmanr(residual, rets_arr)
                    if not np.isnan(res_ic):
                        residual_ics.append(float(res_ic))
                except Exception:
                    pass

                # Store factor loadings (skip intercept betas[0])
                factor_names = factor_df.columns.tolist()
                for i, fn in enumerate(factor_names):
                    all_factor_rows.append({
                        "date": dt,
                        "factor": fn,
                        "loading": float(betas[i + 1]),
                        "r_squared": float(r2),
                    })

            except Exception as e:
                logger.debug("Factor regression failed for %s: %s", dt, e)
                continue

        # Aggregate
        if not all_factor_rows:
            return {
                "factor_loadings": {},
                "factor_r_squared": 0.0,
                "raw_ic": 0.0,
                "factor_neutral_ic": 0.0,
                "alpha_fraction": 0.0,
                "dominant_factor": "none",
                "interpretation": "Insufficient data for factor analysis",
            }

        factor_df_all = pd.DataFrame(all_factor_rows)
        avg_loadings = factor_df_all.groupby("factor")["loading"].mean().to_dict()
        avg_r2 = float(factor_df_all["r_squared"].mean())

        raw_ic_avg = float(np.mean(raw_ics)) if raw_ics else 0.0
        fn_ic_avg = float(np.mean(residual_ics)) if residual_ics else 0.0
        alpha_frac = fn_ic_avg / (raw_ic_avg + 1e-8) if raw_ic_avg > 0 else 0.0
        alpha_frac = max(0.0, min(1.0, alpha_frac))

        # Dominant factor
        dominant = max(avg_loadings, key=lambda k: abs(avg_loadings[k])) if avg_loadings else "none"

        # Interpretation
        if alpha_frac > 0.7:
            interp = (f"Model captures {alpha_frac:.0%} alpha beyond known factors. "
                      f"Strong unique signal.")
        elif alpha_frac > 0.4:
            interp = (f"Model captures {alpha_frac:.0%} alpha beyond known factors. "
                      f"Moderate unique signal — partially explained by {dominant}.")
        else:
            interp = (f"Model captures only {alpha_frac:.0%} alpha beyond factors. "
                      f"Mostly replicating {dominant} factor.")

        return {
            "factor_loadings": {k: round(v, 4) for k, v in avg_loadings.items()},
            "factor_r_squared": round(avg_r2, 3),
            "raw_ic": round(raw_ic_avg, 4),
            "factor_neutral_ic": round(fn_ic_avg, 4),
            "alpha_fraction": round(alpha_frac, 3),
            "dominant_factor": dominant,
            "interpretation": interp,
        }

    # ── Turnover Analysis ─────────────────────────────────────

    def turnover_analysis(self, daily_portfolios: List[Dict]) -> Dict[str, Any]:
        """
        Analyze portfolio turnover.

        daily_portfolios: list of {date, long_tickers, short_tickers}
        """
        if len(daily_portfolios) < 2:
            return {
                "mean_daily_turnover": 0.0,
                "mean_monthly_turnover": 0.0,
                "implied_transaction_cost": 0.0,
                "net_return_after_costs": 0.0,
                "is_tradeable": False,
            }

        daily_turnovers = []
        prev = daily_portfolios[0]

        for curr in daily_portfolios[1:]:
            prev_long = set(prev.get("long_tickers", []))
            curr_long = set(curr.get("long_tickers", []))
            prev_short = set(prev.get("short_tickers", []))
            curr_short = set(curr.get("short_tickers", []))

            # Turnover = fraction of portfolio that changed
            # For each leg: |symmetric_difference| / (|prev| + |curr|) 
            # This gives the fraction of names that changed, 0 to 1
            if len(curr_long) + len(prev_long) > 0:
                long_turnover = len(curr_long.symmetric_difference(prev_long)) / (
                    len(curr_long) + len(prev_long)
                )
            else:
                long_turnover = 0.0

            if len(curr_short) + len(prev_short) > 0:
                short_turnover = len(curr_short.symmetric_difference(prev_short)) / (
                    len(curr_short) + len(prev_short)
                )
            else:
                short_turnover = 0.0

            daily_turnovers.append((long_turnover + short_turnover) / 2)
            prev = curr

        mean_daily = float(np.mean(daily_turnovers))
        mean_monthly = mean_daily * 21  # ~21 trading days / month

        # Transaction cost: 10 bps per trade (conservative for liquid stocks)
        cost_per_day = mean_daily * 0.001  # 0.1% per trade
        cost_per_month = cost_per_day * 21

        return {
            "mean_daily_turnover": round(mean_daily, 4),
            "mean_monthly_turnover": round(min(mean_monthly, 1.0), 4),
            "implied_transaction_cost": round(cost_per_month, 4),
            "net_return_after_costs": 0.0,  # Filled in by caller with L/S return
            "is_tradeable": mean_monthly < 0.8,  # <80% monthly turnover
        }

    # ── Sector Concentration ──────────────────────────────────

    def sector_concentration(
        self,
        daily_portfolios: List[Dict],
        sector_map: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """
        Analyze sector distribution of top/bottom deciles.
        Without a sector map, reports "unavailable".
        """
        if sector_map is None or not daily_portfolios:
            return {
                "long_sector_weights": {},
                "short_sector_weights": {},
                "hhi_concentration": 0.0,
                "is_sector_bet": False,
                "status": "sector_map_not_available",
            }

        # Aggregate sector counts
        long_sectors: Dict[str, int] = {}
        short_sectors: Dict[str, int] = {}

        for port in daily_portfolios:
            for t in port.get("long_tickers", []):
                sec = sector_map.get(t, "Unknown")
                long_sectors[sec] = long_sectors.get(sec, 0) + 1
            for t in port.get("short_tickers", []):
                sec = sector_map.get(t, "Unknown")
                short_sectors[sec] = short_sectors.get(sec, 0) + 1

        # Compute weights
        total_long = sum(long_sectors.values()) or 1
        total_short = sum(short_sectors.values()) or 1
        long_weights = {k: round(v / total_long, 3) for k, v in sorted(long_sectors.items(), key=lambda x: -x[1])}
        short_weights = {k: round(v / total_short, 3) for k, v in sorted(short_sectors.items(), key=lambda x: -x[1])}

        # HHI
        hhi = sum(w ** 2 for w in long_weights.values())
        is_bet = any(w > 0.4 for w in long_weights.values())

        return {
            "long_sector_weights": long_weights,
            "short_sector_weights": short_weights,
            "hhi_concentration": round(hhi, 3),
            "is_sector_bet": is_bet,
        }

    # ── Regime Analysis ───────────────────────────────────────

    def regime_analysis(
        self,
        backtest_results: "BacktestResults",
        polygon_data: "PolygonUniverseData",
    ) -> Dict[str, Any]:
        """
        Split results by market regime (bull/bear/sideways) and volatility.

        Uses SPY as the market proxy. Defines regimes by 63-day return and
        realized volatility.
        """
        if not backtest_results.daily_results:
            return {
                "ic_by_regime": {},
                "ic_by_volatility": {},
                "worst_regime": "unknown",
                "best_regime": "unknown",
                "is_regime_robust": False,
            }

        # Load SPY history
        spy_hist = polygon_data.get_stock_history("SPY", days=600)
        if spy_hist is None or spy_hist.empty:
            logger.warning("SPY data not available for regime analysis")
            return {
                "ic_by_regime": {},
                "ic_by_volatility": {},
                "worst_regime": "data_unavailable",
                "best_regime": "data_unavailable",
                "is_regime_robust": False,
            }

        spy_close = spy_hist["close"].astype(float)
        spy_log_ret = np.log(spy_close / spy_close.shift(1))

        # Compute SPY regime for each date
        spy_63d_ret = np.log(spy_close / spy_close.shift(63))
        spy_21d_vol = spy_log_ret.rolling(21, min_periods=15).std() * np.sqrt(252)
        vol_median = spy_21d_vol.median()

        # Map each backtest date to regime
        regime_ics: Dict[str, List[float]] = {"bull": [], "bear": [], "sideways": []}
        vol_ics: Dict[str, List[float]] = {"high_vol": [], "low_vol": []}

        for dr in backtest_results.daily_results:
            dt = dr["date"]
            ic = dr.get("ic", 0)
            if np.isnan(ic):
                continue

            # Find this date in SPY data
            try:
                if isinstance(spy_close.index, pd.DatetimeIndex):
                    spy_dates = spy_close.index.strftime("%Y-%m-%d")
                else:
                    spy_dates = spy_close.index.astype(str)

                positions = np.where(spy_dates == dt)[0]
                if len(positions) == 0:
                    continue
                pos = positions[0]

                if pos < 63:
                    continue

                ret_63 = float(spy_63d_ret.iloc[pos])
                vol_21 = float(spy_21d_vol.iloc[pos]) if pos < len(spy_21d_vol) else np.nan

                # Regime classification
                if ret_63 > 0.05:
                    regime_ics["bull"].append(ic)
                elif ret_63 < -0.05:
                    regime_ics["bear"].append(ic)
                else:
                    regime_ics["sideways"].append(ic)

                # Volatility classification
                if not np.isnan(vol_21) and not np.isnan(vol_median):
                    if vol_21 > vol_median:
                        vol_ics["high_vol"].append(ic)
                    else:
                        vol_ics["low_vol"].append(ic)

            except Exception:
                continue

        # Aggregate
        ic_by_regime = {}
        for regime, ics in regime_ics.items():
            if ics:
                ic_by_regime[regime] = {
                    "mean_ic": round(float(np.mean(ics)), 4),
                    "n_days": len(ics),
                }
            else:
                ic_by_regime[regime] = {"mean_ic": 0.0, "n_days": 0}

        ic_by_vol = {}
        for regime, ics in vol_ics.items():
            if ics:
                ic_by_vol[regime] = {
                    "mean_ic": round(float(np.mean(ics)), 4),
                    "n_days": len(ics),
                }
            else:
                ic_by_vol[regime] = {"mean_ic": 0.0, "n_days": 0}

        # Worst/best regime
        regime_means = {k: v["mean_ic"] for k, v in ic_by_regime.items() if v["n_days"] > 0}
        worst = min(regime_means, key=regime_means.get) if regime_means else "unknown"
        best = max(regime_means, key=regime_means.get) if regime_means else "unknown"

        # Robust if IC > 0 in all regimes with data
        is_robust = all(v["mean_ic"] > 0 for v in ic_by_regime.values() if v["n_days"] > 5)

        return {
            "ic_by_regime": ic_by_regime,
            "ic_by_volatility": ic_by_vol,
            "worst_regime": worst,
            "best_regime": best,
            "is_regime_robust": is_robust,
        }

    # ── Private Helpers ───────────────────────────────────────

    @staticmethod
    def _forward_return(
        df: pd.DataFrame, date_str: str, horizon: int
    ) -> Optional[float]:
        """Get forward return from a stock's history."""
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

            c0 = float(df.iloc[pos]["close"])
            c1 = float(df.iloc[fwd_pos]["close"])
            if c0 <= 0 or c1 <= 0:
                return None
            return float(np.log(c1 / c0))
        except Exception:
            return None

    @staticmethod
    def _compute_factor_chars(
        df: pd.DataFrame,
        date_str: str,
        spy_hist: Optional[pd.DataFrame] = None,
    ) -> Optional[Dict[str, float]]:
        """
        Compute factor characteristics for a stock as of a given date.

        Factors: market_beta, size, momentum, volatility, short_term_reversal
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

            if pos < 252:
                return None

            close = df["close"].astype(float).values
            volume = df["volume"].astype(float).values

            # Log returns
            log_ret = np.diff(np.log(close))

            # Size: log dollar volume (proxy for market cap)
            size = float(np.log(close[pos] * volume[pos] + 1))

            # Momentum: 252-day return excluding last 21 days
            if pos >= 252:
                momentum = float(np.log(close[pos - 21] / close[pos - 252]))
            else:
                momentum = 0.0

            # Volatility: 21-day realized vol
            if pos >= 21:
                recent_rets = log_ret[pos - 21:pos]
                volatility = float(np.std(recent_rets) * np.sqrt(252))
            else:
                volatility = 0.0

            # Short-term reversal: 21-day return
            if pos >= 21:
                str_reversal = float(np.log(close[pos] / close[pos - 21]))
            else:
                str_reversal = 0.0

            # Market beta: regression of stock returns on SPY returns (63d)
            market_beta = 0.0
            if spy_hist is not None and not spy_hist.empty:
                try:
                    spy_close = spy_hist["close"].astype(float)
                    if isinstance(spy_hist.index, pd.DatetimeIndex):
                        spy_dates = spy_hist.index.strftime("%Y-%m-%d")
                    else:
                        spy_dates = spy_hist.index.astype(str)

                    spy_pos = np.where(spy_dates == date_str)[0]
                    if len(spy_pos) > 0 and spy_pos[0] >= 63:
                        sp = spy_pos[0]
                        spy_rets = np.diff(np.log(spy_close.values[sp - 63:sp + 1]))
                        stock_rets = log_ret[pos - 63:pos]
                        min_len = min(len(spy_rets), len(stock_rets))
                        if min_len >= 30:
                            spy_rets = spy_rets[:min_len]
                            stock_rets = stock_rets[:min_len]
                            cov = np.cov(stock_rets, spy_rets)
                            if cov[1, 1] > 1e-10:
                                market_beta = float(cov[0, 1] / cov[1, 1])
                except Exception:
                    pass

            return {
                "market_beta": market_beta,
                "size": size,
                "momentum": momentum,
                "volatility": volatility,
                "short_term_reversal": str_reversal,
            }

        except Exception:
            return None
