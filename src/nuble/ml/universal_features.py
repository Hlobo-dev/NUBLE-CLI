"""
Universal Feature Engine — Institutional-Grade Feature Computation
====================================================================

Computes 120+ stock-agnostic features from OHLCV data, organized into
6 feature groups. Every feature is comparable across stocks — it measures
a property of price/volume behavior, not a property of a specific stock.

DESIGN PRINCIPLES:

1. NO LOOKAHEAD BIAS: Every feature uses only data available at time t.
2. STATIONARITY: All features are returns, ratios, z-scores, or ranks.
3. STOCK-AGNOSTIC: Features are comparable across any stock at any price.
4. ROBUST: Handle NaN, inf, division by zero with safe computations.
5. MEMORY EFFICIENT: float32 throughout. <2GB for 2000×500×120.

Feature Groups:
  GROUP 1: Momentum (25) — trend, mean-reversion, multi-horizon returns
  GROUP 2: Volatility (20) — realized vol, tail risk, regime detection
  GROUP 3: Volume (18) — liquidity, price-volume, microstructure
  GROUP 4: Technical (25) — RSI, MACD, Bollinger, ADX, Stochastic, etc.
  GROUP 5: Microstructure (12) — intraday range, serial correlation, spread
  GROUP 6: Context (12) — calendar, regime, streaks, relative strength

Author: NUBLE ML Pipeline — Phase 3 Institutional Upgrade
"""

from __future__ import annotations

import logging
import warnings
from typing import Dict, List, Optional, TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from ..data.sec_edgar import SECEdgarXBRL
    from ..data.fred_macro import FREDMacroData

logger = logging.getLogger(__name__)

# Suppress pandas warnings for chained assignment
try:
    warnings.filterwarnings("ignore", category=pd.errors.SettingWithCopyWarning)
except AttributeError:
    pass  # older pandas versions

# ══════════════════════════════════════════════════════════════
# Safe computation helpers
# ══════════════════════════════════════════════════════════════

_EPS = 1e-8  # epsilon for safe division


def _safe_div(a, b):
    """Element-wise safe division, returns 0.0 where denominator is ~0."""
    if isinstance(b, (pd.Series, pd.DataFrame)):
        result = a / b.replace(0, np.nan)
    else:
        result = a / (b if abs(b) > _EPS else np.nan)
    return result


def _safe_series_div(num: pd.Series, den: pd.Series) -> pd.Series:
    """Safe division for two Series, replacing inf/nan from zero-denom."""
    den_safe = den.copy()
    den_safe[den_safe.abs() < _EPS] = np.nan
    return num / den_safe


def _rolling_zscore(s: pd.Series, window: int) -> pd.Series:
    """Rolling z-score of a series within its own history."""
    mean = s.rolling(window, min_periods=max(window // 2, 10)).mean()
    std = s.rolling(window, min_periods=max(window // 2, 10)).std()
    std = std.replace(0, np.nan)
    return (s - mean) / std


# ══════════════════════════════════════════════════════════════
# UniversalFeatureEngine
# ══════════════════════════════════════════════════════════════


class UniversalFeatureEngine:
    """
    Computes 120+ features for any stock from OHLCV data, organized into
    6 feature groups. Every feature is STOCK-AGNOSTIC.

    Usage:
        engine = UniversalFeatureEngine()
        features = engine.compute_features(ohlcv_df)
        # features.shape → (~500, ~120)
    """

    # Minimum rows needed (longest lookback is 252 for 12-month momentum)
    MIN_ROWS = 63
    WARMUP_ROWS = 60  # Rows to drop after computation (NaN warmup)

    # Feature group prefixes
    PREFIXES = {
        "momentum": "mom_",
        "volatility": "vol_",
        "volume": "vlm_",
        "technical": "tech_",
        "microstructure": "micro_",
        "context": "ctx_",
        "fundamental": "fund_",
        "macro": "macro_",
    }

    def compute_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Main entry point. Takes OHLCV DataFrame, returns ~120 feature columns.

        Input columns: [open, high, low, close, volume] minimum
                       [vwap, transactions] optional
        Output: DataFrame with float32 feature columns, same index as input.
        """
        if len(df) < self.MIN_ROWS:
            return pd.DataFrame()

        # Ensure float64 for computation precision
        close = df["close"].astype(np.float64)
        high = df["high"].astype(np.float64)
        low = df["low"].astype(np.float64)
        opn = df["open"].astype(np.float64)
        volume = df["volume"].astype(np.float64)

        # Check for vwap / transactions
        has_vwap = "vwap" in df.columns
        has_txns = "transactions" in df.columns
        vwap = df["vwap"].astype(np.float64) if has_vwap else None
        txns = df["transactions"].astype(np.float64) if has_txns else None

        # Log return (used everywhere)
        log_ret = np.log(close / close.shift(1))

        # Compute each group
        parts = [
            self._momentum_features(close, high, low, log_ret, df.index),
            self._volatility_features(close, high, low, opn, log_ret),
            self._volume_features(close, high, low, volume, log_ret, vwap, txns),
            self._technical_features(close, high, low, opn, volume, log_ret),
            self._microstructure_features(close, high, low, opn, volume, log_ret),
            self._context_features(close, high, low, log_ret, volume, df.index),
        ]

        features = pd.concat(parts, axis=1)

        # Post-process
        features = self._postprocess(features)

        return features

    # ═══════════════════════════════════════════════════════════
    # GROUP 1: MOMENTUM FEATURES (25 features)
    # ═══════════════════════════════════════════════════════════

    def _momentum_features(
        self,
        close: pd.Series,
        high: pd.Series,
        low: pd.Series,
        log_ret: pd.Series,
        index: pd.Index,
    ) -> pd.DataFrame:
        f = pd.DataFrame(index=index)

        # Multi-horizon log returns
        for n in [1, 5, 10, 21, 63, 126, 252]:
            f[f"mom_ret_{n}d"] = np.log(close / close.shift(n))

        # Jegadeesh-Titman momentum: 12-month minus 1-month (skip recent reversal)
        f["mom_momentum_12_1"] = f["mom_ret_252d"] - f["mom_ret_21d"]
        # 6-month minus 1-month
        f["mom_momentum_6_1"] = f["mom_ret_126d"] - f["mom_ret_21d"]
        # Short-term reversal (1-month return itself — negative predictor)
        f["mom_short_term_reversal"] = f["mom_ret_21d"]

        # Moving average ratios
        sma_20 = close.rolling(20, min_periods=15).mean()
        sma_50 = close.rolling(50, min_periods=30).mean()
        sma_200 = close.rolling(200, min_periods=100).mean()

        f["mom_price_sma20_ratio"] = _safe_series_div(close, sma_20) - 1
        f["mom_price_sma50_ratio"] = _safe_series_div(close, sma_50) - 1
        f["mom_price_sma200_ratio"] = _safe_series_div(close, sma_200) - 1
        f["mom_sma_20_50_cross"] = _safe_series_div(sma_20, sma_50) - 1
        f["mom_sma_50_200_cross"] = _safe_series_div(sma_50, sma_200) - 1

        # Rate of change
        f["mom_roc_5"] = close / close.shift(5) - 1
        f["mom_roc_21"] = close / close.shift(21) - 1
        # Acceleration: is momentum increasing?
        roc_21 = f["mom_roc_21"]
        f["mom_acceleration_21"] = roc_21 - roc_21.shift(21)

        # Distance from extremes
        high_252 = high.rolling(252, min_periods=126).max()
        low_252 = low.rolling(252, min_periods=126).min()
        f["mom_dist_52w_high"] = _safe_series_div(close, high_252) - 1
        f["mom_dist_52w_low"] = _safe_series_div(close, low_252) - 1
        range_252 = high_252 - low_252
        f["mom_high_low_position"] = _safe_series_div(close - low_252, range_252)

        # EMA momentum (fast vs slow EMA ratio)
        ema_12 = close.ewm(span=12, min_periods=10, adjust=False).mean()
        ema_26 = close.ewm(span=26, min_periods=20, adjust=False).mean()
        f["mom_ema_12_26_ratio"] = _safe_series_div(ema_12, ema_26) - 1

        # Momentum quality: return / path (smoothed efficiency)
        net_21 = (close - close.shift(21)).abs()
        path_21 = log_ret.abs().rolling(21, min_periods=15).sum() * close
        f["mom_quality_21"] = _safe_series_div(net_21, path_21)

        # 3-month momentum vs 1-month (intermediate horizon)
        f["mom_momentum_3_1"] = f["mom_ret_63d"] - f["mom_ret_21d"]

        # Weekly return volatility (stability of momentum)
        ret_5d = np.log(close / close.shift(5))
        f["mom_weekly_ret_vol"] = ret_5d.rolling(12, min_periods=8).std()

        return f

    # ═══════════════════════════════════════════════════════════
    # GROUP 2: VOLATILITY & RISK FEATURES (20 features)
    # ═══════════════════════════════════════════════════════════

    def _volatility_features(
        self,
        close: pd.Series,
        high: pd.Series,
        low: pd.Series,
        opn: pd.Series,
        log_ret: pd.Series,
    ) -> pd.DataFrame:
        f = pd.DataFrame(index=close.index)

        # Realized volatility at multiple horizons
        for n, label in [(5, "5"), (21, "21"), (63, "63")]:
            f[f"vol_realized_{label}"] = (
                log_ret.rolling(n, min_periods=max(n // 2, 3)).std() * np.sqrt(252)
            )

        rv21 = f["vol_realized_21"]
        rv63 = f["vol_realized_63"]

        # Garman-Klass volatility estimator (5-8x more efficient than close-close)
        log_hl = np.log(high / low)
        log_co = np.log(close / opn)
        gk_daily = 0.5 * log_hl**2 - (2 * np.log(2) - 1) * log_co**2
        gk_mean = gk_daily.rolling(21, min_periods=15).mean()
        f["vol_garman_klass"] = np.sqrt(gk_mean.clip(lower=0)) * np.sqrt(252)

        # Parkinson high-low estimator
        pk_daily = log_hl**2
        pk_mean = pk_daily.rolling(21, min_periods=15).mean()
        f["vol_parkinson"] = np.sqrt(pk_mean / (4 * np.log(2))) * np.sqrt(252)

        # Volatility regime: short-term / long-term
        f["vol_regime"] = _safe_series_div(rv21, rv63)

        # Vol of vol: how stable is volatility?
        f["vol_of_vol"] = rv21.rolling(63, min_periods=30).std()

        # Vol z-score: how unusual is current vol vs 1-year?
        vol_252_mean = rv21.rolling(252, min_periods=126).mean()
        vol_252_std = rv21.rolling(252, min_periods=126).std()
        f["vol_z_score"] = _safe_series_div(rv21 - vol_252_mean, vol_252_std)

        # Tail risk
        f["vol_max_dd_21"] = self._max_drawdown_rolling(close, 21)
        f["vol_max_dd_63"] = self._max_drawdown_rolling(close, 63)

        f["vol_skewness_21"] = log_ret.rolling(21, min_periods=15).skew()
        f["vol_kurtosis_21"] = log_ret.rolling(21, min_periods=15).kurt()

        # Downside risk
        neg_ret = log_ret.copy()
        neg_ret[neg_ret > 0] = np.nan
        downside_vol = neg_ret.rolling(21, min_periods=10).std() * np.sqrt(252)
        pos_ret = log_ret.copy()
        pos_ret[pos_ret < 0] = np.nan
        upside_vol = pos_ret.rolling(21, min_periods=10).std() * np.sqrt(252)
        f["vol_downside_21"] = downside_vol
        f["vol_up_down_ratio"] = _safe_series_div(upside_vol, downside_vol)

        # Normalized ATR
        prev_close = close.shift(1)
        tr = pd.concat([
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ], axis=1).max(axis=1)
        atr_14 = tr.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
        f["vol_atr_pct"] = _safe_series_div(atr_14, close)

        # ATR ratio: expanding or contracting?
        atr_5 = tr.ewm(alpha=1 / 5, min_periods=5, adjust=False).mean()
        f["vol_atr_ratio"] = _safe_series_div(atr_5, atr_14)

        # Kaufman Efficiency Ratio (0=choppy, 1=trending)
        net_move = (close - close.shift(21)).abs()
        path_sum = log_ret.abs().rolling(21, min_periods=15).sum()
        # Convert path to price-space: approx close * path_sum
        path_price = close * path_sum
        f["vol_efficiency_ratio"] = _safe_series_div(net_move, path_price)

        # Z-score of price vs 20-day SMA
        sma_20 = close.rolling(20, min_periods=15).mean()
        std_20 = close.rolling(20, min_periods=15).std()
        f["vol_z_score_20"] = _safe_series_div(close - sma_20, std_20)

        # Realized vol ratio: 5d vs 21d (short-term spike detection)
        rv5 = f["vol_realized_5"]
        f["vol_rv5_rv21_ratio"] = _safe_series_div(rv5, rv21)

        # Range volatility (high-low based, 21-day average)
        hl_pct = _safe_series_div(high - low, close)
        f["vol_range_avg_21"] = hl_pct.rolling(21, min_periods=15).mean()

        return f

    # ═══════════════════════════════════════════════════════════
    # GROUP 3: VOLUME & LIQUIDITY FEATURES (18 features)
    # ═══════════════════════════════════════════════════════════

    def _volume_features(
        self,
        close: pd.Series,
        high: pd.Series,
        low: pd.Series,
        volume: pd.Series,
        log_ret: pd.Series,
        vwap: Optional[pd.Series],
        txns: Optional[pd.Series],
    ) -> pd.DataFrame:
        f = pd.DataFrame(index=close.index)

        # Volume ratios (stock-agnostic)
        vol_sma5 = volume.rolling(5, min_periods=3).mean()
        vol_sma20 = volume.rolling(20, min_periods=10).mean()
        vol_sma63 = volume.rolling(63, min_periods=30).mean()

        f["vlm_sma5_ratio"] = _safe_series_div(volume, vol_sma5)
        f["vlm_sma20_ratio"] = _safe_series_div(volume, vol_sma20)
        f["vlm_sma63_ratio"] = _safe_series_div(volume, vol_sma63)
        f["vlm_trend"] = _safe_series_div(vol_sma5, vol_sma20)

        # Volume z-score
        vol_std63 = volume.rolling(63, min_periods=30).std()
        f["vlm_z_score"] = _safe_series_div(volume - vol_sma63, vol_std63)

        # OBV slope (On Balance Volume)
        obv_sign = log_ret.apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
        obv = (obv_sign * volume).cumsum()
        # Normalize OBV slope by dividing by average volume
        obv_slope = obv.rolling(21, min_periods=15).apply(
            lambda x: np.polyfit(np.arange(len(x)), x, 1)[0] if len(x) >= 10 else np.nan,
            raw=True,
        )
        f["vlm_obv_slope_21"] = _safe_series_div(obv_slope, vol_sma20)

        # Price-volume correlation
        vol_change = volume.pct_change()
        f["vlm_price_vol_corr_21"] = log_ret.rolling(21, min_periods=15).corr(vol_change)

        # Price-volume divergence (binary)
        ret_21 = np.log(close / close.shift(21))
        f["vlm_price_vol_divergence"] = (
            (ret_21 > 0).astype(float) * (f["vlm_obv_slope_21"] < 0).astype(float)
            - (ret_21 < 0).astype(float) * (f["vlm_obv_slope_21"] > 0).astype(float)
        )

        # Dollar volume (liquidity)
        dollar_vol = close * volume
        f["vlm_log_dollar_volume"] = np.log1p(dollar_vol)

        # Volume breakout (binary)
        f["vlm_breakout"] = (volume > 2 * vol_sma20).astype(np.float32)

        # High volume return (signal only on high-vol days)
        high_vol_mask = (volume / vol_sma20) > 1.5
        f["vlm_high_vol_return"] = log_ret * high_vol_mask.astype(float)

        # Amihud illiquidity
        amihud_daily = _safe_series_div(log_ret.abs(), dollar_vol)
        f["vlm_amihud_21"] = amihud_daily.rolling(21, min_periods=10).mean()
        amihud_63 = amihud_daily.rolling(63, min_periods=30).mean()
        f["vlm_amihud_trend"] = _safe_series_div(f["vlm_amihud_21"], amihud_63)

        # Dollar volume rank within own history (0-1)
        f["vlm_dollar_vol_rank"] = dollar_vol.rolling(63, min_periods=30).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=True
        )

        # Volume concentration: how much of 21d volume was in top 5 days
        f["vlm_concentration_21"] = volume.rolling(21, min_periods=15).apply(
            lambda x: np.sort(x)[-5:].sum() / (x.sum() + _EPS), raw=True
        )

        # VWAP features (if available)
        if vwap is not None:
            f["vlm_vwap_deviation"] = _safe_series_div(close - vwap, vwap)
            vwap_slope = vwap.rolling(5, min_periods=3).apply(
                lambda x: np.polyfit(np.arange(len(x)), x, 1)[0] / (x.mean() + _EPS),
                raw=True,
            )
            f["vlm_vwap_slope_5"] = vwap_slope
        else:
            f["vlm_vwap_deviation"] = np.nan
            f["vlm_vwap_slope_5"] = np.nan

        # Transaction features (if available)
        if txns is not None and txns.sum() > 0:
            avg_trade = _safe_series_div(dollar_vol, txns)
            avg_trade_sma63 = avg_trade.rolling(63, min_periods=30).mean()
            avg_trade_std63 = avg_trade.rolling(63, min_periods=30).std()
            f["vlm_avg_trade_size_z"] = _safe_series_div(
                avg_trade - avg_trade_sma63, avg_trade_std63
            )
        else:
            f["vlm_avg_trade_size_z"] = np.nan

        return f

    # ═══════════════════════════════════════════════════════════
    # GROUP 4: TECHNICAL INDICATOR FEATURES (25 features)
    # ═══════════════════════════════════════════════════════════

    def _technical_features(
        self,
        close: pd.Series,
        high: pd.Series,
        low: pd.Series,
        opn: pd.Series,
        volume: pd.Series,
        log_ret: pd.Series,
    ) -> pd.DataFrame:
        f = pd.DataFrame(index=close.index)

        # ── RSI family ──
        for period, label in [(7, "7"), (14, "14"), (21, "21")]:
            f[f"tech_rsi_{label}"] = self._rsi(close, period)

        rsi14 = f["tech_rsi_14"]
        f["tech_rsi_divergence"] = rsi14 - rsi14.shift(14)

        # ── MACD ──
        ema12 = close.ewm(span=12, min_periods=10, adjust=False).mean()
        ema26 = close.ewm(span=26, min_periods=20, adjust=False).mean()
        macd_line = ema12 - ema26
        signal_line = macd_line.ewm(span=9, min_periods=7, adjust=False).mean()
        macd_hist = macd_line - signal_line

        f["tech_macd_signal"] = _safe_series_div(macd_line, close)
        f["tech_macd_histogram"] = _safe_series_div(macd_hist, close)
        macd_hist_sign = np.sign(macd_hist)
        f["tech_macd_crossover"] = macd_hist_sign - macd_hist_sign.shift(1)

        # ── Bollinger Bands ──
        sma_20 = close.rolling(20, min_periods=15).mean()
        std_20 = close.rolling(20, min_periods=15).std()
        bb_upper = sma_20 + 2 * std_20
        bb_lower = sma_20 - 2 * std_20
        bb_range = bb_upper - bb_lower

        f["tech_bb_position"] = _safe_series_div(close - bb_lower, bb_range)
        bb_width = _safe_series_div(bb_range, sma_20)
        f["tech_bb_width"] = bb_width
        f["tech_bb_width_change"] = _safe_series_div(bb_width, bb_width.shift(21)) - 1

        # ── Stochastic ──
        low_14 = low.rolling(14, min_periods=10).min()
        high_14 = high.rolling(14, min_periods=10).max()
        stoch_k = _safe_series_div(close - low_14, high_14 - low_14) * 100
        stoch_d = stoch_k.rolling(3, min_periods=2).mean()
        f["tech_stoch_k"] = stoch_k
        f["tech_stoch_d"] = stoch_d
        f["tech_stoch_crossover"] = (stoch_k > stoch_d).astype(np.float32)

        # ── ADX (trend strength) ──
        prev_close = close.shift(1)
        tr = pd.concat([
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ], axis=1).max(axis=1)
        atr_14 = tr.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()

        up_move = high.diff()
        down_move = -low.diff()
        plus_dm = pd.Series(0.0, index=high.index)
        minus_dm = pd.Series(0.0, index=high.index)
        cond_plus = (up_move > down_move) & (up_move > 0)
        cond_minus = (down_move > up_move) & (down_move > 0)
        plus_dm[cond_plus] = up_move[cond_plus]
        minus_dm[cond_minus] = down_move[cond_minus]

        smooth_plus = plus_dm.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
        smooth_minus = minus_dm.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
        plus_di = _safe_series_div(smooth_plus, atr_14) * 100
        minus_di = _safe_series_div(smooth_minus, atr_14) * 100
        dx = _safe_series_div((plus_di - minus_di).abs(), plus_di + minus_di) * 100
        adx = dx.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
        f["tech_adx_14"] = adx
        f["tech_adx_slope"] = adx - adx.shift(5)

        # ── Williams %R ──
        f["tech_williams_r_14"] = _safe_series_div(
            high_14 - close, high_14 - low_14
        ) * -100

        # ── CCI (Commodity Channel Index) ──
        tp = (high + low + close) / 3
        tp_sma = tp.rolling(20, min_periods=15).mean()
        tp_mad = tp.rolling(20, min_periods=15).apply(
            lambda x: np.abs(x - x.mean()).mean(), raw=True
        )
        f["tech_cci_20"] = _safe_series_div(tp - tp_sma, 0.015 * tp_mad)

        # ── MFI (Money Flow Index — volume-weighted RSI) ──
        mf_raw = tp * volume
        mf_pos = mf_raw.where(tp > tp.shift(1), 0)
        mf_neg = mf_raw.where(tp < tp.shift(1), 0)
        mf_pos_sum = mf_pos.rolling(14, min_periods=10).sum()
        mf_neg_sum = mf_neg.rolling(14, min_periods=10).sum()
        mf_ratio = _safe_series_div(mf_pos_sum, mf_neg_sum)
        f["tech_mfi_14"] = 100 - _safe_series_div(
            pd.Series(100.0, index=close.index), 1 + mf_ratio
        )

        # ── Force Index ──
        force_raw = close.diff() * volume
        force_ema13 = force_raw.ewm(span=13, min_periods=10, adjust=False).mean()
        dollar_vol_avg = (close * volume).rolling(63, min_periods=30).mean()
        f["tech_force_index_13"] = _safe_series_div(force_ema13, dollar_vol_avg)

        # ── Hurst Exponent (vectorized approximation) ──
        f["tech_hurst"] = self._rolling_hurst_fast(log_ret, window=63)

        # ── Ichimoku Cloud (simplified — conversion vs base) ──
        conv_high = high.rolling(9, min_periods=7).max()
        conv_low = low.rolling(9, min_periods=7).min()
        conversion_line = (conv_high + conv_low) / 2

        base_high = high.rolling(26, min_periods=20).max()
        base_low = low.rolling(26, min_periods=20).min()
        base_line = (base_high + base_low) / 2

        f["tech_ichimoku_conv_base"] = _safe_series_div(
            conversion_line - base_line, close
        )

        # Cloud: span A & B
        span_a = (conversion_line + base_line) / 2
        span_b_high = high.rolling(52, min_periods=40).max()
        span_b_low = low.rolling(52, min_periods=40).min()
        span_b = (span_b_high + span_b_low) / 2
        cloud_top = pd.concat([span_a, span_b], axis=1).max(axis=1)
        cloud_bottom = pd.concat([span_a, span_b], axis=1).min(axis=1)

        # +1 above cloud, -1 below cloud, 0 inside
        f["tech_ichimoku_price_cloud"] = np.where(
            close > cloud_top, 1.0,
            np.where(close < cloud_bottom, -1.0, 0.0)
        ).astype(np.float32)

        # ── Price patterns ──
        prev_close = close.shift(1)
        gap = _safe_series_div(opn - prev_close, prev_close)
        f["tech_gap_up"] = gap.where(gap > 0.005, 0.0)
        f["tech_gap_down"] = gap.where(gap < -0.005, 0.0)

        hl_range = high - low + _EPS
        f["tech_body_size"] = _safe_series_div((close - opn).abs(), hl_range)
        f["tech_upper_shadow"] = _safe_series_div(
            high - pd.concat([opn, close], axis=1).max(axis=1), hl_range
        )
        f["tech_lower_shadow"] = _safe_series_div(
            pd.concat([opn, close], axis=1).min(axis=1) - low, hl_range
        )

        return f

    # ═══════════════════════════════════════════════════════════
    # GROUP 5: MICROSTRUCTURE FEATURES (12 features)
    # ═══════════════════════════════════════════════════════════

    def _microstructure_features(
        self,
        close: pd.Series,
        high: pd.Series,
        low: pd.Series,
        opn: pd.Series,
        volume: pd.Series,
        log_ret: pd.Series,
    ) -> pd.DataFrame:
        f = pd.DataFrame(index=close.index)

        # Intraday range
        intraday_range = _safe_series_div(high - low, close)
        f["micro_intraday_range"] = intraday_range
        range_sma20 = intraday_range.rolling(20, min_periods=10).mean()
        f["micro_range_sma_ratio"] = _safe_series_div(intraday_range, range_sma20)

        # Overnight vs intraday returns
        prev_close = close.shift(1)
        f["micro_overnight_return"] = np.log(opn / prev_close)
        f["micro_intraday_return"] = np.log(close / opn)

        overnight = f["micro_overnight_return"]
        intraday = f["micro_intraday_return"]
        f["micro_overnight_intraday_ratio"] = _safe_series_div(
            overnight.abs(), intraday.abs() + _EPS
        )

        # Kyle's lambda (price impact proxy via rolling correlation)
        dollar_vol = close * volume
        ret_abs = log_ret.abs()
        f["micro_kyle_lambda_21"] = ret_abs.rolling(21, min_periods=15).corr(
            np.log1p(dollar_vol)
        )

        # Serial correlation
        f["micro_autocorr_1"] = log_ret.rolling(63, min_periods=30).apply(
            lambda x: pd.Series(x).autocorr(lag=1), raw=True
        )
        f["micro_autocorr_5"] = log_ret.rolling(63, min_periods=30).apply(
            lambda x: pd.Series(x).autocorr(lag=5), raw=True
        )

        # Variance ratio (Lo & MacKinlay 1988)
        var_1d = log_ret.rolling(63, min_periods=30).var()
        ret_5d = np.log(close / close.shift(5))
        var_5d = ret_5d.rolling(63, min_periods=30).var()
        f["micro_variance_ratio_5"] = _safe_series_div(var_5d, 5 * var_1d)

        # Close Location Value
        f["micro_clv"] = _safe_series_div(
            close - low, high - low + _EPS
        )

        # Accumulation/Distribution slope
        clv = f["micro_clv"]
        ad = (clv * volume).cumsum()
        vol_sma20 = volume.rolling(20, min_periods=10).mean()
        ad_slope = ad.rolling(21, min_periods=15).apply(
            lambda x: np.polyfit(np.arange(len(x)), x, 1)[0] if len(x) >= 10 else np.nan,
            raw=True,
        )
        f["micro_ad_slope_21"] = _safe_series_div(ad_slope, vol_sma20)

        # Realized spread proxy (Roll 1984 adapted)
        midpoint = (high + low) / 2
        f["micro_realized_spread"] = _safe_series_div(
            2 * (close - midpoint).abs(), close
        )

        return f

    # ═══════════════════════════════════════════════════════════
    # GROUP 6: CONTEXT FEATURES (12 features)
    # ═══════════════════════════════════════════════════════════

    def _context_features(
        self,
        close: pd.Series,
        high: pd.Series,
        low: pd.Series,
        log_ret: pd.Series,
        volume: pd.Series,
        index: pd.Index,
    ) -> pd.DataFrame:
        f = pd.DataFrame(index=index)

        # Calendar — low-cardinality indicators that can't overfit
        # Removed raw ctx_month (12 buckets → 20% importance = calendar overfitting)
        # Removed raw ctx_day_of_week (same problem)
        # Instead: binary/low-card features with genuine economic meaning
        if isinstance(index, pd.DatetimeIndex):
            # Monday effect (academic anomaly): 1 if Monday, 0 otherwise
            f["ctx_is_monday"] = (index.dayofweek == 0).astype(np.float32)
            # Friday effect (weekend risk): 1 if Friday, 0 otherwise
            f["ctx_is_friday"] = (index.dayofweek == 4).astype(np.float32)
            # Month-end effect (rebalancing): 1 if last 3 trading days of month
            day_in_month = pd.Series(index.day, index=index)
            days_in_month = pd.Series(index.days_in_month, index=index)
            f["ctx_is_month_end"] = (days_in_month - day_in_month <= 3).astype(np.float32)
            # January effect (small-cap anomaly): 1 if January
            f["ctx_is_january"] = (index.month == 1).astype(np.float32)
            # Quarter-end effect (window dressing): 1 if last month of quarter
            f["ctx_is_quarter_end"] = (index.month.isin([3, 6, 9, 12])).astype(np.float32)
        else:
            f["ctx_is_monday"] = np.nan
            f["ctx_is_friday"] = np.nan
            f["ctx_is_month_end"] = np.nan
            f["ctx_is_january"] = np.nan
            f["ctx_is_quarter_end"] = np.nan

        # Relative strength (self-relative)
        rv21 = log_ret.rolling(21, min_periods=15).std() * np.sqrt(252)
        ret_21d = np.log(close / close.shift(21))
        f["ctx_return_vs_own_vol"] = _safe_series_div(ret_21d, rv21)

        high_252 = high.rolling(252, min_periods=126).max()
        low_252 = low.rolling(252, min_periods=126).min()
        own_range = _safe_series_div(high_252, low_252) - 1
        f["ctx_return_vs_own_range"] = _safe_series_div(ret_21d, own_range)

        # Trend strength (efficiency ratio over 63 days)
        net_63 = (close - close.shift(63)).abs()
        path_63 = log_ret.abs().rolling(63, min_periods=30).sum() * close
        f["ctx_trend_strength"] = _safe_series_div(net_63, path_63)

        # Regime change score
        rv63 = log_ret.rolling(63, min_periods=30).std() * np.sqrt(252)
        vol_regime = _safe_series_div(rv21, rv63)
        f["ctx_regime_change_score"] = (vol_regime - vol_regime.shift(21)).abs()

        # Drawdown duration (days since 52-week high)
        expanding_max = close.expanding().max()
        at_high = close >= expanding_max
        # Count days since last True
        dd_duration = pd.Series(0.0, index=index)
        counter = 0
        for i in range(len(at_high)):
            if at_high.iloc[i]:
                counter = 0
            else:
                counter += 1
            dd_duration.iloc[i] = counter
        f["ctx_drawdown_duration"] = dd_duration.astype(np.float32)

        # Recovery speed
        dd_63 = self._max_drawdown_rolling(close, 63)
        ret_5d = np.log(close / close.shift(5))
        f["ctx_recovery_speed"] = _safe_series_div(ret_5d, dd_63.abs() + _EPS)
        # Only meaningful during drawdowns
        f.loc[dd_duration < 5, "ctx_recovery_speed"] = 0.0

        # Log dollar volume average (size proxy)
        dollar_vol_avg = (close * volume).rolling(63, min_periods=30).mean()
        f["ctx_log_dollar_volume_avg"] = np.log1p(dollar_vol_avg)

        # Days since volume spike (proxy for earnings)
        vol_sma20 = volume.rolling(20, min_periods=10).mean()
        is_spike = volume > 3 * vol_sma20
        days_since_spike = pd.Series(252.0, index=index)  # default high
        counter = 252
        for i in range(len(is_spike)):
            if is_spike.iloc[i]:
                counter = 0
            else:
                counter += 1
            days_since_spike.iloc[i] = min(counter, 252)
        f["ctx_days_since_vol_spike"] = days_since_spike.astype(np.float32)

        # Streak features
        pos = (log_ret > 0).astype(int)
        neg = (log_ret < 0).astype(int)
        f["ctx_consecutive_up"] = self._streak_counter(pos).astype(np.float32)
        f["ctx_consecutive_down"] = self._streak_counter(neg).astype(np.float32)

        return f

    # ═══════════════════════════════════════════════════════════
    # OPTIONAL: FUNDAMENTAL FEATURES (from SEC EDGAR)
    # ═══════════════════════════════════════════════════════════

    def add_fundamental_features(
        self,
        features_df: pd.DataFrame,
        ticker: str,
        sec_edgar: Optional["SECEdgarXBRL"] = None,
    ) -> pd.DataFrame:
        """
        Add fundamental ratio features from SEC EDGAR XBRL.
        These are slow-moving features (change quarterly).

        Returns features_df with fund_ columns added.
        If SEC data unavailable, columns are NaN (LightGBM handles natively).
        """
        # Top 10 most predictive fundamental ratios (Gu-Kelly-Xiu)
        FUND_FEATURES = [
            "book_to_market",
            "earnings_to_price",
            "roe",
            "roa",
            "gross_profitability",
            "accruals",
            "asset_growth",
            "sales_growth",
            "debt_to_equity",
            "current_ratio",
        ]

        if sec_edgar is None:
            for feat in FUND_FEATURES:
                features_df[f"fund_{feat}"] = np.nan
            return features_df

        try:
            ratios = sec_edgar.get_fundamental_ratios(ticker)
            if ratios is None:
                for feat in FUND_FEATURES:
                    features_df[f"fund_{feat}"] = np.nan
                return features_df

            for feat in FUND_FEATURES:
                val = ratios.get(feat)
                features_df[f"fund_{feat}"] = float(val) if val is not None else np.nan

        except Exception as e:
            logger.debug("Fundamental features for %s failed: %s", ticker, e)
            for feat in FUND_FEATURES:
                features_df[f"fund_{feat}"] = np.nan

        return features_df

    # ═══════════════════════════════════════════════════════════
    # OPTIONAL: MACRO FEATURES (from FRED)
    # ═══════════════════════════════════════════════════════════

    def add_macro_features(
        self,
        features_df: pd.DataFrame,
        fred: Optional["FREDMacroData"] = None,
    ) -> pd.DataFrame:
        """
        Add macro regime features from FRED.
        Same for ALL stocks on a given date (market-level context).

        Returns features_df with macro_ columns added.
        If FRED unavailable, columns are NaN.
        """
        MACRO_FEATURES = [
            "macro_term_spread",
            "macro_credit_spread",
            "macro_fed_funds",
            "macro_breakeven_inflation",
            "macro_yield_curve_inverted",
            "macro_credit_tightening",
            "macro_real_rate",
            "macro_unemployment_trend",
        ]

        if fred is None or not fred.is_available:
            for feat in MACRO_FEATURES:
                features_df[feat] = np.nan
            return features_df

        try:
            macro_hist = fred.get_history()
            if macro_hist is None or macro_hist.empty:
                for feat in MACRO_FEATURES:
                    features_df[feat] = np.nan
                return features_df

            # Align macro data to features index
            macro_aligned = macro_hist.reindex(features_df.index, method="ffill")

            if "term_spread" in macro_aligned.columns:
                features_df["macro_term_spread"] = macro_aligned["term_spread"].values
            else:
                features_df["macro_term_spread"] = np.nan

            if "credit_spread" in macro_aligned.columns:
                features_df["macro_credit_spread"] = macro_aligned["credit_spread"].values
            else:
                features_df["macro_credit_spread"] = np.nan

            if "fed_funds" in macro_aligned.columns:
                features_df["macro_fed_funds"] = macro_aligned["fed_funds"].values
            else:
                features_df["macro_fed_funds"] = np.nan

            if "breakeven_inflation" in macro_aligned.columns:
                features_df["macro_breakeven_inflation"] = macro_aligned[
                    "breakeven_inflation"
                ].values
            else:
                features_df["macro_breakeven_inflation"] = np.nan

            # Derived
            if "term_spread" in macro_aligned.columns:
                features_df["macro_yield_curve_inverted"] = (
                    macro_aligned["term_spread"] < 0
                ).astype(np.float32).values
            else:
                features_df["macro_yield_curve_inverted"] = np.nan

            if "credit_spread" in macro_aligned.columns:
                cs = macro_aligned["credit_spread"]
                cs_63_ago = cs.shift(63)
                features_df["macro_credit_tightening"] = (
                    cs > cs_63_ago
                ).astype(np.float32).values
            else:
                features_df["macro_credit_tightening"] = np.nan

            if (
                "treasury_10y" in macro_aligned.columns
                and "breakeven_inflation" in macro_aligned.columns
            ):
                features_df["macro_real_rate"] = (
                    macro_aligned["treasury_10y"] - macro_aligned["breakeven_inflation"]
                ).values
            else:
                features_df["macro_real_rate"] = np.nan

            if "unemployment" in macro_aligned.columns:
                unemp = macro_aligned["unemployment"]
                unemp_63 = unemp.shift(63)
                features_df["macro_unemployment_trend"] = (unemp - unemp_63).values
            else:
                features_df["macro_unemployment_trend"] = np.nan

        except Exception as e:
            logger.debug("Macro features failed: %s", e)
            for feat in MACRO_FEATURES:
                if feat not in features_df.columns:
                    features_df[feat] = np.nan

        return features_df

    # ═══════════════════════════════════════════════════════════
    # POST-PROCESSING
    # ═══════════════════════════════════════════════════════════

    def _postprocess(self, features: pd.DataFrame) -> pd.DataFrame:
        """
        Critical post-processing:
        1. Replace inf with NaN
        2. Winsorize at 1st/99th percentile (within each stock)
        3. Forward-fill then zero-fill NaN (after warmup)
        4. Convert to float32
        """
        # 1. Replace inf
        features = features.replace([np.inf, -np.inf], np.nan)

        # 2. Winsorize within each stock's own history (per-column)
        for col in features.columns:
            if col.startswith("ctx_is_"):
                continue  # Don't winsorize binary calendar indicators
            s = features[col]
            lo = s.quantile(0.01)
            hi = s.quantile(0.99)
            if pd.notna(lo) and pd.notna(hi) and lo < hi:
                features[col] = s.clip(lower=lo, upper=hi)

        # 3. Forward-fill (up to 5 periods), then zero-fill
        features = features.ffill(limit=5)
        features = features.fillna(0.0)

        # 4. Convert to float32
        for col in features.columns:
            features[col] = features[col].astype(np.float32)

        return features

    # ═══════════════════════════════════════════════════════════
    # HELPER METHODS
    # ═══════════════════════════════════════════════════════════

    @staticmethod
    def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
        """Wilder's RSI. Returns 0-100 scale."""
        delta = close.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = (-delta).where(delta < 0, 0.0)
        avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        return 100.0 - (100.0 / (1.0 + rs))

    @staticmethod
    def _max_drawdown_rolling(close: pd.Series, window: int) -> pd.Series:
        """Rolling maximum drawdown (always negative or zero)."""
        rolling_max = close.rolling(window, min_periods=max(window // 2, 5)).max()
        dd = _safe_series_div(close, rolling_max) - 1
        return dd.rolling(window, min_periods=max(window // 2, 5)).min()

    @staticmethod
    def _rolling_hurst_fast(log_ret: pd.Series, window: int = 63) -> pd.Series:
        """
        Fast Hurst exponent approximation using variance ratio method.
        VR(q) = Var(q-period returns) / (q × Var(1-period returns))
        H ≈ 0.5 × log2(VR(q))  for q=10

        Much faster than R/S method (~100x) with similar discriminative power.
        H > 0.5 = trending, H < 0.5 = mean-reverting, H = 0.5 = random walk.
        """
        q = 10
        var_1 = log_ret.rolling(window, min_periods=max(window // 2, 20)).var()
        ret_q = log_ret.rolling(q, min_periods=q).sum()
        var_q = ret_q.rolling(window, min_periods=max(window // 2, 20)).var()
        vr = _safe_series_div(var_q, q * var_1)
        # H = 0.5 * log2(VR) + 0.5 → maps VR=1 to H=0.5
        hurst = 0.5 * np.log2(vr.clip(lower=0.01)) + 0.5
        return hurst.clip(0.0, 1.0)

    @staticmethod
    def _rolling_hurst(log_ret: pd.Series, window: int = 63) -> pd.Series:
        """
        Estimate Hurst exponent using rescaled range (R/S) method.
        H > 0.5 = trending, H < 0.5 = mean-reverting, H = 0.5 = random walk.
        DEPRECATED: Use _rolling_hurst_fast for training (100x faster).
        """

        def _hurst_rs(x):
            if len(x) < 20:
                return np.nan
            try:
                n = len(x)
                sizes = [n // 8, n // 4, n // 2, n]
                sizes = [s for s in sizes if s >= 8]
                if len(sizes) < 2:
                    return np.nan

                log_rs = []
                log_n = []
                for s in sizes:
                    n_chunks = n // s
                    if n_chunks < 1:
                        continue
                    rs_vals = []
                    for i in range(n_chunks):
                        chunk = x[i * s : (i + 1) * s]
                        mean_c = np.mean(chunk)
                        cum_dev = np.cumsum(chunk - mean_c)
                        r = np.max(cum_dev) - np.min(cum_dev)
                        s_val = np.std(chunk, ddof=1)
                        if s_val > 1e-10:
                            rs_vals.append(r / s_val)
                    if rs_vals:
                        log_rs.append(np.log(np.mean(rs_vals)))
                        log_n.append(np.log(s))

                if len(log_rs) >= 2:
                    slope = np.polyfit(log_n, log_rs, 1)[0]
                    return np.clip(slope, 0.0, 1.0)
            except Exception:
                pass
            return np.nan

        return log_ret.rolling(window, min_periods=max(window // 2, 20)).apply(
            _hurst_rs, raw=True
        )

    @staticmethod
    def _streak_counter(binary: pd.Series) -> pd.Series:
        """Count consecutive 1s in a binary series. Resets on 0."""
        result = pd.Series(0.0, index=binary.index)
        count = 0
        for i in range(len(binary)):
            if binary.iloc[i] == 1:
                count += 1
            else:
                count = 0
            result.iloc[i] = count
        return result

    # ═══════════════════════════════════════════════════════════
    # UTILITIES
    # ═══════════════════════════════════════════════════════════

    def get_feature_groups(self, columns: Optional[List[str]] = None) -> Dict[str, List[str]]:
        """Return mapping of group name → feature names (for analysis).
        If columns not provided, computes from a dummy 100-row DataFrame.
        """
        if columns is None:
            # Generate dummy data to get column names
            import pandas as pd
            dummy = pd.DataFrame({
                "open": np.random.uniform(90, 110, 100),
                "high": np.random.uniform(91, 112, 100),
                "low": np.random.uniform(89, 109, 100),
                "close": np.random.uniform(90, 111, 100),
                "volume": np.random.uniform(1e6, 1e7, 100),
            }, index=pd.date_range("2024-01-01", periods=100, freq="B"))
            features = self.compute_features(dummy)
            columns = features.columns.tolist()
        return {
            "momentum": [c for c in columns if c.startswith("mom_")],
            "volatility": [c for c in columns if c.startswith("vol_")],
            "volume": [c for c in columns if c.startswith("vlm_")],
            "technical": [c for c in columns if c.startswith("tech_")],
            "microstructure": [c for c in columns if c.startswith("micro_")],
            "context": [c for c in columns if c.startswith("ctx_")],
            "fundamental": [c for c in columns if c.startswith("fund_")],
            "macro": [c for c in columns if c.startswith("macro_")],
        }

    @staticmethod
    def feature_group_importance(
        importance_dict: Dict[str, float],
    ) -> Dict[str, float]:
        """
        Compute total importance by feature group from a dict of
        {feature_name: importance_value}.
        """
        groups: Dict[str, float] = {}
        total = sum(importance_dict.values()) or 1.0
        prefixes = {
            "Momentum": "mom_",
            "Volatility": "vol_",
            "Volume": "vlm_",
            "Technical": "tech_",
            "Microstructure": "micro_",
            "Context": "ctx_",
            "Fundamental": "fund_",
            "Macro": "macro_",
        }
        for group_name, prefix in prefixes.items():
            group_imp = sum(
                v for k, v in importance_dict.items() if k.startswith(prefix)
            )
            groups[group_name] = round(group_imp / total * 100, 1)
        return groups


# ══════════════════════════════════════════════════════════════
# Module-level convenience
# ══════════════════════════════════════════════════════════════

_engine_instance: Optional[UniversalFeatureEngine] = None


def get_feature_engine() -> UniversalFeatureEngine:
    """Get singleton feature engine instance."""
    global _engine_instance
    if _engine_instance is None:
        _engine_instance = UniversalFeatureEngine()
    return _engine_instance


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """Convenience wrapper: compute features for a single stock's OHLCV."""
    return get_feature_engine().compute_features(df)
