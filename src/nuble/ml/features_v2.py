"""
Institutional-Grade Feature Engineering Pipeline (v2)
======================================================

Replaces the naive feature pipeline with proper financial ML methodology
based on de Prado's *Advances in Financial Machine Learning*.

Key innovations over v1:
1. Fractional differentiation (Ch.5): achieve stationarity while preserving
   maximum price memory. Finds minimum d ∈ [0,1] via ADF test.
2. Cyclical calendar encoding: sin/cos transforms preserve adjacency
   (Sunday↔Monday, December↔January).
3. Cross-asset context: SPY, VIX, sector ETFs — individual stocks don't
   move in isolation.
4. Proper Wilder's smoothing for RSI and ATR (the old code used simple
   rolling mean — a well-known bug that distorts indicator values).
5. Hurst exponent: detects trending (H>0.5) vs mean-reverting (H<0.5) regimes.
6. Train/test isolation: features computed INSIDE CV folds. No lookahead bias.

Dependencies:
    pip install statsmodels scipy numpy pandas

Author: NUBLE ML Pipeline
Version: 2.0.0
"""

# ──────────────────────────────────────────────────────────────
# Imports
# ──────────────────────────────────────────────────────────────
from __future__ import annotations

import logging
import warnings
from typing import Any, Callable

import numpy as np
import pandas as pd
from scipy import stats as sp_stats
from statsmodels.tsa.stattools import adfuller

logger = logging.getLogger(__name__)

# Suppress ADF test warnings for small samples
warnings.filterwarnings("ignore", category=RuntimeWarning, module="statsmodels")


# ══════════════════════════════════════════════════════════════
# 1. Fractional Differentiator  (de Prado Ch.5)
# ══════════════════════════════════════════════════════════════


class FractionalDifferentiator:
    """
    Implements de Prado Chapter 5: Fractional Differentiation.

    Price levels are non-stationary (unit root) → spurious regressions.
    First differences (d=1) are stationary but destroy ALL memory.
    Fractional differentiation finds minimum *d* (typically 0.3–0.5) that
    achieves stationarity while preserving maximum memory.

    The fractional differencing operator applies weights:
        w_k = -w_{k-1} * (d - k + 1) / k

    We use the **Fixed-Width Window** method (de Prado §5.4) which
    truncates weights below a threshold for computational efficiency.

    No external ``fracdiff`` library required — implemented from first
    principles for full transparency and compatibility.
    """

    def __init__(self, threshold: float = 1e-5):
        """
        Args:
            threshold: Minimum absolute weight to include in the
                       differencing filter.  Weights below this are
                       truncated (Fixed-Width Window approach).
        """
        self.threshold = threshold
        self.optimal_d: dict[str, float] = {}

    # ── Core math ──────────────────────────────────────────────

    @staticmethod
    def _get_weights(d: float, threshold: float, max_len: int) -> np.ndarray:
        """
        Compute fractional differencing weights (de Prado Snippet 5.1).

        w_0 = 1
        w_k = -w_{k-1} * (d - k + 1) / k   for k >= 1

        Truncate when |w_k| < threshold or k == max_len.

        Returns:
            1-D array of weights, length ≤ max_len.
        """
        weights = [1.0]
        k = 1
        while k < max_len:
            w_next = -weights[-1] * (d - k + 1) / k
            if abs(w_next) < threshold:
                break
            weights.append(w_next)
            k += 1
        return np.array(weights[::-1])   # oldest weight first

    def _fracdiff(self, series: np.ndarray, d: float) -> np.ndarray:
        """
        Apply fractional differencing to a 1-D array.

        Uses the Fixed-Width Window method: for each time step t,
        the differenced value is the dot product of the weight vector
        and the price window ending at t.

        Returns:
            1-D array of length = len(series) - len(weights) + 1
            (the "valid" portion, analogous to np.convolve mode='valid').
        """
        weights = self._get_weights(d, self.threshold, len(series))
        width = len(weights)
        result = np.empty(len(series) - width + 1)
        for t in range(width - 1, len(series)):
            result[t - width + 1] = np.dot(weights, series[t - width + 1: t + 1])
        return result

    # ── Public API ─────────────────────────────────────────────

    def find_optimal_d(
        self,
        series: pd.Series,
        max_d: float = 1.0,
        steps: int = 20,
        significance: float = 0.05,
    ) -> float:
        """
        Find minimum *d* that makes *series* stationary via ADF test.

        Searches d ∈ [0, max_d] in ``steps`` increments.  For each d,
        applies fractional differencing and runs the Augmented
        Dickey–Fuller test.  Returns the smallest d where
        p-value < ``significance``.

        Caches the result in ``self.optimal_d`` keyed by ``series.name``.

        Args:
            series:       Price series (must have a ``.name`` attribute).
            max_d:        Upper bound of the search range.
            steps:        Number of d values to evaluate.
            significance: p-value threshold for stationarity.

        Returns:
            Optimal d ∈ [0, max_d].  Returns 1.0 if no d achieves
            stationarity (full differencing fallback).
        """
        arr = series.dropna().values.astype(np.float64)
        if len(arr) < 30:
            logger.warning(
                "Series '%s' too short (%d) for ADF — defaulting d=1.0",
                series.name, len(arr),
            )
            return 1.0

        d_values = np.linspace(0, max_d, steps + 1)[1:]  # skip d=0

        for d in d_values:
            diffed = self._fracdiff(arr, d)
            if len(diffed) < 20:
                continue  # need enough observations for ADF
            try:
                adf_stat, p_value, *_ = adfuller(diffed, maxlag=1, autolag=None)
                if p_value < significance:
                    key = str(series.name) if series.name is not None else "unnamed"
                    self.optimal_d[key] = float(d)
                    logger.info(
                        "Optimal d for '%s': %.3f (ADF p=%.4f)",
                        key, d, p_value,
                    )
                    return float(d)
            except Exception:
                continue

        # Fallback: full differencing
        key = str(series.name) if series.name is not None else "unnamed"
        self.optimal_d[key] = 1.0
        logger.warning(
            "No d < %.1f achieved stationarity for '%s' — using d=1.0",
            max_d, key,
        )
        return 1.0

    def transform(self, series: pd.Series, d: float) -> pd.Series:
        """
        Apply fractional differencing with given *d*.

        Returns:
            pd.Series aligned to the original index (trimmed for valid
            window — the first ``len(weights)-1`` observations are lost).
        """
        arr = series.dropna().values.astype(np.float64)
        diffed = self._fracdiff(arr, d)
        # Align to the END of the original index (drop leading NaNs)
        valid_idx = series.dropna().index[len(arr) - len(diffed):]
        return pd.Series(diffed, index=valid_idx, name=f"{series.name}_fracdiff")

    def fit_transform(
        self,
        df: pd.DataFrame,
        price_cols: list[str] | None = None,
    ) -> pd.DataFrame:
        """
        For each price column: find optimal d, apply fractional
        differencing, add as ``{col}_fracdiff``.

        Args:
            df:         DataFrame with price columns.
            price_cols: Columns to differentiate.  Defaults to
                        ``["close"]`` if not provided.

        Returns:
            DataFrame with new ``_fracdiff`` columns appended (NaN-padded
            at the top where the filter window hasn't filled).
        """
        if price_cols is None:
            price_cols = ["close"]
        result = df.copy()
        for col in price_cols:
            if col not in df.columns:
                continue
            d = self.find_optimal_d(df[col])
            transformed = self.transform(df[col], d)
            result = result.join(transformed, how="left")
        return result


# ══════════════════════════════════════════════════════════════
# 2. Cyclical Calendar Encoder
# ══════════════════════════════════════════════════════════════


class CyclicalEncoder:
    """
    Encode cyclical features using sin/cos transformation.

    Naive integer encoding (day_of_week = 0,1,…,4) implies that Friday (4)
    is "far" from Monday (0), when they're adjacent in trading time.
    Sin/cos encoding maps each value onto the unit circle so that
    adjacent periods are adjacent in feature space.

    For a value *v* in period *P*:
        sin_feature = sin(2π · v / P)
        cos_feature = cos(2π · v / P)

    The pair (sin, cos) satisfies sin² + cos² = 1 by construction.
    """

    @staticmethod
    def encode(df: pd.DataFrame) -> pd.DataFrame:
        """
        Add cyclical calendar features.

        Requires a ``DatetimeIndex`` on *df*.

        Features created:
        ─────────────────────────────────────────────
        day_of_week_sin/cos    period = 5  (trading days)
        day_of_month_sin/cos   period = 21 (~trading days / month)
        month_sin/cos          period = 12
        quarter_sin/cos        period = 4
        ─────────────────────────────────────────────

        Returns:
            DataFrame with 8 new columns appended.
        """
        result = df.copy()
        idx = pd.DatetimeIndex(result.index)

        def _sincos(values: np.ndarray, period: float, prefix: str) -> None:
            angle = 2 * np.pi * values / period
            result[f"{prefix}_sin"] = np.sin(angle)
            result[f"{prefix}_cos"] = np.cos(angle)

        _sincos(idx.dayofweek.values.astype(float), 5.0, "day_of_week")
        _sincos(idx.day.values.astype(float), 21.0, "day_of_month")
        _sincos(idx.month.values.astype(float), 12.0, "month")
        _sincos(idx.quarter.values.astype(float), 4.0, "quarter")

        return result


# ══════════════════════════════════════════════════════════════
# 3. Cross-Asset Context Features
# ══════════════════════════════════════════════════════════════


class CrossAssetFeatures:
    """
    Individual stocks don't move in isolation.  This class computes
    cross-asset context features that capture market regime and
    relative strength.

    Reference assets:
        SPY — broad equity market
        ^VIX — implied volatility / fear gauge

    Sector ETFs (optional):
        XLK (Tech), XLF (Financials), XLE (Energy), XLV (Healthcare),
        XLI (Industrials), XLP (Consumer Staples), XLY (Consumer Disc),
        XLU (Utilities), XLRE (Real Estate)
    """

    REFERENCE_SYMBOLS: list[str] = ["SPY", "^VIX"]

    SECTOR_ETFS: dict[str, str] = {
        "XLK": "Technology",
        "XLF": "Financials",
        "XLE": "Energy",
        "XLV": "Healthcare",
        "XLI": "Industrials",
        "XLP": "Consumer Staples",
        "XLY": "Consumer Disc",
        "XLU": "Utilities",
        "XLRE": "Real Estate",
    }

    # Map common tickers to their GICS sector ETF
    TICKER_SECTOR: dict[str, str] = {
        "AAPL": "XLK", "MSFT": "XLK", "NVDA": "XLK", "GOOGL": "XLK",
        "AMZN": "XLY", "TSLA": "XLY", "META": "XLK", "AMD": "XLK",
        "JPM": "XLF", "BAC": "XLF", "GS": "XLF",
        "XOM": "XLE", "CVX": "XLE",
        "UNH": "XLV", "JNJ": "XLV", "PFE": "XLV",
        "GE": "XLI", "CAT": "XLI",
    }

    def __init__(self, data_fetcher: Callable | None = None):
        """
        Args:
            data_fetcher: Async or sync callable
                ``(symbol, start_date, end_date) → pd.DataFrame``
                with columns ``['open','high','low','close','volume']``
                and a ``DatetimeIndex``.  If ``None``, cross-asset
                features are silently skipped.
        """
        self.data_fetcher = data_fetcher
        self._cache: dict[str, pd.DataFrame] = {}

    async def fetch_reference_data(
        self,
        start_date: str,
        end_date: str,
    ) -> None:
        """Pre-fetch all reference asset data into the cache."""
        if self.data_fetcher is None:
            return
        import asyncio

        symbols = list(self.REFERENCE_SYMBOLS) + list(self.SECTOR_ETFS.keys())
        for sym in symbols:
            if sym in self._cache:
                continue
            try:
                result = self.data_fetcher(sym, start_date, end_date)
                if asyncio.iscoroutine(result):
                    result = await result
                if result is not None and len(result) > 0:
                    self._cache[sym] = result
            except Exception as exc:
                logger.warning("CrossAsset: failed to fetch %s — %s", sym, exc)

    def compute(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """
        Add cross-asset features to *df*.

        All features use **only past data** (backward-looking rolling
        windows with ``min_periods=window``).

        Features created:
        ─────────────────────────────────────────────
        spy_return_1d / 5d / 21d      SPY momentum
        vix_level                     current VIX close
        vix_change_1d                 1-day VIX change
        vix_percentile_21d            VIX rank in 21-day window
        relative_strength_vs_spy_5d   symbol − SPY 5-day return
        relative_strength_vs_spy_21d  symbol − SPY 21-day return
        sector_etf_return_5d          matching sector ETF momentum
        spy_symbol_corr_21d           rolling 21-day correlation
        market_breadth_proxy          sign(SPY 5d return)
        ─────────────────────────────────────────────

        Returns:
            DataFrame with new columns.
        """
        result = df.copy()

        # ── SPY features ─────────────────────────────────────
        spy = self._cache.get("SPY")
        if spy is not None and "close" in spy.columns:
            spy_close = spy["close"].reindex(result.index, method="ffill")
            for n in (1, 5, 21):
                result[f"spy_return_{n}d"] = spy_close.pct_change(n)

            # Symbol-vs-SPY relative strength
            sym_close = result["close"] if "close" in result.columns else None
            if sym_close is not None:
                for n in (5, 21):
                    sym_ret = sym_close.pct_change(n)
                    spy_ret = spy_close.pct_change(n)
                    result[f"relative_strength_vs_spy_{n}d"] = sym_ret - spy_ret

                # Rolling 21-day correlation
                sym_lr = np.log(sym_close / sym_close.shift(1))
                spy_lr = np.log(spy_close / spy_close.shift(1))
                result["spy_symbol_corr_21d"] = (
                    sym_lr.rolling(21, min_periods=21).corr(spy_lr)
                )

            # Market breadth proxy
            result["market_breadth_proxy"] = np.sign(
                spy_close.pct_change(5)
            )

        # ── VIX features ─────────────────────────────────────
        vix = self._cache.get("^VIX")
        if vix is not None and "close" in vix.columns:
            vix_close = vix["close"].reindex(result.index, method="ffill")
            result["vix_level"] = vix_close
            result["vix_change_1d"] = vix_close.pct_change(1)
            result["vix_percentile_21d"] = vix_close.rolling(
                21, min_periods=21
            ).apply(lambda x: sp_stats.percentileofscore(x, x.iloc[-1]) / 100.0)

        # ── Sector ETF feature ────────────────────────────────
        sector_etf = self.TICKER_SECTOR.get(symbol.upper())
        if sector_etf and sector_etf in self._cache:
            sec_df = self._cache[sector_etf]
            if "close" in sec_df.columns:
                sec_close = sec_df["close"].reindex(result.index, method="ffill")
                result["sector_etf_return_5d"] = sec_close.pct_change(5)

        return result


# ══════════════════════════════════════════════════════════════
# 4. Technical Features  (Advanced)
# ══════════════════════════════════════════════════════════════


class TechnicalFeatures:
    """
    Advanced technical indicators beyond basic RSI / MACD.

    Every indicator uses the **correct** smoothing method:
    - RSI: Wilder's exponential smoothing (α = 1/N), NOT simple rolling mean
    - ATR: Wilder's exponential smoothing, NOT simple rolling mean
    - ADX: Wilder's smoothing on ±DI and DX

    Includes regime-detection features critical for financial ML:
    - Hurst exponent (trending vs mean-reverting detection)
    - Autocorrelation at lag 1 and lag 5
    - ADX (trend strength)
    - Volatility-of-volatility
    """

    # ── Public API ─────────────────────────────────────────────

    @staticmethod
    def compute(df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute a complete set of advanced technical features.

        Input *df* must contain columns:
            ``open``, ``high``, ``low``, ``close``, ``volume``
        with a ``DatetimeIndex``.

        Feature groups (≈ 30 features total):

        **Volatility** — ATR, realized vol, vol-of-vol, Garman-Klass
        **Momentum**   — RSI, ROC, log returns
        **Mean Rev.**  — z-score, Bollinger %B, distance from 52-wk hi/lo
        **Micro.**     — volume SMA ratio, dollar volume, range %
        **Regime**     — Hurst exponent, autocorrelation, ADX

        All rolling computations use ``min_periods=window`` to avoid
        partial-window artefacts.

        Returns:
            DataFrame with all new columns appended.
        """
        result = df.copy()
        close = result["close"].astype(np.float64)
        high = result["high"].astype(np.float64)
        low = result["low"].astype(np.float64)
        opn = result["open"].astype(np.float64)
        volume = result["volume"].astype(np.float64)

        # Pre-compute log returns (used by multiple features)
        log_ret_1d = np.log(close / close.shift(1))

        # ── Volatility ────────────────────────────────────────
        result["atr_14"] = TechnicalFeatures._wilder_atr(high, low, close, 14)

        result["realized_vol_21"] = (
            log_ret_1d.rolling(21, min_periods=21).std() * np.sqrt(252)
        )

        rv21 = result["realized_vol_21"]
        result["vol_of_vol_63"] = rv21.rolling(63, min_periods=63).std()

        # Garman-Klass volatility estimator (uses OHLC — more efficient
        # than close-to-close):
        #   0.5 · ln(H/L)² − (2ln2 − 1) · ln(C/O)²
        log_hl = np.log(high / low)
        log_co = np.log(close / opn)
        gk_daily = 0.5 * log_hl ** 2 - (2 * np.log(2) - 1) * log_co ** 2
        gk_mean = gk_daily.rolling(21, min_periods=21).mean()
        # Garman-Klass daily variance can be negative for extreme cases;
        # clamp at 0 before taking sqrt to avoid NaN propagation.
        result["garman_klass_vol"] = np.sqrt(gk_mean.clip(lower=0)) * np.sqrt(252)

        # ── Momentum ──────────────────────────────────────────
        result["rsi_14"] = TechnicalFeatures._wilder_rsi(close, 14)

        for n in (5, 21):
            result[f"roc_{n}"] = close / close.shift(n) - 1

        for n in (1, 5, 21):
            result[f"log_return_{n}d"] = np.log(close / close.shift(n))

        # ── Mean Reversion ────────────────────────────────────
        sma_20 = close.rolling(20, min_periods=20).mean()
        std_20 = close.rolling(20, min_periods=20).std()
        result["z_score_20"] = (close - sma_20) / std_20

        upper_bb = sma_20 + 2 * std_20
        lower_bb = sma_20 - 2 * std_20
        result["bollinger_pct"] = (close - lower_bb) / (upper_bb - lower_bb)

        high_52w = high.rolling(252, min_periods=252).max()
        low_52w = low.rolling(252, min_periods=252).min()
        result["distance_from_52w_high"] = (close - high_52w) / high_52w
        result["distance_from_52w_low"] = (close - low_52w) / low_52w

        # ── Microstructure ────────────────────────────────────
        vol_sma_21 = volume.rolling(21, min_periods=21).mean()
        result["volume_sma_ratio"] = volume / vol_sma_21

        dollar_vol = close * volume
        result["dollar_volume_21d_avg"] = dollar_vol.rolling(
            21, min_periods=21
        ).mean()

        result["high_low_range_pct"] = (high - low) / close

        # ── Regime Detection ──────────────────────────────────
        result["hurst_exponent_63"] = TechnicalFeatures.compute_hurst(
            log_ret_1d, window=63
        )

        result["autocorrelation_1d"] = log_ret_1d.rolling(
            63, min_periods=63
        ).apply(lambda x: x.autocorr(lag=1), raw=False)

        result["autocorrelation_5d"] = log_ret_1d.rolling(
            63, min_periods=63
        ).apply(lambda x: x.autocorr(lag=5), raw=False)

        result["trend_strength_adx_14"] = TechnicalFeatures._adx(
            high, low, close, period=14
        )

        return result

    # ── Static helpers ────────────────────────────────────────

    @staticmethod
    def _wilder_smooth(series: pd.Series, period: int) -> pd.Series:
        """
        Wilder's exponential smoothing (α = 1/period).

        This is NOT the same as a standard EMA (α = 2/(period+1)).
        Wilder's uses α = 1/N which gives a slower, smoother response.
        Used for RSI, ATR, and ADX.

        Initialization: simple average of first ``period`` observations.
        """
        result = pd.Series(np.nan, index=series.index, dtype=np.float64)
        # Seed with SMA of first `period` valid values
        valid = series.dropna()
        if len(valid) < period:
            return result
        seed_idx = valid.index[period - 1]
        result.loc[seed_idx] = valid.iloc[:period].mean()
        alpha = 1.0 / period
        prev = result.loc[seed_idx]
        for idx in valid.index[valid.index.get_loc(seed_idx) + 1:]:
            val = series.loc[idx]
            if np.isnan(val):
                continue
            prev = prev * (1 - alpha) + val * alpha
            result.loc[idx] = prev
        return result

    @staticmethod
    def _wilder_rsi(close: pd.Series, period: int = 14) -> pd.Series:
        """
        RSI with **Wilder's smoothing** — the correct implementation.

        Common bug: using ``rolling(period).mean()`` for average gain/loss.
        Wilder's original formula uses exponential smoothing with α=1/N:
            AvgGain_t = AvgGain_{t-1} · (N-1)/N + Gain_t · 1/N
            AvgLoss_t = AvgLoss_{t-1} · (N-1)/N + Loss_t · 1/N

        Returns:
            RSI in [0, 100] range.
        """
        delta = close.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = (-delta).where(delta < 0, 0.0)

        avg_gain = TechnicalFeatures._wilder_smooth(gain, period)
        avg_loss = TechnicalFeatures._wilder_smooth(loss, period)

        rs = avg_gain / avg_loss.replace(0, np.nan)
        rsi = 100.0 - (100.0 / (1.0 + rs))
        return rsi

    @staticmethod
    def _wilder_atr(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        period: int = 14,
    ) -> pd.Series:
        """
        Average True Range with **Wilder's smoothing**.

        TR = max(H−L, |H−C_prev|, |L−C_prev|)
        ATR = Wilder smooth of TR with α = 1/period.
        """
        prev_close = close.shift(1)
        tr = pd.concat(
            [
                high - low,
                (high - prev_close).abs(),
                (low - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        return TechnicalFeatures._wilder_smooth(tr, period)

    @staticmethod
    def _adx(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        period: int = 14,
    ) -> pd.Series:
        """
        Average Directional Index (ADX).

        Measures trend strength (0–100):
            +DM = H_t − H_{t-1}  if > 0 and > (L_{t-1} − L_t), else 0
            −DM = L_{t-1} − L_t  if > 0 and > (H_t − H_{t-1}), else 0
            +DI = 100 · Wilder(+DM, N) / ATR_N
            −DI = 100 · Wilder(−DM, N) / ATR_N
            DX  = 100 · |+DI − −DI| / (+DI + −DI)
            ADX = Wilder(DX, N)
        """
        up_move = high.diff()
        down_move = -low.diff()

        plus_dm = pd.Series(0.0, index=high.index)
        minus_dm = pd.Series(0.0, index=high.index)

        cond_plus = (up_move > down_move) & (up_move > 0)
        cond_minus = (down_move > up_move) & (down_move > 0)
        plus_dm[cond_plus] = up_move[cond_plus]
        minus_dm[cond_minus] = down_move[cond_minus]

        atr = TechnicalFeatures._wilder_atr(high, low, close, period)
        smooth_plus = TechnicalFeatures._wilder_smooth(plus_dm, period)
        smooth_minus = TechnicalFeatures._wilder_smooth(minus_dm, period)

        plus_di = 100.0 * smooth_plus / atr.replace(0, np.nan)
        minus_di = 100.0 * smooth_minus / atr.replace(0, np.nan)

        dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
        adx = TechnicalFeatures._wilder_smooth(dx, period)
        return adx

    @staticmethod
    def compute_hurst(
        series: pd.Series,
        window: int = 63,
    ) -> pd.Series:
        """
        Rolling Hurst exponent via Rescaled Range (R/S) analysis.

        For each window of returns:
        1. Divide into sub-periods of sizes [n//8, n//4, n//2, n].
        2. For each sub-period size, compute R/S across all non-overlapping
           sub-periods and average.
        3. Regress log(R/S) on log(sub-period size).
        4. Slope = Hurst exponent H.

        Interpretation:
            H > 0.5 → trending / persistent
            H ≈ 0.5 → random walk
            H < 0.5 → mean-reverting / anti-persistent

        Returns:
            pd.Series of Hurst values, NaN where window is insufficient.
        """

        def _rs_for_subperiod(arr: np.ndarray) -> float:
            """R/S statistic for a single sub-period."""
            n = len(arr)
            if n < 4:
                return np.nan
            mean = np.mean(arr)
            dev = arr - mean
            cum_dev = np.cumsum(dev)
            r = np.max(cum_dev) - np.min(cum_dev)
            s = np.std(arr, ddof=1)
            if s < 1e-12:
                return np.nan
            return r / s

        def _hurst_single(window_data: np.ndarray) -> float:
            """Hurst exponent for one window of returns."""
            n = len(window_data)
            # Sub-period sizes: powers of 2 that fit within the window
            sizes = []
            s = max(4, n // 8)
            while s <= n:
                sizes.append(s)
                s *= 2
            if len(sizes) < 2:
                return np.nan

            log_n = []
            log_rs = []
            for size in sizes:
                n_blocks = n // size
                if n_blocks < 1:
                    continue
                rs_values = []
                for b in range(n_blocks):
                    block = window_data[b * size: (b + 1) * size]
                    rs = _rs_for_subperiod(block)
                    if not np.isnan(rs) and rs > 0:
                        rs_values.append(rs)
                if rs_values:
                    log_n.append(np.log(size))
                    log_rs.append(np.log(np.mean(rs_values)))

            if len(log_n) < 2:
                return np.nan

            # Linear regression: log(R/S) = H · log(n) + c
            slope, _, _, _, _ = sp_stats.linregress(log_n, log_rs)
            return float(np.clip(slope, 0.0, 1.0))

        return series.rolling(window, min_periods=window).apply(
            _hurst_single, raw=True
        )


# ══════════════════════════════════════════════════════════════
# 5. Feature Pipeline  (Main Orchestrator)
# ══════════════════════════════════════════════════════════════


class FeaturePipeline:
    """
    **CRITICAL**: This class enforces train/test isolation.

    The old pipeline computes features on the FULL dataset before
    splitting into train/test — this leaks future information
    (e.g., z-scores computed using future data, optimal-d found on
    test-period prices, percentile ranks that include test values).

    This pipeline computes features **within each fold**:

    1. ``fit()`` — learn statistics on TRAINING data only:
       - Optimal d for fractional differentiation
       - Feature means / stds for standardization

    2. ``transform()`` — apply to train AND test using training-fitted
       statistics.  No future data ever touches computation.

    3. ``fit_transform()`` — convenience wrapper.

    Usage in a CV loop::

        for train_idx, test_idx in cv.split(X):
            train_features, pipe = build_features(
                df.iloc[train_idx], symbol, is_training=True
            )
            test_features, _ = build_features(
                df.iloc[test_idx], symbol,
                is_training=False, pipeline=pipe
            )

    This is the **single most important architectural fix**.
    """

    def __init__(self, data_fetcher: Callable | None = None):
        """
        Args:
            data_fetcher: Async callable for cross-asset data.
                          ``(symbol, start, end) → pd.DataFrame``.
                          Pass ``None`` to skip cross-asset features.
        """
        self.fractional_diff = FractionalDifferentiator()
        self.cyclical_encoder = CyclicalEncoder()
        self.cross_asset = CrossAssetFeatures(data_fetcher) if data_fetcher else None

        # Stores training-fitted parameters.  Populated by ``fit()``.
        self._fitted_params: dict[str, Any] = {}
        self._feature_names: list[str] = []
        self._is_fitted: bool = False

    # ── Fit ────────────────────────────────────────────────────

    def fit(self, df: pd.DataFrame, symbol: str) -> FeaturePipeline:
        """
        Fit on **training data only**.

        Learns and caches:
        - Optimal d for each price column (fractional differencing)
        - Feature means / stds for z-score standardization

        Must be called **once per CV fold** with the training portion.
        """
        self._fitted_params.clear()

        # 1. Optimal d for fractional differencing
        for col in ("close", "volume"):
            if col in df.columns:
                d = self.fractional_diff.find_optimal_d(df[col])
                self._fitted_params[f"{col}_optimal_d"] = d

        # 2. Compute all raw features on training data, then clean
        #    (drop OHLCV, replace inf, drop NaN rows) so that
        #    means/stds are computed on the SAME data transform() produces.
        raw = self._compute_raw_features(df, symbol)
        drop_cols = {"open", "high", "low", "close", "volume"} & set(raw.columns)
        raw = raw.drop(columns=list(drop_cols), errors="ignore")
        raw = raw.replace([np.inf, -np.inf], np.nan).dropna()

        # 2b. Remove collinear features and lock the surviving columns
        raw = self.remove_collinear(raw, threshold=0.95)
        self._fitted_params["fitted_columns"] = raw.columns.tolist()

        numeric_cols = raw.select_dtypes(include=[np.number]).columns.tolist()
        self._fitted_params["feature_means"] = raw[numeric_cols].mean().to_dict()
        self._fitted_params["feature_stds"] = raw[numeric_cols].std().replace(0, 1).to_dict()

        self._is_fitted = True
        return self

    # ── Transform ──────────────────────────────────────────────

    def transform(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """
        Transform data using **only fit-time parameters**.

        Can be called on both train and test data without leakage.

        Pipeline order:
        1. Fractional differentiation (fitted optimal d)
        2. Technical features (pure rolling — backward-looking only)
        3. Cross-asset features (if available)
        4. Cyclical calendar encoding
        5. Drop raw price columns (keep only derived features)
        6. Handle infinities → NaN → drop leading NaN rows
        7. Remove collinear features (|ρ| > 0.95)
        8. Standardize using training-fitted mean / std

        Returns:
            Clean feature DataFrame, NaN-free.
        """
        raw = self._compute_raw_features(df, symbol)

        # Drop raw OHLCV columns — keep only derived features
        drop_cols = {"open", "high", "low", "close", "volume"} & set(raw.columns)
        raw = raw.drop(columns=list(drop_cols), errors="ignore")

        # Replace inf with NaN, then drop leading NaN rows
        raw = raw.replace([np.inf, -np.inf], np.nan)
        raw = raw.dropna()

        if raw.empty:
            logger.warning("All rows dropped after NaN removal for %s", symbol)
            return raw

        # Enforce training-time feature set (if fitted) — guarantees same columns
        fitted_cols = self._fitted_params.get("fitted_columns")
        if self._is_fitted and fitted_cols:
            # Keep only columns that existed at fit time; add missing ones as 0
            missing = [c for c in fitted_cols if c not in raw.columns]
            for c in missing:
                raw[c] = 0.0
            raw = raw[fitted_cols]
        else:
            # No fit record — fall back to collinearity removal
            raw = self.remove_collinear(raw, threshold=0.95)

        # Standardize using training stats (if fitted)
        if self._is_fitted:
            means = self._fitted_params.get("feature_means", {})
            stds = self._fitted_params.get("feature_stds", {})
            for col in raw.columns:
                if col in means and col in stds:
                    raw[col] = (raw[col] - means[col]) / stds[col]

        self._feature_names = raw.columns.tolist()
        return raw

    # ── Convenience ────────────────────────────────────────────

    def fit_transform(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Fit on *df* then transform it.  Convenience wrapper."""
        return self.fit(df, symbol).transform(df, symbol)

    # ── Feature introspection ──────────────────────────────────

    def get_feature_names(self) -> list[str]:
        """Return feature names after the most recent ``transform()``."""
        return list(self._feature_names)

    # ── Internal ───────────────────────────────────────────────

    def _compute_raw_features(
        self,
        df: pd.DataFrame,
        symbol: str,
    ) -> pd.DataFrame:
        """
        Compute all raw (un-standardized) features.

        Ordering:
        1. Fractional differentiation
        2. Technical features
        3. Cross-asset features
        4. Cyclical calendar encoding
        """
        result = df.copy()

        # 1. Fractional differentiation
        for col in ("close", "volume"):
            if col not in result.columns:
                continue
            d = self._fitted_params.get(
                f"{col}_optimal_d",
                self.fractional_diff.optimal_d.get(col, 0.5),
            )
            transformed = self.fractional_diff.transform(result[col], d)
            result = result.join(transformed, how="left")

        # 2. Technical features
        if all(c in result.columns for c in ("open", "high", "low", "close", "volume")):
            result = TechnicalFeatures.compute(result)

        # 3. Cross-asset features
        if self.cross_asset is not None:
            result = self.cross_asset.compute(result, symbol)

        # 4. Cyclical calendar encoding
        if isinstance(result.index, pd.DatetimeIndex):
            result = CyclicalEncoder.encode(result)

        return result

    # ── Collinearity removal ──────────────────────────────────

    @staticmethod
    def remove_collinear(
        df: pd.DataFrame,
        threshold: float = 0.95,
    ) -> pd.DataFrame:
        """
        Remove features with pairwise |Pearson ρ| > ``threshold``.

        For each correlated pair, the feature with **lower variance**
        (less informative) is dropped.

        Returns:
            DataFrame with redundant features removed.
        """
        if df.shape[1] < 2:
            return df

        corr = df.corr().abs()
        upper = corr.where(np.triu(np.ones(corr.shape, dtype=bool), k=1))

        to_drop: set[str] = set()
        for col in upper.columns:
            correlated = upper.index[upper[col] > threshold].tolist()
            for other in correlated:
                if other in to_drop:
                    continue
                # Drop the one with lower variance
                if df[col].var() >= df[other].var():
                    to_drop.add(other)
                else:
                    to_drop.add(col)

        if to_drop:
            logger.info(
                "Removed %d collinear features (|ρ|>%.2f): %s",
                len(to_drop), threshold, sorted(to_drop),
            )
        return df.drop(columns=list(to_drop))


# ══════════════════════════════════════════════════════════════
# 6. Module-Level Convenience Function
# ══════════════════════════════════════════════════════════════


def build_features(
    df: pd.DataFrame,
    symbol: str,
    is_training: bool = True,
    pipeline: FeaturePipeline | None = None,
    data_fetcher: Callable | None = None,
) -> tuple[pd.DataFrame, FeaturePipeline]:
    """
    High-level API for feature construction with train/test isolation.

    Handles the fit/transform lifecycle automatically:

    - ``is_training=True``, ``pipeline=None`` → creates a **new** pipeline,
      fits on *df*, transforms *df*.
    - ``is_training=False``, ``pipeline`` provided → transforms *df* using
      the already-fitted pipeline (no data leakage).
    - ``is_training=True``, ``pipeline`` provided → **re-fits** the pipeline
      on the new training fold, then transforms.

    Usage in a CV loop::

        for train_idx, test_idx in cv.split(X):
            train_feats, pipe = build_features(
                df.iloc[train_idx], symbol, is_training=True
            )
            test_feats, _ = build_features(
                df.iloc[test_idx], symbol,
                is_training=False, pipeline=pipe
            )

    Args:
        df:           OHLCV DataFrame with DatetimeIndex.
        symbol:       Ticker symbol (e.g. "TSLA").
        is_training:  Whether this is a training fold.
        pipeline:     Pre-existing pipeline to reuse / refit.
        data_fetcher: Async callable for cross-asset data.

    Returns:
        ``(feature_df, fitted_pipeline)``
    """
    if pipeline is None:
        pipeline = FeaturePipeline(data_fetcher=data_fetcher)

    if is_training:
        features = pipeline.fit_transform(df, symbol)
    else:
        if not pipeline._is_fitted:
            raise ValueError(
                "Pipeline must be fitted before transforming test data. "
                "Call build_features(…, is_training=True) first."
            )
        features = pipeline.transform(df, symbol)

    return features, pipeline
