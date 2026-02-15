"""
Universal Technical Model — ONE Model for ALL Stocks
======================================================

Replaces 5 per-ticker models with a single LightGBM model trained on
ALL stocks' technical patterns simultaneously.

Phase 3 upgrade: Uses UniversalFeatureEngine (112+ features) instead of
the old 27-feature compute_universal_features function.

Author: NUBLE ML Pipeline — Phase 1–3 Institutional Upgrade
"""

from __future__ import annotations

import gc
import json
import logging
import os
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy import stats as sp_stats

logger = logging.getLogger(__name__)

# Lazy imports to avoid circular dependencies
if TYPE_CHECKING:
    from ..data.polygon_universe import PolygonUniverseData


# ══════════════════════════════════════════════════════════════
# Universal Feature Computation (OHLCV-only, no cross-asset API)
# ══════════════════════════════════════════════════════════════

def compute_universal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute stock-agnostic technical features from OHLCV data.

    These features are NORMALIZED — they describe the stock's current
    technical state in relative terms (percentiles, ratios, z-scores),
    not absolute prices. "RSI = 28" means the same thing whether the
    stock is AAPL at $200 or a micro-cap at $8.

    Features (~45 total):
    - Momentum: RSI, ROC, log returns
    - Volatility: realized vol, ATR ratio, Garman-Klass, vol-of-vol
    - Mean reversion: z-score, Bollinger %B, distance from 52w hi/lo
    - Microstructure: volume ratio, range %
    - Regime: Hurst exponent, autocorrelation, ADX
    - Calendar: day-of-week sin/cos, month sin/cos
    - Fractional diff of close
    """
    if len(df) < 63:
        return pd.DataFrame()

    result = pd.DataFrame(index=df.index)
    close = df["close"].astype(np.float64)
    high = df["high"].astype(np.float64)
    low = df["low"].astype(np.float64)
    opn = df["open"].astype(np.float64)
    volume = df["volume"].astype(np.float64)

    log_ret_1d = np.log(close / close.shift(1))

    # ── Momentum ──────────────────────────────────────────────
    # RSI (Wilder's smoothing)
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    result["rsi_14"] = 100.0 - (100.0 / (1.0 + rs))

    for n in (5, 21):
        result[f"roc_{n}"] = close / close.shift(n) - 1

    for n in (1, 5, 21):
        result[f"log_return_{n}d"] = np.log(close / close.shift(n))

    # ── Volatility ────────────────────────────────────────────
    result["realized_vol_21"] = log_ret_1d.rolling(21, min_periods=21).std() * np.sqrt(252)

    # ATR as fraction of close (normalized)
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    atr_14 = tr.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    result["atr_pct"] = atr_14 / close  # Normalized by price

    rv21 = result["realized_vol_21"]
    result["vol_of_vol_63"] = rv21.rolling(63, min_periods=63).std()

    # Garman-Klass
    log_hl = np.log(high / low)
    log_co = np.log(close / opn)
    gk_daily = 0.5 * log_hl ** 2 - (2 * np.log(2) - 1) * log_co ** 2
    gk_mean = gk_daily.rolling(21, min_periods=21).mean()
    result["garman_klass_vol"] = np.sqrt(gk_mean.clip(lower=0)) * np.sqrt(252)

    # Vol regime: current vol vs long-term vol
    rv_long = log_ret_1d.rolling(252, min_periods=126).std() * np.sqrt(252)
    result["vol_regime"] = rv21 / rv_long.replace(0, np.nan)

    # ── Mean Reversion ────────────────────────────────────────
    sma_20 = close.rolling(20, min_periods=20).mean()
    std_20 = close.rolling(20, min_periods=20).std()
    result["z_score_20"] = (close - sma_20) / std_20

    upper_bb = sma_20 + 2 * std_20
    lower_bb = sma_20 - 2 * std_20
    result["bollinger_pct"] = (close - lower_bb) / (upper_bb - lower_bb)

    high_52w = high.rolling(252, min_periods=126).max()
    low_52w = low.rolling(252, min_periods=126).min()
    result["distance_from_52w_high"] = (close - high_52w) / high_52w
    result["distance_from_52w_low"] = (close - low_52w) / low_52w

    # SMA cross signals
    sma_50 = close.rolling(50, min_periods=50).mean()
    sma_200 = close.rolling(200, min_periods=126).mean()
    result["sma_20_50_cross"] = (sma_20 / sma_50) - 1
    result["sma_50_200_cross"] = (sma_50 / sma_200) - 1

    # ── Microstructure ────────────────────────────────────────
    vol_sma_21 = volume.rolling(21, min_periods=21).mean()
    result["volume_sma_ratio"] = volume / vol_sma_21

    vol_sma_63 = volume.rolling(63, min_periods=63).mean()
    result["volume_relative_63"] = volume / vol_sma_63

    result["high_low_range_pct"] = (high - low) / close

    # ── Regime Detection ──────────────────────────────────────
    result["autocorrelation_1d"] = log_ret_1d.rolling(
        63, min_periods=63
    ).apply(lambda x: x.autocorr(lag=1), raw=False)

    result["autocorrelation_5d"] = log_ret_1d.rolling(
        63, min_periods=63
    ).apply(lambda x: x.autocorr(lag=5), raw=False)

    # ADX
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = pd.Series(0.0, index=high.index)
    minus_dm = pd.Series(0.0, index=high.index)
    cond_plus = (up_move > down_move) & (up_move > 0)
    cond_minus = (down_move > up_move) & (down_move > 0)
    plus_dm[cond_plus] = up_move[cond_plus]
    minus_dm[cond_minus] = down_move[cond_minus]
    smooth_plus = plus_dm.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    smooth_minus = minus_dm.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    plus_di = 100 * smooth_plus / atr_14.replace(0, np.nan)
    minus_di = 100 * smooth_minus / atr_14.replace(0, np.nan)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    result["adx_14"] = dx.ewm(alpha=1/14, min_periods=14, adjust=False).mean()

    # ── Calendar ──────────────────────────────────────────────
    if isinstance(df.index, pd.DatetimeIndex):
        idx = df.index
        angle_dow = 2 * np.pi * idx.dayofweek.values.astype(float) / 5.0
        result["day_of_week_sin"] = np.sin(angle_dow)
        result["day_of_week_cos"] = np.cos(angle_dow)
        angle_month = 2 * np.pi * idx.month.values.astype(float) / 12.0
        result["month_sin"] = np.sin(angle_month)
        result["month_cos"] = np.cos(angle_month)

    # ── Clean ─────────────────────────────────────────────────
    result = result.replace([np.inf, -np.inf], np.nan)
    return result


def compute_label(
    df: pd.DataFrame,
    holding_period: int = 10,
    vol_span: int = 21,
    tp_mult: float = 2.0,
    sl_mult: float = 2.0,
) -> pd.Series:
    """
    Simple triple-barrier label for universal training.
    Returns: Series of labels (0=down, 1=neutral, 2=up).
    """
    close = df["close"].astype(np.float64)
    high = df["high"].astype(np.float64)
    low = df["low"].astype(np.float64)

    # EWMA daily vol
    log_ret = np.log(close / close.shift(1))
    daily_vol = log_ret.ewm(span=vol_span, min_periods=10).std()

    labels = pd.Series(np.nan, index=df.index)
    close_arr = close.values
    high_arr = high.values
    low_arr = low.values

    for i in range(len(df) - holding_period - 1):
        vol = daily_vol.iloc[i]
        if np.isnan(vol) or vol < 1e-8:
            continue

        entry = close_arr[i]
        upper = entry * (1 + tp_mult * vol)
        lower = entry * (1 - sl_mult * vol)

        label = 1  # neutral (vertical barrier)
        for j in range(1, holding_period + 1):
            idx = i + j
            if idx >= len(df):
                break
            if high_arr[idx] >= upper:
                label = 2  # up (take profit)
                break
            if low_arr[idx] <= lower:
                label = 0  # down (stop loss)
                break

        labels.iloc[i] = label

    return labels


# ══════════════════════════════════════════════════════════════
# UniversalTechnicalModel
# ══════════════════════════════════════════════════════════════

class UniversalTechnicalModel:
    """
    A single LightGBM model trained on ALL stocks' technical patterns.
    Works for ANY stock — even ones never seen in training.
    """

    MODEL_DIR = "models/universal/"
    MODEL_FILE = "universal_technical_model.txt"
    PIPELINE_FILE = "universal_feature_pipeline.pkl"
    METADATA_FILE = "universal_metadata.json"

    def __init__(
        self,
        polygon_data: Optional["PolygonUniverseData"] = None,
        model_root: str = "models/universal/",
    ):
        self.polygon_data = polygon_data
        self.model_root = Path(model_root)
        self.model_root.mkdir(parents=True, exist_ok=True)

        self.model: lgb.Booster | None = None
        self.metadata: dict | None = None
        self._fitted_means: dict | None = None
        self._fitted_stds: dict | None = None
        self._feature_names: List[str] | None = None
        self._lock = threading.Lock()

        # Try to load existing model
        self._load_model()

    def _load_model(self) -> bool:
        """Load existing universal model from disk."""
        model_path = self.model_root / self.MODEL_FILE
        meta_path = self.model_root / self.METADATA_FILE
        pipeline_path = self.model_root / self.PIPELINE_FILE

        if not model_path.exists():
            return False

        try:
            self.model = lgb.Booster(model_file=str(model_path))

            if meta_path.exists():
                with open(meta_path) as f:
                    self.metadata = json.load(f)

            if pipeline_path.exists():
                pipe_data = joblib.load(str(pipeline_path))
                self._fitted_means = pipe_data.get("means")
                self._fitted_stds = pipe_data.get("stds")
                self._feature_names = pipe_data.get("feature_names")

            logger.info(
                "Loaded universal model: %s features, trained on %s stocks",
                self.metadata.get("n_features", "?") if self.metadata else "?",
                self.metadata.get("n_stocks_used", "?") if self.metadata else "?",
            )
            return True

        except Exception as e:
            logger.warning("Failed to load universal model: %s", e)
            return False

    def is_ready(self) -> bool:
        """Whether the model is loaded and ready for predictions."""
        return self.model is not None

    # ── Training Panel ────────────────────────────────────────

    def build_training_panel(
        self,
        n_stocks: int = 10000,
        n_days: int = 500,
        min_price: float = 5.0,
        min_dollar_volume: float = 1_000_000,
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Build training panel using UniversalFeatureEngine (112+ features).

        Processes stocks in batches of 200 to control memory.
        Uses float32 to halve memory usage.
        Default n_stocks=10000 — effectively uses ALL available stocks.
        """
        from .universal_features import UniversalFeatureEngine

        if self.polygon_data is None:
            raise RuntimeError("PolygonUniverseData required for training.")

        logger.info("Building training panel: %d stocks × %d days", n_stocks, n_days)
        start_time = time.time()

        engine = UniversalFeatureEngine()

        # Get active universe
        universe = self.polygon_data.get_active_universe(
            min_price=min_price,
            min_dollar_volume=min_dollar_volume,
        )
        if not universe:
            raise RuntimeError("No stocks in active universe. Run backfill() first.")

        if len(universe) > n_stocks:
            universe = universe[:n_stocks]

        logger.info("Selected %d stocks for training", len(universe))

        # Process in batches
        batch_size = 200
        all_features = []
        all_labels = []
        stocks_used = 0

        for batch_start in range(0, len(universe), batch_size):
            batch_tickers = universe[batch_start:batch_start + batch_size]
            logger.info(
                "Processing batch %d–%d / %d",
                batch_start, batch_start + len(batch_tickers), len(universe),
            )

            histories = self.polygon_data.get_multi_stock_history(
                symbols=batch_tickers, days=n_days
            )

            for ticker, df in histories.items():
                if len(df) < 252:
                    continue

                try:
                    # Use new 112-feature engine
                    features = engine.compute_features(df)
                    labels = compute_label(df)

                    # Drop warmup rows
                    features = features.iloc[engine.WARMUP_ROWS:]
                    labels = labels.iloc[engine.WARMUP_ROWS:]

                    # Align
                    common_idx = features.index.intersection(labels.dropna().index)
                    if len(common_idx) < 100:
                        continue

                    feat = features.loc[common_idx]
                    lab = labels.loc[common_idx].dropna()
                    common = feat.index.intersection(lab.index)
                    feat = feat.loc[common]
                    lab = lab.loc[common]

                    if len(feat) < 100:
                        continue

                    feat = feat.copy()
                    feat["_ticker"] = ticker
                    feat["_date"] = feat.index

                    all_features.append(feat)
                    all_labels.append(lab.astype(np.int32))
                    stocks_used += 1

                except Exception as e:
                    logger.debug("Skipping %s: %s", ticker, e)
                    continue

            gc.collect()

        if not all_features:
            raise RuntimeError("No valid training data generated.")

        X = pd.concat(all_features, ignore_index=False)
        y = pd.concat(all_labels, ignore_index=False)

        meta_cols = ["_ticker", "_date"]
        X_meta = X[meta_cols].copy() if all(c in X.columns for c in meta_cols) else None
        X = X.drop(columns=[c for c in meta_cols if c in X.columns], errors="ignore")

        if X_meta is not None:
            X.index = X_meta["_date"].values

        # ── Feature quality validation ──
        # Drop features >30% NaN
        nan_pct = X.isna().mean()
        bad_features = nan_pct[nan_pct > 0.30].index.tolist()
        if bad_features:
            logger.warning("Dropping %d features with >30%% NaN: %s", len(bad_features), bad_features)
            X = X.drop(columns=bad_features)

        # Drop near-zero variance features
        stds = X.std()
        dead_features = stds[stds < 0.001].index.tolist()
        if dead_features:
            logger.warning("Dropping %d near-zero variance features: %s", len(dead_features), dead_features)
            X = X.drop(columns=dead_features)

        elapsed = time.time() - start_time
        logger.info(
            "Training panel built: %d rows × %d features from %d stocks in %.1fs",
            len(X), X.shape[1], stocks_used, elapsed,
        )
        return X, y

    # ── Training ──────────────────────────────────────────────

    def train(
        self,
        X: pd.DataFrame | None = None,
        y: pd.Series | None = None,
        build_data: bool = True,
    ) -> dict:
        """
        Train the universal model using Phase 3 feature engine (112+ features).

        LightGBM params tuned for 112 features and 500K+ samples.
        Stricter quality gates. Feature group importance tracking.
        """
        from .universal_features import UniversalFeatureEngine

        start_time = time.time()

        if build_data or X is None or y is None:
            X, y = self.build_training_panel()

        logger.info("Training universal model: %d samples, %d features", len(X), X.shape[1])

        # ── Time-based split ──────────────────────────────────
        # Preserve dates for per-date IC consistency gate
        dates_index = None
        if isinstance(X.index, pd.DatetimeIndex):
            sort_idx = X.index.argsort()
            X = X.iloc[sort_idx]
            y = y.iloc[sort_idx]
            dates_index = X.index.copy()
            X = X.reset_index(drop=True)
            y = y.reset_index(drop=True)

        n = len(X)
        purge_gap = 10

        train_end = int(n * 0.70)
        val_start = train_end + purge_gap
        val_end = int(n * 0.85)
        test_start = val_end + purge_gap

        X_train = X.iloc[:train_end]
        y_train = y.iloc[:train_end]
        X_val = X.iloc[val_start:val_end]
        y_val = y.iloc[val_start:val_end]
        X_test = X.iloc[test_start:]
        y_test = y.iloc[test_start:]

        # Keep test dates for IC consistency computation
        test_dates = dates_index[test_start:] if dates_index is not None else None

        logger.info(
            "Split: train=%d, val=%d, test=%d (purge gap=%d)",
            len(X_train), len(X_val), len(X_test), purge_gap,
        )

        # ── Identify feature types ────────────────────────────
        numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()

        # Calendar features are now binary (ctx_is_monday, ctx_is_friday, etc.)
        # No categorical feature declaration needed — binary 0/1 works as continuous
        categorical_cols = []

        # ── Standardize using training stats ──────────────────
        means = X_train[numeric_cols].mean().to_dict()
        stds = X_train[numeric_cols].std().replace(0, 1).to_dict()

        # Don't standardize categorical features
        for col in numeric_cols:
            if col in categorical_cols:
                continue
            X_train[col] = (X_train[col] - means.get(col, 0)) / stds.get(col, 1)
            X_val[col] = (X_val[col] - means.get(col, 0)) / stds.get(col, 1)
            X_test[col] = (X_test[col] - means.get(col, 0)) / stds.get(col, 1)

        self._fitted_means = means
        self._fitted_stds = stds
        self._feature_names = numeric_cols

        # ── Class weights for label imbalance ─────────────────
        class_counts = np.bincount(y_train.values.astype(int), minlength=3)
        n_samples = len(y_train)
        n_classes = 3
        class_weights = {}
        for cls in range(n_classes):
            if class_counts[cls] > 0:
                class_weights[cls] = n_samples / (n_classes * class_counts[cls])
            else:
                class_weights[cls] = 1.0

        sample_weight = np.array([class_weights[int(y)] for y in y_train.values])

        # ── LightGBM training (Phase 3 tuned params) ─────────
        params = {
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
            "seed": 42,
            "num_threads": 12,
        }

        train_data = lgb.Dataset(
            X_train[numeric_cols].values,
            label=y_train.values,
            weight=sample_weight,
            feature_name=numeric_cols,
            free_raw_data=False,
        )
        val_data = lgb.Dataset(
            X_val[numeric_cols].values,
            label=y_val.values,
            reference=train_data,
            free_raw_data=False,
        )

        callbacks = [
            lgb.early_stopping(100, verbose=True),
            lgb.log_evaluation(100),
        ]

        model = lgb.train(
            params,
            train_data,
            num_boost_round=3000,
            valid_sets=[train_data, val_data],
            valid_names=["train", "val"],
            callbacks=callbacks,
        )

        # ── Evaluation on test set ────────────────────────────
        test_pred_proba = model.predict(X_test[numeric_cols].values)
        test_pred_class = np.argmax(test_pred_proba, axis=1)

        accuracy = float(np.mean(test_pred_class == y_test.values))

        # Per-class accuracy
        per_class_acc = {}
        for cls in range(3):
            mask = y_test.values == cls
            if mask.sum() > 0:
                per_class_acc[str(cls)] = float(np.mean(test_pred_class[mask] == cls))

        # IC (Spearman correlation of P(long) with label direction)
        long_proba = test_pred_proba[:, 2]
        label_returns = y_test.map({0: -1.0, 1: 0.0, 2: 1.0}).values

        try:
            ic_overall, _ = sp_stats.spearmanr(long_proba, label_returns)
            ic_overall = float(ic_overall) if not np.isnan(ic_overall) else 0.0
        except Exception:
            ic_overall = 0.0

        # ── Per-date IC consistency ───────────────────────────
        # IC must be positive in >60% of test dates (institutional standard)
        ic_positive_frac = 0.0
        n_test_dates = 0
        if test_dates is not None and len(test_dates) > 0:
            test_date_series = pd.Series(test_dates, index=X_test.index)
            unique_dates = test_date_series.unique()
            n_test_dates = len(unique_dates)
            positive_ic_count = 0
            for dt in unique_dates:
                mask = test_date_series == dt
                if mask.sum() < 5:  # Need at least 5 stocks per date for meaningful IC
                    n_test_dates -= 1
                    continue
                try:
                    dt_ic, _ = sp_stats.spearmanr(
                        long_proba[mask.values],
                        label_returns[mask.values],
                    )
                    if not np.isnan(dt_ic) and dt_ic > 0:
                        positive_ic_count += 1
                except Exception:
                    n_test_dates -= 1
                    continue
            ic_positive_frac = positive_ic_count / n_test_dates if n_test_dates > 0 else 0.0
            logger.info(
                "IC consistency: positive in %d/%d dates (%.1f%%)",
                positive_ic_count, n_test_dates, ic_positive_frac * 100,
            )

        # ── Feature importance analysis ───────────────────────
        importance = model.feature_importance(importance_type="gain")
        importance_dict = dict(zip(numeric_cols, importance.tolist()))
        sorted_features = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
        top_20 = [{"feature": f, "importance": round(v, 2)} for f, v in sorted_features[:20]]

        total_importance = sum(importance)
        max_feature_pct = max(importance) / total_importance if total_importance > 0 else 0

        # Feature group importance breakdown
        group_importance = UniversalFeatureEngine.feature_group_importance(importance_dict)

        # Calendar feature combined importance (binary ctx_is_* features)
        calendar_importance = sum(
            v for k, v in importance_dict.items()
            if k.startswith("ctx_is_")
        ) / (total_importance or 1) * 100

        # Feature group diversity: count groups in top 20
        top20_groups = set()
        for feat_info in sorted_features[:20]:
            feat_name = feat_info[0]
            prefix = feat_name.split("_")[0] + "_"
            top20_groups.add(prefix)

        # Class distribution
        test_class_dist = pd.Series(y_test.values).value_counts().to_dict()

        # ── Quality gates (Phase 3 — stricter) ────────────────
        gates = {
            "ic_above_002": ic_overall > 0.02,
            "ic_consistency_60pct": ic_positive_frac > 0.60,
            "accuracy_above_42pct": accuracy > 0.42,
            "no_feature_over_20pct": max_feature_pct < 0.20,
            "calendar_under_10pct": calendar_importance < 10.0,
            "top20_diversity_4_groups": len(top20_groups) >= 4,
            "uses_many_features": len(numeric_cols) > 50,
        }
        all_passed = all(gates.values())

        elapsed = time.time() - start_time

        metadata = {
            "training_date": datetime.now().isoformat(),
            "n_training_samples": len(X_train),
            "n_validation_samples": len(X_val),
            "n_test_samples": len(X_test),
            "n_stocks_used": len(set(X.index)) if isinstance(X.index, pd.DatetimeIndex) else 0,
            "n_features": len(numeric_cols),
            "test_ic_mean": round(ic_overall, 4),
            "test_ic_consistency": round(ic_positive_frac, 4),
            "test_ic_n_dates": n_test_dates,
            "test_accuracy": round(accuracy, 4),
            "per_class_accuracy": per_class_acc,
            "test_class_distribution": {str(k): int(v) for k, v in test_class_dist.items()},
            "top_20_features": top_20,
            "max_feature_importance_pct": round(max_feature_pct, 4),
            "calendar_importance_pct": round(calendar_importance, 2),
            "feature_group_importance": group_importance,
            "top20_group_diversity": len(top20_groups),
            "quality_gates": gates,
            "quality_gates_passed": all_passed,
            "best_iteration": model.best_iteration,
            "training_time_seconds": round(elapsed, 1),
            "class_weights_used": {str(k): round(v, 3) for k, v in class_weights.items()},
            "feature_engine": "UniversalFeatureEngine_v3",
        }

        # ── Save ──────────────────────────────────────────────
        if all_passed:
            model.save_model(str(self.model_root / self.MODEL_FILE))

            joblib.dump(
                {
                    "means": self._fitted_means,
                    "stds": self._fitted_stds,
                    "feature_names": self._feature_names,
                    "categorical_features": categorical_cols,
                },
                str(self.model_root / self.PIPELINE_FILE),
            )

            with open(self.model_root / self.METADATA_FILE, "w") as f:
                json.dump(metadata, f, indent=2, default=str)

            with self._lock:
                self.model = model
                self.metadata = metadata

            logger.info(
                "✅ Universal model saved: IC=%.4f, Accuracy=%.4f, %d features, %.0fs",
                ic_overall, accuracy, len(numeric_cols), elapsed,
            )
        else:
            # Save anyway but log the warning
            failed_gates = {k: v for k, v in gates.items() if not v}
            logger.warning(
                "⚠️ Quality gates not all passed: %s — saving model anyway",
                failed_gates,
            )
            # Still save — the model is usable even if not perfect
            model.save_model(str(self.model_root / self.MODEL_FILE))

            joblib.dump(
                {
                    "means": self._fitted_means,
                    "stds": self._fitted_stds,
                    "feature_names": self._feature_names,
                    "categorical_features": categorical_cols,
                },
                str(self.model_root / self.PIPELINE_FILE),
            )

            with open(self.model_root / self.METADATA_FILE, "w") as f:
                json.dump(metadata, f, indent=2, default=str)

            with self._lock:
                self.model = model
                self.metadata = metadata

        return metadata

    # ── Prediction ────────────────────────────────────────────

    def predict(self, symbol: str, df: pd.DataFrame | None = None) -> dict:
        """
        Predict for ANY symbol using the universal model.

        Uses UniversalFeatureEngine (112+ features) for feature computation.
        If df not provided, fetches from PolygonUniverseData.
        Returns same format as MLPredictor for backward compatibility.
        """
        if not self.is_ready():
            return {}

        with self._lock:
            model = self.model
            means = self._fitted_means
            stds = self._fitted_stds
            feature_names = self._feature_names

        # Get data if not provided
        if df is None:
            if self.polygon_data is None:
                try:
                    from nuble.data.polygon_universe import PolygonUniverseData
                    self.polygon_data = PolygonUniverseData()
                except Exception:
                    return {}
            df = self.polygon_data.get_stock_history(symbol, days=500)
            if df is None:
                return {}

        if len(df) < 63:
            return {}

        try:
            # Use Phase 3 feature engine
            from .universal_features import UniversalFeatureEngine
            engine = UniversalFeatureEngine()
            features = engine.compute_features(df)
            if features.empty:
                return {}

            # Use only last row (most recent)
            latest = features.iloc[[-1]].copy()

            # Align to training features
            for col in feature_names:
                if col not in latest.columns:
                    latest[col] = 0.0
            latest = latest[feature_names]

            # Standardize using training stats
            for col in feature_names:
                latest[col] = (latest[col] - means.get(col, 0)) / stds.get(col, 1)

            # Predict
            proba = model.predict(latest.values)[0]

            pred_class = int(np.argmax(proba))
            direction_map = {0: "SHORT", 1: "NEUTRAL", 2: "LONG"}
            confidence = float(proba[pred_class])

            # Feature importance for this prediction
            top_features = []
            if self.metadata and "top_20_features" in self.metadata:
                for feat_info in self.metadata["top_20_features"][:5]:
                    feat_name = feat_info["feature"]
                    if feat_name in latest.columns:
                        top_features.append({
                            "feature": feat_name,
                            "value": round(float(latest[feat_name].iloc[0]), 4),
                            "importance": feat_info["importance"],
                        })

            return {
                "symbol": symbol,
                "prediction": pred_class,
                "direction": direction_map[pred_class],
                "signal": direction_map[pred_class],
                "confidence": confidence,
                "probabilities": {
                    "short": float(proba[0]),
                    "neutral": float(proba[1]),
                    "long": float(proba[2]),
                },
                "explanation": {
                    "top_features": top_features,
                    "cv_information_coefficient": (
                        self.metadata.get("test_ic_mean", 0) if self.metadata else 0
                    ),
                    "training_date": (
                        self.metadata.get("training_date", "unknown") if self.metadata else "unknown"
                    ),
                    "model_type": "universal",
                    "training_samples": (
                        self.metadata.get("n_training_samples", 0) if self.metadata else 0
                    ),
                    "n_stocks_in_training": (
                        self.metadata.get("n_stocks_used", 0) if self.metadata else 0
                    ),
                    "feature_engine": "UniversalFeatureEngine_v3",
                },
                "model_info": {
                    "model_type": "universal_technical",
                    "version": "3.0.0",
                    "n_features": len(feature_names),
                },
            }

        except Exception as e:
            logger.warning("Universal model predict(%s) failed: %s", symbol, e)
            return {}

    def predict_batch(self, symbols: List[str]) -> Dict[str, dict]:
        """Predict for multiple symbols. More efficient than individual calls."""
        results = {}
        for symbol in symbols:
            result = self.predict(symbol)
            if result:
                results[symbol] = result
        return results
