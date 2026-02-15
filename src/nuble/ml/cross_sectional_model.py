"""
Cross-Sectional Regression Model — Predicts Continuous Forward Excess Returns
===============================================================================

THIS REPLACES classification with regression. Here's why:

Classification: P(direction) → loses magnitude info → inverted deciles
Regression: E(return) → preserves magnitude → proper ranking

The model learns: "Given 112+ rank-normalized features for a stock on date T,
what is the expected EXCESS return over the next N days?"

Excess return = stock_return - universe_median_return
This makes the target market-neutral (predicts relative performance).

KEY INNOVATIONS over universal_model.py:

1. REGRESSION on continuous returns (not classification)
2. CROSS-SECTIONAL RANK NORMALIZATION per date
3. HUBER LOSS (robust to fat-tailed return outliers)
4. Same train/score universe (no mismatch)

Based on: Gu-Kelly-Xiu (2020), "Empirical Asset Pricing via Machine Learning"

Author: NUBLE ML Pipeline — Phase 6 Model Rebuild
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
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

if TYPE_CHECKING:
    from ..data.polygon_universe import PolygonUniverseData
    from .universal_features import UniversalFeatureEngine

logger = logging.getLogger(__name__)

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)


class CrossSectionalModel:
    """
    Cross-sectional regression model — predicts CONTINUOUS forward excess returns.

    Architecture:
      1. Features: 112+ from UniversalFeatureEngine
      2. Cross-sectional rank normalization: percentile ranks across stocks per date
      3. Target: Forward N-day excess log return (continuous)
      4. Model: LightGBM REGRESSION with Huber loss
      5. Evaluation: Spearman IC (rank correlation)
    """

    MODEL_DIR = "models/cross_sectional/"
    MODEL_FILE = "cross_sectional_model.txt"
    METADATA_FILE = "cross_sectional_metadata.json"
    FEATURE_FILE = "cross_sectional_features.json"

    def __init__(
        self,
        polygon_data: Optional["PolygonUniverseData"] = None,
        feature_engine: Optional["UniversalFeatureEngine"] = None,
        model_root: str = "models/cross_sectional/",
    ):
        self.polygon_data = polygon_data
        self._engine = feature_engine
        self.model_root = Path(model_root)
        self.model_root.mkdir(parents=True, exist_ok=True)

        self.model = None
        self.metadata: Optional[dict] = None
        self._feature_names: Optional[List[str]] = None
        self._lock = threading.Lock()

        # Try loading existing model
        self._load_model()

    @property
    def engine(self) -> "UniversalFeatureEngine":
        if self._engine is None:
            from .universal_features import UniversalFeatureEngine
            self._engine = UniversalFeatureEngine()
        return self._engine

    def _load_model(self) -> bool:
        """Load existing cross-sectional model from disk."""
        model_path = self.model_root / self.MODEL_FILE
        meta_path = self.model_root / self.METADATA_FILE
        feat_path = self.model_root / self.FEATURE_FILE

        if not model_path.exists():
            return False

        try:
            import lightgbm as lgb
            self.model = lgb.Booster(model_file=str(model_path))

            if meta_path.exists():
                with open(meta_path) as f:
                    self.metadata = json.load(f)

            if feat_path.exists():
                with open(feat_path) as f:
                    self._feature_names = json.load(f)

            logger.info(
                "Loaded cross-sectional model: %s features, trained %s",
                len(self._feature_names) if self._feature_names else "?",
                self.metadata.get("training_date", "?") if self.metadata else "?",
            )
            return True
        except Exception as e:
            logger.warning("Failed to load cross-sectional model: %s", e)
            return False

    def is_ready(self) -> bool:
        return self.model is not None

    # ══════════════════════════════════════════════════════════
    # TRAINING PANEL
    # ══════════════════════════════════════════════════════════

    def build_training_panel(
        self,
        n_stocks: int = 10000,
        min_history: int = 60,
        forward_horizon: int = 5,
        min_stocks_per_date: int = 100,
        verbose: bool = True,
    ) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
        """
        Build the cross-sectional training panel.

        For each (stock, date):
          raw_return = log(close_{t+horizon} / close_t)
          universe_median = median(raw_return across all stocks on date t)
          excess_return = raw_return - universe_median

        Features are RANK NORMALIZED across all stocks on each date.

        Returns: (X_ranked, y_excess, meta_df)
            X_ranked: DataFrame of rank-normalized features [0, 1]
            y_excess: Series of forward excess returns
            meta_df:  DataFrame with 'ticker' and 'date' columns
        """
        if self.polygon_data is None:
            from ..data.polygon_universe import PolygonUniverseData
            self.polygon_data = PolygonUniverseData()

        start_time = time.time()

        # Step 1: Get all available dates
        all_dates = self.polygon_data._load_index()
        all_dates = sorted([d.isoformat() for d in all_dates])
        if verbose:
            print(f"[Panel] {len(all_dates)} dates available: {all_dates[0]} → {all_dates[-1]}")

        # Step 2: Get universe
        universe = self.polygon_data.get_active_universe(
            min_price=5.0, min_dollar_volume=1_000_000,
        )
        if n_stocks < len(universe):
            universe = universe[:n_stocks]
        if verbose:
            print(f"[Panel] Active universe: {len(universe)} stocks (using {len(universe)})")

        # Step 3: Load all histories
        if verbose:
            print(f"[Panel] Loading stock histories...")
        stock_histories = self._load_histories_batched(universe, verbose=verbose)
        if verbose:
            print(f"[Panel] Loaded {len(stock_histories)} stocks")

        # Step 4: Compute raw features per stock
        if verbose:
            print(f"[Panel] Computing raw features...")
        stock_features: Dict[str, pd.DataFrame] = {}
        computed = 0
        for ticker, df in stock_histories.items():
            if len(df) < max(self.engine.MIN_ROWS, min_history):
                continue
            try:
                feat = self.engine.compute_features(df)
                if feat.empty or len(feat) < min_history:
                    continue
                # Drop warmup
                feat = feat.iloc[self.engine.WARMUP_ROWS:]
                if len(feat) < 20:
                    continue
                stock_features[ticker] = feat
                computed += 1
            except Exception:
                continue

            if computed % 200 == 0 and verbose:
                print(f"  Features: {computed}/{len(stock_histories)}")
            if computed % 500 == 0:
                gc.collect()

        if verbose:
            print(f"[Panel] {computed} stocks with valid features")

        # Step 5: Identify all unique dates across all stocks
        feature_dates = set()
        for feat in stock_features.values():
            for d in feat.index:
                if isinstance(d, pd.Timestamp):
                    feature_dates.add(d.strftime("%Y-%m-%d"))
                else:
                    feature_dates.add(str(d))
        feature_dates = sorted(feature_dates)
        if verbose:
            print(f"[Panel] {len(feature_dates)} unique feature dates")

        # Step 6: Build cross-sectional panel date by date
        panel_rows = []
        panel_targets = []
        panel_meta = []
        dates_processed = 0
        dates_skipped = 0

        for dt_str in feature_dates:
            # Gather all stocks with features on this date
            date_feats = {}
            date_fwd_rets = {}

            for ticker, feat_df in stock_features.items():
                if isinstance(feat_df.index, pd.DatetimeIndex):
                    mask = feat_df.index.strftime("%Y-%m-%d") == dt_str
                else:
                    mask = pd.Series(feat_df.index.astype(str) == dt_str, index=feat_df.index)

                rows = feat_df[mask]
                if len(rows) == 0:
                    continue

                # Compute forward return
                fwd_ret = self._forward_return(stock_histories.get(ticker), dt_str, forward_horizon)
                if fwd_ret is None or np.isnan(fwd_ret):
                    continue

                date_feats[ticker] = rows.iloc[-1]
                date_fwd_rets[ticker] = fwd_ret

            if len(date_feats) < min_stocks_per_date:
                dates_skipped += 1
                continue

            # Build cross-sectional DataFrame for this date
            cs_df = pd.DataFrame(date_feats).T  # rows=stocks, cols=features
            cs_rets = pd.Series(date_fwd_rets)

            # Compute excess returns (market-neutral)
            median_ret = cs_rets.median()
            excess_rets = cs_rets - median_ret

            # RANK NORMALIZE features across stocks on this date
            cs_ranked = cs_df.rank(pct=True)
            # Handle columns with all same values (rank → NaN) → fill with 0.5
            cs_ranked = cs_ranked.fillna(0.5)

            # Append to panel
            for ticker in cs_ranked.index:
                panel_rows.append(cs_ranked.loc[ticker].values)
                panel_targets.append(excess_rets[ticker])
                panel_meta.append({"ticker": ticker, "date": dt_str})

            dates_processed += 1
            if dates_processed % 50 == 0 and verbose:
                print(f"  Dates processed: {dates_processed}/{len(feature_dates)} "
                      f"({len(panel_rows):,} samples)")

            if dates_processed % 100 == 0:
                gc.collect()

        if verbose:
            print(f"[Panel] Done: {dates_processed} dates, {dates_skipped} skipped "
                  f"(< {min_stocks_per_date} stocks)")

        if not panel_rows:
            raise RuntimeError("No valid training data — check data availability")

        # Build final DataFrames
        feature_cols = cs_ranked.columns.tolist()  # from last date
        X = pd.DataFrame(panel_rows, columns=feature_cols, dtype=np.float32)
        y = pd.Series(panel_targets, dtype=np.float32)
        meta = pd.DataFrame(panel_meta)

        # Drop NaN targets
        valid = ~y.isna()
        X = X[valid].reset_index(drop=True)
        y = y[valid].reset_index(drop=True)
        meta = meta[valid].reset_index(drop=True)

        elapsed = time.time() - start_time
        mem_mb = X.memory_usage(deep=True).sum() / 1e6
        if verbose:
            print(f"\n[Panel] Final: {len(X):,} samples × {X.shape[1]} features | "
                  f"Memory: {mem_mb:.0f} MB | Time: {elapsed:.0f}s")
            print(f"[Panel] Target stats: mean={y.mean():.6f}, std={y.std():.4f}, "
                  f"min={y.min():.4f}, max={y.max():.4f}")

        return X, y, meta

    # ══════════════════════════════════════════════════════════
    # TRAINING
    # ══════════════════════════════════════════════════════════

    def train(
        self,
        X: pd.DataFrame = None,
        y: pd.Series = None,
        meta: pd.DataFrame = None,
        build_data: bool = True,
        forward_horizon: int = 5,
        n_stocks: int = 10000,
        verbose: bool = True,
    ) -> dict:
        """
        Train the cross-sectional regression model.

        Split by TIME (never random):
          Train: first 60% of dates
          Validation: next 20% (early stopping)
          Test: last 20% (evaluation)
          Purge: 2× forward_horizon between splits
        """
        import lightgbm as lgb

        start_time = time.time()

        if build_data or X is None:
            X, y, meta = self.build_training_panel(
                n_stocks=n_stocks,
                forward_horizon=forward_horizon,
                verbose=verbose,
            )

        # ── Time-based split ──────────────────────────────────
        dates = sorted(meta["date"].unique())
        n_dates = len(dates)
        purge = 2 * forward_horizon

        train_end_idx = int(n_dates * 0.60)
        val_start_idx = train_end_idx + purge
        val_end_idx = int(n_dates * 0.80)
        test_start_idx = val_end_idx + purge

        if test_start_idx >= n_dates:
            test_start_idx = val_end_idx + 1
        if val_start_idx >= val_end_idx:
            val_start_idx = train_end_idx + 1

        train_dates = set(dates[:train_end_idx])
        val_dates = set(dates[val_start_idx:val_end_idx])
        test_dates = set(dates[test_start_idx:])

        train_mask = meta["date"].isin(train_dates)
        val_mask = meta["date"].isin(val_dates)
        test_mask = meta["date"].isin(test_dates)

        X_train, y_train = X[train_mask].copy(), y[train_mask].copy()
        X_val, y_val = X[val_mask].copy(), y[val_mask].copy()
        X_test, y_test = X[test_mask].copy(), y[test_mask].copy()
        meta_test = meta[test_mask].copy()

        feature_cols = X.columns.tolist()
        self._feature_names = feature_cols

        if verbose:
            print(f"\n[Train] Split by time:")
            print(f"  Train: {len(X_train):,} samples ({len(train_dates)} dates)")
            print(f"  Val:   {len(X_val):,} samples ({len(val_dates)} dates)")
            print(f"  Test:  {len(X_test):,} samples ({len(test_dates)} dates)")
            print(f"  Purge: {purge} days between splits")

        # ── LightGBM Regression ──────────────────────────────
        params = {
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

        train_data = lgb.Dataset(
            X_train.values, label=y_train.values,
            feature_name=feature_cols, free_raw_data=False,
        )
        val_data = lgb.Dataset(
            X_val.values, label=y_val.values,
            reference=train_data, free_raw_data=False,
        )

        if verbose:
            print(f"\n[Train] Training LightGBM regression (Huber loss)...")

        callbacks = [
            lgb.early_stopping(200, verbose=False),
            lgb.log_evaluation(100 if verbose else 0),
        ]

        model = lgb.train(
            params,
            train_data,
            num_boost_round=3000,
            valid_sets=[val_data],
            valid_names=["val"],
            callbacks=callbacks,
        )

        train_elapsed = time.time() - start_time
        if verbose:
            print(f"  Best iteration: {model.best_iteration}")
            print(f"  Training time: {train_elapsed:.0f}s")

        # ── Evaluation on test set ────────────────────────────
        test_preds = model.predict(X_test.values)
        eval_results = self._evaluate_test(
            test_preds, y_test.values, meta_test, verbose=verbose,
        )

        # ── Feature importance ────────────────────────────────
        importance = model.feature_importance(importance_type="gain")
        importance_dict = dict(zip(feature_cols, importance.tolist()))
        sorted_feats = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
        top_20 = [{"feature": f, "importance": round(v, 2)} for f, v in sorted_feats[:20]]

        total_imp = sum(importance) or 1.0
        max_feat_pct = max(importance) / total_imp

        from .universal_features import UniversalFeatureEngine
        group_imp = UniversalFeatureEngine.feature_group_importance(importance_dict)

        if verbose:
            print(f"\n[Train] Feature importance (top 10):")
            for fi in top_20[:10]:
                pct = fi["importance"] / total_imp * 100
                print(f"  {fi['feature']:35s}: {pct:.1f}%")
            print(f"\n[Train] Feature group importance:")
            for grp, pct in sorted(group_imp.items(), key=lambda x: -x[1]):
                if pct > 0:
                    bar = "█" * int(pct / 2)
                    print(f"  {grp:15s}: {pct:5.1f}% {bar}")

        # ── Quality gates ─────────────────────────────────────
        gates = {
            "mean_ic_above_0015": eval_results["mean_ic"] > 0.015,
            "ic_ir_above_03": eval_results["ic_ir"] > 0.3,
            "ic_hit_rate_above_55": eval_results["ic_hit_rate"] > 0.55,
            "decile_mono_above_05": eval_results["decile_monotonicity"] > 0.5,
            "d10_d1_positive": eval_results["d10_d1_spread"] > 0,
            "no_feature_over_15pct": max_feat_pct < 0.15,
        }
        all_passed = all(gates.values())

        if verbose:
            print(f"\n[Train] Quality Gates:")
            for gate, passed in gates.items():
                status = "✅" if passed else "❌"
                print(f"  {status} {gate}")
            print(f"  {'✅ ALL PASSED' if all_passed else '❌ NOT ALL PASSED'}")

        # ── Save model ────────────────────────────────────────
        metadata = {
            "model_type": "cross_sectional_regression",
            "training_date": datetime.now().isoformat(),
            "forward_horizon": forward_horizon,
            "n_training_samples": len(X_train),
            "n_validation_samples": len(X_val),
            "n_test_samples": len(X_test),
            "n_features": len(feature_cols),
            "n_dates_train": len(train_dates),
            "n_dates_test": len(test_dates),
            "best_iteration": model.best_iteration,
            "training_time_seconds": round(train_elapsed, 1),
            # Evaluation
            "mean_ic": eval_results["mean_ic"],
            "ic_std": eval_results["ic_std"],
            "ic_ir": eval_results["ic_ir"],
            "ic_hit_rate": eval_results["ic_hit_rate"],
            "decile_monotonicity": eval_results["decile_monotonicity"],
            "d10_d1_spread": eval_results["d10_d1_spread"],
            "avg_decile_returns": eval_results["avg_decile_returns"],
            "long_short_sharpe": eval_results.get("long_short_sharpe", 0),
            # Features
            "top_20_features": top_20,
            "max_feature_importance_pct": round(max_feat_pct, 4),
            "feature_group_importance": group_imp,
            "quality_gates": gates,
            "quality_gates_passed": all_passed,
        }

        # Save regardless of quality gates
        self.model_root.mkdir(parents=True, exist_ok=True)
        model.save_model(str(self.model_root / self.MODEL_FILE))

        with open(self.model_root / self.METADATA_FILE, "w") as f:
            json.dump(metadata, f, indent=2, default=str)

        with open(self.model_root / self.FEATURE_FILE, "w") as f:
            json.dump(feature_cols, f, indent=2)

        with self._lock:
            self.model = model
            self.metadata = metadata

        if verbose:
            print(f"\n[Train] Model saved to {self.model_root}")

        return metadata

    # ══════════════════════════════════════════════════════════
    # TEST EVALUATION
    # ══════════════════════════════════════════════════════════

    def _evaluate_test(
        self,
        predictions: np.ndarray,
        actuals: np.ndarray,
        meta: pd.DataFrame,
        verbose: bool = True,
    ) -> dict:
        """Compute IC, decile analysis, and Sharpe on the test set."""

        # Per-date IC
        dates = sorted(meta["date"].unique())
        daily_ics = []
        daily_ls_rets = []

        for dt in dates:
            mask = meta["date"].values == dt
            if mask.sum() < 20:
                continue

            preds_dt = predictions[mask]
            rets_dt = actuals[mask]

            try:
                ic, _ = sp_stats.spearmanr(preds_dt, rets_dt)
                if not np.isnan(ic):
                    daily_ics.append(ic)
            except Exception:
                continue

            # Long/short return
            n = len(preds_dt)
            n_decile = max(n // 10, 1)
            sorted_idx = np.argsort(preds_dt)
            short_ret = float(rets_dt[sorted_idx[:n_decile]].mean())
            long_ret = float(rets_dt[sorted_idx[-n_decile:]].mean())
            daily_ls_rets.append(long_ret - short_ret)

        mean_ic = float(np.mean(daily_ics)) if daily_ics else 0.0
        ic_std = float(np.std(daily_ics)) if daily_ics else 1.0
        ic_ir = mean_ic / ic_std if ic_std > 1e-8 else 0.0
        ic_hit_rate = float(np.mean([1 if ic > 0 else 0 for ic in daily_ics])) if daily_ics else 0.0

        # Sharpe of L/S
        if daily_ls_rets and len(daily_ls_rets) > 1:
            ls_sharpe = float(np.mean(daily_ls_rets) / (np.std(daily_ls_rets) + 1e-8) * np.sqrt(252))
        else:
            ls_sharpe = 0.0

        # Overall decile analysis
        n_total = len(predictions)
        ranks = sp_stats.rankdata(predictions) / n_total
        n_deciles = 10
        decile_rets = []
        for d in range(n_deciles):
            lo = d / n_deciles
            hi = (d + 1) / n_deciles
            if d == n_deciles - 1:
                hi = 1.01
            mask = (ranks >= lo) & (ranks < hi)
            if mask.sum() > 0:
                decile_rets.append(float(actuals[mask].mean()))
            else:
                decile_rets.append(0.0)

        d10_d1 = decile_rets[-1] - decile_rets[0] if len(decile_rets) >= 2 else 0.0

        # Monotonicity
        if len(decile_rets) >= 3:
            tau, _ = sp_stats.kendalltau(np.arange(len(decile_rets)), decile_rets)
            mono = float(tau) if not np.isnan(tau) else 0.0
        else:
            mono = 0.0

        if verbose:
            print(f"\n{'='*65}")
            print(f"TEST SET EVALUATION ({len(dates)} dates, {n_total:,} samples)")
            print(f"{'='*65}")
            print(f"  Mean IC:        {mean_ic:.4f}")
            print(f"  IC Std:         {ic_std:.4f}")
            print(f"  IC IR:          {ic_ir:.2f}")
            print(f"  IC Hit Rate:    {ic_hit_rate:.1%}")
            print(f"  L/S Sharpe:     {ls_sharpe:.2f}")
            print(f"  Decile Mono:    {mono:.3f}")
            print(f"  D10-D1 Spread:  {d10_d1:.6f}")
            print(f"\n  DECILE RETURNS (excess):")
            labels = ["D1 (SHORT)", "D2", "D3", "D4", "D5",
                      "D6", "D7", "D8", "D9", "D10 (LONG)"]
            for i, dr in enumerate(decile_rets):
                lbl = labels[i] if i < len(labels) else f"D{i+1}"
                bar = "█" * int(max(0, dr) * 10000) + "▒" * int(max(0, -dr) * 10000)
                print(f"    {lbl:12s}: {dr:+.6f} {bar}")

        return {
            "mean_ic": round(mean_ic, 4),
            "ic_std": round(ic_std, 4),
            "ic_ir": round(ic_ir, 2),
            "ic_hit_rate": round(ic_hit_rate, 3),
            "long_short_sharpe": round(ls_sharpe, 2),
            "decile_monotonicity": round(mono, 3),
            "d10_d1_spread": round(d10_d1, 6),
            "avg_decile_returns": [round(x, 6) for x in decile_rets],
            "n_test_dates": len(dates),
            "n_daily_ics": len(daily_ics),
        }

    # ══════════════════════════════════════════════════════════
    # PREDICTION
    # ══════════════════════════════════════════════════════════

    def predict(
        self,
        symbol: str,
        df: pd.DataFrame = None,
        cross_sectional_ranks: dict = None,
    ) -> dict:
        """
        Predict for a single symbol.

        Mode 1 — With cross_sectional_ranks: use directly as model input.
        Mode 2 — Without: compute raw features, approximate ranks from
                 the stock's own 252-day history.
        """
        if not self.is_ready():
            return {}

        with self._lock:
            model = self.model
            feature_names = self._feature_names

        if feature_names is None:
            return {}

        # Get data if needed
        if df is None:
            if self.polygon_data is None:
                try:
                    from ..data.polygon_universe import PolygonUniverseData
                    self.polygon_data = PolygonUniverseData()
                except Exception:
                    return {}
            df = self.polygon_data.get_stock_history(symbol, days=500)
            if df is None:
                return {}

        if len(df) < self.engine.MIN_ROWS:
            return {}

        try:
            if cross_sectional_ranks is not None:
                # Mode 1: Use provided ranks directly
                row = pd.DataFrame([cross_sectional_ranks])
                for col in feature_names:
                    if col not in row.columns:
                        row[col] = 0.5  # median rank
                row = row[feature_names].fillna(0.5)
            else:
                # Mode 2: Approximate with time-series ranks
                features = self.engine.compute_features(df)
                if features.empty:
                    return {}

                latest_raw = features.iloc[-1:]

                # Approximate cross-sectional rank using 252-day historical percentile
                lookback = min(252, len(features))
                hist = features.iloc[-lookback:]
                row = pd.DataFrame(index=[0])
                for col in feature_names:
                    if col in hist.columns:
                        val = latest_raw[col].iloc[0]
                        # Percentile rank within own history
                        rank_pct = (hist[col] < val).mean()
                        row[col] = float(rank_pct) if not np.isnan(rank_pct) else 0.5
                    else:
                        row[col] = 0.5

                row = row[feature_names].fillna(0.5).astype(np.float32)

            # Predict
            pred_excess = float(model.predict(row.values)[0])

            # Map to direction/confidence
            if pred_excess > 0.001:
                direction = "LONG"
            elif pred_excess < -0.001:
                direction = "SHORT"
            else:
                direction = "NEUTRAL"

            confidence = min(abs(pred_excess) / 0.02, 1.0)

            # Backward-compatible probabilities
            if pred_excess > 0:
                p_long = 0.5 + confidence * 0.4
                p_short = 0.5 - confidence * 0.4
            else:
                p_long = 0.5 - confidence * 0.4
                p_short = 0.5 + confidence * 0.4
            p_neutral = 1.0 - p_long - p_short
            p_neutral = max(0.0, p_neutral)

            top_features = []
            if self.metadata and "top_20_features" in self.metadata:
                for fi in self.metadata["top_20_features"][:5]:
                    fn = fi["feature"]
                    if fn in row.columns:
                        top_features.append({
                            "feature": fn,
                            "rank": round(float(row[fn].iloc[0]), 3),
                            "importance": fi["importance"],
                        })

            return {
                "symbol": symbol,
                "prediction": 2 if direction == "LONG" else 0 if direction == "SHORT" else 1,
                "direction": direction,
                "signal": direction,
                "confidence": round(confidence, 4),
                "predicted_excess_return": round(pred_excess, 6),
                "probabilities": {
                    "short": round(p_short, 4),
                    "neutral": round(p_neutral, 4),
                    "long": round(p_long, 4),
                },
                "explanation": {
                    "top_features": top_features,
                    "model_type": "cross_sectional_regression",
                    "training_date": self.metadata.get("training_date", "unknown") if self.metadata else "unknown",
                    "training_samples": self.metadata.get("n_training_samples", 0) if self.metadata else 0,
                    "test_ic": self.metadata.get("mean_ic", 0) if self.metadata else 0,
                    "test_ic_ir": self.metadata.get("ic_ir", 0) if self.metadata else 0,
                },
                "model_info": {
                    "model_type": "cross_sectional_regression",
                    "version": "6.0.0",
                    "n_features": len(feature_names),
                },
            }

        except Exception as e:
            logger.warning("CrossSectionalModel.predict(%s) failed: %s", symbol, e)
            return {}

    def score_universe(
        self,
        date: str = None,
        universe: List[str] = None,
        stock_histories: Dict[str, pd.DataFrame] = None,
        verbose: bool = True,
    ) -> pd.DataFrame:
        """
        Score ALL stocks in the active universe on a given date.

        1. Compute raw features for each stock
        2. RANK NORMALIZE across all stocks (cross-sectional)
        3. Predict excess return
        4. Sort by predicted return

        Returns DataFrame with columns:
          predicted_excess_return, rank, decile, direction, confidence
        """
        if not self.is_ready():
            return pd.DataFrame()

        if self.polygon_data is None:
            from ..data.polygon_universe import PolygonUniverseData
            self.polygon_data = PolygonUniverseData()

        if universe is None:
            universe = self.polygon_data.get_active_universe(
                min_price=5.0, min_dollar_volume=1_000_000,
            )

        if stock_histories is None:
            if verbose:
                print(f"[Score] Loading histories for {len(universe)} stocks...")
            stock_histories = self._load_histories_batched(universe, verbose=verbose)

        if verbose:
            print(f"[Score] Computing features for {len(stock_histories)} stocks...")

        # Compute raw features for all stocks
        raw_features = {}
        for ticker, df in stock_histories.items():
            if len(df) < self.engine.MIN_ROWS:
                continue
            try:
                feat = self.engine.compute_features(df)
                if not feat.empty:
                    raw_features[ticker] = feat.iloc[-1]  # Latest row
            except Exception:
                continue

        if len(raw_features) < 20:
            return pd.DataFrame()

        # Cross-sectional rank normalization
        cs_df = pd.DataFrame(raw_features).T
        cs_ranked = cs_df.rank(pct=True).fillna(0.5)

        # Predict for each stock
        feature_names = self._feature_names
        results = []
        for ticker in cs_ranked.index:
            ranks = cs_ranked.loc[ticker].to_dict()
            row = pd.DataFrame([{f: ranks.get(f, 0.5) for f in feature_names}])
            row = row.fillna(0.5).astype(np.float32)

            try:
                pred = float(self.model.predict(row.values)[0])
                results.append({
                    "ticker": ticker,
                    "predicted_excess_return": pred,
                })
            except Exception:
                continue

        if not results:
            return pd.DataFrame()

        result_df = pd.DataFrame(results).set_index("ticker")
        result_df = result_df.sort_values("predicted_excess_return", ascending=False)
        result_df["rank"] = range(1, len(result_df) + 1)
        result_df["decile"] = pd.qcut(
            result_df["predicted_excess_return"], 10, labels=False, duplicates="drop"
        ) + 1
        result_df["direction"] = result_df["predicted_excess_return"].apply(
            lambda x: "LONG" if x > 0.001 else "SHORT" if x < -0.001 else "NEUTRAL"
        )
        result_df["confidence"] = result_df["predicted_excess_return"].abs().clip(0, 0.02) / 0.02

        if verbose:
            print(f"[Score] {len(result_df)} stocks scored")
            print(f"  Top 5: {list(result_df.index[:5])}")
            print(f"  Bottom 5: {list(result_df.index[-5:])}")

        return result_df

    # ══════════════════════════════════════════════════════════
    # HELPERS
    # ══════════════════════════════════════════════════════════

    def _load_histories_batched(
        self,
        universe: List[str],
        batch_size: int = 300,
        verbose: bool = False,
    ) -> Dict[str, pd.DataFrame]:
        """Load stock histories in batches."""
        all_hist = {}
        for i in range(0, len(universe), batch_size):
            batch = universe[i:i + batch_size]
            histories = self.polygon_data.get_multi_stock_history(
                symbols=batch, days=600,
            )
            all_hist.update(histories)
            if verbose and (i + batch_size) % 600 == 0:
                print(f"  Loaded: {len(all_hist)} stocks")
            gc.collect()
        return all_hist

    @staticmethod
    def _forward_return(
        df: Optional[pd.DataFrame],
        date_str: str,
        horizon: int,
    ) -> Optional[float]:
        """Compute forward log return from history."""
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
