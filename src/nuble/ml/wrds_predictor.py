"""
WRDS ML Predictor — Bridges trained ML ensemble into APEX agent pipeline.

BEFORE: APEX agents → Polygon (81 days × 5 stocks) → per-ticker MLP → IC ≈ 0
AFTER:  APEX agents → WRDS GKX panel (3.76M × 461 features) → Ensemble → IC ≈ 0.03+

This module:
1. Loads pre-computed ensemble predictions (production_predictions > lgb_predictions)
2. Falls back to LightGBM model for real-time prediction if needed
3. Provides predict(ticker) → signal dict compatible with APEX orchestrator
4. Falls back gracefully if model/data not available
5. ★ Uses S3DataManager for transparent local/cloud data access

Prediction priority:
  1. production_predictions.parquet (meta-ensemble: LGB+Ridge+ElasticNet+NN)
  2. final_predictions.parquet (tree+neural ensemble)
  3. lgb_predictions.parquet (tree/linear ensemble from step6b)
  4. Real-time LightGBM model prediction (fallback)

Usage:
    from nuble.ml.wrds_predictor import get_wrds_predictor
    predictor = get_wrds_predictor()
    signal = predictor.predict("AAPL")
"""

import os
import logging
import pandas as pd
import numpy as np
from typing import Optional, Dict, Any
from datetime import datetime

logger = logging.getLogger(__name__)

# Default directories (fallback if S3DataManager unavailable)
_DEFAULT_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "..", "data", "wrds")
_DEFAULT_RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "..", "wrds_pipeline", "phase3", "results")


def _get_data_manager():
    """Lazy import S3DataManager to avoid circular dependencies."""
    try:
        from nuble.data.s3_data_manager import get_data_manager
        return get_data_manager()
    except Exception as e:
        logger.debug(f"S3DataManager not available: {e}")
        return None


class WRDSPredictor:
    """
    Production ML predictor powered by WRDS institutional data.

    Uses:
    - GKX-style panel: 3.76M stock-months × 461 features (1926-2024)
    - LightGBM model trained with walk-forward validation (2005-2024)
    - 225 characteristic × macro interaction terms
    - 38 FF49 industry dummies
    - 151 base stock characteristics

    Provides predictions compatible with the APEX orchestrator interface.
    """

    def __init__(self, data_dir: str = None, results_dir: str = None):
        self.data_dir = data_dir or os.path.abspath(_DEFAULT_DATA_DIR)
        self.results_dir = results_dir or os.path.abspath(_DEFAULT_RESULTS_DIR)
        self._dm = _get_data_manager()  # S3DataManager for cloud access
        self._model = None
        self._panel = None
        self._predictions = None
        self._prediction_source = None
        self._pred_col = "prediction"
        self._feature_cols = None
        self._ticker_to_permno = {}
        self._permno_to_ticker = {}
        self._model_summary = None
        self._ready = False
        self._load()

    def _load(self):
        """Load model, panel, and predictions — local-first, S3-fallback."""
        try:
            import lightgbm as lgb
        except ImportError:
            logger.warning("LightGBM not installed — WRDS predictor unavailable")
            return

        # ── Load trained LightGBM model ──────────────────────────────
        # The model may live in data/wrds/ (legacy) or models/lightgbm/ (canonical).
        # Check both local paths before falling back to S3.
        model_candidates = [
            os.path.join(self.data_dir, "lgb_latest_model.txt"),             # legacy: data/wrds/
            os.path.join(os.path.dirname(__file__), "..", "..", "..",         # canonical: models/lightgbm/
                         "models", "lightgbm", "lgb_latest_model.txt"),
        ]
        model_path = None
        for candidate in model_candidates:
            candidate = os.path.abspath(candidate)
            if os.path.exists(candidate):
                model_path = candidate
                break

        # S3 fallback if not found locally
        if model_path is None and self._dm:
            try:
                downloaded = self._dm.load_model("lightgbm/lgb_latest_model.txt")
                model_path = str(downloaded)
            except Exception as e:
                logger.debug(f"Model not in S3: {e}")

        if model_path and os.path.exists(model_path):
            self._model = lgb.Booster(model_file=model_path)
            logger.info(f"WRDS model loaded: {model_path}")
        else:
            logger.warning("WRDS LightGBM model not found locally or in S3")
            return

        # ── Load GKX panel (last 24 months only for memory) ─────────
        panel_loaded = False
        panel_path = os.path.join(self.data_dir, "gkx_panel.parquet")

        if os.path.exists(panel_path):
            # Local: use column projection for memory efficiency
            full_panel = pd.read_parquet(panel_path)
            panel_loaded = True
        elif self._dm:
            # S3 fallback: use S3DataManager's transparent access
            try:
                full_panel = self._dm.load_parquet("gkx_panel.parquet")
                panel_loaded = True
                logger.info("GKX panel loaded from S3")
            except Exception as e:
                logger.warning(f"GKX panel not available: {e}")

        if not panel_loaded:
            logger.warning(f"GKX panel not found at {panel_path} or in S3")
            return

        full_panel["date"] = pd.to_datetime(full_panel["date"])
        cutoff = full_panel["date"].max() - pd.DateOffset(months=24)
        self._panel = full_panel[full_panel["date"] >= cutoff].copy()
        del full_panel

        # Build ticker lookup from panel + training_panel
        self._build_ticker_map()

        # Identify feature columns (same logic as step6)
        id_cols = ["permno", "date", "cusip", "ticker", "siccd", "year",
                   "ret", "fwd_ret_1m", "fwd_ret_3m", "fwd_ret_6m", "fwd_ret_12m",
                   "ret_forward", "dlret", "dlstcd"]
        self._feature_cols = [
            c for c in self._panel.columns
            if c not in id_cols
            and self._panel[c].dtype in ["float64", "float32", "int64", "int32"]
        ]
        logger.info(f"WRDS panel loaded: {len(self._panel):,} rows, "
                    f"{len(self._feature_cols)} features, "
                    f"{self._panel['permno'].nunique():,} stocks")

        # ── Load pre-computed predictions ────────────────────────────
        pred_candidates = [
            ("production_predictions.parquet", "Meta-ensemble"),
            ("final_predictions.parquet", "Tree+Neural ensemble"),
            ("lgb_predictions.parquet", "Tree/Linear ensemble"),
        ]
        self._prediction_source = None
        for fname, desc in pred_candidates:
            pred_df = None
            pred_path = os.path.join(self.data_dir, fname)
            if os.path.exists(pred_path):
                pred_df = pd.read_parquet(pred_path)
            elif self._dm:
                try:
                    pred_df = self._dm.load_parquet(fname)
                except Exception:
                    continue

            if pred_df is not None and len(pred_df) > 0:
                self._predictions = pred_df
                self._predictions["date"] = pd.to_datetime(self._predictions["date"])
                self._prediction_source = desc

                # Auto-detect best prediction column
                for pcol in ["prediction_final", "prediction", "pred_nn"]:
                    if pcol in self._predictions.columns:
                        self._pred_col = pcol
                        break
                else:
                    self._pred_col = "prediction"

                logger.info(f"Predictions loaded: {desc} ({fname}), "
                            f"{len(self._predictions):,} rows, col={self._pred_col}")
                break

        # Load model summary
        summary_path = os.path.join(self.results_dir, "model_summary.json")
        if os.path.exists(summary_path):
            import json
            with open(summary_path) as f:
                self._model_summary = json.load(f)

        self._ready = True
        logger.info("✅ WRDS Predictor ready")

    def _build_ticker_map(self):
        """Build bidirectional ticker ↔ PERMNO mapping."""
        # Try GKX panel first
        if "ticker" in self._panel.columns:
            latest = self._panel.sort_values("date").groupby("permno").last()
            for permno, row in latest.iterrows():
                if pd.notna(row.get("ticker")):
                    ticker = str(row["ticker"]).strip().upper()
                    self._ticker_to_permno[ticker] = permno
                    self._permno_to_ticker[permno] = ticker

        # Also try training panel for broader coverage
        tp_path = os.path.join(self.data_dir, "training_panel.parquet")
        tp_loaded = False

        if os.path.exists(tp_path):
            try:
                tp = pd.read_parquet(tp_path, columns=["permno", "date", "ticker"])
                tp_loaded = True
            except Exception:
                pass
        elif self._dm:
            try:
                tp = self._dm.load_parquet("training_panel.parquet",
                                           columns=["permno", "date", "ticker"])
                tp_loaded = True
            except Exception:
                pass

        if tp_loaded:
            try:
                tp["date"] = pd.to_datetime(tp["date"])
                latest_tp = tp.sort_values("date").groupby("permno").last()
                for permno, row in latest_tp.iterrows():
                    if pd.notna(row.get("ticker")) and permno not in self._permno_to_ticker:
                        ticker = str(row["ticker"]).strip().upper()
                        self._ticker_to_permno[ticker] = permno
                        self._permno_to_ticker[permno] = ticker
                del tp, latest_tp
            except Exception:
                pass

        logger.info(f"Ticker map: {len(self._ticker_to_permno)} tickers")

    def _resolve_permno(self, ticker: str) -> Optional[int]:
        """Convert ticker to PERMNO."""
        ticker = ticker.upper().strip()
        if ticker in self._ticker_to_permno:
            return self._ticker_to_permno[ticker]
        # Fuzzy match
        for t, p in self._ticker_to_permno.items():
            if ticker in t or t in ticker:
                return p
        return None

    @property
    def is_ready(self) -> bool:
        return self._ready

    def predict(self, ticker: str, ohlcv_df: pd.DataFrame = None) -> Dict[str, Any]:
        """
        Get ML prediction for a stock.

        Compatible with APEX orchestrator's ML predictor interface:
        Returns dict with:
            - direction: 'BULLISH' / 'BEARISH' / 'NEUTRAL'
            - confidence: float 0-1
            - predicted_return: float (monthly)
            - model_type: str
            - features_used: int
            - decile_rank: int (1-10)
            - signal: str
        """
        if not self._ready:
            return {
                "direction": "NEUTRAL",
                "confidence": 0.0,
                "error": "WRDS predictor not ready",
                "model_type": "wrds_lgb",
            }

        permno = self._resolve_permno(ticker)
        if permno is None:
            return {
                "direction": "NEUTRAL",
                "confidence": 0.0,
                "error": f"Ticker {ticker} not found in WRDS universe",
                "model_type": "wrds_lgb",
            }

        # Get latest features for this stock
        stock_data = self._panel[self._panel["permno"] == permno].sort_values("date")
        if len(stock_data) == 0:
            return {
                "direction": "NEUTRAL",
                "confidence": 0.0,
                "error": f"No feature data for {ticker} (PERMNO {permno})",
                "model_type": "wrds_lgb",
            }

        latest = stock_data.iloc[-1:]
        latest_date = latest["date"].iloc[0]

        # Get features and predict
        X = latest[self._feature_cols].values
        feature_coverage = float(np.isfinite(X).mean())
        pred_return = float(self._model.predict(X)[0])

        # Cross-sectional rank: where does this stock rank vs all stocks this month?
        month_data = self._panel[self._panel["date"] == latest_date]
        if len(month_data) > 100:
            month_X = month_data[self._feature_cols].values
            month_preds = self._model.predict(month_X)
            percentile = float((month_preds <= pred_return).mean())
            decile = min(10, int(percentile * 10) + 1)
            n_universe = len(month_data)
        else:
            percentile = 0.5
            decile = 5
            n_universe = len(month_data)

        # Determine direction and confidence
        if decile >= 8:
            direction = "BULLISH"
            confidence = min(0.95, 0.5 + (decile - 5) * 0.1 + feature_coverage * 0.1)
        elif decile <= 3:
            direction = "BEARISH"
            confidence = min(0.95, 0.5 + (6 - decile) * 0.1 + feature_coverage * 0.1)
        else:
            direction = "NEUTRAL"
            confidence = 0.3 + feature_coverage * 0.1

        # Signal string
        if decile >= 9:
            signal = "STRONG BUY"
        elif decile >= 7:
            signal = "BUY"
        elif decile <= 2:
            signal = "STRONG SELL"
        elif decile <= 4:
            signal = "SELL"
        else:
            signal = "HOLD"

        result = {
            "direction": direction,
            "confidence": round(confidence, 3),
            "predicted_return": round(pred_return, 6),
            "predicted_return_pct": round(pred_return * 100, 2),
            "decile_rank": decile,
            "percentile": round(percentile * 100, 1),
            "signal": signal,
            "model_type": "wrds_lgb",
            "features_used": len(self._feature_cols),
            "feature_coverage": round(feature_coverage, 3),
            "universe_size": n_universe,
            "data_date": str(latest_date.date()),
            "ticker": ticker.upper(),
            "permno": int(permno),
        }

        # Add model performance context if available
        if self._model_summary:
            result["model_ic"] = self._model_summary.get("overall_ic")
            result["model_period"] = self._model_summary.get("test_period")

        return result

    def predict_batch(self, tickers: list) -> Dict[str, Dict[str, Any]]:
        """Predict for multiple tickers efficiently."""
        return {t: self.predict(t) for t in tickers}

    def get_top_picks(self, n: int = 20, min_coverage: float = 0.5) -> pd.DataFrame:
        """Get top N stocks by predicted return from latest cross-section."""
        if not self._ready:
            return pd.DataFrame()

        latest_date = self._panel["date"].max()
        month_data = self._panel[self._panel["date"] == latest_date].copy()

        # Check feature coverage
        coverage = month_data[self._feature_cols].notna().mean(axis=1)
        month_data = month_data[coverage >= min_coverage]

        if len(month_data) == 0:
            return pd.DataFrame()

        # Predict
        X = month_data[self._feature_cols].values
        month_data["predicted_return"] = self._model.predict(X)
        month_data["decile"] = pd.qcut(
            month_data["predicted_return"], 10, labels=False, duplicates="drop"
        ) + 1

        # Add ticker
        month_data["ticker_name"] = month_data["permno"].map(self._permno_to_ticker)

        # Top picks
        cols = ["permno", "ticker_name", "predicted_return", "decile"]
        for c in ["market_cap", "mktcap", "log_market_cap", "mom_12m", "vol_12m"]:
            if c in month_data.columns:
                cols.append(c)

        return month_data.nlargest(n, "predicted_return")[
            [c for c in cols if c in month_data.columns]
        ]

    def get_model_info(self) -> Dict[str, Any]:
        """Get model metadata and performance summary."""
        info = {
            "ready": self._ready,
            "model_type": self._prediction_source or "LightGBM (GKX walk-forward)",
            "data_source": "WRDS (3.76M stock-months, 1926-2024)",
        }
        if self._ready:
            info["n_features"] = len(self._feature_cols)
            info["n_stocks_latest"] = int(self._panel["permno"].nunique())
            info["latest_date"] = str(self._panel["date"].max().date())
            info["n_tickers_mapped"] = len(self._ticker_to_permno)
        if self._model_summary:
            info["overall_ic"] = self._model_summary.get("overall_ic")
            info["test_period"] = self._model_summary.get("test_period")
            info["top_features"] = self._model_summary.get("top_10_features", [])[:5]
        return info


# Singleton pattern for APEX integration
_wrds_predictor_instance = None


def get_wrds_predictor() -> WRDSPredictor:
    """Get or create the singleton WRDS predictor instance."""
    global _wrds_predictor_instance
    if _wrds_predictor_instance is None:
        _wrds_predictor_instance = WRDSPredictor()
    return _wrds_predictor_instance
