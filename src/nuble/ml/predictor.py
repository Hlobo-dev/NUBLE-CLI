#!/usr/bin/env python3
"""
PROMPT F4 — MLPredictor: Production-Grade Prediction Interface
==============================================================

Thread-safe, lazy-loading prediction wrapper around TrainingPipeline.
Discovers the latest model on disk, caches it, and exposes a simple
predict(symbol, df) → dict API that includes SHAP explanations.

Design goals:
1. Zero-crash guarantee: every public method returns a safe default on error.
2. Thread-safe: all mutable state guarded by a lock.
3. Lazy loading: models loaded on first call, not at import time.
4. Auto-retrain detection: if a newer model dir exists on disk, hot-swap.
5. SHAP explanations: every prediction returns top-5 feature attributions.
"""

from __future__ import annotations

import logging
import os
import threading
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional

import pandas as pd

if TYPE_CHECKING:
    from .trainer_v2 import TrainingPipeline

logger = logging.getLogger(__name__)

# ── Lazy import of TrainingPipeline (avoid circular deps) ─────
_TrainingPipeline = None


def _get_training_pipeline():
    """Lazy import to avoid circular dependency at module load."""
    global _TrainingPipeline
    if _TrainingPipeline is None:
        from .trainer_v2 import TrainingPipeline
        _TrainingPipeline = TrainingPipeline
    return _TrainingPipeline


# ═══════════════════════════════════════════════════════════════
# MLPredictor
# ═══════════════════════════════════════════════════════════════

class MLPredictor:
    """
    Production prediction interface.

    Usage::

        predictor = MLPredictor(model_dir="models/")
        result = predictor.predict("SPY", df_recent_ohlcv)
        # result = {
        #     "symbol": "SPY",
        #     "prediction": 1,          # 0=down, 1=neutral, 2=up
        #     "direction": "UP",
        #     "confidence": 0.73,
        #     "probabilities": {"proba_down": 0.10, "proba_neutral": 0.17, "proba_up": 0.73},
        #     "explanation": {"top_features": [...], "base_value": 0.31},
        #     "model_info": {...},
        # }
    """

    _DIRECTION_MAP = {0: "DOWN", 1: "NEUTRAL", 2: "UP"}
    _BINARY_DIRECTION_MAP = {0: "SKIP", 1: "TAKE"}

    def __init__(
        self,
        model_dir: str = "models/",
        auto_reload: bool = True,
    ):
        """
        Args:
            model_dir:   Root directory containing saved model subdirs.
            auto_reload: If True, check for newer model dirs on each call.
        """
        self.model_dir = model_dir
        self.auto_reload = auto_reload

        # Thread safety
        self._lock = threading.Lock()

        # Cached pipelines: symbol → TrainingPipeline
        self._pipelines: Dict[str, "TrainingPipeline"] = {}
        # Track loaded model paths for hot-swap detection
        self._loaded_paths: Dict[str, str] = {}

    # ── Public API ────────────────────────────────────────────

    def predict(self, symbol: str, df: pd.DataFrame) -> dict:
        """
        Make a prediction for *symbol* using the latest data in *df*.

        Args:
            symbol: Ticker (e.g. "SPY").
            df:     Recent OHLCV DataFrame (≥60 rows recommended).

        Returns:
            dict with keys: symbol, prediction, direction, confidence,
            probabilities, explanation, model_info.
            On any error, returns a safe dict with confidence=0.
        """
        try:
            pipeline = self._get_pipeline(symbol)
            if pipeline is None:
                return self._empty_result(symbol, reason="no model found")

            result = pipeline.predict_latest(df)

            # Augment with human-readable direction
            pred = result.get("prediction", 1)
            binary = result.get("model_info", {}).get("binary_mode", False)
            if binary:
                result["direction"] = self._BINARY_DIRECTION_MAP.get(pred, "NEUTRAL")
            else:
                result["direction"] = self._DIRECTION_MAP.get(pred, "NEUTRAL")

            return result

        except Exception as exc:
            logger.warning("MLPredictor.predict(%s) failed: %s", symbol, exc)
            return self._empty_result(symbol, reason=str(exc))

    def predict_batch(
        self, symbols: List[str], df_map: Dict[str, pd.DataFrame]
    ) -> Dict[str, dict]:
        """
        Batch prediction for multiple symbols.

        Args:
            symbols:  List of tickers.
            df_map:   {symbol: DataFrame} mapping.

        Returns:
            {symbol: prediction_dict} mapping.
        """
        results = {}
        for sym in symbols:
            df = df_map.get(sym)
            if df is not None and not df.empty:
                results[sym] = self.predict(sym, df)
            else:
                results[sym] = self._empty_result(sym, reason="no data")
        return results

    def has_model(self, symbol: str) -> bool:
        """Check whether a trained model exists for *symbol*."""
        return self._find_latest_model_dir(symbol) is not None

    def loaded_symbols(self) -> List[str]:
        """Return list of symbols with loaded models."""
        with self._lock:
            return list(self._pipelines.keys())

    def get_model_info(self, symbol: str) -> dict:
        """Return metadata for the loaded model (or empty dict)."""
        pipeline = self._get_pipeline(symbol)
        if pipeline is None:
            return {}
        return {
            "symbol": pipeline.symbol,
            "training_date": pipeline._training_date,
            "binary_mode": pipeline.binary_mode,
            "model_dir": self._loaded_paths.get(symbol, ""),
        }

    # ── Internal helpers ──────────────────────────────────────

    def _get_pipeline(self, symbol: str) -> Optional["TrainingPipeline"]:
        """Load or retrieve cached pipeline. Thread-safe."""
        with self._lock:
            # Hot-swap check: is there a newer model on disk?
            if self.auto_reload and symbol in self._pipelines:
                latest = self._find_latest_model_dir(symbol)
                if latest and latest != self._loaded_paths.get(symbol):
                    logger.info(
                        "Hot-swapping model for %s: %s → %s",
                        symbol,
                        self._loaded_paths.get(symbol, "?"),
                        latest,
                    )
                    self._pipelines.pop(symbol, None)
                    self._loaded_paths.pop(symbol, None)

            if symbol in self._pipelines:
                return self._pipelines[symbol]

            # Try to load from disk
            model_dir = self._find_latest_model_dir(symbol)
            if model_dir is None:
                return None

            try:
                TrainingPipeline = _get_training_pipeline()
                pipeline = TrainingPipeline.load(model_dir)
                self._pipelines[symbol] = pipeline
                self._loaded_paths[symbol] = model_dir
                logger.info("Loaded model for %s from %s", symbol, model_dir)
                return pipeline
            except Exception as exc:
                logger.error("Failed to load model for %s from %s: %s", symbol, model_dir, exc)
                return None

    def _find_latest_model_dir(self, symbol: str) -> Optional[str]:
        """
        Find the most recent model directory for *symbol*.

        Naming convention: ``{model_dir}/{SYMBOL}_{timestamp}/``
        """
        root = Path(self.model_dir)
        if not root.exists():
            return None

        prefix = f"{symbol.upper()}_"
        candidates = sorted(
            [
                d
                for d in root.iterdir()
                if d.is_dir() and d.name.startswith(prefix)
            ],
            key=lambda d: d.name,
            reverse=True,
        )

        for candidate in candidates:
            meta = candidate / "metadata.json"
            model = candidate / "model.txt"
            if meta.exists() and model.exists():
                return str(candidate)

        return None

    @staticmethod
    def _empty_result(symbol: str, reason: str = "") -> dict:
        """Return a safe, zero-confidence result."""
        return {
            "symbol": symbol,
            "prediction": 1,   # neutral
            "direction": "NEUTRAL",
            "confidence": 0.0,
            "probabilities": {},
            "explanation": {"top_features": [], "base_value": 0.0},
            "model_info": {"error": reason},
        }


# ═══════════════════════════════════════════════════════════════
# Module-level singleton
# ═══════════════════════════════════════════════════════════════

_predictor_instance: Optional[MLPredictor] = None
_predictor_lock = threading.Lock()


def get_predictor(model_dir: str = "models/") -> MLPredictor:
    """
    Get or create the global MLPredictor singleton.

    Thread-safe.  Call this from orchestrator / manager.
    """
    global _predictor_instance
    with _predictor_lock:
        if _predictor_instance is None:
            _predictor_instance = MLPredictor(model_dir=model_dir)
        return _predictor_instance
