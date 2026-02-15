"""
Live Predictor — Real-Time Predictions Using Polygon + Trained WRDS Models
==========================================================================
Combines:
  1. PolygonFeatureEngine → computes ~160+ live features
  2. Trained LightGBM from System B → scores the features
  3. Universal Technical Model → daily timing signal
  4. Composite scoring: 70% fundamental (monthly) + 30% timing (daily)

Architecture:
  Polygon Live Data → PolygonFeatureEngine → same 280 feature names as WRDS
  → route by log_market_cap to correct tier model → fundamental_score
  → blend with timing_score → composite → signal

Fallback: When Polygon is unavailable, falls back to WRDS historical data
          via WRDSPredictor (Phase 1 bridge).

Usage:
    from nuble.ml.live_predictor import get_live_predictor
    lp = get_live_predictor()
    result = lp.predict('AAPL')
    # result has: composite_score, fundamental_score, timing_score,
    #             signal, confidence, feature_coverage, data_source, etc.
"""

import os
import logging
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List

logger = logging.getLogger(__name__)

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))

# ── Signal thresholds (decile-based, same as WRDSPredictor v2) ───────
_SIGNAL_THRESHOLDS = [
    (0.9, "D10", "STRONG_BUY"),
    (0.8, "D9",  "BUY"),
    (0.7, "D8",  "BUY"),
    (0.6, "D7",  "BUY"),
    (0.5, "D6",  "HOLD"),
    (0.4, "D5",  "HOLD"),
    (0.3, "D4",  "SELL"),
    (0.2, "D3",  "SELL"),
    (0.1, "D2",  "STRONG_SELL"),
    (0.0, "D1",  "STRONG_SELL"),
]


class LivePredictor:
    """
    Production predictor that uses live Polygon data.
    Falls back to historical WRDS data when Polygon is unavailable.

    Scoring:
      composite = 0.70 × fundamental_score + 0.30 × timing_score
      fundamental_score = LightGBM(live features from Polygon)
      timing_score      = universal_technical_model (P(up) - P(down))
    """

    def __init__(self):
        self._polygon_engine = None   # lazy
        self._wrds_predictor = None   # lazy
        self._timing_model = None     # lazy
        self._timing_features = None  # feature names for timing model

    # ─────────────────────────────────────────────────────────────
    # LAZY LOADERS
    # ─────────────────────────────────────────────────────────────

    def _get_polygon_engine(self):
        """Lazy-load PolygonFeatureEngine."""
        if self._polygon_engine is None:
            try:
                from nuble.data.polygon_feature_engine import PolygonFeatureEngine
                self._polygon_engine = PolygonFeatureEngine()
                logger.info("✅ PolygonFeatureEngine initialized")
            except Exception as e:
                logger.warning(f"PolygonFeatureEngine unavailable: {e}")
                self._polygon_engine = False  # sentinel: tried and failed
        return self._polygon_engine if self._polygon_engine is not False else None

    def _get_wrds_predictor(self):
        """Lazy-load WRDSPredictor (Phase 1 bridge)."""
        if self._wrds_predictor is None:
            from nuble.ml.wrds_predictor import get_wrds_predictor
            self._wrds_predictor = get_wrds_predictor()
        return self._wrds_predictor

    def _load_timing_model(self):
        """Load the universal_technical_model.txt for timing signals."""
        if self._timing_model is not None:
            return
        try:
            import lightgbm as lgb
            model_path = os.path.join(_PROJECT_ROOT, "models", "universal",
                                      "universal_technical_model.txt")
            if os.path.exists(model_path):
                self._timing_model = lgb.Booster(model_file=model_path)
                self._timing_features = self._timing_model.feature_name()
                logger.info(f"✅ Timing model loaded: {len(self._timing_features)} features")
            else:
                logger.warning(f"Timing model not found: {model_path}")
                self._timing_model = False  # sentinel
        except Exception as e:
            logger.warning(f"Timing model load failed: {e}")
            self._timing_model = False

    # ─────────────────────────────────────────────────────────────
    # MAIN PREDICTION
    # ─────────────────────────────────────────────────────────────

    def predict(self, ticker: str) -> Dict[str, Any]:
        """
        Get a live prediction for a ticker.

        Flow:
        1. Compute live features from Polygon
        2. Score with trained LightGBM (per-tier)
        3. Get timing signal from universal technical model
        4. Blend: 70% fundamental + 30% timing
        5. Compare to historical WRDS prediction for sanity check
        """
        engine = self._get_polygon_engine()

        if engine is None:
            # No Polygon → full fallback to WRDS historical
            logger.warning(f"No Polygon engine for {ticker}, falling back to WRDS")
            result = self._get_wrds_predictor().predict(ticker)
            result['data_source'] = 'wrds_historical_fallback'
            return result

        # Compute live features
        try:
            live_features = engine.compute_features(ticker)
        except Exception as e:
            logger.error(f"Polygon feature computation failed for {ticker}: {e}")
            result = self._get_wrds_predictor().predict(ticker)
            result['data_source'] = 'wrds_historical_fallback'
            return result

        if not live_features or len(live_features) < 10:
            logger.warning(f"Insufficient live features for {ticker} "
                           f"({len(live_features) if live_features else 0}), "
                           f"falling back to WRDS")
            result = self._get_wrds_predictor().predict(ticker)
            result['data_source'] = 'wrds_historical_fallback'
            return result

        # Determine tier from live market cap
        lmc = live_features.get('log_market_cap', 0)
        tier = self._classify_tier(lmc)

        # Get the trained tier model from WRDSPredictor
        wrds = self._get_wrds_predictor()
        wrds._ensure_loaded()
        model = wrds._models.get(tier)

        if model is None:
            logger.warning(f"No model for tier {tier}, falling back to WRDS")
            result = wrds.predict(ticker)
            result['data_source'] = 'wrds_historical_fallback'
            return result

        # Build feature vector matching model's expected features
        feature_names = wrds._feature_names.get(tier, model.feature_name())
        feature_vector = np.array(
            [live_features.get(f, 0.0) for f in feature_names],
            dtype=np.float64
        )
        feature_vector = np.nan_to_num(feature_vector.reshape(1, -1), nan=0.0)

        # ── Fundamental score (WRDS-trained model on live features) ──
        fundamental_score = float(model.predict(feature_vector)[0])

        # ── Timing score (universal technical model) ──
        timing_score = self._get_timing_score(ticker, live_features)

        # ── Composite: 70% fundamental + 30% timing ──
        composite_score = 0.70 * fundamental_score + 0.30 * timing_score

        # ── Historical comparison (sanity check) ──
        try:
            historical = wrds.predict(ticker)
            historical_score = historical.get('raw_score', 0) if 'error' not in historical else 0
        except Exception:
            historical_score = 0

        # ── Feature coverage ──
        total_model_features = len(feature_names)
        covered = sum(
            1 for f in feature_names
            if f in live_features and np.isfinite(live_features.get(f, np.nan))
        )
        coverage_pct = (covered / total_model_features * 100) if total_model_features > 0 else 0

        # ── Signal generation ──
        signal, decile, confidence = self._generate_signal(
            composite_score, coverage_pct, tier
        )

        # ── Feature importance drivers ──
        drivers = self._get_top_drivers(model, feature_vector, feature_names, live_features)

        # Market cap in millions
        mcap = live_features.get('market_cap', 0)
        mcap_millions = mcap / 1e6 if mcap > 0 else live_features.get('mktcap', 0) / 1e6

        return {
            'ticker': ticker,
            'tier': tier,
            'fundamental_score': round(fundamental_score, 6),
            'timing_score': round(timing_score, 6),
            'composite_score': round(composite_score, 6),
            'raw_score': round(composite_score, 6),
            'decile': decile,
            'signal': signal,
            'confidence': round(confidence, 4),
            'data_source': 'live_polygon',
            'feature_coverage': f"{covered}/{total_model_features} ({coverage_pct:.1f}%)",
            'feature_coverage_pct': round(coverage_pct, 1),
            'features_missing': [f for f in feature_names if f not in live_features],
            'historical_score': round(historical_score, 6),
            'score_drift': round(composite_score - historical_score, 6),
            'market_cap_millions': round(mcap_millions, 1),
            'sector': live_features.get('gsector'),
            'top_drivers': drivers,
            'macro_regime': wrds.get_market_regime(),
            'timestamp': datetime.now().isoformat(),
        }

    def predict_batch(self, tickers: List[str]) -> List[Dict[str, Any]]:
        """Predict multiple tickers, with rate limiting for Polygon API."""
        import time
        results = []
        for i, ticker in enumerate(tickers):
            try:
                result = self.predict(ticker)
                results.append(result)
            except Exception as e:
                results.append({'ticker': ticker, 'error': str(e)})
            # Rate limit: Polygon free tier = 5 req/min
            if i > 0 and i % 4 == 0:
                time.sleep(12)
        return results

    def get_live_top_picks(self, n: int = 10,
                           tier: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get top N stock picks with live data.
        For large universes, falls back to WRDS historical for ranking
        then re-scores top candidates with live data.
        """
        # Get historical top picks as candidates
        wrds = self._get_wrds_predictor()
        candidates = wrds.get_top_picks(n * 3, tier=tier)

        if not candidates:
            return []

        # Re-score top candidates with live data
        live_results = []
        for candidate in candidates[:n * 2]:
            ticker = candidate.get('ticker', '')
            if not ticker:
                continue
            try:
                live = self.predict(ticker)
                if 'error' not in live:
                    live_results.append(live)
            except Exception:
                live_results.append(candidate)

        # Sort by composite score descending
        live_results.sort(key=lambda x: x.get('composite_score',
                                               x.get('raw_score', 0)),
                          reverse=True)
        return live_results[:n]

    # ─────────────────────────────────────────────────────────────
    # TIMING MODEL
    # ─────────────────────────────────────────────────────────────

    def _get_timing_score(self, ticker: str, features: Dict[str, float]) -> float:
        """
        Score using universal_technical_model.txt.
        Multiclass model → P(up) - P(down) as continuous signal.
        Falls back to 0.0 if model unavailable.
        """
        self._load_timing_model()

        if self._timing_model is False or self._timing_model is None:
            return 0.0

        try:
            vector = np.array(
                [features.get(f, 0.0) for f in self._timing_features],
                dtype=np.float64
            )
            vector = np.nan_to_num(vector.reshape(1, -1), nan=0.0)
            prediction = self._timing_model.predict(vector)

            if prediction.ndim == 2 and prediction.shape[1] == 3:
                # Multiclass: [P(down), P(flat), P(up)]
                probs = prediction[0]
                return float(probs[2] - probs[0])
            elif prediction.ndim == 2 and prediction.shape[1] > 1:
                # Other multiclass
                probs = prediction[0]
                return float(probs[-1] - probs[0])
            else:
                # Regression output
                return float(prediction.ravel()[0])
        except Exception as e:
            logger.debug(f"Timing score failed for {ticker}: {e}")
            return 0.0

    # ─────────────────────────────────────────────────────────────
    # HELPERS
    # ─────────────────────────────────────────────────────────────

    @staticmethod
    def _classify_tier(log_market_cap: float) -> str:
        """Classify stock into tier by log_market_cap."""
        if log_market_cap >= 9.21:
            return "mega"
        elif log_market_cap >= 7.60:
            return "large"
        elif log_market_cap >= 6.21:
            return "mid"
        elif log_market_cap >= 4.61:
            return "small"
        else:
            return "small"  # micro → treat as small

    def _generate_signal(self, score: float, coverage_pct: float,
                         tier: str) -> tuple:
        """
        Generate trading signal from composite score.
        Lower confidence if feature coverage is poor.
        """
        # Base confidence from score extremity
        confidence = min(abs(score) * 5, 0.95)

        # Penalize low coverage
        if coverage_pct < 30:
            confidence *= 0.3
        elif coverage_pct < 50:
            confidence *= 0.6
        elif coverage_pct < 70:
            confidence *= 0.8

        # Determine decile (map score percentile to D1-D10)
        # For live scoring, use absolute thresholds based on typical score ranges
        if score > 0.05:
            decile, signal = "D10", "STRONG_BUY"
        elif score > 0.03:
            decile, signal = "D9", "BUY"
        elif score > 0.02:
            decile, signal = "D8", "BUY"
        elif score > 0.01:
            decile, signal = "D7", "BUY"
        elif score > 0.0:
            decile, signal = "D6", "HOLD"
        elif score > -0.01:
            decile, signal = "D5", "HOLD"
        elif score > -0.02:
            decile, signal = "D4", "SELL"
        elif score > -0.03:
            decile, signal = "D3", "SELL"
        elif score > -0.05:
            decile, signal = "D2", "STRONG_SELL"
        else:
            decile, signal = "D1", "STRONG_SELL"

        return signal, decile, confidence

    @staticmethod
    def _get_top_drivers(model, feature_vector, feature_names, live_features,
                         top_n: int = 5) -> List[Dict]:
        """Get top feature importance drivers for this prediction."""
        try:
            importances = model.feature_importance(importance_type='gain')
            if len(importances) != len(feature_names):
                return []
            total_imp = importances.sum()
            if total_imp == 0:
                return []
            pairs = sorted(
                zip(feature_names, importances),
                key=lambda x: -x[1]
            )
            drivers = []
            for name, imp in pairs[:top_n]:
                val = live_features.get(name, None)
                drivers.append({
                    'feature': name,
                    'importance_pct': round(imp / total_imp * 100, 1),
                    'live_value': round(val, 6) if val is not None else None,
                    'available': name in live_features,
                })
            return drivers
        except Exception:
            return []


# ── Singleton ──────────────────────────────────────────────────
_live_predictor_instance: Optional[LivePredictor] = None


def get_live_predictor() -> LivePredictor:
    """Get or create the singleton LivePredictor instance."""
    global _live_predictor_instance
    if _live_predictor_instance is None:
        _live_predictor_instance = LivePredictor()
    return _live_predictor_instance
