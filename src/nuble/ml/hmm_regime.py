"""
HMM Regime Detection
====================
Replaces hardcoded VIX thresholds with a learned Hidden Markov Model.
Trains on macro features (VIX, credit spread, term spread, market returns)
using 20+ years of data for robust state estimation.

3 States:
  - Bull:    Low VIX, positive term spread, strong returns
  - Neutral: Average macro conditions
  - Crisis:  High VIX, inverted/flat curve, negative returns

Usage:
    from nuble.ml.hmm_regime import get_regime_detector
    detector = get_regime_detector()
    regime = detector.detect_regime()
    # {'state': 'bull', 'state_id': 0, 'probabilities': [0.85, 0.12, 0.03]}
"""

import os
import logging
import pickle
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
_MODEL_PATH = os.path.join(_PROJECT_ROOT, "models", "regime", "hmm_regime_model.pkl")
_DATA_DIR = os.path.join(_PROJECT_ROOT, "data", "wrds")


class HMMRegimeDetector:
    """
    Hidden Markov Model for market regime detection.

    Uses macro features from the GKX panel:
      - VIX (implied volatility)
      - term_spread_10y2y (yield curve slope)
      - corp_spread_bbb (credit risk)
      - market returns (realized SPY-like returns)

    States are post-hoc labeled based on feature means:
      State with lowest VIX mean → bull
      State with highest VIX mean → crisis
      Other → neutral
    """

    def __init__(self):
        self._model = None
        self._scaler_mean = None
        self._scaler_std = None
        self._state_labels = {}
        self._feature_cols = ['vix', 'term_spread_10y2y', 'corp_spread_bbb',
                              'realized_vol', 'mom_1m']
        self._ready = False

    def train(self, force: bool = False):
        """Train HMM on historical macro data from GKX panel."""
        if self._ready and not force:
            return

        # Try loading pre-trained model first
        if os.path.exists(_MODEL_PATH) and not force:
            self._load_model()
            if self._ready:
                return

        # Train from panel data
        try:
            from hmmlearn.hmm import GaussianHMM
        except ImportError:
            logger.warning("hmmlearn not installed — regime detection unavailable")
            return

        logger.info("Training HMM regime model on historical macro data...")

        # Load macro features from panel
        panel_path = os.path.join(_DATA_DIR, "gkx_panel.parquet")
        if not os.path.exists(panel_path):
            logger.warning(f"GKX panel not found at {panel_path}")
            return

        try:
            panel = pd.read_parquet(
                panel_path,
                columns=['date'] + [c for c in self._feature_cols]
            )
        except Exception as e:
            logger.error(f"Failed to load panel for HMM training: {e}")
            return

        # Aggregate: macro vars are cross-sectional constant per month,
        # so take the first non-null value per date for macro vars,
        # and the median for stock-specific vars (realized_vol, mom_1m)
        panel['date'] = pd.to_datetime(panel['date'])
        macro_vars = ['vix', 'term_spread_10y2y', 'corp_spread_bbb']
        stock_vars = ['realized_vol', 'mom_1m']
        
        monthly_macro = panel.groupby(panel['date'].dt.to_period('M'))[macro_vars].first()
        monthly_stock = panel.groupby(panel['date'].dt.to_period('M'))[stock_vars].median()
        monthly = pd.concat([monthly_macro, monthly_stock], axis=1)
        monthly = monthly.dropna()
        
        # Filter to post-1990 (VIX only available from ~1990, earlier values are backfilled)
        monthly = monthly[monthly.index >= pd.Period('1990-01', freq='M')]
        logger.info(f"Training on {len(monthly)} months (1990+)")

        if len(monthly) < 60:
            logger.warning(f"Insufficient data for HMM training: {len(monthly)} months")
            return

        # Standardize
        X = monthly.values
        self._scaler_mean = X.mean(axis=0)
        self._scaler_std = X.std(axis=0) + 1e-8
        X_scaled = (X - self._scaler_mean) / self._scaler_std

        # Fit 3-state Gaussian HMM
        # Use diagonal covariance for more stable estimation with limited data
        model = GaussianHMM(
            n_components=3,
            covariance_type='diag',
            n_iter=500,
            random_state=42,
            tol=1e-6,
            init_params='stmc',
        )
        model.fit(X_scaled)

        self._model = model

        # Label states based on VIX means
        state_means = model.means_
        vix_idx = self._feature_cols.index('vix')

        vix_means = state_means[:, vix_idx]
        bull_state = int(np.argmin(vix_means))
        crisis_state = int(np.argmax(vix_means))
        neutral_state = [i for i in range(3) if i not in [bull_state, crisis_state]][0]

        self._state_labels = {
            bull_state: 'bull',
            neutral_state: 'neutral',
            crisis_state: 'crisis',
        }

        # Log state characteristics
        for state_id, label in self._state_labels.items():
            means = model.means_[state_id]
            raw_means = means * self._scaler_std + self._scaler_mean
            logger.info(
                f"  State {state_id} ({label}): "
                f"VIX={raw_means[vix_idx]:.1f}, "
                f"TermSpread={raw_means[1]:.2f}, "
                f"CreditSpread={raw_means[2]:.2f}"
            )

        self._ready = True

        # Save model
        self._save_model()
        logger.info(f"✅ HMM regime model trained on {len(monthly)} months of data")

    def detect_regime(self, macro_features: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Detect current market regime.

        Args:
            macro_features: Optional dict with current values for feature_cols.
                           If None, uses the latest panel data.

        Returns:
            Dict with state label, probabilities, and feature values.
        """
        if not self._ready:
            self.train()

        if not self._ready:
            return self._fallback_regime(macro_features)

        # Build feature vector
        if macro_features:
            X = np.array([[macro_features.get(f, 0.0) for f in self._feature_cols]])
        else:
            X = self._get_latest_features()
            if X is None:
                return self._fallback_regime(macro_features)

        # Standardize
        X_scaled = (X - self._scaler_mean) / self._scaler_std

        # Predict state
        state_id = int(self._model.predict(X_scaled)[0])
        state_label = self._state_labels.get(state_id, 'neutral')

        # State probabilities
        log_probs = self._model.score_samples(X_scaled)
        posteriors = self._model.predict_proba(X_scaled)[0]

        return {
            'state': state_label,
            'state_id': state_id,
            'probabilities': {
                self._state_labels.get(i, f'state_{i}'): round(float(p), 4)
                for i, p in enumerate(posteriors)
            },
            'features': {
                f: round(float(X[0, i]), 4) for i, f in enumerate(self._feature_cols)
            },
            'confidence': round(float(posteriors[state_id]), 4),
            'timestamp': datetime.now().isoformat(),
        }

    def get_vix_exposure(self, macro_features: Optional[Dict] = None) -> float:
        """
        Get VIX-scaled exposure factor based on detected regime.
        Replaces hardcoded thresholds in the ensemble.

        Returns:
          bull:    1.0 (full exposure)
          neutral: 0.8 (standard exposure)
          crisis:  0.3 (defensive)
        """
        regime = self.detect_regime(macro_features)
        exposure_map = {
            'bull': 1.0,
            'neutral': 0.8,
            'crisis': 0.3,
        }
        return exposure_map.get(regime['state'], 0.8)

    def _get_latest_features(self) -> Optional[np.ndarray]:
        """Get latest macro features from panel."""
        panel_path = os.path.join(_DATA_DIR, "gkx_panel.parquet")
        try:
            panel = pd.read_parquet(
                panel_path,
                columns=['date'] + self._feature_cols,
            )
            panel['date'] = pd.to_datetime(panel['date'])
            latest = panel[panel['date'] == panel['date'].max()]
            values = latest[self._feature_cols].median()
            return values.values.reshape(1, -1)
        except Exception as e:
            logger.error(f"Failed to get latest features: {e}")
            return None

    def _fallback_regime(self, macro_features: Optional[Dict] = None) -> Dict[str, Any]:
        """Rule-based fallback when HMM is unavailable."""
        vix = 20.0
        if macro_features:
            vix = macro_features.get('vix', 20.0)
        elif os.path.exists(os.path.join(_DATA_DIR, "gkx_panel.parquet")):
            try:
                panel = pd.read_parquet(
                    os.path.join(_DATA_DIR, "gkx_panel.parquet"),
                    columns=['date', 'vix'],
                )
                vix = panel['vix'].dropna().iloc[-1]
            except Exception:
                pass

        if vix < 15:
            state = 'bull'
        elif vix > 30:
            state = 'crisis'
        else:
            state = 'neutral'

        return {
            'state': state,
            'state_id': {'bull': 0, 'neutral': 1, 'crisis': 2}.get(state, 1),
            'probabilities': {'bull': 0.33, 'neutral': 0.34, 'crisis': 0.33},
            'features': {'vix': round(float(vix), 2)},
            'confidence': 0.5,
            'method': 'rule_based_fallback',
            'timestamp': datetime.now().isoformat(),
        }

    def _save_model(self):
        """Save trained HMM model to disk."""
        os.makedirs(os.path.dirname(_MODEL_PATH), exist_ok=True)
        state = {
            'model': self._model,
            'scaler_mean': self._scaler_mean,
            'scaler_std': self._scaler_std,
            'state_labels': self._state_labels,
            'feature_cols': self._feature_cols,
            'trained_at': datetime.now().isoformat(),
        }
        with open(_MODEL_PATH, 'wb') as f:
            pickle.dump(state, f)
        logger.info(f"HMM model saved to {_MODEL_PATH}")

    def _load_model(self):
        """Load pre-trained HMM model from disk."""
        try:
            with open(_MODEL_PATH, 'rb') as f:
                state = pickle.load(f)
            self._model = state['model']
            self._scaler_mean = state['scaler_mean']
            self._scaler_std = state['scaler_std']
            self._state_labels = state['state_labels']
            self._feature_cols = state.get('feature_cols', self._feature_cols)
            self._ready = True
            logger.info(f"HMM model loaded from {_MODEL_PATH}")
        except Exception as e:
            logger.warning(f"Failed to load HMM model: {e}")


# ── Singleton ──────────────────────────────────────────────────
_detector_instance: Optional[HMMRegimeDetector] = None


def get_regime_detector() -> HMMRegimeDetector:
    """Get or create the singleton HMMRegimeDetector."""
    global _detector_instance
    if _detector_instance is None:
        _detector_instance = HMMRegimeDetector()
    return _detector_instance
