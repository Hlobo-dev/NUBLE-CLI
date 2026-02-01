"""
Regime Detection Module
========================

Hidden Markov Models for market regime detection:
- HMM-based regime classification
- Regime filtering for trading
- Online regime prediction

These methods identify market states to adapt trading strategies.
"""

from .hmm_detector import (
    HMMRegimeDetector,
    RegimeConfig,
    RegimeState,
    create_regime_features,
    plot_regimes,
)

__all__ = [
    'HMMRegimeDetector',
    'RegimeConfig',
    'RegimeState',
    'create_regime_features',
    'plot_regimes',
]
