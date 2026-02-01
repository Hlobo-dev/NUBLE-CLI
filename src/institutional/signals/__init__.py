"""
Enhanced Signals Module

Multi-timeframe, regime-adaptive signal generation.
"""

from .enhanced_signals import (
    EnhancedSignalGenerator,
    EnhancedSignal,
    SignalStrength,
    CrossAssetMomentum
)

__all__ = [
    'EnhancedSignalGenerator',
    'EnhancedSignal',
    'SignalStrength',
    'CrossAssetMomentum'
]
