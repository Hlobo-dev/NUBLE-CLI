"""
NUBLE Decision Engine V2 - Institutional Grade
==================================================

Multi-layer decision engine that uses 15+ data points to make trading decisions.

Layers:
1. Signal Layer (40%) - Technical signals from multiple sources
2. Context Layer (30%) - Market regime, sentiment, volatility
3. Validation Layer (20%) - Historical win rate, backtest confidence
4. Risk Layer (10% + VETO) - Position limits, drawdown, correlation

Author: NUBLE ELITE
Version: 2.0.0
"""

from .engine_v2 import DecisionEngineV2, TradingDecision, TradeStrength
from .data_classes import (
    SignalLayerScore,
    ContextLayerScore,
    ValidationLayerScore,
    RiskLayerScore,
    Regime,
)

__all__ = [
    'DecisionEngineV2',
    'TradingDecision',
    'TradeStrength',
    'SignalLayerScore',
    'ContextLayerScore',
    'ValidationLayerScore',
    'RiskLayerScore',
    'Regime',
]
