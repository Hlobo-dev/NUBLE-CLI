"""
Institutional Backtesting Framework
====================================

Production-grade backtesting with walk-forward optimization.

Modules:
- engine: Core backtesting engine
- analytics: Performance attribution and risk analytics
"""

from .engine import (
    BacktestEngine,
    BacktestConfig,
    BacktestResults,
    Portfolio,
    Position,
    Broker,
    Order,
    OrderType,
    OrderSide,
    OrderStatus,
    Fill,
    MarketEvent,
    Strategy,
    ModelStrategy,
    TransactionCostModel,
    WalkForwardOptimizer,
    MonteCarloSimulator,
)

__all__ = [
    'BacktestEngine',
    'BacktestConfig',
    'BacktestResults',
    'Portfolio',
    'Position',
    'Broker',
    'Order',
    'OrderType',
    'OrderSide',
    'OrderStatus',
    'Fill',
    'MarketEvent',
    'Strategy',
    'ModelStrategy',
    'TransactionCostModel',
    'WalkForwardOptimizer',
    'MonteCarloSimulator',
]
