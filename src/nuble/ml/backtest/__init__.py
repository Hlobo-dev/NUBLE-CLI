"""
NUBLE Walk-Forward Backtest & Signal Analysis
===============================================

Institutional-grade model validation framework.

- WalkForwardBacktest: Expanding-window walk-forward validation
- BacktestResults: Result container with analysis and plotting
- SignalAnalysis: Signal decay, factor exposure, turnover, regime analysis
"""

from .walk_forward import WalkForwardBacktest, BacktestResults
from .signal_analysis import SignalAnalysis

__all__ = [
    "WalkForwardBacktest",
    "BacktestResults",
    "SignalAnalysis",
]
