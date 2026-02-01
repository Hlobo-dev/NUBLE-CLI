"""
Labeling Module for Financial ML
=================================

Implements advanced labeling methods from Lopez de Prado (2018):
- Triple Barrier Method
- Meta-Labeling
- Event-driven labeling

These methods transform raw price data into meaningful labels that
simulate realistic trading conditions with profit targets and stop losses.
"""

from .triple_barrier import (
    TripleBarrierLabeler,
    TripleBarrierConfig,
    BarrierEvent,
    apply_triple_barrier,
    get_daily_volatility,
)

__all__ = [
    'TripleBarrierLabeler',
    'TripleBarrierConfig',
    'BarrierEvent',
    'apply_triple_barrier',
    'get_daily_volatility',
]
