"""
Feature Engineering Module
==========================

Advanced feature engineering methods from Lopez de Prado (2018):
- Fractional Differentiation
- Feature importance
- Information-driven bars

These methods preserve predictive memory while achieving stationarity.
"""

from .frac_diff import (
    FractionalDifferentiator,
    frac_diff_ffd,
    find_min_ffd,
    get_weights_ffd,
    FracDiffConfig,
    plot_min_ffd,
    compute_memory_preservation,
)

__all__ = [
    'FractionalDifferentiator',
    'frac_diff_ffd',
    'find_min_ffd',
    'get_weights_ffd',
    'FracDiffConfig',
    'plot_min_ffd',
    'compute_memory_preservation',
]
