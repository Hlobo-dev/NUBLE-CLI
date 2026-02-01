"""
Beta Hedging Module

Dynamic beta hedging to achieve market-neutral exposure.
"""

from .beta_hedge import (
    DynamicBetaHedge,
    MultiAssetBetaHedge,
    HedgeConfig,
    HedgeState
)

__all__ = [
    'DynamicBetaHedge',
    'MultiAssetBetaHedge', 
    'HedgeConfig',
    'HedgeState'
]
