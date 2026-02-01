"""
KYPERIAN Multi-Asset Module

Universal asset handling for stocks, ETFs, crypto, and more.
"""

from .detector import AssetDetector, AssetClass
from .base import BaseAssetAnalyzer

# Optional imports - only load if modules exist
try:
    from .crypto_analyzer import CryptoAnalyzer
except ImportError:
    CryptoAnalyzer = None

__all__ = [
    'AssetDetector',
    'AssetClass',
    'BaseAssetAnalyzer',
    'CryptoAnalyzer',
]
