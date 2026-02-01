"""
KYPERIAN News Intelligence Module

Real-time news processing with sentiment analysis.
Integrates with ML trading signals for enhanced predictions.
"""

from .client import StockNewsClient
from .sentiment import SentimentAnalyzer
from .pipeline import NewsPipeline
from .integrator import NewsSignalIntegrator

__all__ = [
    'StockNewsClient',
    'SentimentAnalyzer', 
    'NewsPipeline',
    'NewsSignalIntegrator'
]
