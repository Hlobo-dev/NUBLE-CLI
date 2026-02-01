"""
Base Asset Analyzer for KYPERIAN
Abstract base class for all asset type analyzers
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, List, Any
from dataclasses import dataclass
from datetime import datetime
from enum import Enum


class SignalType(Enum):
    """Trading signal types"""
    STRONG_LONG = 2
    LONG = 1
    NEUTRAL = 0
    SHORT = -1
    STRONG_SHORT = -2


@dataclass
class AnalysisResult:
    """Base analysis result"""
    symbol: str
    asset_class: str
    timestamp: datetime
    
    # Price info
    current_price: Optional[float] = None
    price_change: Optional[float] = None
    
    # Signal info
    signal: SignalType = SignalType.NEUTRAL
    signal_strength: float = 0.0
    confidence: float = 0.0
    
    # Metadata
    data_sources: List[str] = None
    notes: List[str] = None
    
    def __post_init__(self):
        if self.data_sources is None:
            self.data_sources = []
        if self.notes is None:
            self.notes = []
    
    def to_dict(self) -> Dict:
        return {
            "symbol": self.symbol,
            "asset_class": self.asset_class,
            "timestamp": self.timestamp.isoformat(),
            "current_price": self.current_price,
            "price_change": self.price_change,
            "signal": self.signal.name,
            "signal_strength": self.signal_strength,
            "confidence": self.confidence,
            "data_sources": self.data_sources,
            "notes": self.notes
        }


class BaseAssetAnalyzer(ABC):
    """
    Abstract base class for asset analyzers.
    
    Subclasses implement asset-specific analysis logic for:
    - Stocks
    - ETFs
    - Crypto
    - Commodities
    - Forex
    """
    
    def __init__(self, name: str = "BaseAnalyzer"):
        self.name = name
        self._cache = {}
    
    @abstractmethod
    def analyze(self, symbol: str) -> AnalysisResult:
        """
        Perform analysis on a symbol.
        
        Args:
            symbol: Asset symbol to analyze
            
        Returns:
            AnalysisResult with signal and data
        """
        pass
    
    @abstractmethod
    def get_price(self, symbol: str) -> Optional[float]:
        """Get current price for symbol"""
        pass
    
    @abstractmethod
    def get_historical(self, symbol: str, days: int = 30) -> Optional[List[Dict]]:
        """Get historical price data"""
        pass
    
    def analyze_multiple(self, symbols: List[str]) -> Dict[str, AnalysisResult]:
        """Analyze multiple symbols"""
        results = {}
        for symbol in symbols:
            try:
                results[symbol] = self.analyze(symbol)
            except Exception as e:
                print(f"Error analyzing {symbol}: {e}")
        return results
    
    def clear_cache(self):
        """Clear internal cache"""
        self._cache = {}
