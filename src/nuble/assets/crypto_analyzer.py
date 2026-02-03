"""
NUBLE Crypto Asset Analyzer
Combines market data + news sentiment for crypto assets
"""

import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum

# Import our clients
import sys
sys.path.insert(0, "/Users/humbertolobo/Desktop/bolt.new-main/NUBLE-CLI")

from src.nuble.news.crypto_client import CryptoNewsClient
from src.nuble.news.coindesk_client import CoinDeskClient
from src.nuble.news.sentiment import FinBERTAnalyzer


class CryptoSignal(Enum):
    """Trading signal strength"""
    STRONG_LONG = 2
    LONG = 1
    NEUTRAL = 0
    SHORT = -1
    STRONG_SHORT = -2


@dataclass
class CryptoAnalysis:
    """Complete crypto analysis result"""
    symbol: str
    timestamp: datetime
    
    # Price data
    current_price: Optional[float]
    price_change_24h: Optional[float]
    
    # Sentiment data
    news_sentiment: float  # -1 to 1
    sentiment_confidence: float  # 0 to 1
    news_count: int
    
    # Combined signal
    signal: CryptoSignal
    signal_strength: float  # -1 to 1
    
    # Raw data
    news_headlines: List[str]
    analysis_notes: List[str]


class CryptoAnalyzer:
    """
    Unified Crypto Analyzer
    
    Combines:
    - CryptoNews API for news data
    - CoinDesk API for price data
    - FinBERT for sentiment analysis
    """
    
    def __init__(
        self,
        crypto_news_key: str = None,
        coindesk_key: str = None,
        load_finbert: bool = True
    ):
        # Initialize clients
        self.crypto_news = CryptoNewsClient(crypto_news_key)
        self.coindesk = CoinDeskClient(coindesk_key)
        
        # Initialize FinBERT (optional for testing)
        self.finbert = None
        if load_finbert:
            try:
                self.finbert = FinBERTAnalyzer()
            except Exception as e:
                print(f"⚠️ FinBERT not loaded: {e}")
        
        # Signal thresholds
        self.strong_threshold = 0.6
        self.weak_threshold = 0.2
        
        # Crypto symbol mappings
        self.symbol_map = {
            "BTC": ["BTC", "Bitcoin", "BTCUSD"],
            "ETH": ["ETH", "Ethereum", "ETHUSD"],
            "SOL": ["SOL", "Solana", "SOLUSD"],
            "XRP": ["XRP", "Ripple", "XRPUSD"],
            "ADA": ["ADA", "Cardano", "ADAUSD"],
            "DOGE": ["DOGE", "Dogecoin", "DOGEUSD"],
            "DOT": ["DOT", "Polkadot", "DOTUSD"],
            "AVAX": ["AVAX", "Avalanche", "AVAXUSD"],
            "MATIC": ["MATIC", "Polygon", "MATICUSD"],
            "LINK": ["LINK", "Chainlink", "LINKUSD"]
        }
    
    def analyze(self, symbol: str) -> CryptoAnalysis:
        """
        Complete analysis for a crypto asset
        
        Args:
            symbol: Crypto symbol (BTC, ETH, etc.)
            
        Returns:
            CryptoAnalysis with all data and signals
        """
        notes = []
        headlines = []
        
        # Normalize symbol
        symbol = symbol.upper().replace("USD", "").replace("-USD", "")
        notes.append(f"Analyzing {symbol}")
        
        # 1. Get price data
        current_price = None
        price_change = None
        
        price_data = self.coindesk.get_current_price(symbol)
        if price_data and price_data.get("price"):
            current_price = price_data["price"]
            notes.append(f"Price: ${current_price:,.2f}")
        
        # 2. Get news
        news_items = []
        try:
            news_data = self.crypto_news.get_ticker_news(symbol, items=10)
            if news_data and news_data.get("data"):
                news_items = news_data["data"]
                notes.append(f"Found {len(news_items)} news articles")
        except Exception as e:
            notes.append(f"News fetch error: {e}")
        
        # 3. Extract headlines
        for item in news_items:
            title = item.get("title", "")
            if title:
                headlines.append(title)
        
        # 4. Sentiment analysis
        sentiment_score = 0.0
        sentiment_confidence = 0.0
        
        if headlines and self.finbert:
            sentiments = []
            confidences = []
            
            for headline in headlines[:10]:  # Limit to 10
                result = self.finbert.analyze(headline)
                if result:
                    sentiments.append(result["compound_score"])
                    confidences.append(result["confidence"])
            
            if sentiments:
                sentiment_score = np.mean(sentiments)
                sentiment_confidence = np.mean(confidences)
                notes.append(f"Sentiment: {sentiment_score:.3f} (conf: {sentiment_confidence:.2f})")
        elif headlines:
            # Simple keyword-based fallback
            sentiment_score = self._keyword_sentiment(headlines)
            sentiment_confidence = 0.5
            notes.append(f"Keyword sentiment: {sentiment_score:.3f}")
        
        # 5. Generate signal
        signal, signal_strength = self._calculate_signal(
            sentiment_score,
            sentiment_confidence,
            price_change
        )
        notes.append(f"Signal: {signal.name} ({signal_strength:.3f})")
        
        return CryptoAnalysis(
            symbol=symbol,
            timestamp=datetime.now(),
            current_price=current_price,
            price_change_24h=price_change,
            news_sentiment=sentiment_score,
            sentiment_confidence=sentiment_confidence,
            news_count=len(headlines),
            signal=signal,
            signal_strength=signal_strength,
            news_headlines=headlines,
            analysis_notes=notes
        )
    
    def analyze_multiple(self, symbols: List[str]) -> Dict[str, CryptoAnalysis]:
        """Analyze multiple crypto assets"""
        results = {}
        for symbol in symbols:
            try:
                results[symbol] = self.analyze(symbol)
            except Exception as e:
                print(f"Error analyzing {symbol}: {e}")
        return results
    
    def _keyword_sentiment(self, headlines: List[str]) -> float:
        """Fallback keyword-based sentiment"""
        positive_words = [
            "surge", "soar", "rally", "bullish", "gain", "rise", "jump",
            "breakthrough", "adoption", "partnership", "launch", "success",
            "upgrade", "milestone", "record", "growth", "positive", "optimistic"
        ]
        
        negative_words = [
            "crash", "plunge", "dump", "bearish", "drop", "fall", "decline",
            "hack", "scam", "fraud", "ban", "regulation", "lawsuit", "fail",
            "loss", "risk", "warning", "concern", "negative", "pessimistic"
        ]
        
        pos_count = 0
        neg_count = 0
        
        text = " ".join(headlines).lower()
        
        for word in positive_words:
            if word in text:
                pos_count += 1
        
        for word in negative_words:
            if word in text:
                neg_count += 1
        
        total = pos_count + neg_count
        if total == 0:
            return 0.0
        
        return (pos_count - neg_count) / total
    
    def _calculate_signal(
        self,
        sentiment: float,
        confidence: float,
        price_change: Optional[float]
    ) -> Tuple[CryptoSignal, float]:
        """Calculate trading signal from sentiment and price"""
        
        # Weight sentiment by confidence
        weighted_sentiment = sentiment * confidence
        
        # Incorporate price momentum if available
        if price_change is not None:
            # Normalize price change (assuming ±10% is significant)
            norm_price = np.clip(price_change / 10, -1, 1)
            # Combine with sentiment (60% sentiment, 40% momentum)
            combined = 0.6 * weighted_sentiment + 0.4 * norm_price
        else:
            combined = weighted_sentiment
        
        # Determine signal
        if combined >= self.strong_threshold:
            return CryptoSignal.STRONG_LONG, combined
        elif combined >= self.weak_threshold:
            return CryptoSignal.LONG, combined
        elif combined <= -self.strong_threshold:
            return CryptoSignal.STRONG_SHORT, combined
        elif combined <= -self.weak_threshold:
            return CryptoSignal.SHORT, combined
        else:
            return CryptoSignal.NEUTRAL, combined
    
    def get_market_overview(self) -> Dict[str, Any]:
        """Get overview of major crypto assets"""
        symbols = ["BTC", "ETH", "SOL", "XRP", "ADA"]
        
        print("🔍 Fetching market overview...")
        
        # Get prices
        prices = {}
        bpi = self.coindesk.get_bitcoin_index()
        if bpi:
            prices["BTC"] = bpi.get("bpi", {}).get("USD", {}).get("rate_float")
        
        # Get general crypto news
        general_news = None
        try:
            general_news = self.crypto_news.get_general_news(items=20)
        except:
            pass
        
        # Analyze top assets
        analyses = {}
        for symbol in symbols[:3]:  # Top 3 to avoid rate limits
            try:
                analyses[symbol] = self.analyze(symbol)
            except Exception as e:
                print(f"  {symbol}: Error - {e}")
        
        return {
            "timestamp": datetime.now().isoformat(),
            "bitcoin_price": prices.get("BTC"),
            "analyses": analyses,
            "general_news_count": len(general_news.get("data", [])) if general_news else 0
        }


def test_crypto_analyzer():
    """Test the crypto analyzer"""
    print("=" * 70)
    print("🚀 NUBLE CRYPTO ANALYZER TEST")
    print("=" * 70)
    
    # Initialize without FinBERT for quick test
    print("\n📦 Initializing analyzer...")
    analyzer = CryptoAnalyzer(load_finbert=False)
    
    # Test individual analysis
    print("\n" + "=" * 70)
    print("📊 INDIVIDUAL ASSET ANALYSIS")
    print("=" * 70)
    
    for symbol in ["BTC", "ETH", "SOL"]:
        print(f"\n🪙 {symbol} Analysis:")
        print("-" * 50)
        
        try:
            result = analyzer.analyze(symbol)
            
            if result.current_price:
                print(f"   💰 Price: ${result.current_price:,.2f}")
            else:
                print(f"   💰 Price: Not available")
            
            print(f"   📰 News: {result.news_count} articles")
            print(f"   📊 Sentiment: {result.news_sentiment:.3f}")
            print(f"   🎯 Signal: {result.signal.name}")
            print(f"   💪 Strength: {result.signal_strength:.3f}")
            
            if result.news_headlines:
                print(f"\n   📰 Latest Headlines:")
                for headline in result.news_headlines[:3]:
                    print(f"      • {headline[:70]}...")
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    print("\n" + "=" * 70)
    print("✅ CRYPTO ANALYZER TEST COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    test_crypto_analyzer()
