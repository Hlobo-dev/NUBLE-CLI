"""
NUBLE DATA AGGREGATOR
========================

Unified data aggregation layer that pulls from ALL available sources:
- Polygon.io (real-time quotes, OHLCV, options, news)
- DynamoDB (LuxAlgo signals)
- FinBERT (sentiment)
- HMM (regime detection)

This module ensures the Ultimate Decision Engine has access to all data.
"""

import asyncio
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta, date
from typing import Any, Dict, List, Optional, Tuple
import json

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class MarketSnapshot:
    """Complete market snapshot for a symbol."""
    symbol: str
    timestamp: datetime
    
    # Price data
    current_price: float = 0
    bid: Optional[float] = None
    ask: Optional[float] = None
    volume_24h: int = 0
    change_pct: float = 0
    
    # Technical
    sma_20: Optional[float] = None
    sma_50: Optional[float] = None
    sma_200: Optional[float] = None
    rsi_14: Optional[float] = None
    macd: Optional[Dict] = None
    bollinger: Optional[Dict] = None
    atr_14: Optional[float] = None
    
    # Options
    put_call_ratio: Optional[float] = None
    unusual_calls: int = 0
    unusual_puts: int = 0
    max_pain: Optional[float] = None
    implied_volatility: Optional[float] = None
    
    # Sentiment
    news_sentiment: Optional[float] = None
    news_count: int = 0
    social_sentiment: Optional[float] = None
    
    # Regime
    regime: str = "UNKNOWN"
    regime_probability: float = 0.5
    volatility: float = 0.02
    
    # LuxAlgo signals
    luxalgo_weekly: Optional[str] = None
    luxalgo_daily: Optional[str] = None
    luxalgo_4h: Optional[str] = None
    luxalgo_aligned: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "timestamp": self.timestamp.isoformat(),
            "price": {
                "current": self.current_price,
                "bid": self.bid,
                "ask": self.ask,
                "volume_24h": self.volume_24h,
                "change_pct": self.change_pct,
            },
            "technical": {
                "sma_20": self.sma_20,
                "sma_50": self.sma_50,
                "sma_200": self.sma_200,
                "rsi_14": self.rsi_14,
                "macd": self.macd,
                "bollinger": self.bollinger,
                "atr_14": self.atr_14,
            },
            "options": {
                "put_call_ratio": self.put_call_ratio,
                "unusual_calls": self.unusual_calls,
                "unusual_puts": self.unusual_puts,
                "max_pain": self.max_pain,
                "implied_volatility": self.implied_volatility,
            },
            "sentiment": {
                "news": self.news_sentiment,
                "news_count": self.news_count,
                "social": self.social_sentiment,
            },
            "regime": {
                "state": self.regime,
                "probability": self.regime_probability,
                "volatility": self.volatility,
            },
            "luxalgo": {
                "weekly": self.luxalgo_weekly,
                "daily": self.luxalgo_daily,
                "4h": self.luxalgo_4h,
                "aligned": self.luxalgo_aligned,
            },
        }


class DataAggregator:
    """
    Unified data aggregator for all NUBLE data sources.
    
    Sources:
    - Polygon.io: Real-time quotes, OHLCV, options chain, news
    - DynamoDB: LuxAlgo signals from TradingView webhooks
    - FinBERT: NLP sentiment analysis
    - HMM: Regime detection
    
    Usage:
        aggregator = DataAggregator()
        await aggregator.initialize()
        
        snapshot = await aggregator.get_snapshot("BTCUSD")
        print(f"Current regime: {snapshot.regime}")
    """
    
    def __init__(self, polygon_api_key: str = None):
        """Initialize data aggregator."""
        self.polygon_api_key = polygon_api_key or os.getenv("POLYGON_API_KEY", "JHKwAdyIOeExkYOxh3LwTopmqqVVFeBY")
        
        # Lazy-loaded components
        self._polygon = None
        self._dynamodb = None
        self._signals_table = None
        self._finbert = None
        self._hmm = None
        
        self._initialized = False
        
        # Cache
        self._cache = {}
        self._cache_ttl = 60  # seconds
    
    async def initialize(self):
        """Initialize all data sources."""
        if self._initialized:
            return
        
        logger.info("Initializing Data Aggregator...")
        
        # Polygon.io
        if self.polygon_api_key:
            try:
                from institutional.providers.polygon import PolygonProvider
                self._polygon = PolygonProvider(self.polygon_api_key)
                logger.info("✅ Polygon.io initialized")
            except ImportError:
                logger.warning("⚠️ Polygon provider not available")
        
        # DynamoDB
        try:
            import boto3
            self._dynamodb = boto3.resource('dynamodb')
            self._signals_table = self._dynamodb.Table('nuble-production-signals')
            logger.info("✅ DynamoDB initialized")
        except Exception as e:
            logger.warning(f"⚠️ DynamoDB not available: {e}")
        
        self._initialized = True
        logger.info("Data Aggregator initialized")
    
    async def get_snapshot(self, symbol: str, use_cache: bool = True) -> MarketSnapshot:
        """
        Get complete market snapshot for a symbol.
        
        Aggregates data from all sources in parallel.
        """
        if not self._initialized:
            await self.initialize()
        
        # Check cache
        cache_key = f"snapshot_{symbol}"
        if use_cache and cache_key in self._cache:
            cached, ts = self._cache[cache_key]
            if (datetime.now(timezone.utc) - ts).total_seconds() < self._cache_ttl:
                return cached
        
        now = datetime.now(timezone.utc)
        snapshot = MarketSnapshot(symbol=symbol, timestamp=now)
        
        # Gather all data in parallel
        tasks = [
            self._get_price_data(symbol, snapshot),
            self._get_technical_data(symbol, snapshot),
            self._get_options_data(symbol, snapshot),
            self._get_sentiment_data(symbol, snapshot),
            self._get_regime_data(symbol, snapshot),
            self._get_luxalgo_signals(symbol, snapshot),
        ]
        
        await asyncio.gather(*tasks, return_exceptions=True)
        
        # Cache result
        self._cache[cache_key] = (snapshot, now)
        
        return snapshot
    
    async def _get_price_data(self, symbol: str, snapshot: MarketSnapshot):
        """Get real-time price data from Polygon."""
        if not self._polygon:
            return
        
        try:
            response = await self._polygon.get_quote(symbol)
            if response.success and response.data:
                quote = response.data
                snapshot.current_price = quote.price
                snapshot.bid = quote.bid
                snapshot.ask = quote.ask
                snapshot.volume_24h = quote.volume or 0
                snapshot.change_pct = quote.change_percent or 0
        except Exception as e:
            logger.debug(f"Error getting price data: {e}")
    
    async def _get_technical_data(self, symbol: str, snapshot: MarketSnapshot):
        """Calculate technical indicators from OHLCV data."""
        if not self._polygon:
            return
        
        try:
            end_date = date.today()
            start_date = end_date - timedelta(days=250)
            
            response = await self._polygon.get_historical(symbol, start_date, end_date, "1d")
            
            if not response.success or not response.data:
                return
            
            df = pd.DataFrame([{
                "open": bar.open,
                "high": bar.high,
                "low": bar.low,
                "close": bar.close,
                "volume": bar.volume,
            } for bar in response.data])
            
            if len(df) < 50:
                return
            
            close = df["close"]
            high = df["high"]
            low = df["low"]
            
            # SMAs
            snapshot.sma_20 = float(close.rolling(20).mean().iloc[-1])
            snapshot.sma_50 = float(close.rolling(50).mean().iloc[-1])
            if len(df) >= 200:
                snapshot.sma_200 = float(close.rolling(200).mean().iloc[-1])
            
            # RSI
            delta = close.diff()
            gain = delta.where(delta > 0, 0).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            snapshot.rsi_14 = float(rsi.iloc[-1])
            
            # MACD
            ema12 = close.ewm(span=12).mean()
            ema26 = close.ewm(span=26).mean()
            macd = ema12 - ema26
            signal_line = macd.ewm(span=9).mean()
            histogram = macd - signal_line
            
            snapshot.macd = {
                "value": float(macd.iloc[-1]),
                "signal": float(signal_line.iloc[-1]),
                "histogram": float(histogram.iloc[-1]),
                "bullish": bool(histogram.iloc[-1] > 0),
            }
            
            # Bollinger Bands
            ma20 = close.rolling(20).mean()
            std20 = close.rolling(20).std()
            
            snapshot.bollinger = {
                "upper": float(ma20.iloc[-1] + 2 * std20.iloc[-1]),
                "middle": float(ma20.iloc[-1]),
                "lower": float(ma20.iloc[-1] - 2 * std20.iloc[-1]),
                "position": "above" if close.iloc[-1] > ma20.iloc[-1] + 2 * std20.iloc[-1] else
                           "below" if close.iloc[-1] < ma20.iloc[-1] - 2 * std20.iloc[-1] else "inside",
            }
            
            # ATR
            tr1 = high - low
            tr2 = abs(high - close.shift())
            tr3 = abs(low - close.shift())
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr = tr.rolling(14).mean()
            snapshot.atr_14 = float(atr.iloc[-1])
            
            # Volatility (annualized)
            returns = close.pct_change()
            snapshot.volatility = float(returns.std() * np.sqrt(252))
            
            # Update current price if not set
            if snapshot.current_price == 0:
                snapshot.current_price = float(close.iloc[-1])
                
        except Exception as e:
            logger.debug(f"Error getting technical data: {e}")
    
    async def _get_options_data(self, symbol: str, snapshot: MarketSnapshot):
        """Get options flow data from Polygon."""
        if not self._polygon:
            return
        
        # Skip for crypto (no options)
        if symbol.endswith("USD"):
            return
        
        try:
            response = await self._polygon.get_unusual_options_activity(symbol)
            
            if response.success and response.data:
                unusual = response.data
                
                calls = [o for o in unusual if o.get("contract_type") == "call"]
                puts = [o for o in unusual if o.get("contract_type") == "put"]
                
                snapshot.unusual_calls = len(calls)
                snapshot.unusual_puts = len(puts)
                
                if calls or puts:
                    snapshot.put_call_ratio = len(puts) / len(calls) if len(calls) > 0 else None
                
                # Get average IV
                ivs = [o.get("implied_volatility") for o in unusual if o.get("implied_volatility")]
                if ivs:
                    snapshot.implied_volatility = sum(ivs) / len(ivs)
                    
        except Exception as e:
            logger.debug(f"Error getting options data: {e}")
    
    async def _get_sentiment_data(self, symbol: str, snapshot: MarketSnapshot):
        """Get sentiment from news using FinBERT."""
        if not self._polygon:
            return
        
        try:
            response = await self._polygon.get_news(symbol, limit=10)
            
            if not response.success or not response.data:
                return
            
            snapshot.news_count = len(response.data)
            
            # Get FinBERT
            if self._finbert is None:
                try:
                    from nuble.news.sentiment import SentimentAnalyzer
                    self._finbert = SentimentAnalyzer()
                except:
                    pass
            
            if self._finbert:
                headlines = [article.title for article in response.data]
                results = self._finbert.analyze_batch(headlines)
                
                if results:
                    avg_sentiment = sum(r.normalized_score for r in results) / len(results)
                    snapshot.news_sentiment = float(avg_sentiment)
            else:
                # Keyword-based fallback
                positive_keywords = ["surge", "beat", "upgrade", "record", "bullish", "growth"]
                negative_keywords = ["miss", "downgrade", "decline", "warning", "bearish", "risk"]
                
                positive_count = 0
                negative_count = 0
                
                for article in response.data:
                    title_lower = article.title.lower()
                    if any(kw in title_lower for kw in positive_keywords):
                        positive_count += 1
                    if any(kw in title_lower for kw in negative_keywords):
                        negative_count += 1
                
                total = positive_count + negative_count
                if total > 0:
                    snapshot.news_sentiment = (positive_count - negative_count) / total
                    
        except Exception as e:
            logger.debug(f"Error getting sentiment data: {e}")
    
    async def _get_regime_data(self, symbol: str, snapshot: MarketSnapshot):
        """Detect market regime using HMM."""
        if not self._polygon:
            return
        
        try:
            end_date = date.today()
            start_date = end_date - timedelta(days=365)
            
            response = await self._polygon.get_historical(symbol, start_date, end_date, "1d")
            
            if not response.success or not response.data:
                return
            
            closes = pd.Series([bar.close for bar in response.data])
            returns = closes.pct_change().dropna()
            
            if len(returns) < 100:
                return
            
            # Get HMM detector
            if self._hmm is None:
                try:
                    from institutional.regime.hmm_detector import HMMRegimeDetector
                    self._hmm = HMMRegimeDetector(n_regimes=3)
                except:
                    pass
            
            if self._hmm:
                if not getattr(self._hmm, 'is_fitted', False):
                    self._hmm.fit(returns)
                
                regime_idx = self._hmm.predict(returns).iloc[-1]
                regime_name = self._hmm.regime_names.get(regime_idx, "UNKNOWN")
                
                snapshot.regime = regime_name
                snapshot.regime_probability = 0.8
            else:
                # Simple regime detection fallback
                recent_returns = returns.tail(20).mean() * 252  # Annualized
                recent_vol = returns.tail(20).std() * np.sqrt(252)
                
                if recent_returns > 0.15 and recent_vol < 0.25:
                    snapshot.regime = "BULL"
                elif recent_returns < -0.10:
                    snapshot.regime = "BEAR"
                elif recent_vol > 0.35:
                    snapshot.regime = "VOLATILE"
                else:
                    snapshot.regime = "SIDEWAYS"
                    
        except Exception as e:
            logger.debug(f"Error getting regime data: {e}")
    
    async def _get_luxalgo_signals(self, symbol: str, snapshot: MarketSnapshot):
        """Get LuxAlgo signals from DynamoDB."""
        if not self._signals_table:
            return
        
        try:
            now = datetime.now(timezone.utc)
            
            timeframes = {
                "1W": ("luxalgo_weekly", 168),
                "1D": ("luxalgo_daily", 48),
                "4h": ("luxalgo_4h", 12),
            }
            
            signals = []
            
            for tf, (attr, max_age) in timeframes.items():
                response = self._signals_table.query(
                    KeyConditionExpression='pk = :pk AND begins_with(sk, :tf)',
                    ExpressionAttributeValues={
                        ':pk': f'SIGNAL#{symbol}',
                        ':tf': f'{tf}#',
                    },
                    ScanIndexForward=False,
                    Limit=1,
                )
                
                if response.get('Items'):
                    item = response['Items'][0]
                    ts = float(item.get('timestamp', 0))
                    if ts > 1e12:
                        ts = ts / 1000
                    
                    signal_time = datetime.fromtimestamp(ts, tz=timezone.utc)
                    age_hours = (now - signal_time).total_seconds() / 3600
                    
                    if age_hours <= max_age:
                        action = item.get('action', 'NEUTRAL')
                        setattr(snapshot, attr, action)
                        signals.append(action)
            
            # Check alignment
            if signals:
                all_bullish = all(a in ['BUY', 'STRONG_BUY'] for a in signals)
                all_bearish = all(a in ['SELL', 'STRONG_SELL'] for a in signals)
                snapshot.luxalgo_aligned = all_bullish or all_bearish
                
        except Exception as e:
            logger.debug(f"Error getting LuxAlgo signals: {e}")
    
    async def get_multi_snapshot(
        self, symbols: List[str], use_cache: bool = True
    ) -> Dict[str, MarketSnapshot]:
        """Get snapshots for multiple symbols in parallel."""
        tasks = [self.get_snapshot(symbol, use_cache) for symbol in symbols]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        return {
            symbol: result if not isinstance(result, Exception) else MarketSnapshot(symbol=symbol, timestamp=datetime.now(timezone.utc))
            for symbol, result in zip(symbols, results)
        }
    
    def clear_cache(self):
        """Clear the data cache."""
        self._cache = {}


# ============================================================
# REAL-TIME STREAMING
# ============================================================

class RealTimeDataStream:
    """
    Real-time data streaming using Polygon WebSocket.
    
    Usage:
        async with RealTimeDataStream(api_key) as stream:
            await stream.subscribe(['BTCUSD', 'AAPL'])
            
            async for tick in stream.ticks():
                print(f"{tick['symbol']}: ${tick['price']}")
    """
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("POLYGON_API_KEY", "JHKwAdyIOeExkYOxh3LwTopmqqVVFeBY")
        self._stream = None
        self._connected = False
        self._subscriptions = set()
    
    async def __aenter__(self):
        await self.connect()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.disconnect()
    
    async def connect(self):
        """Connect to Polygon WebSocket."""
        try:
            from institutional.streaming.realtime import PolygonStream
            self._stream = PolygonStream(self.api_key)
            await self._stream.connect()
            self._connected = True
            logger.info("Connected to Polygon WebSocket")
        except Exception as e:
            logger.error(f"Failed to connect: {e}")
            raise
    
    async def disconnect(self):
        """Disconnect from WebSocket."""
        if self._stream:
            await self._stream.disconnect()
        self._connected = False
    
    async def subscribe(self, symbols: List[str]):
        """Subscribe to symbols."""
        if not self._connected:
            raise RuntimeError("Not connected")
        
        for symbol in symbols:
            await self._stream.subscribe([symbol])
            self._subscriptions.add(symbol)
    
    async def ticks(self):
        """Async generator for real-time ticks."""
        if not self._connected:
            raise RuntimeError("Not connected")
        
        async for tick in self._stream.ticks():
            yield tick


# ============================================================
# CONVENIENCE FUNCTIONS
# ============================================================

async def get_snapshot(symbol: str) -> MarketSnapshot:
    """Quick function to get a snapshot."""
    aggregator = DataAggregator()
    await aggregator.initialize()
    return await aggregator.get_snapshot(symbol)


async def get_all_snapshots(symbols: List[str] = None) -> Dict[str, MarketSnapshot]:
    """Get snapshots for all watched symbols."""
    if symbols is None:
        symbols = ['BTCUSD', 'ETHUSD', 'SPY', 'QQQ', 'AAPL', 'TSLA', 'NVDA', 'AMD']
    
    aggregator = DataAggregator()
    await aggregator.initialize()
    return await aggregator.get_multi_snapshot(symbols)


if __name__ == "__main__":
    import asyncio
    import sys
    
    async def main():
        symbol = sys.argv[1] if len(sys.argv) > 1 else "BTCUSD"
        
        print(f"\n📊 NUBLE DATA AGGREGATOR")
        print(f"{'=' * 50}")
        print(f"Fetching snapshot for {symbol}...\n")
        
        snapshot = await get_snapshot(symbol)
        
        print(f"💰 Price: ${snapshot.current_price:,.2f}")
        print(f"   Change: {snapshot.change_pct:+.2f}%")
        
        print(f"\n📈 Technical:")
        print(f"   SMA 20: {snapshot.sma_20:,.2f}" if snapshot.sma_20 else "   SMA 20: N/A")
        print(f"   SMA 50: {snapshot.sma_50:,.2f}" if snapshot.sma_50 else "   SMA 50: N/A")
        print(f"   RSI 14: {snapshot.rsi_14:.1f}" if snapshot.rsi_14 else "   RSI 14: N/A")
        if snapshot.macd:
            print(f"   MACD: {'Bullish 📈' if snapshot.macd['bullish'] else 'Bearish 📉'}")
        
        print(f"\n🎯 Options:")
        print(f"   P/C Ratio: {snapshot.put_call_ratio:.2f}" if snapshot.put_call_ratio else "   P/C Ratio: N/A")
        print(f"   Unusual Calls: {snapshot.unusual_calls}")
        print(f"   Unusual Puts: {snapshot.unusual_puts}")
        
        print(f"\n📰 Sentiment:")
        print(f"   News Score: {snapshot.news_sentiment:+.2f}" if snapshot.news_sentiment else "   News Score: N/A")
        print(f"   Article Count: {snapshot.news_count}")
        
        print(f"\n🌍 Regime:")
        print(f"   State: {snapshot.regime}")
        print(f"   Volatility: {snapshot.volatility:.1%}")
        
        print(f"\n📡 LuxAlgo:")
        print(f"   Weekly: {snapshot.luxalgo_weekly or 'N/A'}")
        print(f"   Daily:  {snapshot.luxalgo_daily or 'N/A'}")
        print(f"   4H:     {snapshot.luxalgo_4h or 'N/A'}")
        print(f"   Aligned: {'Yes ✅' if snapshot.luxalgo_aligned else 'No ❌'}")
    
    asyncio.run(main())
