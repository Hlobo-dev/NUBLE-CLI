"""
News Pipeline

Combines news fetching with sentiment analysis.
Real-time processing with caching and deduplication.
"""

import asyncio
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Set
import logging
from dataclasses import dataclass, field
import json
from pathlib import Path

from .client import StockNewsClient
from .sentiment import SentimentAnalyzer

logger = logging.getLogger(__name__)


@dataclass
class NewsSignal:
    """Trading signal derived from news."""
    timestamp: datetime
    symbol: str
    sentiment_score: float  # -1 to +1
    confidence: float  # 0 to 1
    headline: str
    source: str
    article_count: int
    actionable: bool
    signal_type: str  # 'BULLISH', 'BEARISH', 'NEUTRAL'
    
    def to_dict(self) -> Dict:
        return {
            'timestamp': self.timestamp.isoformat(),
            'symbol': self.symbol,
            'sentiment_score': self.sentiment_score,
            'confidence': self.confidence,
            'headline': self.headline,
            'source': self.source,
            'article_count': self.article_count,
            'actionable': self.actionable,
            'signal_type': self.signal_type
        }


class NewsPipeline:
    """
    Complete news processing pipeline.
    
    Features:
    - Real-time news fetching
    - FinBERT sentiment analysis
    - Signal generation
    - Deduplication
    - Historical storage
    
    Usage:
        pipeline = NewsPipeline()
        await pipeline.start()
        
        # Get latest signal for a symbol
        signal = await pipeline.get_signal('AAPL')
        
        # Stream signals
        async for signal in pipeline.stream_signals(['AAPL', 'MSFT', 'NVDA']):
            print(f"{signal.symbol}: {signal.signal_type}")
    """
    
    def __init__(
        self,
        symbols: Optional[List[str]] = None,
        poll_interval: int = 60,  # seconds
        storage_path: Optional[str] = None
    ):
        """
        Initialize news pipeline.
        
        Args:
            symbols: List of symbols to monitor
            poll_interval: How often to fetch news (seconds)
            storage_path: Path to store historical signals
        """
        self.symbols = symbols or []
        self.poll_interval = poll_interval
        self.storage_path = Path(storage_path) if storage_path else None
        
        # Components
        self.news_client = StockNewsClient()
        self.sentiment_analyzer = SentimentAnalyzer()
        
        # State
        self.seen_articles: Set[str] = set()
        self.signals: Dict[str, NewsSignal] = {}  # Latest signal per symbol
        self.signal_history: Dict[str, List[NewsSignal]] = {}  # Historical signals
        
        # Control
        self._running = False
        self._task: Optional[asyncio.Task] = None
        
        logger.info(f"NewsPipeline initialized for {len(self.symbols)} symbols")
    
    async def start(self):
        """Start the pipeline."""
        if self._running:
            logger.warning("Pipeline already running")
            return
        
        self._running = True
        self._task = asyncio.create_task(self._run_loop())
        logger.info("NewsPipeline started")
    
    async def stop(self):
        """Stop the pipeline."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        await self.news_client.close()
        logger.info("NewsPipeline stopped")
    
    async def _run_loop(self):
        """Main processing loop."""
        while self._running:
            try:
                await self._process_cycle()
            except Exception as e:
                logger.error(f"Error in news pipeline: {e}")
            
            await asyncio.sleep(self.poll_interval)
    
    async def _process_cycle(self):
        """Single processing cycle."""
        if not self.symbols:
            return
        
        # Fetch news for all symbols
        articles = await self.news_client.get_ticker_news(
            self.symbols,
            items=50,
            date_range='last60min'
        )
        
        # Filter new articles
        new_articles = []
        for article in articles:
            article_id = article.get('news_url', '') or article.get('title', '')
            if article_id not in self.seen_articles:
                self.seen_articles.add(article_id)
                new_articles.append(article)
        
        if not new_articles:
            return
        
        logger.info(f"Processing {len(new_articles)} new articles")
        
        # Analyze sentiment
        enriched = self.sentiment_analyzer.analyze_news_articles(new_articles)
        
        # Group by symbol and generate signals
        symbol_articles: Dict[str, List[Dict]] = {}
        for article in enriched:
            tickers = article.get('tickers', [])
            for ticker in tickers:
                if ticker in self.symbols:
                    if ticker not in symbol_articles:
                        symbol_articles[ticker] = []
                    symbol_articles[ticker].append(article)
        
        # Generate signal for each symbol
        for symbol, articles in symbol_articles.items():
            signal = self._generate_signal(symbol, articles)
            if signal:
                self.signals[symbol] = signal
                
                # Store in history
                if symbol not in self.signal_history:
                    self.signal_history[symbol] = []
                self.signal_history[symbol].append(signal)
                
                # Persist if configured
                if self.storage_path:
                    await self._persist_signal(signal)
                
                logger.info(f"Signal: {symbol} -> {signal.signal_type} ({signal.sentiment_score:+.2f})")
    
    def _generate_signal(
        self,
        symbol: str,
        articles: List[Dict]
    ) -> Optional[NewsSignal]:
        """Generate trading signal from articles."""
        if not articles:
            return None
        
        # Get aggregate sentiment
        aggregate = self.sentiment_analyzer.get_aggregate_sentiment(articles)
        
        # Get most impactful headline
        sorted_articles = sorted(
            articles,
            key=lambda a: abs(a.get('ml_sentiment', {}).get('sentiment_score', 0)),
            reverse=True
        )
        top_article = sorted_articles[0]
        
        # Determine if actionable
        actionable = (
            abs(aggregate['aggregate_score']) > 0.3 and
            aggregate['confidence'] > 0.6 and
            len(articles) >= 2  # Multiple sources confirm
        )
        
        return NewsSignal(
            timestamp=datetime.utcnow(),
            symbol=symbol,
            sentiment_score=aggregate['aggregate_score'],
            confidence=aggregate['confidence'],
            headline=top_article.get('title', 'Unknown'),
            source=top_article.get('source_name', 'Unknown'),
            article_count=len(articles),
            actionable=actionable,
            signal_type=aggregate['signal']
        )
    
    async def _persist_signal(self, signal: NewsSignal):
        """Save signal to storage."""
        if not self.storage_path:
            return
        
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Daily file
        date_str = signal.timestamp.strftime('%Y-%m-%d')
        file_path = self.storage_path / f"signals_{date_str}.jsonl"
        
        with open(file_path, 'a') as f:
            f.write(json.dumps(signal.to_dict()) + '\n')
    
    # ==========================================
    # PUBLIC API
    # ==========================================
    
    async def get_signal(self, symbol: str) -> Optional[NewsSignal]:
        """Get latest signal for a symbol."""
        return self.signals.get(symbol)
    
    async def get_all_signals(self) -> Dict[str, NewsSignal]:
        """Get all current signals."""
        return self.signals.copy()
    
    async def fetch_and_analyze(
        self,
        symbol: str,
        lookback_hours: int = 24
    ) -> NewsSignal:
        """
        Fetch and analyze news for a symbol on-demand.
        
        Args:
            symbol: Ticker symbol
            lookback_hours: How far back to look
            
        Returns:
            NewsSignal with sentiment analysis
        """
        # Determine date range
        if lookback_hours <= 1:
            date_range = 'last60min'
        elif lookback_hours <= 24:
            date_range = 'today'
        elif lookback_hours <= 48:
            date_range = 'yesterday'
        else:
            date_range = 'last7days'
        
        # Fetch news
        articles = await self.news_client.get_ticker_news(
            [symbol],
            items=50,
            date_range=date_range
        )
        
        if not articles:
            return NewsSignal(
                timestamp=datetime.utcnow(),
                symbol=symbol,
                sentiment_score=0.0,
                confidence=0.0,
                headline="No recent news",
                source="N/A",
                article_count=0,
                actionable=False,
                signal_type='NEUTRAL'
            )
        
        # Analyze
        enriched = self.sentiment_analyzer.analyze_news_articles(articles)
        
        # Generate signal
        signal = self._generate_signal(symbol, enriched)
        
        return signal or NewsSignal(
            timestamp=datetime.utcnow(),
            symbol=symbol,
            sentiment_score=0.0,
            confidence=0.0,
            headline="Analysis failed",
            source="N/A",
            article_count=len(articles),
            actionable=False,
            signal_type='NEUTRAL'
        )
    
    async def get_sentiment_summary(
        self,
        symbol: str
    ) -> Dict:
        """
        Get comprehensive sentiment summary.
        
        Returns:
            - Current signal
            - Historical trend
            - Key headlines
        """
        # Get current signal
        signal = await self.fetch_and_analyze(symbol, lookback_hours=24)
        
        # Get sentiment stats from API
        stats = await self.news_client.get_sentiment_stats(
            symbol,
            date_range='last7days'
        )
        
        # Get trending headlines
        trending = await self.news_client.get_trending_headlines(ticker=symbol)
        
        # Compute trend
        trend = 'STABLE'
        if 'data' in stats and len(stats['data']) >= 2:
            recent_scores = [d.get('sentiment_score', 0) for d in stats['data'][:3]]
            older_scores = [d.get('sentiment_score', 0) for d in stats['data'][3:7]]
            
            if recent_scores and older_scores:
                recent_avg = sum(recent_scores) / len(recent_scores)
                older_avg = sum(older_scores) / len(older_scores)
                
                if recent_avg > older_avg + 0.2:
                    trend = 'IMPROVING'
                elif recent_avg < older_avg - 0.2:
                    trend = 'DETERIORATING'
        
        return {
            'symbol': symbol,
            'timestamp': datetime.utcnow().isoformat(),
            'current_signal': signal.to_dict(),
            'sentiment_trend': trend,
            'daily_stats': stats.get('data', [])[:7],
            'trending_headlines': [
                {'title': h.get('title', ''), 'sentiment': h.get('sentiment', '')}
                for h in trending[:5]
            ],
            'recommendation': self._get_recommendation(signal, trend)
        }
    
    def _get_recommendation(
        self,
        signal: NewsSignal,
        trend: str
    ) -> str:
        """Generate human-readable recommendation."""
        if signal.signal_type == 'BULLISH':
            if trend == 'IMPROVING':
                return "Strong bullish sentiment with improving trend. News flow is supportive."
            elif trend == 'DETERIORATING':
                return "Bullish sentiment but trend weakening. Monitor for reversal."
            else:
                return "Bullish sentiment, stable trend. Positive news environment."
        
        elif signal.signal_type == 'BEARISH':
            if trend == 'DETERIORATING':
                return "Strong bearish sentiment with worsening trend. Exercise caution."
            elif trend == 'IMPROVING':
                return "Bearish sentiment but trend improving. Potential reversal."
            else:
                return "Bearish sentiment, stable trend. Negative news environment."
        
        else:  # NEUTRAL
            if trend == 'IMPROVING':
                return "Neutral sentiment but improving. Watch for bullish catalyst."
            elif trend == 'DETERIORATING':
                return "Neutral sentiment but deteriorating. Watch for bearish catalyst."
            else:
                return "Neutral sentiment, no clear direction. Wait for catalyst."
    
    def add_symbol(self, symbol: str):
        """Add a symbol to monitor."""
        if symbol not in self.symbols:
            self.symbols.append(symbol)
            logger.info(f"Added {symbol} to monitoring")
    
    def remove_symbol(self, symbol: str):
        """Remove a symbol from monitoring."""
        if symbol in self.symbols:
            self.symbols.remove(symbol)
            logger.info(f"Removed {symbol} from monitoring")


# Quick test function
async def test_pipeline():
    """Test the news pipeline."""
    print("Testing News Pipeline...")
    print("=" * 60)
    
    pipeline = NewsPipeline(symbols=['AAPL', 'MSFT', 'NVDA'])
    
    try:
        # Test on-demand analysis
        print("\n1. Fetching and analyzing AAPL news...")
        signal = await pipeline.fetch_and_analyze('AAPL', lookback_hours=24)
        
        print(f"   Symbol: {signal.symbol}")
        print(f"   Sentiment: {signal.sentiment_score:+.2f}")
        print(f"   Confidence: {signal.confidence:.1%}")
        print(f"   Signal: {signal.signal_type}")
        print(f"   Actionable: {signal.actionable}")
        print(f"   Articles: {signal.article_count}")
        print(f"   Top Headline: {signal.headline[:60]}...")
        
        # Test sentiment summary
        print("\n2. Getting NVDA sentiment summary...")
        summary = await pipeline.get_sentiment_summary('NVDA')
        
        print(f"   Trend: {summary['sentiment_trend']}")
        print(f"   Recommendation: {summary['recommendation']}")
        print(f"   Trending headlines:")
        for h in summary['trending_headlines'][:3]:
            print(f"      - {h['title'][:50]}...")
        
        print("\n" + "=" * 60)
        print("✅ News Pipeline working!")
        
    finally:
        await pipeline.stop()


if __name__ == "__main__":
    asyncio.run(test_pipeline())
