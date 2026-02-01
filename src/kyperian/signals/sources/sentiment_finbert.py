"""
Sentiment Signal Source - FinBERT

Uses FinBERT NLP model for financial sentiment analysis.
Integrates with the existing NewsPipeline and SentimentAnalyzer.
"""

from datetime import datetime
from typing import Dict, Optional, Any, List
import logging

from ..base_source import SignalSource, NormalizedSignal

logger = logging.getLogger(__name__)


class SentimentFinBERTSource(SignalSource):
    """
    Sentiment signal source using FinBERT.
    
    This source analyzes news and social media sentiment
    using the FinBERT model (fine-tuned for financial text).
    
    Data Sources:
    - News headlines
    - News articles
    - Social media (optional)
    - Earnings call sentiment
    
    Example:
        source = SentimentFinBERTSource()
        
        # From pre-computed sentiment
        signal = source.generate_signal('AAPL', context={'sentiment': 0.3})
        
        # From headlines
        signal = source.generate_signal('AAPL', data=["Apple beats earnings"])
    """
    
    name = "sentiment"
    base_weight = 0.10
    
    def __init__(
        self,
        weight: float = None,
        min_articles: int = 1,
        decay_hours: float = 24.0
    ):
        """
        Initialize sentiment source.
        
        Args:
            weight: Custom weight
            min_articles: Minimum articles for actionable signal
            decay_hours: Hours after which sentiment decays
        """
        super().__init__(weight)
        self.min_articles = min_articles
        self.decay_hours = decay_hours
        self._analyzer = None
        self._pipeline = None
    
    def _get_analyzer(self):
        """Lazy load the sentiment analyzer."""
        if self._analyzer is None:
            try:
                from kyperian.news.sentiment import SentimentAnalyzer
                self._analyzer = SentimentAnalyzer()
            except ImportError as e:
                logger.debug(f"Could not import SentimentAnalyzer: {e}")
        return self._analyzer
    
    def _get_pipeline(self):
        """Lazy load the news pipeline."""
        if self._pipeline is None:
            try:
                from kyperian.news.pipeline import NewsPipeline
                self._pipeline = NewsPipeline()
            except ImportError as e:
                logger.debug(f"Could not import NewsPipeline: {e}")
        return self._pipeline
    
    def generate_signal(
        self,
        symbol: str,
        data: Any = None,
        context: Dict = None
    ) -> Optional[NormalizedSignal]:
        """
        Generate normalized signal from sentiment analysis.
        
        Args:
            symbol: Asset symbol
            data: Headlines/articles to analyze (list of strings)
                  or None to use pre-computed sentiment
            context: Additional context (pre-computed sentiment, etc.)
            
        Returns:
            NormalizedSignal or None
        """
        context = context or {}
        
        # Option 1: Use pre-computed sentiment from context
        if 'sentiment' in context:
            sentiment_score = context['sentiment']
            confidence = context.get('sentiment_confidence', 0.6)
            
            signal = NormalizedSignal(
                source_name=self.name,
                symbol=symbol,
                timestamp=datetime.now(),
                direction=sentiment_score,
                confidence=confidence,
                raw_data={'source': 'pre-computed'},
                reasoning=f"Sentiment: {sentiment_score:+.2f} (pre-computed)"
            )
            self._last_signal = signal
            return signal
        
        # Option 2: Analyze provided headlines
        if data and isinstance(data, list) and len(data) > 0:
            return self._analyze_headlines(symbol, data)
        
        # Option 3: Use news pipeline
        pipeline = self._get_pipeline()
        if pipeline and symbol in pipeline.signals:
            news_signal = pipeline.signals[symbol]
            
            signal = NormalizedSignal(
                source_name=self.name,
                symbol=symbol,
                timestamp=datetime.now(),
                direction=news_signal.sentiment_score,
                confidence=news_signal.confidence,
                raw_data=news_signal.to_dict(),
                reasoning=f"Sentiment from {news_signal.article_count} articles"
            )
            self._last_signal = signal
            return signal
        
        # No sentiment data available
        return None
    
    def _analyze_headlines(
        self, 
        symbol: str, 
        headlines: List[str]
    ) -> Optional[NormalizedSignal]:
        """Analyze headlines with FinBERT."""
        analyzer = self._get_analyzer()
        if analyzer is None:
            logger.debug("Sentiment analyzer not available")
            return None
        
        try:
            # Analyze each headline
            results = analyzer.analyze_batch(headlines)
            
            if not results:
                return None
            
            # Aggregate sentiment
            total_score = sum(r.normalized_score for r in results)
            avg_score = total_score / len(results)
            
            # Confidence based on agreement and number of articles
            scores = [r.normalized_score for r in results]
            if len(scores) > 1:
                import numpy as np
                std = np.std(scores)
                agreement = 1 - min(1, std)  # Lower std = higher agreement
            else:
                agreement = 0.5
            
            confidence = min(1.0, agreement * (0.5 + len(results) * 0.1))
            
            signal = NormalizedSignal(
                source_name=self.name,
                symbol=symbol,
                timestamp=datetime.now(),
                direction=avg_score,
                confidence=confidence,
                raw_data={
                    'articles_analyzed': len(results),
                    'avg_sentiment': avg_score,
                    'agreement': agreement
                },
                reasoning=f"Sentiment: {avg_score:+.2f} from {len(results)} headlines"
            )
            
            self._last_signal = signal
            return signal
            
        except Exception as e:
            logger.warning(f"Sentiment analysis failed: {e}")
            return None
    
    def get_confidence(self) -> float:
        """Get confidence in sentiment signals."""
        # Sentiment is supplementary
        base_confidence = 0.5
        
        # Boost from recent accuracy
        recent_accuracy = self.get_recent_accuracy(20)
        accuracy_boost = (recent_accuracy - 0.5) * 0.3
        
        return min(1.0, max(0.2, base_confidence + accuracy_boost))
    
    async def analyze_symbol(self, symbol: str) -> Optional[NormalizedSignal]:
        """
        Async method to fetch and analyze news for a symbol.
        
        Useful for real-time updates.
        """
        pipeline = self._get_pipeline()
        if pipeline is None:
            return None
        
        try:
            news_signal = await pipeline.get_signal(symbol)
            if news_signal:
                signal = NormalizedSignal(
                    source_name=self.name,
                    symbol=symbol,
                    timestamp=datetime.now(),
                    direction=news_signal.sentiment_score,
                    confidence=news_signal.confidence,
                    raw_data=news_signal.to_dict(),
                    reasoning=f"Live sentiment from {news_signal.article_count} articles"
                )
                self._last_signal = signal
                return signal
        except Exception as e:
            logger.warning(f"Async sentiment analysis failed: {e}")
        
        return None
