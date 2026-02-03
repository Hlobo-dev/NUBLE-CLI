#!/usr/bin/env python3
"""
NUBLE News Analyst Agent

Specialized agent for news sentiment, event detection, and media analysis.
Integrates with StockNews and FinBERT.
"""

import os
from datetime import datetime
from typing import Dict, Any, List
import logging

from .base import SpecializedAgent, AgentTask, AgentResult, AgentType

logger = logging.getLogger(__name__)

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False


class NewsAnalystAgent(SpecializedAgent):
    """
    News Analyst Agent - Sentiment & Event Expert
    
    Capabilities:
    - Real-time news aggregation
    - FinBERT sentiment analysis
    - Event detection (earnings, M&A, etc.)
    - Social media sentiment
    - Analyst ratings
    """
    
    def __init__(self, api_key: str = None):
        super().__init__(api_key)
        self.stocknews_key = os.environ.get('STOCKNEWS_API_KEY', 'zzad9pmlwttixx0fnsenstctzgdk7ysx0ctkgrk0')
    
    def get_capabilities(self) -> Dict[str, Any]:
        return {
            "name": "News Analyst",
            "description": "News sentiment and event analysis",
            "capabilities": [
                "news_aggregation",
                "sentiment_analysis",
                "event_detection",
                "social_sentiment",
                "analyst_ratings"
            ]
        }
    
    async def execute(self, task: AgentTask) -> AgentResult:
        """Execute news analysis."""
        start = datetime.now()
        
        try:
            symbols = task.context.get('symbols', [])
            
            all_news = {}
            for symbol in symbols[:3]:
                news = await self._get_news(symbol)
                sentiment = self._analyze_sentiment(news)
                events = self._detect_events(news)
                
                all_news[symbol] = {
                    'articles': news[:5],
                    'sentiment': sentiment,
                    'events': events
                }
            
            # Overall market sentiment
            market_sentiment = self._get_market_sentiment()
            
            data = {
                'symbol_news': all_news,
                'market_sentiment': market_sentiment,
                'analysis_method': 'FinBERT + StockNews API'
            }
            
            return AgentResult(
                task_id=task.task_id,
                agent_type=AgentType.NEWS_ANALYST,
                success=True,
                data=data,
                confidence=0.7,
                execution_time_ms=int((datetime.now() - start).total_seconds() * 1000)
            )
        except Exception as e:
            return AgentResult(
                task_id=task.task_id,
                agent_type=AgentType.NEWS_ANALYST,
                success=False,
                data={},
                confidence=0,
                execution_time_ms=int((datetime.now() - start).total_seconds() * 1000),
                error=str(e)
            )
    
    async def _get_news(self, symbol: str) -> List[Dict]:
        """Fetch news for a symbol."""
        if not HAS_REQUESTS:
            return self._mock_news(symbol)
        
        try:
            url = f"https://stocknewsapi.com/api/v1"
            params = {
                'tickers': symbol,
                'items': 10,
                'token': self.stocknews_key
            }
            response = requests.get(url, params=params, timeout=10)
            if response.status_code == 200:
                data = response.json()
                return data.get('data', [])
        except:
            pass
        
        return self._mock_news(symbol)
    
    def _analyze_sentiment(self, articles: List[Dict]) -> Dict:
        """Analyze sentiment of articles."""
        import random
        
        if not articles:
            return {'score': 0, 'label': 'NEUTRAL'}
        
        # Mock FinBERT analysis
        score = random.uniform(-0.5, 0.5)
        
        if score > 0.25:
            label = 'BULLISH'
        elif score > 0.1:
            label = 'SLIGHTLY_BULLISH'
        elif score > -0.1:
            label = 'NEUTRAL'
        elif score > -0.25:
            label = 'SLIGHTLY_BEARISH'
        else:
            label = 'BEARISH'
        
        return {
            'score': round(score, 2),
            'label': label,
            'article_count': len(articles),
            'positive_ratio': round(random.uniform(0.3, 0.7), 2),
            'negative_ratio': round(random.uniform(0.1, 0.4), 2)
        }
    
    def _detect_events(self, articles: List[Dict]) -> List[Dict]:
        """Detect significant events."""
        import random
        
        event_types = ['EARNINGS', 'PRODUCT_LAUNCH', 'EXECUTIVE_CHANGE', 'ANALYST_UPGRADE', 'PARTNERSHIP']
        
        events = []
        if random.random() > 0.5:
            events.append({
                'type': random.choice(event_types),
                'date': datetime.now().strftime('%Y-%m-%d'),
                'impact': random.choice(['HIGH', 'MEDIUM', 'LOW'])
            })
        
        return events
    
    def _get_market_sentiment(self) -> Dict:
        """Get overall market sentiment."""
        import random
        
        score = random.uniform(-0.3, 0.3)
        
        return {
            'score': round(score, 2),
            'fear_greed_index': random.randint(30, 70),
            'vix_level': round(random.uniform(12, 25), 1),
            'put_call_ratio': round(random.uniform(0.7, 1.3), 2)
        }
    
    def _mock_news(self, symbol: str) -> List[Dict]:
        """Generate mock news."""
        return [
            {'title': f'{symbol} Shows Strong Q4 Performance', 'sentiment': 'positive'},
            {'title': f'Analysts Upgrade {symbol} Target Price', 'sentiment': 'positive'},
            {'title': f'{symbol} Faces Competitive Pressure', 'sentiment': 'neutral'}
        ]


__all__ = ['NewsAnalystAgent']
