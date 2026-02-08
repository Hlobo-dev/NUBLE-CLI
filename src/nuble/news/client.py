"""
StockNews API Client

Handles all interactions with stocknewsapi.com
API Key: zzad9pmlwttixx0fnsenstctzgdk7ysx0ctkgrk0

Documentation: https://stocknewsapi.com/documentation
"""

import asyncio
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
import logging

try:
    import aiohttp
    HAS_AIOHTTP = True
except ImportError:
    aiohttp = None  # type: ignore
    HAS_AIOHTTP = False

logger = logging.getLogger(__name__)


class StockNewsClient:
    """
    Client for StockNews API.
    
    Features:
    - Ticker-specific news
    - General market news
    - Sentiment statistics
    - Trending headlines
    - Historical news (back to March 2019)
    
    Usage:
        client = StockNewsClient()
        news = await client.get_ticker_news(['AAPL', 'MSFT'], items=10)
        sentiment = await client.get_sentiment_stats('TSLA', date_range='last7days')
    """
    
    BASE_URL = "https://stocknewsapi.com/api/v1"
    API_KEY = "zzad9pmlwttixx0fnsenstctzgdk7ysx0ctkgrk0"
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or self.API_KEY
        self._session = None  # aiohttp.ClientSession, created lazily
        
    async def _get_session(self):
        """Get or create aiohttp session."""
        if not HAS_AIOHTTP:
            raise ImportError("aiohttp is required. Install with: pip install aiohttp")
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session
    
    async def close(self):
        """Close the session."""
        if self._session and not self._session.closed:
            await self._session.close()
    
    async def _request(self, endpoint: str, params: Dict[str, Any]) -> Dict:
        """Make API request."""
        params['token'] = self.api_key
        
        session = await self._get_session()
        url = f"{self.BASE_URL}{endpoint}"
        
        try:
            async with session.get(url, params=params) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    logger.error(f"API error: {response.status} - {await response.text()}")
                    return {'data': [], 'error': f"HTTP {response.status}"}
        except Exception as e:
            logger.error(f"Request failed: {e}")
            return {'data': [], 'error': str(e)}
    
    # ==========================================
    # TICKER NEWS
    # ==========================================
    
    async def get_ticker_news(
        self,
        tickers: List[str],
        items: int = 50,
        page: int = 1,
        date_range: Optional[str] = None,
        sector: Optional[str] = None
    ) -> List[Dict]:
        """
        Get news for specific tickers.
        
        Args:
            tickers: List of ticker symbols (e.g., ['AAPL', 'MSFT'])
            items: Number of items per page (max 50)
            page: Page number
            date_range: e.g., 'last7days', 'today', 'yesterday', '03152019-03252019'
            sector: Filter by sector (e.g., 'technology')
            
        Returns:
            List of news articles with sentiment
        """
        params = {
            'tickers': ','.join(tickers),
            'items': min(items, 50),
            'page': page
        }
        
        if date_range:
            params['date'] = date_range
        if sector:
            params['sector'] = sector
            
        result = await self._request('', params)
        return result.get('data', [])
    
    async def get_ticker_only_news(
        self,
        ticker: str,
        items: int = 50,
        page: int = 1
    ) -> List[Dict]:
        """
        Get news where ONLY this ticker is mentioned.
        More focused, fewer results.
        """
        params = {
            'tickers-only': ticker,
            'items': min(items, 50),
            'page': page
        }
        
        result = await self._request('', params)
        return result.get('data', [])
    
    async def get_multi_ticker_news(
        self,
        tickers: List[str],
        items: int = 50,
        page: int = 1
    ) -> List[Dict]:
        """
        Get news where ALL tickers are mentioned together.
        Useful for correlation analysis.
        """
        params = {
            'tickers-include': ','.join(tickers),
            'items': min(items, 50),
            'page': page
        }
        
        result = await self._request('', params)
        return result.get('data', [])
    
    # ==========================================
    # GENERAL MARKET NEWS
    # ==========================================
    
    async def get_general_news(
        self,
        items: int = 50,
        page: int = 1,
        date_range: Optional[str] = None
    ) -> List[Dict]:
        """Get general market news."""
        params = {
            'section': 'general',
            'items': min(items, 50),
            'page': page
        }
        
        if date_range:
            params['date'] = date_range
            
        result = await self._request('/category', params)
        return result.get('data', [])
    
    async def get_all_ticker_news(
        self,
        items: int = 50,
        page: int = 1,
        sector: Optional[str] = None,
        industry: Optional[str] = None
    ) -> List[Dict]:
        """Get news from all tickers."""
        params = {
            'section': 'alltickers',
            'items': min(items, 50),
            'page': page
        }
        
        if sector:
            params['sector'] = sector
        if industry:
            params['industry'] = industry
            
        result = await self._request('/category', params)
        return result.get('data', [])
    
    # ==========================================
    # SENTIMENT STATISTICS
    # ==========================================
    
    async def get_sentiment_stats(
        self,
        ticker: str,
        date_range: str = 'last30days',
        page: int = 1,
        use_cache: bool = True
    ) -> Dict:
        """
        Get daily sentiment statistics for a ticker.
        
        Sentiment Score ranges from -1.5 (Negative) to +1.5 (Positive)
        
        Args:
            ticker: Ticker symbol
            date_range: 'last7days', 'last30days', 'last60days', etc.
            page: Page number
            use_cache: Whether to use 1-hour cache (faster but delayed)
            
        Returns:
            Sentiment statistics by day
        """
        params = {
            'tickers': ticker,
            'date': date_range,
            'page': page
        }
        
        if not use_cache:
            params['cache'] = 'false'
            
        result = await self._request('/stat', params)
        return result
    
    async def get_market_sentiment(
        self,
        date_range: str = 'last7days',
        page: int = 1
    ) -> Dict:
        """Get overall market sentiment."""
        params = {
            'section': 'general',
            'date': date_range,
            'page': page
        }
        
        result = await self._request('/stat', params)
        return result
    
    # ==========================================
    # TRENDING HEADLINES
    # ==========================================
    
    async def get_trending_headlines(
        self,
        ticker: Optional[str] = None,
        page: int = 1
    ) -> List[Dict]:
        """
        Get trending headlines (filtered noise).
        
        Args:
            ticker: Optional ticker to filter
            page: Page number
            
        Returns:
            List of important trending headlines
        """
        params = {'page': page}
        
        if ticker:
            params['ticker'] = ticker
            
        result = await self._request('/trending-headlines', params)
        return result.get('data', [])
    
    # ==========================================
    # CONVENIENCE METHODS
    # ==========================================
    
    async def get_latest_news(
        self,
        tickers: List[str],
        minutes: int = 60
    ) -> List[Dict]:
        """
        Get news from the last N minutes.
        
        Args:
            tickers: List of tickers
            minutes: How far back to look (5, 10, 15, 30, 45, 60)
            
        Returns:
            Recent news articles
        """
        # Map minutes to API date parameter
        if minutes <= 5:
            date_range = 'last5min'
        elif minutes <= 10:
            date_range = 'last10min'
        elif minutes <= 15:
            date_range = 'last15min'
        elif minutes <= 30:
            date_range = 'last30min'
        elif minutes <= 45:
            date_range = 'last45min'
        else:
            date_range = 'last60min'
        
        return await self.get_ticker_news(tickers, items=50, date_range=date_range)
    
    async def get_news_summary(
        self,
        ticker: str
    ) -> Dict:
        """
        Get comprehensive news summary for a ticker.
        
        Returns:
            - Recent news (last 24h)
            - Sentiment stats (last 7 days)
            - Trending headlines
        """
        # Parallel fetch
        news_task = self.get_ticker_news([ticker], items=20, date_range='today')
        sentiment_task = self.get_sentiment_stats(ticker, date_range='last7days')
        trending_task = self.get_trending_headlines(ticker=ticker)
        
        news, sentiment, trending = await asyncio.gather(
            news_task, sentiment_task, trending_task,
            return_exceptions=True
        )
        
        # Handle exceptions
        if isinstance(news, Exception):
            news = []
        if isinstance(sentiment, Exception):
            sentiment = {}
        if isinstance(trending, Exception):
            trending = []
        
        return {
            'ticker': ticker,
            'timestamp': datetime.utcnow().isoformat(),
            'recent_news': news,
            'sentiment_stats': sentiment,
            'trending': trending,
            'news_count_24h': len(news) if isinstance(news, list) else 0
        }


# Synchronous wrapper for non-async contexts
class StockNewsClientSync:
    """Synchronous wrapper for StockNewsClient."""
    
    def __init__(self, api_key: Optional[str] = None):
        self._async_client = StockNewsClient(api_key)
    
    def _run(self, coro):
        """Run async coroutine synchronously."""
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(coro)
        finally:
            loop.close()
    
    def get_ticker_news(self, tickers: List[str], **kwargs) -> List[Dict]:
        return self._run(self._async_client.get_ticker_news(tickers, **kwargs))
    
    def get_sentiment_stats(self, ticker: str, **kwargs) -> Dict:
        return self._run(self._async_client.get_sentiment_stats(ticker, **kwargs))
    
    def get_trending_headlines(self, **kwargs) -> List[Dict]:
        return self._run(self._async_client.get_trending_headlines(**kwargs))
    
    def get_news_summary(self, ticker: str) -> Dict:
        return self._run(self._async_client.get_news_summary(ticker))


# Quick test function
async def test_client():
    """Test the client."""
    client = StockNewsClient()
    
    try:
        print("Testing StockNews API Client...")
        print("=" * 50)
        
        # Test ticker news
        print("\n1. Getting AAPL news...")
        news = await client.get_ticker_news(['AAPL'], items=3)
        for article in news[:3]:
            print(f"   - {article.get('title', 'No title')[:60]}...")
            print(f"     Sentiment: {article.get('sentiment', 'N/A')}")
        
        # Test sentiment stats
        print("\n2. Getting AAPL sentiment stats...")
        stats = await client.get_sentiment_stats('AAPL', date_range='last7days')
        if 'data' in stats:
            for day in stats['data'][:3]:
                print(f"   - {day.get('date', 'N/A')}: Score {day.get('sentiment_score', 'N/A')}")
        
        # Test trending
        print("\n3. Getting trending headlines...")
        trending = await client.get_trending_headlines()
        for headline in trending[:3]:
            print(f"   - {headline.get('title', 'No title')[:60]}...")
        
        print("\n" + "=" * 50)
        print("✅ All tests passed!")
        
    finally:
        await client.close()


if __name__ == "__main__":
    asyncio.run(test_client())
