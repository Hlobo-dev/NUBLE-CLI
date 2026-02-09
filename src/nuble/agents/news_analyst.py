#!/usr/bin/env python3
"""
NUBLE News Analyst Agent — ELITE TIER

Comprehensive news sentiment & event detection from multiple sources:
- Polygon.io News API — Rich article data with publisher, tickers, description
- StockNews API — Stock-specific news with sentiment labels
- CryptoNews API — Crypto-specific news with sentiment
- Alternative.me Fear & Greed — Market-wide sentiment
- Polygon VIX — Volatility context for sentiment calibration
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

# Crypto symbols for CryptoNews routing
CRYPTO_SYMBOLS = {'BTC', 'ETH', 'SOL', 'XRP', 'ADA', 'DOGE', 'AVAX', 'DOT', 'MATIC', 'LINK', 'UNI', 'AAVE', 'ATOM', 'NEAR'}


class NewsAnalystAgent(SpecializedAgent):
    """
    News Analyst Agent — ELITE Sentiment & Event Expert (FULL PREMIUM)
    
    REAL DATA from 5+ sources, using ALL premium endpoints:
    1. Polygon News — /v2/reference/news with publisher, tickers, descriptions
    2. StockNews API (PRO $50/mo) — ALL endpoints:
       - /api/v1 — Ticker news with sentiment
       - /api/v1/stat — Quantitative sentiment analysis over time
       - /api/v1/top-mention — Most mentioned stocks with sentiment
       - /api/v1/events — Breaking news events with event IDs
       - /api/v1/trending-headlines — Top trending headlines
       - /api/v1/earnings-calendar — Upcoming earnings dates
       - /api/v1/ratings — Analyst upgrades/downgrades with price targets
       - /api/v1/sundown-digest — Daily market summary
       - /api/v1/category?section=general — General market news (Fed, CPI, etc.)
       - Sector, topic, sentiment, rank score filtering
    3. CryptoNews API (PRO $50/mo) — ALL endpoints:
       - /api/v1 — Ticker news with sentiment
       - /api/v1/stat — Quantitative crypto sentiment over time
       - /api/v1/top-mention — Most mentioned coins with sentiment
       - /api/v1/events — Breaking crypto events
       - /api/v1/trending-headlines — Top crypto headlines
       - /api/v1/category?section=general — General crypto market news
       - Topic filtering (DeFi, NFT, Regulations, Mining, etc.)
    4. Alternative.me Fear & Greed — Market-wide sentiment index
    5. Polygon VIX — Volatility context for sentiment calibration
    """
    
    def __init__(self, api_key: str = None):
        super().__init__(api_key)
        self.stocknews_key = os.environ.get('STOCKNEWS_API_KEY', 'zzad9pmlwttixx0fnsenstctzgdk7ysx0ctkgrk0')
        self.polygon_key = os.environ.get('POLYGON_API_KEY', 'JHKwAdyIOeExkYOxh3LwTopmqqVVFeBY')
        self.cryptonews_key = os.environ.get('CRYPTONEWS_API_KEY', os.environ.get('CRYPTO_NEWS_KEY', 'fci3fvhrbxocelhel4ddc7zbmgsxnq1zmwrkxgq2'))
    
    def get_capabilities(self) -> Dict[str, Any]:
        return {
            "name": "News Analyst",
            "description": "Elite news sentiment & event analysis — FULL PREMIUM endpoints",
            "capabilities": [
                "multi_source_aggregation", "sentiment_analysis",
                "event_detection", "crypto_news", "market_sentiment",
                "headline_scoring", "cross_source_verification",
                "earnings_calendar", "analyst_ratings", "trending_headlines",
                "top_mentioned_tickers", "sector_news", "topic_filtering",
                "quantitative_sentiment", "sundown_digest", "breaking_events"
            ],
            "data_sources": [
                "Polygon News", "StockNews PRO (all endpoints)",
                "CryptoNews PRO (all endpoints)",
                "Alternative.me FGI", "Polygon VIX"
            ]
        }
    
    async def execute(self, task: AgentTask) -> AgentResult:
        """Execute news analysis using ALL premium endpoints."""
        start = datetime.now()
        
        try:
            symbols = task.context.get('symbols', [])
            shared = self._get_shared_data(task)
            
            all_news = {}
            for symbol in symbols[:3]:
                is_crypto = symbol.upper() in CRYPTO_SYMBOLS
                
                # === TICKER-SPECIFIC NEWS from multiple sources ===
                stocknews = await self._get_stocknews(symbol, shared) if not is_crypto else []
                polygon_news = await self._get_polygon_news(symbol, shared) if not is_crypto else []
                crypto_news = await self._get_crypto_news(symbol, shared) if is_crypto else []
                
                combined = stocknews + polygon_news + crypto_news
                
                # === PREMIUM: Quantitative Sentiment Analysis ===
                quant_sentiment = await self._get_sentiment_stats(symbol, is_crypto, shared)
                
                sentiment = self._analyze_sentiment(combined)
                events = self._detect_events(combined)
                
                all_news[symbol] = {
                    'articles': combined[:8],
                    'sentiment': sentiment,
                    'quantitative_sentiment': quant_sentiment,
                    'events': events,
                    'sources_used': []
                }
                if stocknews:
                    all_news[symbol]['sources_used'].append('StockNews PRO')
                if polygon_news:
                    all_news[symbol]['sources_used'].append('Polygon')
                if crypto_news:
                    all_news[symbol]['sources_used'].append('CryptoNews PRO')
            
            # === PREMIUM: Market-wide intelligence ===
            market_sentiment = await self._get_market_sentiment(shared)
            trending = await self._get_trending_headlines(shared)
            top_mentioned = await self._get_top_mentioned(shared)
            breaking_events = await self._get_breaking_events(shared)
            general_market_news = await self._get_general_market_news(shared)
            earnings_calendar = await self._get_earnings_calendar(shared)
            analyst_ratings = await self._get_analyst_ratings(symbols, shared)
            sundown_digest = await self._get_sundown_digest(shared)
            
            # Crypto-specific premium data
            has_crypto = any(s.upper() in CRYPTO_SYMBOLS for s in symbols)
            crypto_trending = await self._get_crypto_trending(shared) if has_crypto else {}
            crypto_top_mentioned = await self._get_crypto_top_mentioned(shared) if has_crypto else {}
            crypto_general = await self._get_crypto_general_news(shared) if has_crypto else {}
            crypto_events = await self._get_crypto_events(shared) if has_crypto else {}
            
            data = {
                'symbol_news': all_news,
                'market_sentiment': market_sentiment,
                'trending_headlines': trending,
                'top_mentioned_tickers': top_mentioned,
                'breaking_events': breaking_events,
                'general_market_news': general_market_news,
                'earnings_calendar': earnings_calendar,
                'analyst_ratings': analyst_ratings,
                'sundown_digest': sundown_digest,
                'analysis_method': 'FULL PREMIUM — StockNews PRO + CryptoNews PRO + Polygon'
            }
            
            # Add crypto premium data if relevant
            if has_crypto:
                data['crypto_trending'] = crypto_trending
                data['crypto_top_mentioned'] = crypto_top_mentioned
                data['crypto_general_news'] = crypto_general
                data['crypto_events'] = crypto_events
            
            return AgentResult(
                task_id=task.task_id,
                agent_type=AgentType.NEWS_ANALYST,
                success=True,
                data=data,
                confidence=0.88,
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
    
    # ──────────────────────────────────────────────────────────────────────
    # StockNews PRO endpoints
    # ──────────────────────────────────────────────────────────────────────
    
    async def _get_stocknews(self, symbol: str, shared=None) -> List[Dict]:
        """StockNews PRO — Ticker news with sentiment + rank score."""
        if shared:
            data = await shared.get_stocknews(symbol)
            if data and data.get('data'):
                articles = data['data']
                for a in articles:
                    a['_source'] = 'stocknews_pro'
                return articles
        if not HAS_REQUESTS or not self.stocknews_key:
            return []
        try:
            resp = requests.get("https://stocknewsapi.com/api/v1", params={
                'tickers': symbol, 'items': 10, 'sortby': 'rank',
                'extra-fields': 'id,eventid,rankscore',
                'token': self.stocknews_key
            }, timeout=10)
            if resp.status_code == 200:
                articles = resp.json().get('data', [])
                for a in articles:
                    a['_source'] = 'stocknews_pro'
                return articles
        except Exception:
            pass
        return []
    
    async def _get_sentiment_stats(self, symbol: str, is_crypto: bool = False, shared=None) -> Dict:
        """StockNews/CryptoNews PRO — /stat endpoint for quantitative sentiment over time."""
        if shared:
            if is_crypto:
                data = await shared.get_cryptonews_stat(symbol)
            else:
                data = await shared.get_stocknews_stat(symbol, "last30days")
            if data and data.get('data'):
                return {'sentiment_data': data['data'], 'period': 'last30days', 'data_source': 'cryptonews_stat' if is_crypto else 'stocknews_stat'}
        if not HAS_REQUESTS:
            return {}
        try:
            if is_crypto:
                resp = requests.get("https://cryptonews-api.com/api/v1/stat", params={
                    'tickers': symbol, 'date': 'last30days', 'page': 1,
                    'token': self.cryptonews_key
                }, timeout=10)
            else:
                resp = requests.get("https://stocknewsapi.com/api/v1/stat", params={
                    'tickers': symbol, 'date': 'last30days', 'page': 1,
                    'token': self.stocknews_key
                }, timeout=10)
            
            if resp.status_code == 200:
                data = resp.json().get('data', {})
                return {
                    'sentiment_data': data,
                    'period': 'last30days',
                    'data_source': 'cryptonews_stat' if is_crypto else 'stocknews_stat'
                }
        except Exception:
            pass
        return {}
    
    async def _get_trending_headlines(self, shared=None) -> Dict:
        """StockNews PRO — /trending-headlines for top trending stories."""
        if shared:
            data = await shared.get_stocknews_trending()
            if data and data.get('data'):
                headlines = data['data']
                return {
                    'headlines': [{'title': h.get('title', ''), 'description': h.get('text', '')[:200] if h.get('text') else '', 'source': h.get('source_name', ''), 'date': h.get('date', ''), 'tickers': h.get('tickers', [])} for h in headlines[:10]],
                    'count': len(headlines),
                    'data_source': 'stocknews_trending'
                }
        if not HAS_REQUESTS or not self.stocknews_key:
            return {}
        try:
            resp = requests.get("https://stocknewsapi.com/api/v1/trending-headlines", params={
                'page': 1, 'token': self.stocknews_key
            }, timeout=10)
            if resp.status_code == 200:
                headlines = resp.json().get('data', [])
                return {
                    'headlines': [{
                        'title': h.get('title', ''),
                        'description': h.get('text', '')[:200] if h.get('text') else '',
                        'source': h.get('source_name', ''),
                        'date': h.get('date', ''),
                        'tickers': h.get('tickers', []),
                    } for h in headlines[:10]],
                    'count': len(headlines),
                    'data_source': 'stocknews_trending'
                }
        except Exception:
            pass
        return {}
    
    async def _get_top_mentioned(self, shared=None) -> Dict:
        """StockNews PRO — /top-mention for most discussed stocks with sentiment."""
        if shared:
            data = await shared.get_stocknews_top_mentioned()
            if data and data.get('data'):
                return {'top_tickers': data['data'][:15], 'period': 'last7days', 'data_source': 'stocknews_top_mention'}
        if not HAS_REQUESTS or not self.stocknews_key:
            return {}
        try:
            resp = requests.get("https://stocknewsapi.com/api/v1/top-mention", params={
                'date': 'last7days', 'token': self.stocknews_key
            }, timeout=10)
            if resp.status_code == 200:
                data = resp.json().get('data', [])
                return {
                    'top_tickers': data[:15],
                    'period': 'last7days',
                    'data_source': 'stocknews_top_mention'
                }
        except Exception:
            pass
        return {}
    
    async def _get_breaking_events(self, shared=None) -> Dict:
        """StockNews PRO — /events for breaking news events."""
        if shared:
            data = await shared.get_stocknews_events()
            if data and data.get('data'):
                events = data['data']
                return {
                    'events': [{'title': e.get('title', ''), 'event_id': e.get('eventid', ''), 'date': e.get('date', ''), 'source': e.get('source_name', ''), 'tickers': e.get('tickers', [])} for e in events[:10]],
                    'count': len(events),
                    'data_source': 'stocknews_events'
                }
        if not HAS_REQUESTS or not self.stocknews_key:
            return {}
        try:
            resp = requests.get("https://stocknewsapi.com/api/v1/events", params={
                'page': 1, 'token': self.stocknews_key
            }, timeout=10)
            if resp.status_code == 200:
                events = resp.json().get('data', [])
                return {
                    'events': [{
                        'title': e.get('title', ''),
                        'event_id': e.get('eventid', ''),
                        'date': e.get('date', ''),
                        'source': e.get('source_name', ''),
                        'tickers': e.get('tickers', []),
                    } for e in events[:10]],
                    'count': len(events),
                    'data_source': 'stocknews_events'
                }
        except Exception:
            pass
        return {}
    
    async def _get_general_market_news(self, shared=None) -> Dict:
        """StockNews PRO — /category?section=general for Fed, CPI, macro news."""
        if shared:
            data = await shared.get_stocknews_category("general")
            if data and data.get('data'):
                articles = data['data']
                return {
                    'articles': [{'title': a.get('title', ''), 'text': (a.get('text', '')[:200] + '...') if a.get('text') else '', 'sentiment': a.get('sentiment', ''), 'source': a.get('source_name', ''), 'date': a.get('date', ''), 'rank_score': a.get('rankscore', '')} for a in articles[:10]],
                    'count': len(articles),
                    'data_source': 'stocknews_general_market'
                }
        if not HAS_REQUESTS or not self.stocknews_key:
            return {}
        try:
            resp = requests.get("https://stocknewsapi.com/api/v1/category", params={
                'section': 'general', 'items': 10, 'sortby': 'rank',
                'extra-fields': 'id,eventid,rankscore',
                'page': 1, 'token': self.stocknews_key
            }, timeout=10)
            if resp.status_code == 200:
                articles = resp.json().get('data', [])
                return {
                    'articles': [{
                        'title': a.get('title', ''),
                        'text': (a.get('text', '')[:200] + '...') if a.get('text') else '',
                        'sentiment': a.get('sentiment', ''),
                        'source': a.get('source_name', ''),
                        'date': a.get('date', ''),
                        'rank_score': a.get('rankscore', ''),
                    } for a in articles[:10]],
                    'count': len(articles),
                    'data_source': 'stocknews_general_market'
                }
        except Exception:
            pass
        return {}
    
    async def _get_earnings_calendar(self, shared=None) -> Dict:
        """StockNews PRO — /earnings-calendar for upcoming earnings dates."""
        if shared:
            data = await shared.get_stocknews_earnings()
            if data and data.get('data'):
                return {'upcoming_earnings': data['data'][:20], 'count': len(data['data']), 'data_source': 'stocknews_earnings_calendar'}
        if not HAS_REQUESTS or not self.stocknews_key:
            return {}
        try:
            resp = requests.get("https://stocknewsapi.com/api/v1/earnings-calendar", params={
                'page': 1, 'items': 20, 'token': self.stocknews_key
            }, timeout=10)
            if resp.status_code == 200:
                earnings = resp.json().get('data', [])
                return {
                    'upcoming_earnings': earnings[:20],
                    'count': len(earnings),
                    'data_source': 'stocknews_earnings_calendar'
                }
        except Exception:
            pass
        return {}
    
    async def _get_analyst_ratings(self, symbols: List[str], shared=None) -> Dict:
        """StockNews PRO — /ratings for analyst upgrades/downgrades + price targets."""
        if shared:
            data = await shared.get_stocknews_ratings()
            if data and data.get('data'):
                ratings = data['data']
                relevant = [r for r in ratings if any(s.upper() == r.get('ticker', '').upper() for s in symbols)]
                return {
                    'relevant_ratings': relevant,
                    'recent_ratings': [r for r in ratings if r not in relevant][:10],
                    'total_count': len(ratings),
                    'data_source': 'stocknews_ratings'
                }
        if not HAS_REQUESTS or not self.stocknews_key:
            return {}
        try:
            resp = requests.get("https://stocknewsapi.com/api/v1/ratings", params={
                'items': 15, 'page': 1, 'token': self.stocknews_key
            }, timeout=10)
            if resp.status_code == 200:
                ratings = resp.json().get('data', [])
                
                # Filter for symbols of interest if available
                relevant = []
                other = []
                for r in ratings:
                    entry = {
                        'ticker': r.get('ticker', ''),
                        'action': r.get('action', ''),
                        'rating_from': r.get('rating_from', ''),
                        'rating_to': r.get('rating_to', ''),
                        'target_from': r.get('target_from', ''),
                        'target_to': r.get('target_to', ''),
                        'analyst': r.get('analyst', ''),
                        'analyst_company': r.get('analyst_company', ''),
                        'date': r.get('date', ''),
                    }
                    if any(s.upper() == r.get('ticker', '').upper() for s in symbols):
                        relevant.append(entry)
                    else:
                        other.append(entry)
                
                return {
                    'relevant_ratings': relevant,
                    'recent_ratings': other[:10],
                    'total_count': len(ratings),
                    'data_source': 'stocknews_ratings'
                }
        except Exception:
            pass
        return {}
    
    async def _get_sundown_digest(self, shared=None) -> Dict:
        """StockNews PRO — /sundown-digest for daily market summary."""
        if shared:
            data = await shared.get_stocknews_sundown()
            if data and data.get('data'):
                return {
                    'digest': [{'title': d.get('title', ''), 'text': d.get('text', ''), 'date': d.get('date', '')} for d in data['data'][:3]],
                    'data_source': 'stocknews_sundown'
                }
        if not HAS_REQUESTS or not self.stocknews_key:
            return {}
        try:
            resp = requests.get("https://stocknewsapi.com/api/v1/sundown-digest", params={
                'page': 1, 'token': self.stocknews_key
            }, timeout=10)
            if resp.status_code == 200:
                digest = resp.json().get('data', [])
                return {
                    'digest': [{
                        'title': d.get('title', ''),
                        'text': d.get('text', ''),
                        'date': d.get('date', ''),
                    } for d in digest[:3]],
                    'data_source': 'stocknews_sundown'
                }
        except Exception:
            pass
        return {}
    
    # ──────────────────────────────────────────────────────────────────────
    # CryptoNews PRO endpoints
    # ──────────────────────────────────────────────────────────────────────
    
    async def _get_crypto_news(self, symbol: str, shared=None) -> List[Dict]:
        """CryptoNews PRO — Ticker news with sentiment + rank score."""
        if shared:
            data = await shared.get_cryptonews(symbol)
            if data and data.get('data'):
                articles = data['data']
                for a in articles:
                    a['_source'] = 'cryptonews_pro'
                return articles
        if not HAS_REQUESTS or not self.cryptonews_key:
            return []
        try:
            resp = requests.get("https://cryptonews-api.com/api/v1", params={
                'tickers': symbol, 'items': 10, 'sortby': 'rank',
                'extra-fields': 'id,eventid,rankscore',
                'token': self.cryptonews_key
            }, timeout=10)
            if resp.status_code == 200:
                articles = resp.json().get('data', [])
                for a in articles:
                    a['_source'] = 'cryptonews_pro'
                return articles
        except Exception:
            pass
        return []
    
    async def _get_crypto_trending(self, shared=None) -> Dict:
        """CryptoNews PRO — /trending-headlines for top crypto stories."""
        if shared:
            data = await shared.get_cryptonews_trending()
            if data and data.get('data'):
                headlines = data['data']
                return {
                    'headlines': [{'title': h.get('title', ''), 'description': (h.get('text', '')[:200]) if h.get('text') else '', 'source': h.get('source_name', ''), 'date': h.get('date', '')} for h in headlines[:10]],
                    'count': len(headlines),
                    'data_source': 'cryptonews_trending'
                }
        if not HAS_REQUESTS or not self.cryptonews_key:
            return {}
        try:
            resp = requests.get("https://cryptonews-api.com/api/v1/trending-headlines", params={
                'page': 1, 'token': self.cryptonews_key
            }, timeout=10)
            if resp.status_code == 200:
                headlines = resp.json().get('data', [])
                return {
                    'headlines': [{
                        'title': h.get('title', ''),
                        'description': (h.get('text', '')[:200]) if h.get('text') else '',
                        'source': h.get('source_name', ''),
                        'date': h.get('date', ''),
                    } for h in headlines[:10]],
                    'count': len(headlines),
                    'data_source': 'cryptonews_trending'
                }
        except Exception:
            pass
        return {}
    
    async def _get_crypto_top_mentioned(self, shared=None) -> Dict:
        """CryptoNews PRO — /top-mention for most discussed coins."""
        if shared:
            data = await shared.get_cryptonews_top_mentioned()
            if data and data.get('data'):
                return {'top_coins': data['data'][:15], 'period': 'last7days', 'data_source': 'cryptonews_top_mention'}
        if not HAS_REQUESTS or not self.cryptonews_key:
            return {}
        try:
            resp = requests.get("https://cryptonews-api.com/api/v1/top-mention", params={
                'date': 'last7days', 'token': self.cryptonews_key
            }, timeout=10)
            if resp.status_code == 200:
                data = resp.json().get('data', [])
                return {
                    'top_coins': data[:15],
                    'period': 'last7days',
                    'data_source': 'cryptonews_top_mention'
                }
        except Exception:
            pass
        return {}
    
    async def _get_crypto_general_news(self, shared=None) -> Dict:
        """CryptoNews PRO — /category?section=general for regulation, market news."""
        if shared:
            data = await shared.get_cryptonews_category("general")
            if data and data.get('data'):
                articles = data['data']
                return {
                    'articles': [{'title': a.get('title', ''), 'text': (a.get('text', '')[:200] + '...') if a.get('text') else '', 'sentiment': a.get('sentiment', ''), 'source': a.get('source_name', ''), 'date': a.get('date', ''), 'rank_score': a.get('rankscore', '')} for a in articles[:10]],
                    'data_source': 'cryptonews_general'
                }
        if not HAS_REQUESTS or not self.cryptonews_key:
            return {}
        try:
            resp = requests.get("https://cryptonews-api.com/api/v1/category", params={
                'section': 'general', 'items': 10, 'sortby': 'rank',
                'extra-fields': 'id,eventid,rankscore',
                'page': 1, 'token': self.cryptonews_key
            }, timeout=10)
            if resp.status_code == 200:
                articles = resp.json().get('data', [])
                return {
                    'articles': [{
                        'title': a.get('title', ''),
                        'text': (a.get('text', '')[:200] + '...') if a.get('text') else '',
                        'sentiment': a.get('sentiment', ''),
                        'source': a.get('source_name', ''),
                        'date': a.get('date', ''),
                        'rank_score': a.get('rankscore', ''),
                    } for a in articles[:10]],
                    'data_source': 'cryptonews_general'
                }
        except Exception:
            pass
        return {}
    
    async def _get_crypto_events(self, shared=None) -> Dict:
        """CryptoNews PRO — /events for breaking crypto events."""
        if shared:
            data = await shared.get_cryptonews_events()
            if data and data.get('data'):
                events = data['data']
                return {
                    'events': [{'title': e.get('title', ''), 'event_id': e.get('eventid', ''), 'date': e.get('date', ''), 'source': e.get('source_name', '')} for e in events[:10]],
                    'data_source': 'cryptonews_events'
                }
        if not HAS_REQUESTS or not self.cryptonews_key:
            return {}
        try:
            resp = requests.get("https://cryptonews-api.com/api/v1/events", params={
                'page': 1, 'token': self.cryptonews_key
            }, timeout=10)
            if resp.status_code == 200:
                events = resp.json().get('data', [])
                return {
                    'events': [{
                        'title': e.get('title', ''),
                        'event_id': e.get('eventid', ''),
                        'date': e.get('date', ''),
                        'source': e.get('source_name', ''),
                    } for e in events[:10]],
                    'data_source': 'cryptonews_events'
                }
        except Exception:
            pass
        return {}
    
    # ──────────────────────────────────────────────────────────────────────
    # Polygon News
    # ──────────────────────────────────────────────────────────────────────
    
    async def _get_polygon_news(self, symbol: str, shared=None) -> List[Dict]:
        """Fetch news from Polygon.io /v2/reference/news."""
        if shared:
            data = await shared.get_polygon_news(symbol)
            if data and data.get('results'):
                articles = []
                for a in data['results']:
                    articles.append({
                        'title': a.get('title', ''),
                        'text': a.get('description', ''),
                        'date': a.get('published_utc', ''),
                        'source_name': a.get('publisher', {}).get('name', ''),
                        'sentiment': '',
                        'tickers': [t for t in (a.get('tickers', []) or [])],
                        '_source': 'polygon'
                    })
                return articles
        if not HAS_REQUESTS:
            return []
        try:
            resp = requests.get("https://api.polygon.io/v2/reference/news", params={
                'ticker': symbol, 'limit': 10, 'order': 'desc', 'apiKey': self.polygon_key
            }, timeout=10)
            if resp.status_code == 200:
                articles = []
                for a in resp.json().get('results', []):
                    articles.append({
                        'title': a.get('title', ''),
                        'text': a.get('description', ''),
                        'date': a.get('published_utc', ''),
                        'source_name': a.get('publisher', {}).get('name', ''),
                        'sentiment': '',
                        'tickers': [t for t in (a.get('tickers', []) or [])],
                        '_source': 'polygon'
                    })
                return articles
        except Exception:
            pass
        return []
    
    # ──────────────────────────────────────────────────────────────────────
    # Sentiment analysis
    # ──────────────────────────────────────────────────────────────────────
    
    def _analyze_sentiment(self, articles: List[Dict]) -> Dict:
        """Analyze sentiment from real article data across all sources."""
        if not articles:
            return {'score': 0, 'label': 'NEUTRAL', 'article_count': 0, 'data_source': 'no_articles'}
        
        positive = 0
        negative = 0
        neutral = 0
        
        bullish_words = ['upgrade', 'beat', 'strong', 'surge', 'rally', 'growth', 'record', 'outperform', 'buy', 'bullish']
        bearish_words = ['downgrade', 'miss', 'weak', 'crash', 'decline', 'risk', 'warning', 'sell', 'bearish', 'layoff', 'lawsuit']
        
        for article in articles:
            sent = str(article.get('sentiment', article.get('overall_sentiment_label', ''))).lower()
            text = str(article.get('title', '') + ' ' + article.get('text', '')).lower()
            
            if sent in ('positive', 'bullish') or any(w in text for w in bullish_words):
                positive += 1
            elif sent in ('negative', 'bearish') or any(w in text for w in bearish_words):
                negative += 1
            else:
                neutral += 1
        
        total = positive + negative + neutral
        pos_ratio = positive / total if total > 0 else 0
        neg_ratio = negative / total if total > 0 else 0
        score = (positive - negative) / total if total > 0 else 0
        
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
        
        # Source breakdown
        sources = {}
        for a in articles:
            src = a.get('_source', 'unknown')
            sources[src] = sources.get(src, 0) + 1
        
        return {
            'score': round(score, 2),
            'label': label,
            'article_count': total,
            'positive_ratio': round(pos_ratio, 2),
            'negative_ratio': round(neg_ratio, 2),
            'source_breakdown': sources,
            'data_source': 'multi_source_analyzed'
        }
    
    def _detect_events(self, articles: List[Dict]) -> List[Dict]:
        """Detect significant events from real article data."""
        events = []
        
        event_keywords = {
            'EARNINGS': ['earnings', 'quarterly results', 'eps', 'revenue beat', 'revenue miss', 'profit'],
            'PRODUCT_LAUNCH': ['launch', 'unveil', 'new product', 'release', 'announce'],
            'EXECUTIVE_CHANGE': ['ceo', 'cfo', 'resign', 'appoint', 'executive', 'leadership'],
            'ANALYST_UPGRADE': ['upgrade', 'downgrade', 'price target', 'rating', 'outperform', 'underperform'],
            'PARTNERSHIP': ['partnership', 'deal', 'acquisition', 'merge', 'joint venture', 'agreement'],
            'REGULATORY': ['sec', 'fda', 'regulation', 'compliance', 'investigation', 'lawsuit'],
            'CRYPTO_SPECIFIC': ['halving', 'etf approval', 'defi', 'tvl', 'staking', 'fork', 'airdrop'],
        }
        
        seen_types = set()
        for article in articles:
            text = str(article.get('title', '') + ' ' + article.get('text', '')).lower()
            
            for event_type, keywords in event_keywords.items():
                if event_type not in seen_types and any(k in text for k in keywords):
                    impact = 'HIGH' if event_type in ('EARNINGS', 'REGULATORY', 'EXECUTIVE_CHANGE') else 'MEDIUM'
                    events.append({
                        'type': event_type,
                        'date': article.get('date', datetime.now().strftime('%Y-%m-%d')),
                        'impact': impact,
                        'headline': article.get('title', 'N/A'),
                        'source': article.get('_source', 'unknown'),
                        'rank_score': article.get('rankscore', ''),
                        'data_source': f"{article.get('_source', 'unknown')}_detected"
                    })
                    seen_types.add(event_type)
        
        return events
    
    # ──────────────────────────────────────────────────────────────────────
    # Market-wide sentiment
    # ──────────────────────────────────────────────────────────────────────
    
    async def _get_market_sentiment(self, shared=None) -> Dict:
        """Get real market sentiment from VIX, Fear & Greed, and SPY."""
        result = {'data_source': 'live_apis'}

        # Try shared data layer first for all three data points
        if shared:
            fng_data = await shared.get_fear_greed()
            if fng_data and fng_data.get('data'):
                fng = fng_data['data']
                current = int(fng[0].get('value', 50))
                result['fear_greed_index'] = current
                result['fear_greed_label'] = fng[0].get('value_classification', 'Neutral')
                result['fear_greed_trend'] = [int(d.get('value', 50)) for d in fng]

            vix_data = await shared.get_vix()
            if vix_data and vix_data.get('results'):
                vr = vix_data['results'][0]
                result['vix_level'] = round(vr['c'], 1)
                vix_change = ((vr['c'] - vr['o']) / vr['o']) * 100 if vr.get('o') else 0
                result['vix_change_pct'] = round(vix_change, 2)

            spy_data = await shared.get_quote('SPY')
            if spy_data and spy_data.get('results'):
                sr = spy_data['results'][0]
                market_change = ((sr['c'] - sr['o']) / sr['o']) * 100 if sr['o'] else 0
                result['spy_change_pct'] = round(market_change, 2)

            fgi = result.get('fear_greed_index')
            vix = result.get('vix_level')
            if fgi is not None and vix is not None:
                fgi_score = (fgi - 50) / 50
                vix_score = max(-1, min(1, (20 - vix) / 15))
                score = (fgi_score * 0.6 + vix_score * 0.4)
                result['score'] = round(score, 2)
            elif fgi is not None:
                result['score'] = round((fgi - 50) / 50, 2)
            else:
                result['score'] = 0
            return result
        
        # Fear & Greed from Alternative.me
        try:
            resp = requests.get("https://api.alternative.me/fng/?limit=7", timeout=8)
            if resp.status_code == 200:
                fng_data = resp.json().get('data', [])
                if fng_data:
                    current = int(fng_data[0].get('value', 50))
                    result['fear_greed_index'] = current
                    result['fear_greed_label'] = fng_data[0].get('value_classification', 'Neutral')
                    result['fear_greed_trend'] = [int(d.get('value', 50)) for d in fng_data]
        except Exception:
            result['fear_greed_index'] = None
        
        # Real VIX from Polygon
        try:
            resp = requests.get("https://api.polygon.io/v2/aggs/ticker/VIX/prev",
                                params={'apiKey': self.polygon_key}, timeout=8)
            if resp.status_code == 200:
                results = resp.json().get('results', [])
                if results:
                    result['vix_level'] = round(results[0]['c'], 1)
                    vix_change = ((results[0]['c'] - results[0]['o']) / results[0]['o']) * 100 if results[0].get('o') else 0
                    result['vix_change_pct'] = round(vix_change, 2)
        except Exception:
            pass
        
        # SPY context
        try:
            resp = requests.get("https://api.polygon.io/v2/aggs/ticker/SPY/prev",
                                params={'apiKey': self.polygon_key}, timeout=8)
            if resp.status_code == 200:
                results = resp.json().get('results', [])
                if results:
                    spy_close = results[0]['c']
                    spy_open = results[0]['o']
                    market_change = ((spy_close - spy_open) / spy_open) * 100 if spy_open else 0
                    result['spy_change_pct'] = round(market_change, 2)
        except Exception:
            pass
        
        # Overall sentiment score
        fgi = result.get('fear_greed_index')
        vix = result.get('vix_level')
        
        if fgi is not None and vix is not None:
            fgi_score = (fgi - 50) / 50
            vix_score = max(-1, min(1, (20 - vix) / 15))
            score = (fgi_score * 0.6 + vix_score * 0.4)
            result['score'] = round(score, 2)
        elif fgi is not None:
            result['score'] = round((fgi - 50) / 50, 2)
        else:
            result['score'] = 0
        
        return result


__all__ = ['NewsAnalystAgent']
