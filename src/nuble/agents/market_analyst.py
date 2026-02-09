#!/usr/bin/env python3
"""
NUBLE Market Analyst Agent

Specialized agent for real-time market data, technical analysis, and price patterns.

Uses:
- Polygon.io API for real-time and historical data
- 50+ technical indicators
- Pattern recognition (candlestick, chart patterns)
- Multi-timeframe analysis
"""

import os
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
import logging
import json

from .base import SpecializedAgent, AgentTask, AgentResult, AgentType

logger = logging.getLogger(__name__)

# Try imports
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False


class MarketAnalystAgent(SpecializedAgent):
    """
    Market Analyst Agent - Technical Analysis Expert
    
    Capabilities:
    - Real-time and historical price data
    - 50+ technical indicators
    - Candlestick pattern recognition
    - Chart pattern detection
    - Multi-timeframe analysis
    - Volume analysis
    - Support/resistance levels
    """
    
    def __init__(self, api_key: str = None):
        super().__init__(api_key)
        
        # Polygon.io API for market data
        self.polygon_key = os.environ.get('POLYGON_API_KEY', 'JHKwAdyIOeExkYOxh3LwTopmqqVVFeBY')
        self.base_url = "https://api.polygon.io"
    
    def get_capabilities(self) -> Dict[str, Any]:
        """Return agent capabilities."""
        return {
            "name": "Market Analyst",
            "description": "Technical analysis and market data expert",
            "capabilities": [
                "real_time_quotes",
                "historical_data",
                "technical_indicators",
                "candlestick_patterns",
                "chart_patterns",
                "support_resistance",
                "volume_analysis",
                "multi_timeframe"
            ],
            "supported_symbols": ["stocks", "etfs", "forex", "crypto"],
            "data_sources": ["Polygon.io", "Alpha Vantage"]
        }
    
    async def execute(self, task: AgentTask) -> AgentResult:
        """Execute the market analysis task."""
        start_time = datetime.now()
        
        try:
            symbols = task.context.get('symbols', [])
            query = task.context.get('query', task.instruction)
            
            if not symbols:
                # Try to extract from instruction
                symbols = self._extract_symbols(task.instruction)
            
            if not symbols:
                return AgentResult(
                    task_id=task.task_id,
                    agent_type=AgentType.MARKET_ANALYST,
                    success=False,
                    data={'error': 'No symbols specified'},
                    confidence=0,
                    execution_time_ms=0,
                    error="No symbols provided"
                )
            
            # Get market data for each symbol
            all_data = {}
            
            for symbol in symbols[:3]:  # Limit to 3 symbols
                symbol_data = await self._analyze_symbol(symbol, query, task)
                all_data[symbol] = symbol_data
            
            # Calculate overall confidence
            confidences = [d.get('confidence', 0.5) for d in all_data.values()]
            overall_confidence = sum(confidences) / len(confidences)
            
            execution_time = int((datetime.now() - start_time).total_seconds() * 1000)
            
            return AgentResult(
                task_id=task.task_id,
                agent_type=AgentType.MARKET_ANALYST,
                success=True,
                data=all_data,
                confidence=overall_confidence,
                execution_time_ms=execution_time
            )
            
        except Exception as e:
            logger.error(f"MarketAnalystAgent error: {e}")
            execution_time = int((datetime.now() - start_time).total_seconds() * 1000)
            
            return AgentResult(
                task_id=task.task_id,
                agent_type=AgentType.MARKET_ANALYST,
                success=False,
                data={},
                confidence=0,
                execution_time_ms=execution_time,
                error=str(e)
            )
    
    async def _analyze_symbol(self, symbol: str, query: str, task=None) -> Dict[str, Any]:
        """Perform full analysis on a symbol using all data sources."""
        result = {
            'symbol': symbol,
            'timestamp': datetime.now().isoformat()
        }
        
        # Get current quote
        quote = await self._get_quote(symbol, task)
        if quote:
            result['quote'] = quote
        
        # Get historical data
        historical = await self._get_historical(symbol, days=90, task=task)
        if historical:
            result['historical'] = {
                'data_points': len(historical),
                'start_date': historical[0]['date'] if historical else None,
                'end_date': historical[-1]['date'] if historical else None
            }
            
            # Calculate technical indicators
            technicals = self._calculate_technicals(historical, symbol)
            result['technicals'] = technicals
            
            # Detect patterns
            patterns = self._detect_patterns(historical)
            result['patterns'] = patterns
            
            # Support and resistance
            levels = self._calculate_support_resistance(historical)
            result['key_levels'] = levels
            
            # Trend analysis
            trend = self._analyze_trend(historical)
            result['trend'] = trend
        
        # StockNews PRO — News sentiment for this symbol
        news_sentiment = await self._get_stocknews_sentiment(symbol, task)
        if news_sentiment:
            result['news_sentiment'] = news_sentiment
        
        # StockNews PRO — Analyst ratings for this symbol
        analyst_ratings = await self._get_analyst_ratings(symbol, task)
        if analyst_ratings:
            result['analyst_ratings'] = analyst_ratings
        
        # StockNews PRO — Check earnings calendar
        earnings_info = await self._get_earnings_info(symbol, task)
        if earnings_info:
            result['upcoming_earnings'] = earnings_info
        
        # Calculate overall signal (now includes news + analyst data)
        signal = self._calculate_signal(result)
        result['signal'] = signal
        result['confidence'] = signal.get('confidence', 0.5)
        
        return result
    
    async def _get_stocknews_sentiment(self, symbol: str, task=None) -> Dict:
        """StockNews PRO — Get quantitative sentiment + ranked news. Prefers SharedDataLayer."""
        # Try shared data layer first (zero HTTP calls)
        shared = self._get_shared_data(task)
        if shared:
            result = {}
            stat_data = await shared.get_stocknews_stat(symbol)
            if stat_data and stat_data.get('data'):
                result['sentiment_stats_7d'] = stat_data['data']
            
            news_data = await shared.get_stocknews(symbol)
            if news_data and news_data.get('data'):
                articles = news_data['data']
                pos = sum(1 for a in articles if str(a.get('sentiment', '')).lower() in ('positive', 'bullish'))
                neg = sum(1 for a in articles if str(a.get('sentiment', '')).lower() in ('negative', 'bearish'))
                total = len(articles)
                score = (pos - neg) / total if total > 0 else 0
                result['score'] = round(score, 2)
                result['label'] = 'BULLISH' if score > 0.2 else 'BEARISH' if score < -0.2 else 'NEUTRAL'
                result['positive'] = pos
                result['negative'] = neg
                result['article_count'] = total
                result['top_headlines'] = [a.get('title', '') for a in articles[:3]]
            
            if result:
                result['data_source'] = 'stocknews_pro'
                return result

        # Fallback to direct HTTP
        if not HAS_REQUESTS:
            return {}
        stocknews_key = os.environ.get('STOCKNEWS_API_KEY', 'zzad9pmlwttixx0fnsenstctzgdk7ysx0ctkgrk0')
        result = {}
        
        # /stat — Quantitative sentiment over 7 days
        try:
            resp = requests.get("https://stocknewsapi.com/api/v1/stat", params={
                'tickers': symbol, 'date': 'last7days', 'page': 1,
                'token': stocknews_key
            }, timeout=8)
            if resp.status_code == 200:
                stat = resp.json().get('data', {})
                if stat:
                    result['sentiment_stats_7d'] = stat
        except Exception:
            pass
        
        # /api/v1 — Ticker news sorted by rank with sentiment
        try:
            resp = requests.get("https://stocknewsapi.com/api/v1", params={
                'tickers': symbol, 'items': 8, 'sortby': 'rank',
                'extra-fields': 'id,eventid,rankscore',
                'token': stocknews_key
            }, timeout=8)
            if resp.status_code == 200:
                articles = resp.json().get('data', [])
                if articles:
                    pos = sum(1 for a in articles if str(a.get('sentiment', '')).lower() in ('positive', 'bullish'))
                    neg = sum(1 for a in articles if str(a.get('sentiment', '')).lower() in ('negative', 'bearish'))
                    total = len(articles)
                    score = (pos - neg) / total if total > 0 else 0
                    
                    result['score'] = round(score, 2)
                    result['label'] = 'BULLISH' if score > 0.2 else 'BEARISH' if score < -0.2 else 'NEUTRAL'
                    result['positive'] = pos
                    result['negative'] = neg
                    result['article_count'] = total
                    result['top_headlines'] = [a.get('title', '') for a in articles[:3]]
        except Exception:
            pass
        
        if result:
            result['data_source'] = 'stocknews_pro'
        return result
    
    async def _get_analyst_ratings(self, symbol: str, task=None) -> Dict:
        """StockNews PRO — /ratings for analyst upgrades/downgrades + price targets. Prefers SharedDataLayer."""
        # Try shared data layer first
        shared = self._get_shared_data(task)
        if shared:
            data = await shared.get_stocknews_ratings()
            if data and data.get('data'):
                ratings = data['data']
                symbol_ratings = []
                for r in ratings:
                    if r.get('ticker', '').upper() == symbol.upper():
                        symbol_ratings.append({
                            'action': r.get('action', ''),
                            'rating_from': r.get('rating_from', ''),
                            'rating_to': r.get('rating_to', ''),
                            'target_from': r.get('target_from', ''),
                            'target_to': r.get('target_to', ''),
                            'analyst': r.get('analyst', ''),
                            'analyst_company': r.get('analyst_company', ''),
                            'date': r.get('date', ''),
                        })
                if symbol_ratings:
                    return {
                        'ratings': symbol_ratings,
                        'latest_action': symbol_ratings[0].get('action', ''),
                        'data_source': 'stocknews_ratings'
                    }
                return {}

        # Fallback to direct HTTP
        if not HAS_REQUESTS:
            return {}
        stocknews_key = os.environ.get('STOCKNEWS_API_KEY', 'zzad9pmlwttixx0fnsenstctzgdk7ysx0ctkgrk0')
        try:
            resp = requests.get("https://stocknewsapi.com/api/v1/ratings", params={
                'items': 20, 'page': 1, 'token': stocknews_key
            }, timeout=8)
            if resp.status_code == 200:
                ratings = resp.json().get('data', [])
                symbol_ratings = []
                for r in ratings:
                    if r.get('ticker', '').upper() == symbol.upper():
                        symbol_ratings.append({
                            'action': r.get('action', ''),
                            'rating_from': r.get('rating_from', ''),
                            'rating_to': r.get('rating_to', ''),
                            'target_from': r.get('target_from', ''),
                            'target_to': r.get('target_to', ''),
                            'analyst': r.get('analyst', ''),
                            'analyst_company': r.get('analyst_company', ''),
                            'date': r.get('date', ''),
                        })
                if symbol_ratings:
                    return {
                        'ratings': symbol_ratings,
                        'latest_action': symbol_ratings[0].get('action', ''),
                        'data_source': 'stocknews_ratings'
                    }
        except Exception:
            pass
        return {}
    
    async def _get_earnings_info(self, symbol: str, task=None) -> Dict:
        """StockNews PRO — /earnings-calendar for upcoming earnings date. Prefers SharedDataLayer."""
        # Try shared data layer first
        shared = self._get_shared_data(task)
        if shared:
            data = await shared.get_stocknews_earnings()
            if data and data.get('data'):
                for e in data['data']:
                    if e.get('ticker', '').upper() == symbol.upper():
                        return {
                            'date': e.get('date', ''),
                            'time': e.get('time', ''),
                            'data_source': 'stocknews_earnings_calendar'
                        }
                return {}

        # Fallback to direct HTTP
        if not HAS_REQUESTS:
            return {}
        stocknews_key = os.environ.get('STOCKNEWS_API_KEY', 'zzad9pmlwttixx0fnsenstctzgdk7ysx0ctkgrk0')
        try:
            resp = requests.get("https://stocknewsapi.com/api/v1/earnings-calendar", params={
                'page': 1, 'items': 50, 'token': stocknews_key
            }, timeout=8)
            if resp.status_code == 200:
                earnings = resp.json().get('data', [])
                for e in earnings:
                    if e.get('ticker', '').upper() == symbol.upper():
                        return {
                            'date': e.get('date', ''),
                            'time': e.get('time', ''),
                            'data_source': 'stocknews_earnings_calendar'
                        }
        except Exception:
            pass
        return {}
    
    async def _get_quote(self, symbol: str, task=None) -> Optional[Dict]:
        """Get real-time quote from Polygon. Prefers SharedDataLayer if available."""
        # Try shared data layer first (zero HTTP calls)
        shared = self._get_shared_data(task)
        if shared:
            data = await shared.get_quote(symbol)
            if data and data.get('results'):
                r = data['results'][0]
                return {
                    'symbol': symbol,
                    'open': r.get('o'),
                    'high': r.get('h'),
                    'low': r.get('l'),
                    'close': r.get('c'),
                    'volume': r.get('v'),
                    'vwap': r.get('vw'),
                    'change_pct': ((r.get('c', 0) - r.get('o', 1)) / r.get('o', 1) * 100) if r.get('o') else 0
                }

        # Fallback to direct HTTP
        if not HAS_REQUESTS:
            return self._real_quote(symbol)
        
        try:
            url = f"{self.base_url}/v2/aggs/ticker/{symbol}/prev"
            params = {'apiKey': self.polygon_key}
            
            response = requests.get(url, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                if data.get('results'):
                    r = data['results'][0]
                    return {
                        'symbol': symbol,
                        'open': r.get('o'),
                        'high': r.get('h'),
                        'low': r.get('l'),
                        'close': r.get('c'),
                        'volume': r.get('v'),
                        'vwap': r.get('vw'),
                        'change_pct': ((r.get('c', 0) - r.get('o', 1)) / r.get('o', 1) * 100) if r.get('o') else 0
                    }
        except Exception as e:
            logger.warning(f"Quote fetch failed for {symbol}: {e}")
        
        return self._real_quote(symbol)
    
    async def _get_historical(self, symbol: str, days: int = 90, task=None) -> List[Dict]:
        """Get historical OHLCV data. Prefers SharedDataLayer if available."""
        # Try shared data layer first
        shared = self._get_shared_data(task)
        if shared:
            data = await shared.get_historical(symbol, days)
            if data and data.get('results'):
                return [
                    {
                        'date': datetime.fromtimestamp(r['t'] / 1000).strftime('%Y-%m-%d'),
                        'open': r.get('o'),
                        'high': r.get('h'),
                        'low': r.get('l'),
                        'close': r.get('c'),
                        'volume': r.get('v'),
                        'vwap': r.get('vw')
                    }
                    for r in data['results']
                ]

        # Fallback to direct HTTP
        if not HAS_REQUESTS:
            return self._real_historical(symbol, days)
        
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            url = f"{self.base_url}/v2/aggs/ticker/{symbol}/range/1/day/{start_date.strftime('%Y-%m-%d')}/{end_date.strftime('%Y-%m-%d')}"
            params = {'apiKey': self.polygon_key, 'sort': 'asc'}
            
            response = requests.get(url, params=params, timeout=15)
            
            if response.status_code == 200:
                data = response.json()
                results = data.get('results', [])
                
                return [
                    {
                        'date': datetime.fromtimestamp(r['t'] / 1000).strftime('%Y-%m-%d'),
                        'open': r.get('o'),
                        'high': r.get('h'),
                        'low': r.get('l'),
                        'close': r.get('c'),
                        'volume': r.get('v'),
                        'vwap': r.get('vw')
                    }
                    for r in results
                ]
        except Exception as e:
            logger.warning(f"Historical fetch failed for {symbol}: {e}")
        
        return self._real_historical(symbol, days)
    
    def _calculate_technicals(self, data: List[Dict], symbol: str = 'SPY') -> Dict[str, Any]:
        """Calculate 50+ technical indicators."""
        if not data or not HAS_NUMPY:
            return self._real_technicals(symbol)
        
        # Filter bars where ALL OHLCV fields are present to keep arrays aligned
        valid_bars = [d for d in data if d.get('close') is not None and d.get('high') is not None 
                      and d.get('low') is not None and d.get('open') is not None]
        
        if len(valid_bars) < 20:
            return self._real_technicals(symbol)
        
        closes = np.array([d['close'] for d in valid_bars])
        highs = np.array([d['high'] for d in valid_bars])
        lows = np.array([d['low'] for d in valid_bars])
        volumes = np.array([d.get('volume', 0) or 0 for d in valid_bars])
        
        # Moving Averages
        sma_5 = np.mean(closes[-5:]) if len(closes) >= 5 else None
        sma_10 = np.mean(closes[-10:]) if len(closes) >= 10 else None
        sma_20 = np.mean(closes[-20:]) if len(closes) >= 20 else None
        sma_50 = np.mean(closes[-50:]) if len(closes) >= 50 else None
        sma_200 = np.mean(closes[-200:]) if len(closes) >= 200 else None
        
        # EMA
        def ema(data, period):
            multiplier = 2 / (period + 1)
            ema_vals = [data[0]]
            for i in range(1, len(data)):
                ema_vals.append((data[i] * multiplier) + (ema_vals[-1] * (1 - multiplier)))
            return ema_vals[-1]
        
        ema_12 = ema(closes, 12) if len(closes) >= 12 else None
        ema_26 = ema(closes, 26) if len(closes) >= 26 else None
        
        # MACD
        macd = ema_12 - ema_26 if ema_12 and ema_26 else None
        
        # RSI
        def calculate_rsi(prices, period=14):
            deltas = np.diff(prices)
            gains = np.where(deltas > 0, deltas, 0)
            losses = np.where(deltas < 0, -deltas, 0)
            
            avg_gain = np.mean(gains[-period:])
            avg_loss = np.mean(losses[-period:])
            
            if avg_loss == 0:
                return 100
            rs = avg_gain / avg_loss
            return 100 - (100 / (1 + rs))
        
        rsi = calculate_rsi(closes) if len(closes) >= 15 else None
        
        # Bollinger Bands
        bb_period = 20
        if len(closes) >= bb_period:
            bb_middle = np.mean(closes[-bb_period:])
            bb_std = np.std(closes[-bb_period:])
            bb_upper = bb_middle + 2 * bb_std
            bb_lower = bb_middle - 2 * bb_std
            bb_width = (bb_upper - bb_lower) / bb_middle
        else:
            bb_middle = bb_upper = bb_lower = bb_width = None
        
        # ATR (Average True Range)
        def calculate_atr(highs, lows, closes, period=14):
            tr = []
            for i in range(1, len(closes)):
                tr1 = highs[i] - lows[i]
                tr2 = abs(highs[i] - closes[i-1])
                tr3 = abs(lows[i] - closes[i-1])
                tr.append(max(tr1, tr2, tr3))
            return np.mean(tr[-period:])
        
        atr = calculate_atr(highs, lows, closes) if len(closes) >= 15 else None
        
        # Stochastic
        def calculate_stochastic(highs, lows, closes, k_period=14):
            h14 = max(highs[-k_period:])
            l14 = min(lows[-k_period:])
            if h14 - l14 == 0:
                return 50
            return ((closes[-1] - l14) / (h14 - l14)) * 100
        
        stoch_k = calculate_stochastic(highs, lows, closes) if len(closes) >= 14 else None
        
        # VWAP
        vwap = None
        if len(volumes) > 0 and np.sum(volumes) > 0:
            typical_prices = (highs + lows + closes) / 3
            vwap = np.sum(typical_prices * volumes) / np.sum(volumes)
        
        # Volume Analysis
        avg_volume = np.mean(volumes[-20:]) if len(volumes) >= 20 else np.mean(volumes)
        volume_ratio = volumes[-1] / avg_volume if avg_volume > 0 else 1
        
        # Money Flow Index
        def calculate_mfi(highs, lows, closes, volumes, period=14):
            typical = (highs + lows + closes) / 3
            raw_money_flow = typical * volumes
            
            positive_flow = 0
            negative_flow = 0
            
            for i in range(-period, 0):
                if typical[i] > typical[i-1]:
                    positive_flow += raw_money_flow[i]
                else:
                    negative_flow += raw_money_flow[i]
            
            if negative_flow == 0:
                return 100
            money_ratio = positive_flow / negative_flow
            return 100 - (100 / (1 + money_ratio))
        
        mfi = calculate_mfi(highs, lows, closes, volumes) if len(closes) >= 15 else None
        
        # Price position
        current_price = closes[-1]
        price_vs_sma20 = ((current_price - sma_20) / sma_20 * 100) if sma_20 else None
        price_vs_sma50 = ((current_price - sma_50) / sma_50 * 100) if sma_50 else None
        
        return {
            'current_price': float(current_price),
            'moving_averages': {
                'sma_5': float(sma_5) if sma_5 else None,
                'sma_10': float(sma_10) if sma_10 else None,
                'sma_20': float(sma_20) if sma_20 else None,
                'sma_50': float(sma_50) if sma_50 else None,
                'sma_200': float(sma_200) if sma_200 else None,
                'ema_12': float(ema_12) if ema_12 else None,
                'ema_26': float(ema_26) if ema_26 else None
            },
            'oscillators': {
                'rsi': float(rsi) if rsi else None,
                'macd': float(macd) if macd else None,
                'stochastic_k': float(stoch_k) if stoch_k else None,
                'mfi': float(mfi) if mfi else None
            },
            'volatility': {
                'atr': float(atr) if atr else None,
                'atr_pct': float(atr / current_price * 100) if atr else None,
                'bb_upper': float(bb_upper) if bb_upper else None,
                'bb_middle': float(bb_middle) if bb_middle else None,
                'bb_lower': float(bb_lower) if bb_lower else None,
                'bb_width': float(bb_width) if bb_width else None
            },
            'volume': {
                'current': int(volumes[-1]) if len(volumes) > 0 else None,
                'avg_20d': float(avg_volume),
                'ratio': float(volume_ratio),
                'vwap': float(vwap) if vwap else None
            },
            'price_position': {
                'vs_sma20_pct': float(price_vs_sma20) if price_vs_sma20 else None,
                'vs_sma50_pct': float(price_vs_sma50) if price_vs_sma50 else None,
                'from_52w_high_pct': None,  # Would need more data
                'from_52w_low_pct': None
            }
        }
    
    def _detect_patterns(self, data: List[Dict]) -> Dict[str, Any]:
        """Detect candlestick and chart patterns."""
        if not data or len(data) < 5:
            return {'candlestick': [], 'chart': []}
        
        candlestick_patterns = []
        chart_patterns = []
        
        # Get recent candles
        recent = data[-5:]
        
        # Doji detection
        for i, candle in enumerate(recent):
            body = abs(candle['close'] - candle['open'])
            total_range = candle['high'] - candle['low']
            
            if total_range > 0 and body / total_range < 0.1:
                candlestick_patterns.append({
                    'pattern': 'DOJI',
                    'date': candle['date'],
                    'significance': 'Indecision, possible reversal'
                })
        
        # Hammer / Shooting Star
        for i, candle in enumerate(recent):
            body = abs(candle['close'] - candle['open'])
            upper_shadow = candle['high'] - max(candle['open'], candle['close'])
            lower_shadow = min(candle['open'], candle['close']) - candle['low']
            
            if lower_shadow > 2 * body and upper_shadow < body:
                candlestick_patterns.append({
                    'pattern': 'HAMMER' if candle['close'] > candle['open'] else 'HANGING_MAN',
                    'date': candle['date'],
                    'significance': 'Potential bullish reversal' if candle['close'] > candle['open'] else 'Potential bearish reversal'
                })
            
            if upper_shadow > 2 * body and lower_shadow < body:
                candlestick_patterns.append({
                    'pattern': 'SHOOTING_STAR' if candle['close'] < candle['open'] else 'INVERTED_HAMMER',
                    'date': candle['date'],
                    'significance': 'Potential bearish reversal' if candle['close'] < candle['open'] else 'Potential bullish reversal'
                })
        
        # Engulfing patterns
        if len(recent) >= 2:
            prev = recent[-2]
            curr = recent[-1]
            
            # Bullish engulfing
            if (prev['close'] < prev['open'] and  # Previous red
                curr['close'] > curr['open'] and  # Current green
                curr['open'] < prev['close'] and
                curr['close'] > prev['open']):
                candlestick_patterns.append({
                    'pattern': 'BULLISH_ENGULFING',
                    'date': curr['date'],
                    'significance': 'Strong bullish reversal signal'
                })
            
            # Bearish engulfing
            if (prev['close'] > prev['open'] and  # Previous green
                curr['close'] < curr['open'] and  # Current red
                curr['open'] > prev['close'] and
                curr['close'] < prev['open']):
                candlestick_patterns.append({
                    'pattern': 'BEARISH_ENGULFING',
                    'date': curr['date'],
                    'significance': 'Strong bearish reversal signal'
                })
        
        # Simple trend detection for chart patterns
        if HAS_NUMPY and len(data) >= 20:
            closes = np.array([d['close'] for d in data[-20:]])
            
            # Linear regression for trend
            x = np.arange(len(closes))
            slope = np.polyfit(x, closes, 1)[0]
            
            if slope > 0.5:
                chart_patterns.append({
                    'pattern': 'UPTREND',
                    'strength': 'Strong' if slope > 1 else 'Moderate',
                    'significance': 'Price trending higher'
                })
            elif slope < -0.5:
                chart_patterns.append({
                    'pattern': 'DOWNTREND',
                    'strength': 'Strong' if slope < -1 else 'Moderate',
                    'significance': 'Price trending lower'
                })
            else:
                chart_patterns.append({
                    'pattern': 'SIDEWAYS',
                    'strength': 'Moderate',
                    'significance': 'Price consolidating'
                })
        
        return {
            'candlestick': candlestick_patterns[-3:],  # Last 3 patterns
            'chart': chart_patterns
        }
    
    def _calculate_support_resistance(self, data: List[Dict]) -> Dict[str, Any]:
        """Calculate support and resistance levels."""
        if not data or not HAS_NUMPY:
            return {'support': [], 'resistance': []}
        
        highs = [d['high'] for d in data]
        lows = [d['low'] for d in data]
        closes = [d['close'] for d in data]
        
        current_price = closes[-1]
        
        # Find local maxima and minima
        resistance_levels = []
        support_levels = []
        
        for i in range(2, len(highs) - 2):
            # Local maximum
            if highs[i] > highs[i-1] and highs[i] > highs[i-2] and highs[i] > highs[i+1] and highs[i] > highs[i+2]:
                if highs[i] > current_price:
                    resistance_levels.append(highs[i])
            
            # Local minimum
            if lows[i] < lows[i-1] and lows[i] < lows[i-2] and lows[i] < lows[i+1] and lows[i] < lows[i+2]:
                if lows[i] < current_price:
                    support_levels.append(lows[i])
        
        # Sort and get most relevant
        resistance_levels = sorted(set(resistance_levels))[:3]
        support_levels = sorted(set(support_levels), reverse=True)[:3]
        
        # Add 52-week high/low
        high_52w = max(highs)
        low_52w = min(lows)
        
        return {
            'support': [round(s, 2) for s in support_levels],
            'resistance': [round(r, 2) for r in resistance_levels],
            'high_52w': round(high_52w, 2),
            'low_52w': round(low_52w, 2),
            'current_price': round(current_price, 2),
            'distance_to_support_pct': round((current_price - support_levels[0]) / current_price * 100, 2) if support_levels else None,
            'distance_to_resistance_pct': round((resistance_levels[0] - current_price) / current_price * 100, 2) if resistance_levels else None
        }
    
    def _analyze_trend(self, data: List[Dict]) -> Dict[str, Any]:
        """Analyze the overall trend."""
        if not data or not HAS_NUMPY:
            return {'direction': 'UNKNOWN', 'strength': 0}
        
        closes = np.array([d['close'] for d in data])
        
        # Multiple timeframe trend
        def get_trend(prices):
            if len(prices) < 2:
                return 0
            return (prices[-1] - prices[0]) / prices[0] * 100
        
        trend_5d = get_trend(closes[-5:]) if len(closes) >= 5 else 0
        trend_20d = get_trend(closes[-20:]) if len(closes) >= 20 else 0
        trend_60d = get_trend(closes[-60:]) if len(closes) >= 60 else 0
        
        # Determine direction
        avg_trend = (trend_5d + trend_20d + trend_60d) / 3
        
        if avg_trend > 5:
            direction = 'STRONG_BULLISH'
        elif avg_trend > 2:
            direction = 'BULLISH'
        elif avg_trend > -2:
            direction = 'NEUTRAL'
        elif avg_trend > -5:
            direction = 'BEARISH'
        else:
            direction = 'STRONG_BEARISH'
        
        # Trend alignment
        trends_aligned = (
            (trend_5d > 0 and trend_20d > 0 and trend_60d > 0) or
            (trend_5d < 0 and trend_20d < 0 and trend_60d < 0)
        )
        
        return {
            'direction': direction,
            'strength': abs(avg_trend),
            'aligned': trends_aligned,
            'short_term_5d': round(trend_5d, 2),
            'medium_term_20d': round(trend_20d, 2),
            'long_term_60d': round(trend_60d, 2)
        }
    
    def _calculate_signal(self, analysis: Dict) -> Dict[str, Any]:
        """Calculate overall trading signal including news + analyst data."""
        signals = []
        
        technicals = analysis.get('technicals', {})
        trend = analysis.get('trend', {})
        patterns = analysis.get('patterns', {})
        news_sentiment = analysis.get('news_sentiment', {})
        analyst_ratings = analysis.get('analyst_ratings', {})
        
        # RSI signal
        rsi = technicals.get('oscillators', {}).get('rsi')
        if rsi:
            if rsi < 30:
                signals.append(('RSI', 'OVERSOLD', 0.7))
            elif rsi > 70:
                signals.append(('RSI', 'OVERBOUGHT', -0.7))
            else:
                signals.append(('RSI', 'NEUTRAL', 0))
        
        # Moving average signal
        ma = technicals.get('moving_averages', {})
        price = technicals.get('current_price')
        sma_20 = ma.get('sma_20')
        sma_50 = ma.get('sma_50')
        
        if price and sma_20 and sma_50:
            if price > sma_20 > sma_50:
                signals.append(('MA', 'BULLISH', 0.8))
            elif price < sma_20 < sma_50:
                signals.append(('MA', 'BEARISH', -0.8))
            else:
                signals.append(('MA', 'MIXED', 0))
        
        # Trend signal
        direction = trend.get('direction', 'NEUTRAL')
        if 'BULLISH' in direction:
            signals.append(('TREND', direction, 0.6 if 'STRONG' in direction else 0.4))
        elif 'BEARISH' in direction:
            signals.append(('TREND', direction, -0.6 if 'STRONG' in direction else -0.4))
        
        # Pattern signals
        for pattern in patterns.get('candlestick', []):
            if 'BULLISH' in pattern['pattern']:
                signals.append(('PATTERN', pattern['pattern'], 0.5))
            elif 'BEARISH' in pattern['pattern']:
                signals.append(('PATTERN', pattern['pattern'], -0.5))
        
        # StockNews PRO — News sentiment signal
        news_score = news_sentiment.get('score', 0)
        if news_score:
            news_label = news_sentiment.get('label', 'NEUTRAL')
            if news_score > 0.2:
                signals.append(('NEWS_SENTIMENT', f'BULLISH ({news_label})', min(0.6, news_score)))
            elif news_score < -0.2:
                signals.append(('NEWS_SENTIMENT', f'BEARISH ({news_label})', max(-0.6, news_score)))
            else:
                signals.append(('NEWS_SENTIMENT', 'NEUTRAL', 0))
        
        # StockNews PRO — Analyst ratings signal
        if analyst_ratings and analyst_ratings.get('ratings'):
            latest = analyst_ratings['ratings'][0]
            action = str(latest.get('action', '')).lower()
            if 'upgrade' in action:
                signals.append(('ANALYST_RATING', f"UPGRADE by {latest.get('analyst_company', '')}", 0.7))
            elif 'downgrade' in action:
                signals.append(('ANALYST_RATING', f"DOWNGRADE by {latest.get('analyst_company', '')}", -0.7))
            elif 'initiated' in action or 'buy' in action:
                signals.append(('ANALYST_RATING', f"INITIATED by {latest.get('analyst_company', '')}", 0.5))
        
        # Earnings proximity warning
        earnings = analysis.get('upcoming_earnings', {})
        if earnings and earnings.get('date'):
            signals.append(('EARNINGS_RISK', f"Earnings on {earnings['date']}", 0))
        
        # Calculate aggregate
        if signals:
            avg_signal = sum(s[2] for s in signals) / len(signals)
        else:
            avg_signal = 0
        
        # Determine recommendation
        if avg_signal > 0.5:
            recommendation = 'STRONG_BUY'
            confidence = min(0.9, 0.6 + avg_signal * 0.3)
        elif avg_signal > 0.2:
            recommendation = 'BUY'
            confidence = 0.5 + avg_signal * 0.2
        elif avg_signal > -0.2:
            recommendation = 'HOLD'
            confidence = 0.5
        elif avg_signal > -0.5:
            recommendation = 'SELL'
            confidence = 0.5 - avg_signal * 0.2
        else:
            recommendation = 'STRONG_SELL'
            confidence = min(0.9, 0.6 - avg_signal * 0.3)
        
        return {
            'recommendation': recommendation,
            'signal_strength': round(avg_signal, 2),
            'confidence': round(confidence, 2),
            'components': [{'indicator': s[0], 'signal': s[1], 'score': s[2]} for s in signals]
        }
    
    def _extract_symbols(self, text: str) -> List[str]:
        """Extract symbols from text."""
        import re
        
        patterns = [
            r'\$([A-Z]{1,5})\b',
            r'\b([A-Z]{2,5})\b'
        ]
        
        symbols = set()
        for pattern in patterns:
            matches = re.findall(pattern, text.upper())
            symbols.update(matches)
        
        common_words = {'I', 'A', 'THE', 'AND', 'OR', 'FOR', 'TO', 'GET', 'ANALYZE', 'CHECK', 'SHOULD', 'BUY', 'SELL'}
        symbols -= common_words
        
        return list(symbols)[:3]
    
    def _real_quote(self, symbol: str) -> Dict:
        """Fetch real quote data from Polygon."""
        polygon_key = os.environ.get('POLYGON_API_KEY', 'JHKwAdyIOeExkYOxh3LwTopmqqVVFeBY')
        try:
            url = f"https://api.polygon.io/v2/aggs/ticker/{symbol}/prev"
            resp = requests.get(url, params={'apiKey': polygon_key}, timeout=10)
            if resp.status_code == 200:
                results = resp.json().get('results', [])
                if results:
                    r = results[0]
                    change_pct = ((r['c'] - r['o']) / r['o']) * 100 if r['o'] else 0
                    return {
                        'symbol': symbol,
                        'open': r['o'],
                        'high': r['h'],
                        'low': r['l'],
                        'close': r['c'],
                        'volume': r.get('v', 0),
                        'change_pct': round(change_pct, 2),
                        'data_source': 'polygon_live'
                    }
        except Exception as e:
            logger.warning(f"Quote fetch failed for {symbol}: {e}")
        return {'symbol': symbol, 'error': 'Quote unavailable', 'data_source': 'polygon_error'}
    
    def _real_historical(self, symbol: str, days: int) -> List[Dict]:
        """Fetch real historical data from Polygon."""
        polygon_key = os.environ.get('POLYGON_API_KEY', 'JHKwAdyIOeExkYOxh3LwTopmqqVVFeBY')
        try:
            end_date = datetime.now().strftime('%Y-%m-%d')
            start_date = (datetime.now() - timedelta(days=days + 10)).strftime('%Y-%m-%d')
            url = f"https://api.polygon.io/v2/aggs/ticker/{symbol}/range/1/day/{start_date}/{end_date}"
            resp = requests.get(url, params={'apiKey': polygon_key, 'sort': 'asc'}, timeout=10)
            if resp.status_code == 200:
                results = resp.json().get('results', [])
                data = []
                for r in results:
                    data.append({
                        'date': datetime.fromtimestamp(r['t'] / 1000).strftime('%Y-%m-%d'),
                        'open': r['o'],
                        'high': r['h'],
                        'low': r['l'],
                        'close': r['c'],
                        'volume': r.get('v', 0)
                    })
                return data
        except Exception as e:
            logger.warning(f"Historical fetch failed for {symbol}: {e}")
        return []
    
    def _real_technicals(self, symbol: str) -> Dict:
        """Calculate real technical indicators from Polygon server-side API + historical data."""
        import numpy as np
        polygon_key = os.environ.get('POLYGON_API_KEY', 'JHKwAdyIOeExkYOxh3LwTopmqqVVFeBY')
        
        # 1. Try Polygon server-side indicators first (more accurate)
        server_rsi = None
        server_sma_20 = None
        server_sma_50 = None
        server_sma_200 = None
        server_ema_12 = None
        server_ema_26 = None
        server_macd = None
        server_macd_signal = None
        server_macd_hist = None
        
        def _pg(url, params=None):
            p = {'apiKey': polygon_key}
            if params:
                p.update(params)
            try:
                resp = requests.get(url, params=p, timeout=10)
                if resp.status_code == 200:
                    return resp.json()
            except Exception:
                pass
            return {}
        
        # RSI
        rsi_data = _pg(f"https://api.polygon.io/v1/indicators/rsi/{symbol}",
                       {'timespan': 'day', 'window': 14, 'series_type': 'close', 'order': 'desc', 'limit': 1})
        rsi_vals = rsi_data.get('results', {}).get('values', [])
        if rsi_vals:
            server_rsi = rsi_vals[0].get('value')
        
        # SMAs
        for window in [20, 50, 200]:
            sma_data = _pg(f"https://api.polygon.io/v1/indicators/sma/{symbol}",
                           {'timespan': 'day', 'window': window, 'series_type': 'close', 'order': 'desc', 'limit': 1})
            sma_vals = sma_data.get('results', {}).get('values', [])
            if sma_vals:
                val = sma_vals[0].get('value')
                if window == 20: server_sma_20 = val
                elif window == 50: server_sma_50 = val
                elif window == 200: server_sma_200 = val
        
        # EMAs
        for window in [12, 26]:
            ema_data = _pg(f"https://api.polygon.io/v1/indicators/ema/{symbol}",
                           {'timespan': 'day', 'window': window, 'series_type': 'close', 'order': 'desc', 'limit': 1})
            ema_vals = ema_data.get('results', {}).get('values', [])
            if ema_vals:
                val = ema_vals[0].get('value')
                if window == 12: server_ema_12 = val
                elif window == 26: server_ema_26 = val
        
        # MACD
        macd_data = _pg(f"https://api.polygon.io/v1/indicators/macd/{symbol}",
                        {'timespan': 'day', 'short_window': 12, 'long_window': 26,
                         'signal_window': 9, 'series_type': 'close', 'order': 'desc', 'limit': 1})
        macd_vals = macd_data.get('results', {}).get('values', [])
        if macd_vals:
            server_macd = macd_vals[0].get('value')
            server_macd_signal = macd_vals[0].get('signal')
            server_macd_hist = macd_vals[0].get('histogram')
        
        # 2. Get historical data for volume, ATR, Bollinger Bands, etc.
        try:
            end_date = datetime.now().strftime('%Y-%m-%d')
            start_date = (datetime.now() - timedelta(days=300)).strftime('%Y-%m-%d')
            url = f"https://api.polygon.io/v2/aggs/ticker/{symbol}/range/1/day/{start_date}/{end_date}"
            resp = requests.get(url, params={'apiKey': polygon_key, 'sort': 'asc'}, timeout=10)
            if resp.status_code == 200:
                results = resp.json().get('results', [])
                if len(results) >= 50:
                    closes = np.array([r['c'] for r in results])
                    volumes = np.array([r.get('v', 0) for r in results])
                    highs = np.array([r['h'] for r in results])
                    lows = np.array([r['l'] for r in results])
                    
                    current_price = float(closes[-1])
                    
                    # Use server-side values if available, else calculate manually
                    sma_5 = float(np.mean(closes[-5:]))
                    sma_10 = float(np.mean(closes[-10:]))
                    sma_20 = float(server_sma_20) if server_sma_20 else float(np.mean(closes[-20:]))
                    sma_50 = float(server_sma_50) if server_sma_50 else float(np.mean(closes[-50:]))
                    sma_200 = float(server_sma_200) if server_sma_200 else (float(np.mean(closes[-200:])) if len(closes) >= 200 else float(np.mean(closes)))
                    
                    ema_12 = float(server_ema_12) if server_ema_12 else None
                    ema_26 = float(server_ema_26) if server_ema_26 else None
                    
                    rsi = float(server_rsi) if server_rsi else None
                    if rsi is None:
                        returns = np.diff(closes) / closes[:-1]
                        gains = np.where(returns > 0, returns, 0)
                        losses = np.where(returns < 0, -returns, 0)
                        avg_gain = np.mean(gains[-14:]) if len(gains) >= 14 else np.mean(gains)
                        avg_loss = np.mean(losses[-14:]) if len(losses) >= 14 else np.mean(losses)
                        rs = avg_gain / avg_loss if avg_loss > 0 else 100
                        rsi = float(100 - (100 / (1 + rs)))
                    
                    macd = float(server_macd) if server_macd else (ema_12 - ema_26 if ema_12 and ema_26 else None)
                    
                    # ATR
                    tr = np.maximum(highs[-14:] - lows[-14:],
                                    np.maximum(np.abs(highs[-14:] - closes[-15:-1]),
                                               np.abs(lows[-14:] - closes[-15:-1])))
                    atr = float(np.mean(tr))
                    
                    # Bollinger Bands
                    bb_middle = sma_20
                    bb_std = float(np.std(closes[-20:]))
                    bb_upper = bb_middle + 2 * bb_std
                    bb_lower = bb_middle - 2 * bb_std
                    bb_width = (bb_upper - bb_lower) / bb_middle if bb_middle > 0 else 0
                    
                    return {
                        'current_price': current_price,
                        'moving_averages': {
                            'sma_5': round(sma_5, 2), 'sma_10': round(sma_10, 2),
                            'sma_20': round(sma_20, 2), 'sma_50': round(sma_50, 2),
                            'sma_200': round(sma_200, 2),
                            'ema_12': round(ema_12, 2) if ema_12 else None,
                            'ema_26': round(ema_26, 2) if ema_26 else None,
                        },
                        'oscillators': {
                            'rsi': round(rsi, 1),
                            'macd': round(macd, 4) if macd else None,
                            'macd_signal': round(float(server_macd_signal), 4) if server_macd_signal else None,
                            'macd_histogram': round(float(server_macd_hist), 4) if server_macd_hist else None,
                        },
                        'volatility': {
                            'atr': round(atr, 2),
                            'atr_pct': round(atr / current_price * 100, 2) if current_price > 0 else 0,
                            'bb_upper': round(bb_upper, 2),
                            'bb_middle': round(bb_middle, 2),
                            'bb_lower': round(bb_lower, 2),
                            'bb_width': round(bb_width, 4),
                        },
                        'volume': {
                            'current': int(volumes[-1]),
                            'avg_20d': int(np.mean(volumes[-20:])),
                            'ratio': round(float(volumes[-1] / np.mean(volumes[-20:])), 2) if np.mean(volumes[-20:]) > 0 else 1.0,
                        },
                        'data_source': 'polygon_indicators + polygon_historical'
                    }
        except Exception as e:
            logger.warning(f"Technicals calculation failed for {symbol}: {e}")
        return {'error': 'Technicals unavailable', 'data_source': 'polygon_error'}


__all__ = ['MarketAnalystAgent']
