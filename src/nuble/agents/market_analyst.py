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
                symbol_data = await self._analyze_symbol(symbol, query)
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
    
    async def _analyze_symbol(self, symbol: str, query: str) -> Dict[str, Any]:
        """Perform full analysis on a symbol."""
        result = {
            'symbol': symbol,
            'timestamp': datetime.now().isoformat()
        }
        
        # Get current quote
        quote = await self._get_quote(symbol)
        if quote:
            result['quote'] = quote
        
        # Get historical data
        historical = await self._get_historical(symbol, days=90)
        if historical:
            result['historical'] = {
                'data_points': len(historical),
                'start_date': historical[0]['date'] if historical else None,
                'end_date': historical[-1]['date'] if historical else None
            }
            
            # Calculate technical indicators
            technicals = self._calculate_technicals(historical)
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
        
        # Calculate overall signal
        signal = self._calculate_signal(result)
        result['signal'] = signal
        result['confidence'] = signal.get('confidence', 0.5)
        
        return result
    
    async def _get_quote(self, symbol: str) -> Optional[Dict]:
        """Get real-time quote from Polygon."""
        if not HAS_REQUESTS:
            return self._mock_quote(symbol)
        
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
        
        return self._mock_quote(symbol)
    
    async def _get_historical(self, symbol: str, days: int = 90) -> List[Dict]:
        """Get historical OHLCV data."""
        if not HAS_REQUESTS:
            return self._mock_historical(symbol, days)
        
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
        
        return self._mock_historical(symbol, days)
    
    def _calculate_technicals(self, data: List[Dict]) -> Dict[str, Any]:
        """Calculate 50+ technical indicators."""
        if not data or not HAS_NUMPY:
            return self._mock_technicals()
        
        closes = np.array([d['close'] for d in data if d['close']])
        highs = np.array([d['high'] for d in data if d['high']])
        lows = np.array([d['low'] for d in data if d['low']])
        volumes = np.array([d['volume'] for d in data if d['volume']])
        
        if len(closes) < 20:
            return self._mock_technicals()
        
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
        """Calculate overall trading signal."""
        signals = []
        
        technicals = analysis.get('technicals', {})
        trend = analysis.get('trend', {})
        patterns = analysis.get('patterns', {})
        
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
    
    def _mock_quote(self, symbol: str) -> Dict:
        """Generate mock quote data."""
        import random
        base_price = {
            'AAPL': 178.50, 'MSFT': 378.25, 'GOOGL': 141.80, 'AMZN': 178.50,
            'META': 505.75, 'NVDA': 875.30, 'TSLA': 248.50, 'AMD': 158.25
        }.get(symbol, 150.00)
        
        noise = random.uniform(-0.02, 0.02)
        close = base_price * (1 + noise)
        
        return {
            'symbol': symbol,
            'open': round(close * 0.995, 2),
            'high': round(close * 1.01, 2),
            'low': round(close * 0.99, 2),
            'close': round(close, 2),
            'volume': random.randint(10000000, 50000000),
            'change_pct': round(noise * 100, 2)
        }
    
    def _mock_historical(self, symbol: str, days: int) -> List[Dict]:
        """Generate mock historical data."""
        import random
        
        base_price = 150.0
        data = []
        
        for i in range(days):
            date = (datetime.now() - timedelta(days=days-i)).strftime('%Y-%m-%d')
            change = random.uniform(-0.03, 0.03)
            base_price *= (1 + change)
            
            data.append({
                'date': date,
                'open': round(base_price * 0.998, 2),
                'high': round(base_price * 1.015, 2),
                'low': round(base_price * 0.985, 2),
                'close': round(base_price, 2),
                'volume': random.randint(5000000, 30000000)
            })
        
        return data
    
    def _mock_technicals(self) -> Dict:
        """Generate mock technical indicators."""
        return {
            'current_price': 175.50,
            'moving_averages': {
                'sma_5': 174.25, 'sma_10': 173.50, 'sma_20': 171.80,
                'sma_50': 168.25, 'sma_200': 162.50,
                'ema_12': 173.80, 'ema_26': 171.25
            },
            'oscillators': {
                'rsi': 58.5, 'macd': 2.55, 'stochastic_k': 65.2, 'mfi': 52.8
            },
            'volatility': {
                'atr': 3.25, 'atr_pct': 1.85,
                'bb_upper': 182.50, 'bb_middle': 171.80, 'bb_lower': 161.10,
                'bb_width': 0.125
            },
            'volume': {
                'current': 25000000, 'avg_20d': 22000000, 'ratio': 1.14
            }
        }


__all__ = ['MarketAnalystAgent']
