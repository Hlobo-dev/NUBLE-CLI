#!/usr/bin/env python3
"""
NUBLE Quant Analyst Agent

Specialized agent for ML signals, factor models, and quantitative analysis.
Integrates with the production ML pipeline (46M+ parameters).
"""

import os
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional

import numpy as np

from .base import SpecializedAgent, AgentTask, AgentResult, AgentType

logger = logging.getLogger(__name__)

# Lazy-load ML components
_ml_predictor = None
_regime_detector = None


def _get_ml_predictor():
    """Lazy load ML predictor."""
    global _ml_predictor
    if _ml_predictor is None:
        try:
            from ...institutional.ml import get_predictor
            _ml_predictor = get_predictor()
            logger.info("ML predictor loaded for QuantAnalyst")
        except Exception as e:
            logger.warning(f"Could not load ML predictor: {e}")
    return _ml_predictor


def _get_regime_detector():
    """Lazy load regime detector."""
    global _regime_detector
    if _regime_detector is None:
        try:
            from ...institutional.ml.regime import MarketRegimeDetector
            _regime_detector = MarketRegimeDetector()
            logger.info("Regime detector loaded for QuantAnalyst")
        except Exception as e:
            logger.warning(f"Could not load regime detector: {e}")
    return _regime_detector


class QuantAnalystAgent(SpecializedAgent):
    """
    Quant Analyst Agent - ML Signals & Factor Models
    
    PRODUCTION INTEGRATION:
    - Uses real ML models (LSTM, Transformer, Ensemble) from institutional/ml/
    - Uses real regime detection (HMM-based)
    - Calculates real factor exposures from market data
    
    Capabilities:
    - ML signal generation (AFML methodology)
    - Factor model analysis
    - Meta-labeling signals
    - Regime detection
    - Statistical arbitrage signals
    """
    
    def __init__(self, api_key: str = None):
        super().__init__(api_key)
        self.agent_type = AgentType.QUANT_ANALYST
        self.name = "Quant Analyst"
        self.description = "ML signals and quantitative analysis"
        
        # Polygon API for market data
        self.polygon_key = os.getenv('POLYGON_API_KEY', 'JHKwAdyIOeExkYOxh3LwTopmqqVVFeBY')
    
    def get_capabilities(self) -> Dict[str, Any]:
        return {
            "name": "Quant Analyst",
            "description": "ML signals and quantitative analysis",
            "capabilities": [
                "ml_signals",
                "factor_models",
                "meta_labeling",
                "regime_detection",
                "stat_arb",
                "backtesting"
            ],
            "models": [
                "LSTM (3.2M params)",
                "Transformer (5.8M params)",
                "Ensemble Network",
                "HMM Regime Classifier"
            ]
        }
    
    async def execute(self, task: AgentTask) -> AgentResult:
        """Execute quant analysis using real ML models."""
        start = datetime.now()
        
        try:
            symbols = task.context.get('symbols', [])
            shared = self._get_shared_data(task)
            
            # Generate ML signals using real models
            signals = {}
            for symbol in symbols[:3]:
                signals[symbol] = await self._generate_signals(symbol, shared)
            
            # Regime analysis using real HMM
            regime = await self._detect_regime(shared)
            
            # Factor exposures from real data
            factors = await self._analyze_factors(symbols, shared)
            
            # Calculate overall confidence from signals
            confidences = [s.get('confidence', 0.5) for s in signals.values()]
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0.5
            
            data = {
                'ml_signals': signals,
                'regime': regime,
                'factors': factors,
                'methodology': 'AFML (Advances in Financial Machine Learning)',
                'models_used': self._get_models_used()
            }
            
            return AgentResult(
                task_id=task.task_id,
                agent_type=AgentType.QUANT_ANALYST,
                success=True,
                data=data,
                confidence=avg_confidence,
                execution_time_ms=int((datetime.now() - start).total_seconds() * 1000)
            )
        except Exception as e:
            logger.error(f"QuantAnalystAgent error: {e}")
            return AgentResult(
                task_id=task.task_id,
                agent_type=AgentType.QUANT_ANALYST,
                success=False,
                data={},
                confidence=0,
                execution_time_ms=int((datetime.now() - start).total_seconds() * 1000),
                error=str(e)
            )
    
    async def _generate_signals(self, symbol: str, shared=None) -> Dict:
        """Generate ML signals using real models."""
        
        # Try to use the real ML predictor
        predictor = _get_ml_predictor()
        
        if predictor:
            try:
                prediction = predictor.predict(symbol)
                
                # Map direction to signal
                direction_map = {
                    'bullish': 'LONG',
                    'bearish': 'SHORT',
                    'neutral': 'NEUTRAL'
                }
                
                primary_signal = direction_map.get(prediction.direction, 'NEUTRAL')
                
                return {
                    'symbol': symbol,
                    'primary_signal': primary_signal,
                    'confidence': round(prediction.direction_confidence, 3),
                    'predictions': prediction.predictions,
                    'model_used': prediction.model_used if hasattr(prediction, 'model_used') else 'ensemble',
                    'regime': prediction.current_regime,
                    'uncertainty': round(prediction.uncertainty, 3),
                    'features_used': prediction.feature_count if hasattr(prediction, 'feature_count') else 64,
                    'timestamp': datetime.now().isoformat()
                }
                
            except Exception as e:
                logger.warning(f"ML prediction failed for {symbol}: {e}")
        
        # Fallback: use technical analysis from real data
        return await self._technical_fallback_signal(symbol, shared)
    
    def _polygon_get(self, url: str, params: Dict = None) -> Dict:
        """Helper for Polygon API calls."""
        p = {'apiKey': self.polygon_key}
        if params:
            p.update(params)
        try:
            import requests
            resp = requests.get(url, params=p, timeout=12)
            if resp.status_code == 200:
                return resp.json()
        except Exception as e:
            logger.warning(f"Polygon call failed: {e}")
        return {}
    
    def _get_polygon_indicator(self, symbol: str, indicator: str, window: int = 14) -> Optional[float]:
        """Get server-side indicator from Polygon (sma, ema, rsi, macd)."""
        data = self._polygon_get(
            f"https://api.polygon.io/v1/indicators/{indicator}/{symbol}",
            {'timespan': 'day', 'window': window, 'series_type': 'close', 'order': 'desc', 'limit': 1}
        )
        values = data.get('results', {}).get('values', [])
        if values:
            return values[0].get('value')
        return None
    
    async def _get_news_sentiment_factor(self, symbol: str, shared=None) -> float:
        """Get news sentiment as a quantitative factor (-1 to +1) using premium endpoints."""
        import requests
        stocknews_key = os.getenv('STOCKNEWS_API_KEY', 'zzad9pmlwttixx0fnsenstctzgdk7ysx0ctkgrk0')
        sentiment_score = 0.0
        sources_used = 0
        
        # 1. StockNews PRO — /stat endpoint for quantitative sentiment over 7 days
        try:
            stat_data = None
            if shared:
                stat_resp = await shared.get_stocknews_stat(symbol, 'last7days')
                if stat_resp:
                    stat_data = stat_resp.get('data', {})
            else:
                resp = requests.get("https://stocknewsapi.com/api/v1/stat", params={
                    'tickers': symbol, 'date': 'last7days', 'page': 1,
                    'token': stocknews_key
                }, timeout=8)
                if resp.status_code == 200:
                    stat_data = resp.json().get('data', {})
            
            if stat_data:
                total = stat_data.get('total', 0)
                positive = stat_data.get('positive', 0) or stat_data.get('Positive', 0)
                negative = stat_data.get('negative', 0) or stat_data.get('Negative', 0)
                if total and total > 0:
                    stat_score = (positive - negative) / total
                    sentiment_score += stat_score
                    sources_used += 1
        except Exception:
            pass
        
        # 2. StockNews PRO — Ticker news with sentiment labels (fallback/supplement)
        try:
            articles = None
            if shared:
                news_resp = await shared.get_stocknews(symbol)
                if news_resp:
                    articles = news_resp.get('data', [])
            else:
                resp = requests.get("https://stocknewsapi.com/api/v1", params={
                    'tickers': symbol, 'items': 10, 'sortby': 'rank',
                    'extra-fields': 'id,eventid,rankscore',
                    'token': stocknews_key
                }, timeout=8)
                if resp.status_code == 200:
                    articles = resp.json().get('data', [])
            
            if articles:
                pos = sum(1 for a in articles if str(a.get('sentiment', '')).lower() in ('positive', 'bullish'))
                neg = sum(1 for a in articles if str(a.get('sentiment', '')).lower() in ('negative', 'bearish'))
                total = len(articles)
                ticker_score = (pos - neg) / total if total > 0 else 0
                sentiment_score += ticker_score
                sources_used += 1
        except Exception:
            pass
        
        # 3. StockNews PRO — Check if symbol has upcoming earnings (risk factor)
        try:
            earnings_data = None
            if shared:
                earnings_data = await shared.get_stocknews_earnings()
            else:
                resp = requests.get("https://stocknewsapi.com/api/v1/earnings-calendar", params={
                    'page': 1, 'items': 50, 'token': stocknews_key
                }, timeout=8)
                if resp.status_code == 200:
                    earnings_data = resp.json()
            
            if earnings_data:
                earnings = earnings_data.get('data', [])
                for e in earnings:
                    if e.get('ticker', '').upper() == symbol.upper():
                        sentiment_score -= 0.1
                        break
        except Exception:
            pass
        
        # Average across sources
        if sources_used > 0:
            return max(-1.0, min(1.0, sentiment_score / sources_used))
        return 0.0
    
    async def _technical_fallback_signal(self, symbol: str, shared=None) -> Dict:
        """Generate signal from Polygon server-side indicators + historical data."""
        import requests
        
        if not self.polygon_key:
            return self._minimal_fallback_signal(symbol)
        
        try:
            from datetime import timedelta
            
            # 1. Get Polygon server-side indicators (more accurate than manual)
            if shared:
                rsi_data = await shared.get_rsi(symbol)
                rsi_vals = (rsi_data or {}).get('results', {}).get('values', [])
                rsi = float(rsi_vals[0].get('value')) if rsi_vals else None
                
                sma20_data = await shared.get_sma(symbol, 20)
                sma_20 = float((sma20_data or {}).get('results', {}).get('values', [{}])[0].get('value', 0)) if sma20_data and (sma20_data or {}).get('results', {}).get('values') else None
                
                sma50_data = await shared.get_sma(symbol, 50)
                sma_50 = float((sma50_data or {}).get('results', {}).get('values', [{}])[0].get('value', 0)) if sma50_data and (sma50_data or {}).get('results', {}).get('values') else None
                
                ema12_data = await shared.get_ema(symbol, 12)
                ema_12 = float((ema12_data or {}).get('results', {}).get('values', [{}])[0].get('value', 0)) if ema12_data and (ema12_data or {}).get('results', {}).get('values') else None
                
                ema26_data = await shared.get_ema(symbol, 26)
                ema_26 = float((ema26_data or {}).get('results', {}).get('values', [{}])[0].get('value', 0)) if ema26_data and (ema26_data or {}).get('results', {}).get('values') else None
                
                macd_data = await shared.get_macd(symbol)
            else:
                rsi = self._get_polygon_indicator(symbol, 'rsi', 14)
                sma_20 = self._get_polygon_indicator(symbol, 'sma', 20)
                sma_50 = self._get_polygon_indicator(symbol, 'sma', 50)
                ema_12 = self._get_polygon_indicator(symbol, 'ema', 12)
                ema_26 = self._get_polygon_indicator(symbol, 'ema', 26)
                
                macd_data = self._polygon_get(
                    f"https://api.polygon.io/v1/indicators/macd/{symbol}",
                    {'timespan': 'day', 'short_window': 12, 'long_window': 26,
                     'signal_window': 9, 'series_type': 'close', 'order': 'desc', 'limit': 1}
                )
            
            # MACD parsing
            macd_val = None
            macd_signal = None
            macd_histogram = None
            macd_values = (macd_data or {}).get('results', {}).get('values', [])
            if macd_values:
                macd_val = macd_values[0].get('value')
                macd_signal = macd_values[0].get('signal')
                macd_histogram = macd_values[0].get('histogram')
            
            # 2. Get historical data for momentum & volatility
            end_date = datetime.now().strftime('%Y-%m-%d')
            start_date = (datetime.now() - timedelta(days=60)).strftime('%Y-%m-%d')
            
            closes = None
            returns = None
            if shared:
                hist = await shared.get_historical(symbol, 60)
                results = (hist or {}).get('results', [])
                if len(results) >= 20:
                    closes = np.array([r['c'] for r in results])
                    returns = np.diff(closes) / closes[:-1]
            else:
                url = f"https://api.polygon.io/v2/aggs/ticker/{symbol}/range/1/day/{start_date}/{end_date}"
                response = requests.get(url, params={'apiKey': self.polygon_key, 'sort': 'asc'}, timeout=10)
                
                if response.status_code == 200:
                    results = response.json().get('results', [])
                    if len(results) >= 20:
                        closes = np.array([r['c'] for r in results])
                        returns = np.diff(closes) / closes[:-1]
            
            # Fallback RSI if server-side failed
            if rsi is None and returns is not None and len(returns) >= 14:
                gains = np.where(returns > 0, returns, 0)
                losses = np.where(returns < 0, -returns, 0)
                avg_gain = np.mean(gains[-14:])
                avg_loss = np.mean(losses[-14:])
                rs = avg_gain / avg_loss if avg_loss > 0 else 100
                rsi = 100 - (100 / (1 + rs))
            
            if rsi is None:
                return self._minimal_fallback_signal(symbol)
            
            # 3. News sentiment factor
            news_factor = await self._get_news_sentiment_factor(symbol, shared)
            
            # 4. Calculate momentum & volatility
            momentum_5d = 0
            momentum_20d = 0
            volatility = 0.2
            
            if closes is not None and len(closes) > 20:
                momentum_5d = (closes[-1] - closes[-6]) / closes[-6]
                momentum_20d = (closes[-1] - closes[-21]) / closes[-21]
            if returns is not None and len(returns) >= 20:
                volatility = float(np.std(returns[-20:]) * np.sqrt(252))
            
            # 5. Multi-factor signal generation
            bullish_score = 0
            total_factors = 8
            
            if rsi < 40: bullish_score += 1
            if rsi < 30: bullish_score += 1
            if momentum_5d > 0: bullish_score += 1
            if momentum_20d > 0: bullish_score += 1
            if closes is not None and sma_20 and closes[-1] > sma_20: bullish_score += 1
            if sma_20 and sma_50 and sma_20 > sma_50: bullish_score += 1
            if macd_histogram and macd_histogram > 0: bullish_score += 1
            if news_factor > 0.1: bullish_score += 1
            
            if bullish_score >= 5:
                primary_signal = 'LONG'
            elif bullish_score <= 2:
                primary_signal = 'SHORT'
            else:
                primary_signal = 'NEUTRAL'
            
            confidence = min(0.9, 0.4 + (abs(bullish_score - 4) * 0.08))
            
            return {
                'symbol': symbol,
                'primary_signal': primary_signal,
                'confidence': round(confidence, 3),
                'model_used': 'polygon_indicators + news_sentiment',
                'features': {
                    'rsi': round(float(rsi), 2) if rsi else None,
                    'sma_20': round(float(sma_20), 2) if sma_20 else None,
                    'sma_50': round(float(sma_50), 2) if sma_50 else None,
                    'ema_12': round(float(ema_12), 2) if ema_12 else None,
                    'ema_26': round(float(ema_26), 2) if ema_26 else None,
                    'macd': round(float(macd_val), 4) if macd_val else None,
                    'macd_signal': round(float(macd_signal), 4) if macd_signal else None,
                    'macd_histogram': round(float(macd_histogram), 4) if macd_histogram else None,
                    'momentum_5d': round(momentum_5d * 100, 2),
                    'momentum_20d': round(momentum_20d * 100, 2),
                    'volatility': round(volatility, 3),
                    'news_sentiment_factor': round(news_factor, 2),
                },
                'bullish_score': bullish_score,
                'total_factors': total_factors,
                'data_sources': ['polygon_sma', 'polygon_ema', 'polygon_rsi', 'polygon_macd', 'stocknews_sentiment'],
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.warning(f"Technical fallback failed for {symbol}: {e}")
            return self._minimal_fallback_signal(symbol)
    
    def _minimal_fallback_signal(self, symbol: str) -> Dict:
        """Minimal signal when no data available."""
        return {
            'symbol': symbol,
            'primary_signal': 'NEUTRAL',
            'confidence': 0.5,
            'model_used': 'none',
            'error': 'No data available',
            'timestamp': datetime.now().isoformat()
        }
    
    async def _detect_regime(self, shared=None) -> Dict:
        """Detect current market regime using real HMM or VIX-based detection."""
        
        # Try to use real regime detector
        detector = _get_regime_detector()
        
        if detector:
            try:
                regime = detector.detect()
                return {
                    'current_regime': regime.state.name,
                    'regime_probability': round(regime.probability, 3),
                    'regime_duration_days': regime.duration_days,
                    'transition_probabilities': regime.transition_probs,
                    'model': 'HMM'
                }
            except Exception as e:
                logger.warning(f"HMM regime detection failed: {e}")
        
        # Fallback: VIX-based regime detection
        return await self._vix_based_regime(shared)
    
    async def _vix_based_regime(self, shared=None) -> Dict:
        """Detect regime from VIX and SPY trend."""
        import requests
        
        if not self.polygon_key:
            return {
                'current_regime': 'UNKNOWN',
                'regime_probability': 0.5,
                'model': 'none'
            }
        
        try:
            # Get VIX
            vix = 20  # Default
            if shared:
                vix_data = await shared.get_vix()
                vix_results = (vix_data or {}).get('results', [])
                if vix_results:
                    vix = vix_results[0].get('c', 20)
            else:
                vix_url = f"https://api.polygon.io/v2/aggs/ticker/VIX/prev"
                vix_response = requests.get(vix_url, params={'apiKey': self.polygon_key}, timeout=5)
                if vix_response.status_code == 200:
                    vix_data = vix_response.json().get('results', [{}])
                    if vix_data:
                        vix = vix_data[0].get('c', 20)
            
            # Get SPY trend
            from datetime import timedelta
            end_date = datetime.now().strftime('%Y-%m-%d')
            start_date = (datetime.now() - timedelta(days=60)).strftime('%Y-%m-%d')
            
            trend = 'MIXED'
            if shared:
                spy_hist = await shared.get_historical('SPY', 60)
                spy_data = (spy_hist or {}).get('results', [])
            else:
                spy_url = f"https://api.polygon.io/v2/aggs/ticker/SPY/range/1/day/{start_date}/{end_date}"
                spy_response = requests.get(spy_url, params={'apiKey': self.polygon_key}, timeout=5)
                spy_data = spy_response.json().get('results', []) if spy_response.status_code == 200 else []
            
            if len(spy_data) >= 20:
                closes = [r['c'] for r in spy_data]
                sma_20 = sum(closes[-20:]) / 20
                sma_50 = sum(closes[-50:]) / 50 if len(closes) >= 50 else sma_20
                
                if closes[-1] > sma_20 > sma_50:
                    trend = 'BULLISH'
                elif closes[-1] < sma_20 < sma_50:
                    trend = 'BEARISH'
            
            # Determine regime
            if vix > 35:
                regime = 'CRISIS'
                probability = 0.9
            elif vix > 25:
                regime = 'HIGH_VOL'
                probability = 0.8
            elif trend == 'BULLISH' and vix < 20:
                regime = 'BULL'
                probability = 0.85
            elif trend == 'BEARISH' and vix < 25:
                regime = 'BEAR'
                probability = 0.8
            else:
                regime = 'SIDEWAYS'
                probability = 0.7
            
            return {
                'current_regime': regime,
                'regime_probability': probability,
                'vix': round(vix, 2),
                'spy_trend': trend,
                'model': 'vix_based'
            }
            
        except Exception as e:
            logger.warning(f"VIX-based regime detection failed: {e}")
            return {
                'current_regime': 'UNKNOWN',
                'regime_probability': 0.5,
                'model': 'none'
            }
    
    async def _analyze_factors(self, symbols: List[str], shared=None) -> Dict:
        """Analyze factor exposures using real market data + Polygon indicators."""
        import requests
        
        if not self.polygon_key or not symbols:
            return self._default_factors()
        
        try:
            from datetime import timedelta
            end_date = datetime.now().strftime('%Y-%m-%d')
            start_date = (datetime.now() - timedelta(days=252)).strftime('%Y-%m-%d')
            
            # Get SPY for market beta
            if shared:
                spy_hist = await shared.get_historical('SPY', 252)
                spy_data = (spy_hist or {}).get('results', [])
            else:
                spy_url = f"https://api.polygon.io/v2/aggs/ticker/SPY/range/1/day/{start_date}/{end_date}"
                spy_response = requests.get(spy_url, params={'apiKey': self.polygon_key}, timeout=10)
                spy_data = spy_response.json().get('results', []) if spy_response.status_code == 200 else []
            
            if not spy_data:
                return self._default_factors()
            
            spy_closes = np.array([r['c'] for r in spy_data])
            spy_returns = np.diff(spy_closes) / spy_closes[:-1]
            
            # Calculate beta for each symbol
            betas = []
            for symbol in symbols[:3]:
                try:
                    if shared:
                        sym_hist = await shared.get_historical(symbol, 252)
                        sym_data = (sym_hist or {}).get('results', [])
                    else:
                        sym_url = f"https://api.polygon.io/v2/aggs/ticker/{symbol}/range/1/day/{start_date}/{end_date}"
                        sym_response = requests.get(sym_url, params={'apiKey': self.polygon_key}, timeout=10)
                        sym_data = sym_response.json().get('results', []) if sym_response.status_code == 200 else []
                    
                    if len(sym_data) >= len(spy_data):
                        sym_closes = np.array([r['c'] for r in sym_data[:len(spy_data)]])
                        sym_returns = np.diff(sym_closes) / sym_closes[:-1]
                        
                        if len(sym_returns) == len(spy_returns):
                            cov = np.cov(sym_returns, spy_returns)[0, 1]
                            var = np.var(spy_returns)
                            beta = cov / var if var > 0 else 1.0
                            betas.append(beta)
                except Exception:
                    continue
            
            avg_beta = sum(betas) / len(betas) if betas else 1.0
            
            # Momentum factor from Polygon SMA
            if shared:
                sma50_data = await shared.get_sma('SPY', 50)
                spy_sma_50 = float((sma50_data or {}).get('results', {}).get('values', [{}])[0].get('value', 0)) if sma50_data and (sma50_data or {}).get('results', {}).get('values') else None
                sma200_data = await shared.get_sma('SPY', 200)
                spy_sma_200 = float((sma200_data or {}).get('results', {}).get('values', [{}])[0].get('value', 0)) if sma200_data and (sma200_data or {}).get('results', {}).get('values') else None
            else:
                spy_sma_50 = self._get_polygon_indicator('SPY', 'sma', 50)
                spy_sma_200 = self._get_polygon_indicator('SPY', 'sma', 200)
            
            momentum = round(float((spy_closes[-1] - spy_closes[-20]) / spy_closes[-20]), 3) if len(spy_closes) > 20 else 0
            
            # RSI factor from Polygon
            if shared:
                rsi_data = await shared.get_rsi('SPY')
                rsi_vals = (rsi_data or {}).get('results', {}).get('values', [])
                spy_rsi = float(rsi_vals[0].get('value')) if rsi_vals else None
            else:
                spy_rsi = self._get_polygon_indicator('SPY', 'rsi', 14)
            
            # News sentiment factor
            news_factor = await self._get_news_sentiment_factor(symbols[0], shared) if symbols else 0
            
            factors = {
                'market_beta': round(float(avg_beta), 3),
                'momentum': momentum,
                'volatility': round(float(np.std(spy_returns) * np.sqrt(252)), 3),
                'spy_rsi': round(float(spy_rsi), 1) if spy_rsi else None,
                'spy_sma_50': round(float(spy_sma_50), 2) if spy_sma_50 else None,
                'spy_sma_200': round(float(spy_sma_200), 2) if spy_sma_200 else None,
                'spy_above_sma50': spy_closes[-1] > spy_sma_50 if spy_sma_50 else None,
                'spy_above_sma200': spy_closes[-1] > spy_sma_200 if spy_sma_200 else None,
                'news_sentiment': round(float(news_factor), 2),
                'model': 'polygon_indicators + stocknews',
                'symbols_analyzed': len(betas),
                'data_sources': ['polygon_historical', 'polygon_sma', 'polygon_rsi', 'stocknews']
            }
            
            return factors
            
        except Exception as e:
            logger.warning(f"Factor analysis failed: {e}")
            return self._default_factors()
    
    def _default_factors(self) -> Dict:
        """Default factor values when calculation fails."""
        return {
            'market_beta': 1.0,
            'momentum': 0.0,
            'volatility': 0.2,
            'model': 'default'
        }
    
    def _get_models_used(self) -> List[str]:
        """Get list of models that were used."""
        models = []
        if _ml_predictor is not None:
            models.extend(['LSTM', 'Transformer', 'Ensemble'])
        if _regime_detector is not None:
            models.append('HMM Regime')
        if not models:
            models.append('Technical Fallback')
        return models


__all__ = ['QuantAnalystAgent']
