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
        self.polygon_key = os.getenv('POLYGON_API_KEY')
    
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
            
            # Generate ML signals using real models
            signals = {}
            for symbol in symbols[:3]:
                signals[symbol] = await self._generate_signals(symbol)
            
            # Regime analysis using real HMM
            regime = await self._detect_regime()
            
            # Factor exposures from real data
            factors = await self._analyze_factors(symbols)
            
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
    
    async def _generate_signals(self, symbol: str) -> Dict:
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
        return await self._technical_fallback_signal(symbol)
    
    async def _technical_fallback_signal(self, symbol: str) -> Dict:
        """Generate signal from technical analysis when ML unavailable."""
        import requests
        
        if not self.polygon_key:
            return self._minimal_fallback_signal(symbol)
        
        try:
            from datetime import timedelta
            end_date = datetime.now().strftime('%Y-%m-%d')
            start_date = (datetime.now() - timedelta(days=60)).strftime('%Y-%m-%d')
            
            url = f"https://api.polygon.io/v2/aggs/ticker/{symbol}/range/1/day/{start_date}/{end_date}"
            response = requests.get(url, params={'apiKey': self.polygon_key, 'limit': 60}, timeout=10)
            
            if response.status_code != 200:
                return self._minimal_fallback_signal(symbol)
            
            data = response.json()
            results = data.get('results', [])
            
            if len(results) < 20:
                return self._minimal_fallback_signal(symbol)
            
            closes = np.array([r['c'] for r in results])
            
            # Calculate indicators
            returns = np.diff(closes) / closes[:-1]
            
            # RSI
            gains = np.where(returns > 0, returns, 0)
            losses = np.where(returns < 0, -returns, 0)
            avg_gain = np.mean(gains[-14:])
            avg_loss = np.mean(losses[-14:])
            rs = avg_gain / avg_loss if avg_loss > 0 else 100
            rsi = 100 - (100 / (1 + rs))
            
            # Momentum
            momentum_5d = (closes[-1] - closes[-6]) / closes[-6] if len(closes) > 5 else 0
            momentum_20d = (closes[-1] - closes[-21]) / closes[-21] if len(closes) > 20 else 0
            
            # Volatility
            volatility = np.std(returns[-20:]) * np.sqrt(252)
            
            # Trend
            sma_20 = np.mean(closes[-20:])
            sma_50 = np.mean(closes[-50:]) if len(closes) >= 50 else sma_20
            trend_strength = (sma_20 - sma_50) / sma_50 if sma_50 > 0 else 0
            
            # Generate signal from technicals
            bullish_score = 0
            if rsi < 40: bullish_score += 1
            if rsi < 30: bullish_score += 1
            if momentum_5d > 0: bullish_score += 1
            if momentum_20d > 0: bullish_score += 1
            if closes[-1] > sma_20: bullish_score += 1
            if sma_20 > sma_50: bullish_score += 1
            
            if bullish_score >= 4:
                primary_signal = 'LONG'
            elif bullish_score <= 2:
                primary_signal = 'SHORT'
            else:
                primary_signal = 'NEUTRAL'
            
            # Confidence based on indicator agreement
            confidence = min(0.9, 0.4 + (abs(bullish_score - 3) * 0.1))
            
            return {
                'symbol': symbol,
                'primary_signal': primary_signal,
                'confidence': round(confidence, 3),
                'model_used': 'technical_fallback',
                'features': {
                    'rsi': round(rsi, 2),
                    'momentum_5d': round(momentum_5d * 100, 2),
                    'momentum_20d': round(momentum_20d * 100, 2),
                    'volatility': round(volatility, 3),
                    'trend_strength': round(trend_strength * 100, 2)
                },
                'bullish_score': bullish_score,
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
    
    async def _detect_regime(self) -> Dict:
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
        return await self._vix_based_regime()
    
    async def _vix_based_regime(self) -> Dict:
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
            vix_url = f"https://api.polygon.io/v2/aggs/ticker/VIX/prev"
            vix_response = requests.get(vix_url, params={'apiKey': self.polygon_key}, timeout=5)
            
            vix = 20  # Default
            if vix_response.status_code == 200:
                vix_data = vix_response.json().get('results', [{}])
                if vix_data:
                    vix = vix_data[0].get('c', 20)
            
            # Get SPY trend
            from datetime import timedelta
            end_date = datetime.now().strftime('%Y-%m-%d')
            start_date = (datetime.now() - timedelta(days=60)).strftime('%Y-%m-%d')
            
            spy_url = f"https://api.polygon.io/v2/aggs/ticker/SPY/range/1/day/{start_date}/{end_date}"
            spy_response = requests.get(spy_url, params={'apiKey': self.polygon_key}, timeout=5)
            
            trend = 'MIXED'
            if spy_response.status_code == 200:
                spy_data = spy_response.json().get('results', [])
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
    
    async def _analyze_factors(self, symbols: List[str]) -> Dict:
        """Analyze factor exposures using real market data."""
        import requests
        
        if not self.polygon_key or not symbols:
            return self._default_factors()
        
        try:
            from datetime import timedelta
            end_date = datetime.now().strftime('%Y-%m-%d')
            start_date = (datetime.now() - timedelta(days=252)).strftime('%Y-%m-%d')
            
            # Get SPY for market beta calculation
            spy_url = f"https://api.polygon.io/v2/aggs/ticker/SPY/range/1/day/{start_date}/{end_date}"
            spy_response = requests.get(spy_url, params={'apiKey': self.polygon_key}, timeout=10)
            
            if spy_response.status_code != 200:
                return self._default_factors()
            
            spy_data = spy_response.json().get('results', [])
            spy_closes = np.array([r['c'] for r in spy_data])
            spy_returns = np.diff(spy_closes) / spy_closes[:-1]
            
            # Calculate beta for each symbol
            betas = []
            for symbol in symbols[:3]:
                try:
                    sym_url = f"https://api.polygon.io/v2/aggs/ticker/{symbol}/range/1/day/{start_date}/{end_date}"
                    sym_response = requests.get(sym_url, params={'apiKey': self.polygon_key}, timeout=10)
                    
                    if sym_response.status_code == 200:
                        sym_data = sym_response.json().get('results', [])
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
            
            # Calculate other factor proxies from the data
            factors = {
                'market_beta': round(float(avg_beta), 3),
                'momentum': round(float((spy_closes[-1] - spy_closes[-20]) / spy_closes[-20]), 3) if len(spy_closes) > 20 else 0,
                'volatility': round(float(np.std(spy_returns) * np.sqrt(252)), 3),
                'model': 'calculated',
                'symbols_analyzed': len(betas)
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
