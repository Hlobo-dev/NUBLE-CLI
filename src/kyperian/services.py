"""
KYPERIAN Unified Services Layer
================================

This module bridges the gap between the three subsystems:
1. kyperian/ (Core CLI)
2. institutional/ (Pro Platform)  
3. TENK_SOURCE/ (SEC Filings)

It provides a single, unified interface for all capabilities.
"""

import os
import asyncio
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path

logger = logging.getLogger(__name__)


class ServiceType(Enum):
    """Types of services available."""
    MARKET_DATA = "market_data"
    TECHNICAL = "technical"
    FILINGS = "filings"
    ML_PREDICTION = "ml_prediction"
    SENTIMENT = "sentiment"
    PATTERNS = "patterns"
    ANALYTICS = "analytics"


@dataclass
class ServiceStatus:
    """Status of a service."""
    name: str
    available: bool
    details: str = ""
    last_check: datetime = field(default_factory=datetime.now)


@dataclass
class UnifiedResponse:
    """Unified response from any service."""
    success: bool
    service: ServiceType
    data: Any = None
    error: str = ""
    latency_ms: float = 0
    source: str = ""
    cached: bool = False
    
    def to_dict(self) -> Dict:
        return {
            "success": self.success,
            "service": self.service.value,
            "data": self.data,
            "error": self.error,
            "latency_ms": self.latency_ms,
            "source": self.source,
            "cached": self.cached,
        }


class UnifiedServices:
    """
    Unified services layer that integrates all KYPERIAN capabilities.
    
    This is the single point of access for:
    - Market data (Polygon, Alpha Vantage, Finnhub)
    - SEC filings (EDGAR, TENK integration)
    - ML predictions (trained models)
    - Technical analysis (50+ indicators)
    - Sentiment analysis
    - Pattern recognition
    """
    
    def __init__(self):
        self._initialized = False
        self._services: Dict[ServiceType, ServiceStatus] = {}
        
        # Lazy-loaded components
        self._orchestrator = None
        self._filings_analyzer = None
        self._filings_search = None
        self._ml_predictor = None
        self._technical_analyzer = None
        self._sentiment_analyzer = None
        self._pattern_recognizer = None
        
        # Cache
        self._cache: Dict[str, Tuple[Any, datetime]] = {}
        self._cache_ttl = 60  # seconds
        
        # API keys
        self._polygon_key = os.getenv("POLYGON_API_KEY")
        self._anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    
    async def initialize(self) -> Dict[ServiceType, ServiceStatus]:
        """
        Initialize all services and report status.
        
        Returns dict of service -> status for display.
        """
        if self._initialized:
            return self._services
        
        # Check market data
        self._services[ServiceType.MARKET_DATA] = await self._init_market_data()
        
        # Check filings
        self._services[ServiceType.FILINGS] = await self._init_filings()
        
        # Check ML
        self._services[ServiceType.ML_PREDICTION] = await self._init_ml()
        
        # Check technical (always available, no external deps)
        self._services[ServiceType.TECHNICAL] = ServiceStatus(
            name="Technical Analysis",
            available=True,
            details="50+ indicators ready"
        )
        
        # Check sentiment
        self._services[ServiceType.SENTIMENT] = ServiceStatus(
            name="Sentiment Analysis",
            available=True,
            details="Lexicon-based (FinBERT not loaded)"
        )
        
        # Check patterns
        self._services[ServiceType.PATTERNS] = ServiceStatus(
            name="Pattern Recognition",
            available=True,
            details="Classical patterns ready"
        )
        
        self._initialized = True
        return self._services
    
    async def _init_market_data(self) -> ServiceStatus:
        """Initialize market data providers."""
        try:
            if not self._polygon_key:
                return ServiceStatus(
                    name="Market Data",
                    available=False,
                    details="POLYGON_API_KEY not set"
                )
            
            # Try to import and init orchestrator
            from institutional.core import Orchestrator
            from institutional.config import load_config
            
            config = load_config()
            self._orchestrator = Orchestrator(config=config)
            
            providers = self._orchestrator.get_available_providers()
            
            return ServiceStatus(
                name="Market Data",
                available=True,
                details=f"Providers: {', '.join(providers)}"
            )
        except Exception as e:
            logger.warning(f"Market data init failed: {e}")
            return ServiceStatus(
                name="Market Data",
                available=False,
                details=str(e)
            )
    
    async def _init_filings(self) -> ServiceStatus:
        """Initialize SEC filings service."""
        try:
            from institutional.filings import FilingsAnalyzer, FilingsSearch
            
            # Check if we have an Anthropic key for analysis
            if self._anthropic_key:
                self._filings_analyzer = FilingsAnalyzer()
            
            # Search doesn't need API key
            self._filings_search = FilingsSearch()
            
            # Check if any filings are indexed
            try:
                indexed = self._filings_search.db.get_indexed_filings()
                count = len(indexed) if indexed else 0
            except:
                count = 0
            
            return ServiceStatus(
                name="SEC Filings",
                available=True,
                details=f"{count} filings indexed, analyzer {'ready' if self._filings_analyzer else 'needs ANTHROPIC_API_KEY'}"
            )
        except Exception as e:
            logger.warning(f"Filings init failed: {e}")
            return ServiceStatus(
                name="SEC Filings",
                available=False,
                details=str(e)
            )
    
    async def _init_ml(self) -> ServiceStatus:
        """Initialize ML prediction service."""
        try:
            # First try to use pre-trained model registry
            try:
                from institutional.ml.registry import get_registry
                self._model_registry = get_registry()
                registry_status = self._model_registry.get_status()
                
                if registry_status['total_models'] > 0:
                    models = registry_status['models']
                    return ServiceStatus(
                        name="ML Predictions",
                        available=True,
                        details=f"Pre-trained: {', '.join(models[:5])}"
                    )
            except ImportError:
                self._model_registry = None
            
            # Fall back to RealTimePredictor
            from institutional.ml.prediction import RealTimePredictor
            
            if not self._polygon_key:
                return ServiceStatus(
                    name="ML Predictions",
                    available=False,
                    details="POLYGON_API_KEY needed for data"
                )
            
            self._ml_predictor = RealTimePredictor(
                api_key=self._polygon_key,
                models_dir="models"
            )
            
            # Check for pre-trained models
            models_dir = Path("models")
            pretrained_dir = Path("models/pretrained")
            
            model_files = []
            for d in [models_dir, pretrained_dir]:
                if d.exists():
                    model_files.extend(list(d.glob("*.pt")))
            
            if model_files:
                # Extract symbols from model names
                symbols = set()
                for f in model_files:
                    parts = f.stem.split("_")
                    if len(parts) >= 2:
                        symbols.add(parts[1])  # mlp_SPY_date.pt -> SPY
                
                return ServiceStatus(
                    name="ML Predictions",
                    available=True,
                    details=f"{len(model_files)} models: {', '.join(sorted(symbols)[:5])}"
                )
            else:
                return ServiceStatus(
                    name="ML Predictions",
                    available=True,
                    details="No pre-trained models (will train on demand)"
                )
                
        except Exception as e:
            logger.warning(f"ML init failed: {e}")
            return ServiceStatus(
                name="ML Predictions",
                available=False,
                details=str(e)
            )
    
    def get_status(self) -> Dict[ServiceType, ServiceStatus]:
        """Get current service status."""
        return self._services
    
    def _get_cache(self, key: str) -> Optional[Any]:
        """Get from cache if not expired."""
        if key in self._cache:
            data, timestamp = self._cache[key]
            if (datetime.now() - timestamp).total_seconds() < self._cache_ttl:
                return data
            del self._cache[key]
        return None
    
    def _set_cache(self, key: str, data: Any):
        """Set cache entry."""
        self._cache[key] = (data, datetime.now())
    
    # ==================== MARKET DATA ====================
    
    async def get_quote(self, symbol: str) -> UnifiedResponse:
        """Get real-time quote for a symbol."""
        import time
        start = time.time()
        
        cache_key = f"quote:{symbol}"
        cached = self._get_cache(cache_key)
        if cached:
            return UnifiedResponse(
                success=True,
                service=ServiceType.MARKET_DATA,
                data=cached,
                source="cache",
                cached=True,
                latency_ms=0
            )
        
        if not self._orchestrator:
            return UnifiedResponse(
                success=False,
                service=ServiceType.MARKET_DATA,
                error="Market data not initialized"
            )
        
        try:
            from institutional.providers.base import DataType
            result = await self._orchestrator.execute_query(f"quote for {symbol}")
            
            latency = (time.time() - start) * 1000
            
            if result.status.value == "completed":
                self._set_cache(cache_key, result.data)
                return UnifiedResponse(
                    success=True,
                    service=ServiceType.MARKET_DATA,
                    data=result.data,
                    source="polygon",
                    latency_ms=latency
                )
            else:
                return UnifiedResponse(
                    success=False,
                    service=ServiceType.MARKET_DATA,
                    error="; ".join(result.errors),
                    latency_ms=latency
                )
        except Exception as e:
            return UnifiedResponse(
                success=False,
                service=ServiceType.MARKET_DATA,
                error=str(e),
                latency_ms=(time.time() - start) * 1000
            )
    
    async def get_historical(
        self, 
        symbol: str, 
        days: int = 365
    ) -> UnifiedResponse:
        """Get historical OHLCV data."""
        import time
        start = time.time()
        
        if not self._orchestrator:
            return UnifiedResponse(
                success=False,
                service=ServiceType.MARKET_DATA,
                error="Market data not initialized"
            )
        
        try:
            result = await self._orchestrator.execute_query(
                f"historical data for {symbol} last {days} days"
            )
            
            return UnifiedResponse(
                success=result.status.value == "completed",
                service=ServiceType.MARKET_DATA,
                data=result.data,
                error="; ".join(result.errors) if result.errors else "",
                latency_ms=(time.time() - start) * 1000,
                source="polygon"
            )
        except Exception as e:
            return UnifiedResponse(
                success=False,
                service=ServiceType.MARKET_DATA,
                error=str(e)
            )
    
    # ==================== TECHNICAL ANALYSIS ====================
    
    async def get_technical_indicators(
        self, 
        symbol: str,
        indicators: List[str] = None
    ) -> UnifiedResponse:
        """Get technical indicators for a symbol."""
        import time
        start = time.time()
        
        try:
            # Get historical data first
            hist_response = await self.get_historical(symbol, days=100)
            if not hist_response.success:
                return UnifiedResponse(
                    success=False,
                    service=ServiceType.TECHNICAL,
                    error=f"Failed to get data: {hist_response.error}"
                )
            
            # Lazy load analyzer
            if not self._technical_analyzer:
                from institutional.analytics.technical import TechnicalAnalyzer
                self._technical_analyzer = TechnicalAnalyzer()
            
            # Extract prices from data
            data = hist_response.data
            if "ohlcv" in data and data["ohlcv"]:
                ohlcv = data["ohlcv"]
                if hasattr(ohlcv[0], 'close'):
                    closes = [bar.close for bar in ohlcv]
                    highs = [bar.high for bar in ohlcv]
                    lows = [bar.low for bar in ohlcv]
                    volumes = [bar.volume for bar in ohlcv]
                else:
                    closes = [bar.get('close', bar.get('c', 0)) for bar in ohlcv]
                    highs = [bar.get('high', bar.get('h', 0)) for bar in ohlcv]
                    lows = [bar.get('low', bar.get('l', 0)) for bar in ohlcv]
                    volumes = [bar.get('volume', bar.get('v', 0)) for bar in ohlcv]
            else:
                return UnifiedResponse(
                    success=False,
                    service=ServiceType.TECHNICAL,
                    error="No OHLCV data available"
                )
            
            # Calculate indicators
            result = {}
            
            # RSI
            if not indicators or 'rsi' in indicators:
                rsi = self._technical_analyzer.rsi(closes, 14)
                result['rsi'] = rsi[-1] if rsi else None
            
            # MACD
            if not indicators or 'macd' in indicators:
                macd_line, signal, hist = self._technical_analyzer.macd(closes)
                result['macd'] = {
                    'macd': macd_line[-1] if macd_line else None,
                    'signal': signal[-1] if signal else None,
                    'histogram': hist[-1] if hist else None
                }
            
            # Bollinger Bands
            if not indicators or 'bollinger' in indicators:
                upper, middle, lower = self._technical_analyzer.bollinger_bands(closes, 20, 2)
                result['bollinger'] = {
                    'upper': upper[-1] if upper else None,
                    'middle': middle[-1] if middle else None,
                    'lower': lower[-1] if lower else None
                }
            
            # Moving Averages
            if not indicators or 'sma' in indicators:
                sma20 = self._technical_analyzer.sma(closes, 20)
                sma50 = self._technical_analyzer.sma(closes, 50)
                result['sma_20'] = sma20[-1] if sma20 else None
                result['sma_50'] = sma50[-1] if sma50 else None
            
            # EMA
            if not indicators or 'ema' in indicators:
                ema12 = self._technical_analyzer.ema(closes, 12)
                ema26 = self._technical_analyzer.ema(closes, 26)
                result['ema_12'] = ema12[-1] if ema12 else None
                result['ema_26'] = ema26[-1] if ema26 else None
            
            # ATR
            if not indicators or 'atr' in indicators:
                atr = self._technical_analyzer.atr(highs, lows, closes, 14)
                result['atr'] = atr[-1] if atr else None
            
            # Current price context
            result['current_price'] = closes[-1]
            result['symbol'] = symbol
            
            # Add signals interpretation
            result['signals'] = self._interpret_signals(result)
            
            return UnifiedResponse(
                success=True,
                service=ServiceType.TECHNICAL,
                data=result,
                latency_ms=(time.time() - start) * 1000,
                source="calculated"
            )
            
        except Exception as e:
            return UnifiedResponse(
                success=False,
                service=ServiceType.TECHNICAL,
                error=str(e)
            )
    
    def _interpret_signals(self, indicators: Dict) -> Dict:
        """Interpret technical indicators into signals."""
        signals = {}
        
        # RSI interpretation
        rsi = indicators.get('rsi')
        if rsi:
            if rsi > 70:
                signals['rsi'] = {'signal': 'OVERBOUGHT', 'value': rsi}
            elif rsi < 30:
                signals['rsi'] = {'signal': 'OVERSOLD', 'value': rsi}
            else:
                signals['rsi'] = {'signal': 'NEUTRAL', 'value': rsi}
        
        # MACD interpretation
        macd = indicators.get('macd', {})
        if macd.get('macd') and macd.get('signal'):
            if macd['macd'] > macd['signal']:
                signals['macd'] = {'signal': 'BULLISH', 'value': macd['histogram']}
            else:
                signals['macd'] = {'signal': 'BEARISH', 'value': macd['histogram']}
        
        # Bollinger interpretation
        bb = indicators.get('bollinger', {})
        price = indicators.get('current_price')
        if bb.get('upper') and bb.get('lower') and price:
            if price > bb['upper']:
                signals['bollinger'] = {'signal': 'OVERBOUGHT', 'position': 'above_upper'}
            elif price < bb['lower']:
                signals['bollinger'] = {'signal': 'OVERSOLD', 'position': 'below_lower'}
            else:
                signals['bollinger'] = {'signal': 'NEUTRAL', 'position': 'within_bands'}
        
        # Trend (SMA)
        sma20 = indicators.get('sma_20')
        sma50 = indicators.get('sma_50')
        if sma20 and sma50:
            if sma20 > sma50:
                signals['trend'] = {'signal': 'UPTREND', 'sma20': sma20, 'sma50': sma50}
            else:
                signals['trend'] = {'signal': 'DOWNTREND', 'sma20': sma20, 'sma50': sma50}
        
        return signals
    
    # ==================== SEC FILINGS ====================
    
    async def search_filings(
        self,
        query: str,
        symbol: str = None,
        form_types: List[str] = None,
        limit: int = 10
    ) -> UnifiedResponse:
        """Search SEC filings with semantic search."""
        import time
        start = time.time()
        
        if not self._filings_search:
            return UnifiedResponse(
                success=False,
                service=ServiceType.FILINGS,
                error="Filings search not initialized"
            )
        
        try:
            results = self._filings_search.search(
                query=query,
                tickers=[symbol] if symbol else None,
                forms=form_types,
                limit=limit
            )
            
            return UnifiedResponse(
                success=True,
                service=ServiceType.FILINGS,
                data=[{
                    'ticker': r.ticker,
                    'form': r.form,
                    'year': r.year,
                    'text': r.text[:500] + "..." if len(r.text) > 500 else r.text,
                    'similarity': r.similarity,
                    'source': r.source_label
                } for r in results],
                latency_ms=(time.time() - start) * 1000,
                source="duckdb"
            )
        except Exception as e:
            return UnifiedResponse(
                success=False,
                service=ServiceType.FILINGS,
                error=str(e)
            )
    
    async def analyze_filing(
        self,
        symbol: str,
        analysis_type: str = "risk_factors",
        form: str = "10-K",
        year: int = None
    ) -> UnifiedResponse:
        """Analyze SEC filing with Claude."""
        import time
        start = time.time()
        
        if not self._filings_analyzer:
            return UnifiedResponse(
                success=False,
                service=ServiceType.FILINGS,
                error="Filings analyzer not initialized (need ANTHROPIC_API_KEY)"
            )
        
        try:
            result = await self._filings_analyzer.analyze(
                ticker=symbol,
                form=form,
                year=year or datetime.now().year - 1,
                analysis_type=analysis_type
            )
            
            return UnifiedResponse(
                success=True,
                service=ServiceType.FILINGS,
                data={
                    'analysis': result.content,
                    'key_points': result.key_points,
                    'confidence': result.confidence,
                    'sources': result.sources,
                    'model': result.model
                },
                latency_ms=(time.time() - start) * 1000,
                source="claude"
            )
        except Exception as e:
            return UnifiedResponse(
                success=False,
                service=ServiceType.FILINGS,
                error=str(e)
            )
    
    # ==================== ML PREDICTIONS ====================
    
    async def get_prediction(
        self,
        symbol: str,
        model_type: str = "mlp"
    ) -> UnifiedResponse:
        """Get ML prediction for a symbol."""
        import time
        start = time.time()
        
        # Try pre-trained registry first
        if hasattr(self, '_model_registry') and self._model_registry:
            try:
                if self._model_registry.is_available(symbol):
                    # Get features for this symbol
                    features = await self._get_prediction_features(symbol)
                    if features is not None:
                        prediction = self._model_registry.predict(symbol, features)
                        if prediction:
                            return UnifiedResponse(
                                success=True,
                                service=ServiceType.ML_PREDICTION,
                                data=prediction,
                                latency_ms=(time.time() - start) * 1000,
                                source=f"pretrained:{prediction.get('model_type', 'mlp')}"
                            )
            except Exception as e:
                logger.warning(f"Pre-trained prediction failed: {e}")
        
        # Fall back to real-time predictor
        if not self._ml_predictor:
            return UnifiedResponse(
                success=False,
                service=ServiceType.ML_PREDICTION,
                error="ML predictor not initialized"
            )
        
        try:
            prediction = await self._ml_predictor.predict(symbol, model_type)
            
            return UnifiedResponse(
                success=True,
                service=ServiceType.ML_PREDICTION,
                data={
                    'symbol': prediction.symbol,
                    'direction': prediction.direction,
                    'confidence': prediction.confidence,
                    'predicted_return': prediction.predicted_return,
                    'current_price': prediction.current_price,
                    'price_target': prediction.price_target_1d,
                    'model_type': prediction.model_type,
                    'historical_sharpe': prediction.historical_sharpe,
                    'historical_accuracy': prediction.historical_dir_acc,
                    'signals': prediction.signals,
                    'timestamp': prediction.timestamp
                },
                latency_ms=(time.time() - start) * 1000,
                source=f"model:{model_type}"
            )
        except FileNotFoundError:
            return UnifiedResponse(
                success=False,
                service=ServiceType.ML_PREDICTION,
                error=f"No trained model for {symbol}. Run 'train {symbol}' first."
            )
        except Exception as e:
            return UnifiedResponse(
                success=False,
                service=ServiceType.ML_PREDICTION,
                error=str(e)
            )
    
    async def _get_prediction_features(self, symbol: str) -> Any:
        """Get features needed for prediction."""
        try:
            # Get recent price data
            if self._orchestrator:
                data = await self._orchestrator.get_historical_data(
                    symbol, timeframe='day', limit=60
                )
                if data is not None and len(data) > 0:
                    # Build features from price data
                    from institutional.ml.features import FeatureEngineer
                    engineer = FeatureEngineer()
                    features = engineer.build_features(data)
                    return features.iloc[-1].values
        except Exception as e:
            logger.warning(f"Could not get features for {symbol}: {e}")
        return None
    
    async def get_ensemble_prediction(
        self,
        symbol: str
    ) -> UnifiedResponse:
        """Get ensemble prediction from multiple models."""
        import time
        start = time.time()
        
        if not self._ml_predictor:
            return UnifiedResponse(
                success=False,
                service=ServiceType.ML_PREDICTION,
                error="ML predictor not initialized"
            )
        
        try:
            ensemble = await self._ml_predictor.ensemble_predict(symbol)
            
            return UnifiedResponse(
                success=True,
                service=ServiceType.ML_PREDICTION,
                data={
                    'symbol': ensemble.symbol,
                    'direction': ensemble.ensemble_direction,
                    'confidence': ensemble.ensemble_confidence,
                    'expected_return': ensemble.ensemble_return,
                    'model_agreement': ensemble.model_agreement,
                    'individual_predictions': [
                        {
                            'model': p.model_type,
                            'direction': p.direction,
                            'confidence': p.confidence,
                            'sharpe': p.historical_sharpe
                        }
                        for p in ensemble.predictions
                    ],
                    'timestamp': ensemble.timestamp
                },
                latency_ms=(time.time() - start) * 1000,
                source="ensemble"
            )
        except Exception as e:
            return UnifiedResponse(
                success=False,
                service=ServiceType.ML_PREDICTION,
                error=str(e)
            )
    
    # ==================== PATTERN RECOGNITION ====================
    
    async def detect_patterns(self, symbol: str) -> UnifiedResponse:
        """Detect chart patterns for a symbol."""
        import time
        start = time.time()
        
        try:
            # Get historical data
            hist_response = await self.get_historical(symbol, days=100)
            if not hist_response.success:
                return UnifiedResponse(
                    success=False,
                    service=ServiceType.PATTERNS,
                    error=f"Failed to get data: {hist_response.error}"
                )
            
            # Lazy load pattern recognizer
            if not self._pattern_recognizer:
                from institutional.analytics.patterns import PatternRecognizer
                self._pattern_recognizer = PatternRecognizer()
            
            # Extract OHLC
            data = hist_response.data
            if "ohlcv" in data and data["ohlcv"]:
                ohlcv = data["ohlcv"]
                if hasattr(ohlcv[0], 'open'):
                    opens = [bar.open for bar in ohlcv]
                    highs = [bar.high for bar in ohlcv]
                    lows = [bar.low for bar in ohlcv]
                    closes = [bar.close for bar in ohlcv]
                else:
                    opens = [bar.get('open', bar.get('o', 0)) for bar in ohlcv]
                    highs = [bar.get('high', bar.get('h', 0)) for bar in ohlcv]
                    lows = [bar.get('low', bar.get('l', 0)) for bar in ohlcv]
                    closes = [bar.get('close', bar.get('c', 0)) for bar in ohlcv]
            else:
                return UnifiedResponse(
                    success=False,
                    service=ServiceType.PATTERNS,
                    error="No OHLCV data"
                )
            
            # Detect patterns
            patterns = self._pattern_recognizer.detect_all(opens, highs, lows, closes)
            
            return UnifiedResponse(
                success=True,
                service=ServiceType.PATTERNS,
                data={
                    'symbol': symbol,
                    'patterns': [
                        {
                            'type': p.pattern_type.value,
                            'confidence': p.confidence,
                            'direction': p.direction,
                            'price_target': p.price_target,
                            'description': p.description
                        }
                        for p in patterns
                    ],
                    'count': len(patterns)
                },
                latency_ms=(time.time() - start) * 1000,
                source="calculated"
            )
        except Exception as e:
            return UnifiedResponse(
                success=False,
                service=ServiceType.PATTERNS,
                error=str(e)
            )
    
    # ==================== SENTIMENT ====================
    
    async def get_sentiment(self, symbol: str) -> UnifiedResponse:
        """Get sentiment analysis for a symbol."""
        import time
        start = time.time()
        
        try:
            # Lazy load sentiment analyzer
            if not self._sentiment_analyzer:
                from institutional.analytics.sentiment import SentimentAnalyzer
                self._sentiment_analyzer = SentimentAnalyzer()
            
            # Get news if available
            news_text = []
            if self._orchestrator:
                try:
                    from institutional.providers.base import DataType
                    result = await self._orchestrator.execute_query(f"news for {symbol}")
                    if result.data and "news" in result.data:
                        for article in result.data["news"][:10]:
                            if hasattr(article, 'title'):
                                news_text.append(article.title)
                            elif isinstance(article, dict):
                                news_text.append(article.get('title', ''))
                except:
                    pass
            
            if news_text:
                # Analyze each headline
                results = []
                for text in news_text:
                    result = self._sentiment_analyzer.analyze(text)
                    results.append(result)
                
                # Aggregate
                avg_score = sum(r.score for r in results) / len(results)
                if avg_score > 0.1:
                    overall = "BULLISH"
                elif avg_score < -0.1:
                    overall = "BEARISH"
                else:
                    overall = "NEUTRAL"
                
                return UnifiedResponse(
                    success=True,
                    service=ServiceType.SENTIMENT,
                    data={
                        'symbol': symbol,
                        'overall_sentiment': overall,
                        'score': avg_score,
                        'articles_analyzed': len(results),
                        'bullish_count': sum(1 for r in results if r.score > 0),
                        'bearish_count': sum(1 for r in results if r.score < 0),
                        'headlines': news_text[:5]
                    },
                    latency_ms=(time.time() - start) * 1000,
                    source="lexicon"
                )
            else:
                return UnifiedResponse(
                    success=True,
                    service=ServiceType.SENTIMENT,
                    data={
                        'symbol': symbol,
                        'overall_sentiment': 'UNKNOWN',
                        'score': 0,
                        'articles_analyzed': 0,
                        'note': 'No news data available'
                    },
                    latency_ms=(time.time() - start) * 1000,
                    source="none"
                )
        except Exception as e:
            return UnifiedResponse(
                success=False,
                service=ServiceType.SENTIMENT,
                error=str(e)
            )


# Global singleton
_services: Optional[UnifiedServices] = None


def get_services() -> UnifiedServices:
    """Get the global services instance."""
    global _services
    if _services is None:
        _services = UnifiedServices()
    return _services


async def init_services() -> Dict[ServiceType, ServiceStatus]:
    """Initialize services and return status."""
    services = get_services()
    return await services.initialize()
