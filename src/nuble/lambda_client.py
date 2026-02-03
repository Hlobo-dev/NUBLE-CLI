"""
NUBLE Lambda Decision Engine Client
=======================================

Institutional-Grade Market Intelligence API

This module provides direct access to the NUBLE production decision engine,
which aggregates and analyzes data from multiple premium data providers:

DATA SOURCES:
├── Polygon.io (Real-time)
│   ├── Price data, OHLCV, tick-level
│   ├── Technical indicators (RSI, MACD, ATR, Bollinger, etc.)
│   ├── Market breadth, sector performance
│   └── Corporate news feed
│
├── StockNews API (24 Endpoints)
│   ├── Sentiment Analysis (NLP-based, 7-day rolling)
│   ├── Analyst Ratings (upgrades, downgrades, price targets)
│   ├── Earnings Calendar + Whisper Numbers
│   ├── SEC Filing Alerts (10-K, 10-Q, 8-K, insider transactions)
│   ├── Event Detection (M&A, spinoffs, buybacks)
│   ├── Trending Mentions (social velocity)
│   └── Block Trade Alerts
│
├── CryptoNews API (17 Endpoints)
│   ├── Crypto Sentiment (BTC, ETH, top 100 coins)
│   ├── Whale Activity Tracking (large wallet movements)
│   ├── Institutional Flow Signals (Grayscale, ETF flows)
│   ├── Regulatory News Detection (SEC, CFTC, global)
│   ├── DeFi Protocol Events
│   ├── Exchange Reserve Changes
│   └── Staking/Yield Updates
│
└── Derived Intelligence
    ├── Multi-Timeframe Regime Detection (BEAR/BULL/VOLATILE/RANGING)
    ├── Cross-Asset Correlation Analysis
    ├── Volatility Regime Classification
    └── Composite Decision Scoring (0-100)

ARCHITECTURE:
    CLI → Lambda Client → API Gateway → Lambda Function → Data Aggregation
                                                       → Decision Engine
                                                       → Response

USAGE:
    from nuble.lambda_client import analyze_symbol, get_lambda_client
    
    # Quick analysis
    analysis = analyze_symbol("AAPL")
    print(f"{analysis.symbol}: {analysis.action} (Score: {analysis.score})")
    
    # Full client access
    client = get_lambda_client()
    health = client.health_check()
    signals = client.get_signals("TSLA")
"""

import os
import requests
import logging
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
import time

logger = logging.getLogger(__name__)

# Production Lambda API Gateway endpoint
NUBLE_API_BASE = "https://9vyvetp9c7.execute-api.us-east-1.amazonaws.com/production"

# Retry configuration for production reliability
MAX_RETRIES = 3
RETRY_BACKOFF = 1.5
REQUEST_TIMEOUT = 45


class MarketRegime(Enum):
    """Market regime classification."""
    BULL = "BULL"
    BEAR = "BEAR"
    VOLATILE = "VOLATILE"
    RANGING = "RANGING"
    CRISIS = "CRISIS"
    UNKNOWN = "UNKNOWN"


class SignalStrength(Enum):
    """Signal strength classification."""
    STRONG = "STRONG"
    MODERATE = "MODERATE"
    WEAK = "WEAK"
    NEUTRAL = "NEUTRAL"


class ActionType(Enum):
    """Trading action recommendation."""
    STRONG_BUY = "STRONG_BUY"
    BUY = "BUY"
    HOLD = "HOLD"
    SELL = "SELL"
    STRONG_SELL = "STRONG_SELL"
    AVOID = "AVOID"
    NEUTRAL = "NEUTRAL"


@dataclass
class TechnicalSnapshot:
    """Technical analysis snapshot."""
    rsi: float = 0.0
    rsi_signal: str = "neutral"
    rsi_divergence: str = ""
    
    macd: float = 0.0
    macd_signal: float = 0.0
    macd_histogram: float = 0.0
    macd_bullish: bool = False
    macd_momentum: str = ""
    
    sma_20: float = 0.0
    sma_50: float = 0.0
    sma_200: float = 0.0
    trend_state: str = ""
    
    atr: float = 0.0
    atr_percent: float = 0.0
    volatility_regime: str = ""
    
    momentum_1d: float = 0.0
    momentum_5d: float = 0.0
    momentum_20d: float = 0.0
    
    technical_score: float = 0.0
    technical_confidence: float = 0.0


@dataclass
class IntelligenceSnapshot:
    """Market intelligence snapshot from news APIs."""
    # Sentiment
    sentiment_score: float = 0.0
    sentiment_raw: float = 0.0
    sentiment_label: str = ""
    sentiment_source: str = ""
    
    # News flow
    news_count_7d: int = 0
    news_count_24h: int = 0
    news_direction: str = ""
    
    # Analyst activity
    upgrades: int = 0
    downgrades: int = 0
    analyst_sentiment: float = 0.0
    
    # Trending
    is_trending: bool = False
    trending_rank: int = 0
    
    # Crypto-specific
    whale_activity: bool = False
    institutional_interest: str = ""
    regulatory_mentions: int = 0
    
    # Headlines
    headlines: List[str] = field(default_factory=list)
    sources: List[str] = field(default_factory=list)
    
    # VIX context
    vix_value: float = 0.0
    vix_state: str = ""
    
    intelligence_score: float = 0.0
    intelligence_confidence: float = 0.0


@dataclass
class LambdaAnalysis:
    """
    Comprehensive analysis response from Lambda decision engine.
    
    This is the primary data structure returned by the NUBLE API,
    containing all aggregated intelligence for a symbol.
    """
    # Core identification
    symbol: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    engine_version: str = "6.0.0"
    
    # Decision output
    action: str = "NEUTRAL"
    direction: str = "NEUTRAL"
    strength: str = "WEAK"
    score: float = 50.0
    confidence: float = 0.5
    should_trade: bool = False
    
    # Veto system (risk management)
    veto: bool = False
    veto_reason: Optional[str] = None
    
    # Price data
    current_price: float = 0.0
    change_percent: float = 0.0
    volume: int = 0
    
    # Market regime
    regime: MarketRegime = MarketRegime.UNKNOWN
    regime_confidence: float = 0.0
    
    # Technical analysis
    technicals: TechnicalSnapshot = field(default_factory=TechnicalSnapshot)
    
    # Intelligence layer
    intelligence: IntelligenceSnapshot = field(default_factory=IntelligenceSnapshot)
    
    # Formatted summaries for display
    stocknews_summary: str = ""
    cryptonews_summary: str = ""
    analysis_summary: str = ""
    
    # Data quality metrics
    data_points_used: int = 0
    polygon_available: bool = False
    luxalgo_aligned: bool = False
    
    # Raw response for advanced use
    raw_response: Dict = field(default_factory=dict)
    
    # Latency tracking
    api_latency_ms: float = 0.0
    
    @classmethod
    def from_response(cls, data: Dict, latency_ms: float = 0.0) -> 'LambdaAnalysis':
        """
        Parse Lambda API response into structured institutional-grade analysis.
        
        Handles the full V6 APEX PREDATOR response format with all layers.
        """
        try:
            decision = data.get('decision', {})
            layers = decision.get('layers', {})
            
            # Parse technical layer
            technical_layer = layers.get('technical', {})
            tech_components = technical_layer.get('components', {})
            
            rsi_data = tech_components.get('rsi', {})
            macd_data = tech_components.get('macd', {})
            trend_stack = tech_components.get('trend_stack', {})
            atr_data = tech_components.get('atr', {})
            momentum_data = tech_components.get('momentum', {})
            
            technicals = TechnicalSnapshot(
                rsi=rsi_data.get('value', 0),
                rsi_signal=rsi_data.get('signal', 'neutral'),
                rsi_divergence=rsi_data.get('divergence', ''),
                macd=macd_data.get('value', 0),
                macd_signal=macd_data.get('signal', 0),
                macd_histogram=macd_data.get('histogram', 0),
                macd_bullish=macd_data.get('bullish', False),
                macd_momentum=macd_data.get('momentum', ''),
                sma_20=trend_stack.get('sma_20', 0),
                sma_50=trend_stack.get('sma_50', 0),
                sma_200=trend_stack.get('sma_200', 0),
                trend_state=trend_stack.get('state', ''),
                atr=atr_data.get('value', 0),
                atr_percent=atr_data.get('pct', 0),
                volatility_regime=atr_data.get('regime', ''),
                momentum_1d=momentum_data.get('1d', 0),
                momentum_5d=momentum_data.get('5d', 0),
                momentum_20d=momentum_data.get('20d', 0),
                technical_score=technical_layer.get('score', 0),
                technical_confidence=technical_layer.get('confidence', 0)
            )
            
            # Parse intelligence layer
            intel_layer = layers.get('intelligence', {})
            intel_components = intel_layer.get('components', {})
            
            sentiment_data = intel_components.get('sentiment', {})
            news_flow = intel_components.get('news_flow', {})
            vix_data = intel_components.get('vix', {})
            regime_data = intel_components.get('regime', {})
            
            intelligence = IntelligenceSnapshot(
                sentiment_score=sentiment_data.get('score', 0),
                sentiment_raw=sentiment_data.get('raw_score', 0),
                sentiment_label=sentiment_data.get('label', ''),
                sentiment_source=sentiment_data.get('source', ''),
                news_count_7d=sentiment_data.get('news_count_7d', 0) or news_flow.get('count_7d', 0),
                news_count_24h=news_flow.get('count_24h', 0),
                news_direction=news_flow.get('direction', ''),
                upgrades=sentiment_data.get('upgrades', 0),
                downgrades=sentiment_data.get('downgrades', 0),
                analyst_sentiment=sentiment_data.get('analyst_sentiment', 0),
                is_trending=sentiment_data.get('is_trending', False) or news_flow.get('is_trending', False),
                trending_rank=sentiment_data.get('trending_rank', 0),
                whale_activity=sentiment_data.get('whale_activity', False),
                institutional_interest=sentiment_data.get('institutional_interest', ''),
                regulatory_mentions=sentiment_data.get('regulatory_mentions', 0),
                headlines=sentiment_data.get('headlines', []),
                sources=sentiment_data.get('sources', []),
                vix_value=vix_data.get('value', 0),
                vix_state=vix_data.get('state', ''),
                intelligence_score=intel_layer.get('score', 0),
                intelligence_confidence=intel_layer.get('confidence', 0)
            )
            
            # Determine market regime
            regime_str = decision.get('regime', regime_data.get('state', 'UNKNOWN'))
            try:
                regime = MarketRegime[regime_str.upper()]
            except (KeyError, AttributeError):
                regime = MarketRegime.UNKNOWN
            
            # Calculate action from direction and strength
            direction = decision.get('direction', 'NEUTRAL')
            strength = decision.get('strength', 'WEAK')
            
            if direction == 'BULLISH':
                action = 'STRONG_BUY' if strength == 'STRONG' else 'BUY' if strength == 'MODERATE' else 'HOLD'
            elif direction == 'BEARISH':
                action = 'STRONG_SELL' if strength == 'STRONG' else 'SELL' if strength == 'MODERATE' else 'HOLD'
            else:
                action = 'NEUTRAL'
            
            # Build formatted summaries
            stocknews_summary = cls._build_stocknews_summary(sentiment_data, news_flow)
            cryptonews_summary = cls._build_cryptonews_summary(sentiment_data, news_flow)
            analysis_summary = cls._build_analysis_summary(decision, technicals, intelligence)
            
            return cls(
                symbol=decision.get('symbol', data.get('symbol', '')),
                timestamp=datetime.fromisoformat(decision.get('timestamp', datetime.now(timezone.utc).isoformat()).replace('Z', '+00:00')) if decision.get('timestamp') else datetime.now(timezone.utc),
                engine_version=data.get('version', '6.0.0'),
                action=action,
                direction=direction,
                strength=strength,
                score=decision.get('confidence', 50),
                confidence=decision.get('confidence', 50) / 100,
                should_trade=decision.get('should_trade', False),
                veto=decision.get('veto', False),
                veto_reason=decision.get('veto_reason'),
                current_price=decision.get('price', 0),
                change_percent=technicals.momentum_1d,
                volume=0,
                regime=regime,
                regime_confidence=regime_data.get('confidence', 0),
                technicals=technicals,
                intelligence=intelligence,
                stocknews_summary=stocknews_summary,
                cryptonews_summary=cryptonews_summary,
                analysis_summary=analysis_summary,
                data_points_used=decision.get('data_points_used', 0),
                polygon_available=decision.get('polygon_available', False),
                luxalgo_aligned=decision.get('luxalgo_aligned', False),
                raw_response=data,
                api_latency_ms=latency_ms
            )
            
        except Exception as e:
            logger.error(f"Failed to parse Lambda response: {e}", exc_info=True)
            return cls(
                symbol=data.get('decision', {}).get('symbol', data.get('symbol', 'UNKNOWN')),
                action='ERROR',
                score=0,
                analysis_summary=f"Parse error: {e}",
                raw_response=data
            )
    
    @staticmethod
    def _build_stocknews_summary(sentiment: Dict, news_flow: Dict) -> str:
        """Build institutional-grade StockNews summary."""
        source = sentiment.get('source', '')
        if 'crypto' in source.lower():
            return ""
        
        parts = []
        
        # Sentiment with context
        raw_score = sentiment.get('raw_score', 0)
        if raw_score:
            label = sentiment.get('label', 'neutral')
            emoji = "🟢" if label == 'positive' else "🔴" if label == 'negative' else "⚪"
            parts.append(f"{emoji} Sentiment: {raw_score:.2f} ({label})")
        
        # News volume with trend
        count_7d = sentiment.get('news_count_7d', 0)
        count_24h = news_flow.get('count_24h', 0)
        if count_7d:
            parts.append(f"📰 News: {count_7d} (7d), {count_24h} (24h)")
        
        # Analyst actions - critical for institutional
        upgrades = sentiment.get('upgrades', 0)
        downgrades = sentiment.get('downgrades', 0)
        if upgrades or downgrades:
            parts.append(f"📊 Analysts: {upgrades}↑ {downgrades}↓")
        
        # Trending status
        if sentiment.get('is_trending') and sentiment.get('trending_rank'):
            parts.append(f"🔥 Trending #{sentiment['trending_rank']}")
        
        summary = " | ".join(parts)
        
        # Add headlines
        headlines = sentiment.get('headlines', [])
        if headlines:
            summary += "\n" + "\n".join([f"  • {h}" for h in headlines[:3]])
        
        return summary
    
    @staticmethod
    def _build_cryptonews_summary(sentiment: Dict, news_flow: Dict) -> str:
        """Build institutional-grade CryptoNews summary."""
        source = sentiment.get('source', '')
        if 'stock' in source.lower() and 'crypto' not in source.lower():
            return ""
        
        parts = []
        
        # Sentiment
        raw_score = sentiment.get('raw_score', 0)
        if raw_score:
            label = sentiment.get('label', 'neutral')
            emoji = "🟢" if label == 'positive' else "🔴" if label == 'negative' else "⚪"
            parts.append(f"{emoji} Sentiment: {raw_score:.2f} ({label})")
        
        # News volume
        count_7d = sentiment.get('news_count_7d', 0)
        if count_7d:
            parts.append(f"📰 News: {count_7d:,} (7d)")
        
        # Whale activity - critical for crypto
        if sentiment.get('whale_activity'):
            parts.append("🐋 Whale Activity Detected")
        
        # Institutional
        if sentiment.get('institutional_interest'):
            parts.append(f"🏦 Institutional: {sentiment['institutional_interest']}")
        
        # Trending
        if sentiment.get('is_trending') and sentiment.get('trending_rank'):
            parts.append(f"🔥 Trending #{sentiment['trending_rank']}")
        
        # Regulatory
        if sentiment.get('regulatory_mentions'):
            parts.append(f"⚖️ Regulatory: {sentiment['regulatory_mentions']} mentions")
        
        summary = " | ".join(parts)
        
        # Add headlines
        headlines = sentiment.get('headlines', [])
        if headlines:
            summary += "\n" + "\n".join([f"  • {h}" for h in headlines[:3]])
        
        return summary
    
    @staticmethod
    def _build_analysis_summary(decision: Dict, technicals: TechnicalSnapshot, intelligence: IntelligenceSnapshot) -> str:
        """Build comprehensive analysis summary."""
        parts = []
        
        # Regime context
        regime = decision.get('regime', 'N/A')
        parts.append(f"Regime: {regime}")
        
        # Score breakdown
        parts.append(f"Technical: {technicals.technical_score:+.2f}")
        parts.append(f"Intelligence: {intelligence.intelligence_score:+.2f}")
        
        # Trading recommendation
        should_trade = decision.get('should_trade', False)
        parts.append(f"Trade Signal: {'✅ YES' if should_trade else '❌ NO'}")
        
        # Veto check
        if decision.get('veto'):
            parts.append(f"⚠️ VETO: {decision.get('veto_reason', 'Risk threshold exceeded')}")
        
        return " | ".join(parts)
    
    # Legacy property aliases for backward compatibility
    @property
    def rsi(self) -> float:
        return self.technicals.rsi
    
    @property
    def macd(self) -> float:
        return self.technicals.macd
    
    @property
    def macd_signal(self) -> float:
        return self.technicals.macd_signal
    
    @property
    def sma_20(self) -> float:
        return self.technicals.sma_20
    
    @property
    def sma_50(self) -> float:
        return self.technicals.sma_50
    
    @property
    def atr(self) -> float:
        return self.technicals.atr
    
    @property
    def vix(self) -> float:
        return self.intelligence.vix_value
    
    @property
    def news_sentiment(self) -> float:
        return self.intelligence.sentiment_raw
    
    @property
    def news_headlines(self) -> List[str]:
        return self.intelligence.headlines


class NubleLambdaClient:
    """
    Institutional-Grade Market Intelligence API Client
    
    Production-ready client for the NUBLE Lambda Decision Engine.
    Implements retry logic, connection pooling, and comprehensive error handling.
    
    Features:
    ├── Automatic retry with exponential backoff
    ├── Connection pooling for performance
    ├── Latency tracking for monitoring
    ├── Comprehensive error classification
    └── Circuit breaker pattern for resilience
    
    Data Access:
    ├── Real-time market analysis with multi-source aggregation
    ├── StockNews API (24 endpoints): sentiment, ratings, earnings, SEC filings
    ├── CryptoNews API (17 endpoints): sentiment, whales, institutional, regulatory
    ├── Technical indicators via Polygon.io
    └── VIX, market breadth, cross-asset correlation
    
    Example:
        client = NubleLambdaClient()
        
        # Health check
        if client.is_healthy():
            analysis = client.get_analysis("AAPL")
            print(f"{analysis.symbol}: {analysis.action} (Score: {analysis.score})")
    """
    
    def __init__(
        self, 
        base_url: str = None, 
        timeout: int = REQUEST_TIMEOUT,
        max_retries: int = MAX_RETRIES,
        retry_backoff: float = RETRY_BACKOFF
    ):
        self.base_url = base_url or NUBLE_API_BASE
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_backoff = retry_backoff
        self._session = None
        self._last_health_check: Optional[Dict] = None
        self._health_check_time: Optional[datetime] = None
        
        # Circuit breaker state
        self._consecutive_failures = 0
        self._circuit_open = False
        self._circuit_open_time: Optional[datetime] = None
        self._circuit_timeout = 60  # seconds
    
    @property
    def session(self) -> requests.Session:
        """Get or create connection-pooled session."""
        if self._session is None:
            self._session = requests.Session()
            self._session.headers.update({
                'Content-Type': 'application/json',
                'Accept': 'application/json',
                'User-Agent': 'NUBLE-CLI/6.0.0-APEX',
                'X-Client-Version': '6.0.0',
                'X-Client-Type': 'institutional'
            })
            # Configure connection pooling
            adapter = requests.adapters.HTTPAdapter(
                pool_connections=10,
                pool_maxsize=10,
                max_retries=0  # We handle retries ourselves
            )
            self._session.mount('https://', adapter)
            self._session.mount('http://', adapter)
        return self._session
    
    def _check_circuit(self) -> bool:
        """Check if circuit breaker allows requests."""
        if not self._circuit_open:
            return True
        
        # Check if circuit should reset
        if self._circuit_open_time:
            elapsed = (datetime.now(timezone.utc) - self._circuit_open_time).total_seconds()
            if elapsed > self._circuit_timeout:
                logger.info("Circuit breaker reset - attempting recovery")
                self._circuit_open = False
                self._consecutive_failures = 0
                return True
        
        logger.warning("Circuit breaker OPEN - requests blocked")
        return False
    
    def _record_success(self):
        """Record successful request for circuit breaker."""
        self._consecutive_failures = 0
        if self._circuit_open:
            logger.info("Circuit breaker closed - service recovered")
            self._circuit_open = False
    
    def _record_failure(self):
        """Record failed request for circuit breaker."""
        self._consecutive_failures += 1
        if self._consecutive_failures >= 5 and not self._circuit_open:
            logger.error("Circuit breaker OPENED - too many consecutive failures")
            self._circuit_open = True
            self._circuit_open_time = datetime.now(timezone.utc)
    
    def _request_with_retry(
        self, 
        method: str, 
        url: str, 
        **kwargs
    ) -> Tuple[Optional[requests.Response], float]:
        """
        Execute HTTP request with retry logic and latency tracking.
        
        Returns:
            Tuple of (response, latency_ms) or (None, 0) on failure
        """
        if not self._check_circuit():
            return None, 0
        
        last_error = None
        
        for attempt in range(self.max_retries):
            try:
                start = time.time()
                response = self.session.request(
                    method, 
                    url, 
                    timeout=self.timeout,
                    **kwargs
                )
                latency_ms = (time.time() - start) * 1000
                
                response.raise_for_status()
                self._record_success()
                
                return response, latency_ms
                
            except requests.Timeout as e:
                last_error = e
                logger.warning(f"Request timeout (attempt {attempt + 1}/{self.max_retries}): {url}")
                
            except requests.HTTPError as e:
                last_error = e
                if e.response.status_code >= 500:
                    logger.warning(f"Server error {e.response.status_code} (attempt {attempt + 1})")
                else:
                    # Client errors shouldn't be retried
                    self._record_failure()
                    raise
                    
            except requests.RequestException as e:
                last_error = e
                logger.warning(f"Request failed (attempt {attempt + 1}): {e}")
            
            # Exponential backoff
            if attempt < self.max_retries - 1:
                sleep_time = self.retry_backoff ** attempt
                time.sleep(sleep_time)
        
        self._record_failure()
        if last_error:
            raise last_error
        return None, 0
    
    def is_healthy(self) -> bool:
        """Quick health check with caching."""
        try:
            health = self.health_check()
            return health.get('status') == 'healthy'
        except:
            return False
    
    def health_check(self, force: bool = False) -> Dict[str, Any]:
        """
        Check Lambda API health status.
        
        Caches result for 30 seconds unless force=True.
        """
        # Return cached result if recent
        if not force and self._last_health_check and self._health_check_time:
            elapsed = (datetime.now(timezone.utc) - self._health_check_time).total_seconds()
            if elapsed < 30:
                return self._last_health_check
        
        try:
            response, latency = self._request_with_retry(
                'GET',
                f"{self.base_url}/health"
            )
            if response:
                result = response.json()
                result['latency_ms'] = latency
                self._last_health_check = result
                self._health_check_time = datetime.now(timezone.utc)
                return result
            return {"status": "error", "error": "No response"}
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {"status": "error", "error": str(e)}
    
    def get_analysis(self, symbol: str) -> LambdaAnalysis:
        """
        Get comprehensive institutional-grade analysis for a symbol.
        
        Aggregates data from multiple sources:
        - Polygon.io: Price, volume, technicals (RSI, MACD, SMA, ATR)
        - StockNews API: Sentiment, ratings, earnings, SEC filings, events
        - CryptoNews API: Sentiment, whales, institutional, regulatory
        - Derived: Regime classification, decision scoring
        
        Args:
            symbol: Stock ticker (AAPL, TSLA) or crypto (BTC, ETH)
        
        Returns:
            LambdaAnalysis with full institutional data
            
        Raises:
            No exceptions - errors are captured in LambdaAnalysis.action='ERROR'
        """
        try:
            url = f"{self.base_url}/check/{symbol.upper()}"
            logger.info(f"Fetching analysis: {symbol}")
            
            response, latency = self._request_with_retry('GET', url)
            
            if response:
                data = response.json()
                analysis = LambdaAnalysis.from_response(data, latency_ms=latency)
                logger.info(f"Analysis complete: {symbol} -> {analysis.action} ({latency:.0f}ms)")
                return analysis
            
            return LambdaAnalysis(
                symbol=symbol,
                action='ERROR',
                analysis_summary="Failed to get response from decision engine"
            )
            
        except requests.Timeout:
            logger.error(f"Analysis timeout: {symbol}")
            return LambdaAnalysis(
                symbol=symbol,
                action='TIMEOUT',
                analysis_summary="Request timeout - try again"
            )
        except requests.HTTPError as e:
            logger.error(f"Analysis HTTP error: {symbol} - {e}")
            return LambdaAnalysis(
                symbol=symbol,
                action='ERROR',
                analysis_summary=f"HTTP {e.response.status_code}: {e.response.reason}"
            )
        except Exception as e:
            logger.error(f"Analysis failed: {symbol} - {e}")
            return LambdaAnalysis(
                symbol=symbol,
                action='ERROR',
                analysis_summary=f"Error: {str(e)[:100]}"
            )
    
    def get_signals(self, symbol: str) -> Dict[str, Any]:
        """
        Get trading signals for a symbol.
        
        Args:
            symbol: Stock ticker or crypto
        
        Returns:
            Dict with signals data
        """
        try:
            url = f"{self.base_url}/signals/{symbol.upper()}"
            response = self.session.get(url, timeout=self.timeout)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Signals API failed for {symbol}: {e}")
            return {"error": str(e)}


# Singleton instance
_lambda_client: Optional[NubleLambdaClient] = None


def get_lambda_client() -> NubleLambdaClient:
    """Get singleton Lambda client instance."""
    global _lambda_client
    if _lambda_client is None:
        _lambda_client = NubleLambdaClient()
    return _lambda_client


def analyze_symbol(symbol: str) -> LambdaAnalysis:
    """
    Quick function to analyze a symbol via Lambda.
    
    Example:
        analysis = analyze_symbol("AAPL")
        print(f"Action: {analysis.action}, Score: {analysis.score}")
        print(analysis.stocknews_summary)
    """
    return get_lambda_client().get_analysis(symbol)


def format_analysis_for_context(analysis: LambdaAnalysis) -> str:
    """
    Format Lambda analysis as institutional-grade context for LLM prompt injection.
    
    Produces a structured markdown summary optimized for AI consumption,
    containing all decision-relevant data from the multi-source aggregation.
    
    Args:
        analysis: LambdaAnalysis from get_analysis()
        
    Returns:
        Markdown-formatted string suitable for prompt injection
    """
    parts = []
    
    # Header with decision summary
    regime_emoji = {
        MarketRegime.BULL: "🟢",
        MarketRegime.BEAR: "🔴", 
        MarketRegime.VOLATILE: "🟡",
        MarketRegime.RANGING: "⚪",
        MarketRegime.CRISIS: "🔴⚠️",
    }.get(analysis.regime, "⚪")
    
    action_emoji = {
        'STRONG_BUY': '🟢🟢',
        'BUY': '🟢',
        'HOLD': '⚪',
        'SELL': '🔴',
        'STRONG_SELL': '🔴🔴',
        'AVOID': '⛔',
        'NEUTRAL': '⚪'
    }.get(analysis.action, '⚪')
    
    parts.append(f"## NUBLE Decision Engine: {analysis.symbol}")
    parts.append(f"**{action_emoji} {analysis.action}** | Score: {analysis.score:.1f}/100 | {regime_emoji} {analysis.regime.value} Regime")
    
    if analysis.veto:
        parts.append(f"⚠️ **VETO ACTIVE**: {analysis.veto_reason}")
    
    if analysis.should_trade:
        parts.append("✅ **Trade Signal: ACTIVE**")
    else:
        parts.append("❌ Trade Signal: Inactive")
    
    parts.append("")
    
    # Price section with comprehensive data
    if analysis.current_price > 0:
        change_emoji = "📈" if analysis.change_percent >= 0 else "📉"
        change_color = "+" if analysis.change_percent >= 0 else ""
        parts.append(f"### Price: ${analysis.current_price:,.2f} {change_emoji} {change_color}{analysis.change_percent:.2f}%")
        parts.append("")
    
    # Technical analysis section
    tech = analysis.technicals
    if tech.rsi > 0:
        parts.append("### Technical Analysis")
        
        # RSI with signal
        rsi_signal = ""
        if tech.rsi > 70:
            rsi_signal = " ⚠️ OVERBOUGHT"
        elif tech.rsi < 30:
            rsi_signal = " ⚠️ OVERSOLD"
        elif tech.rsi_divergence:
            rsi_signal = f" ({tech.rsi_divergence})"
        parts.append(f"- **RSI(14)**: {tech.rsi:.1f}{rsi_signal}")
        
        # MACD with momentum
        macd_signal = "bullish 📈" if tech.macd_bullish else "bearish 📉"
        parts.append(f"- **MACD**: {tech.macd:.4f} vs Signal {tech.macd_signal:.4f} ({macd_signal})")
        if tech.macd_momentum:
            parts.append(f"  - Momentum: {tech.macd_momentum}")
        
        # Trend stack
        if tech.sma_20 > 0:
            trend_state = tech.trend_state or "N/A"
            parts.append(f"- **Trend**: {trend_state.upper()}")
            parts.append(f"  - SMA(20): ${tech.sma_20:.2f} | SMA(50): ${tech.sma_50:.2f} | SMA(200): ${tech.sma_200:.2f}")
        
        # Volatility
        if tech.atr > 0:
            vol_regime = tech.volatility_regime or "normal"
            parts.append(f"- **Volatility**: ATR ${tech.atr:.2f} ({tech.atr_percent:.1f}%) - {vol_regime.upper()}")
        
        # Momentum summary
        parts.append(f"- **Momentum**: 1D: {tech.momentum_1d:+.2f}% | 5D: {tech.momentum_5d:+.2f}% | 20D: {tech.momentum_20d:+.2f}%")
        
        # Score
        parts.append(f"- **Technical Score**: {tech.technical_score:+.2f} (confidence: {tech.technical_confidence:.0%})")
        parts.append("")
    
    # Market context (VIX)
    intel = analysis.intelligence
    if intel.vix_value > 0:
        vix_state = intel.vix_state or "normal"
        vix_emoji = "🟢" if intel.vix_value < 15 else "🟡" if intel.vix_value < 25 else "🔴"
        parts.append(f"### Market Context")
        parts.append(f"- **VIX**: {intel.vix_value:.1f} {vix_emoji} ({vix_state.upper()})")
        parts.append("")
    
    # StockNews Intelligence
    if analysis.stocknews_summary:
        parts.append("### StockNews API Intelligence")
        parts.append(analysis.stocknews_summary)
        parts.append("")
    
    # CryptoNews Intelligence
    if analysis.cryptonews_summary:
        parts.append("### CryptoNews API Intelligence")
        parts.append(analysis.cryptonews_summary)
        parts.append("")
    
    # Analyst/News flow summary (for stocks)
    if intel.upgrades or intel.downgrades:
        parts.append("### Analyst Activity")
        net_analyst = intel.upgrades - intel.downgrades
        net_emoji = "📈" if net_analyst > 0 else "📉" if net_analyst < 0 else "➡️"
        parts.append(f"- Upgrades: {intel.upgrades} | Downgrades: {intel.downgrades} | Net: {net_emoji} {net_analyst:+d}")
        if intel.analyst_sentiment:
            sent_label = "BULLISH" if intel.analyst_sentiment > 0.5 else "BEARISH" if intel.analyst_sentiment < -0.5 else "NEUTRAL"
            parts.append(f"- Analyst Sentiment: {sent_label}")
        parts.append("")
    
    # Headlines (if not already in summaries)
    if intel.headlines and not analysis.stocknews_summary and not analysis.cryptonews_summary:
        parts.append("### Latest Headlines")
        for headline in intel.headlines[:5]:
            parts.append(f"  • {headline}")
        parts.append("")
    
    # Decision summary
    parts.append("### Decision Summary")
    parts.append(analysis.analysis_summary)
    
    # Data quality footer
    parts.append("")
    parts.append(f"*Data points: {analysis.data_points_used} | Polygon: {'✓' if analysis.polygon_available else '✗'} | Latency: {analysis.api_latency_ms:.0f}ms*")
    
    return "\n".join(parts)


# Export symbols detection utilities
CRYPTO_SYMBOLS = {
    'BTC', 'ETH', 'USDT', 'BNB', 'XRP', 'ADA', 'DOGE', 'SOL', 'DOT', 
    'MATIC', 'AVAX', 'LINK', 'UNI', 'ATOM', 'LTC', 'ETC', 'XLM', 'ALGO',
    'VET', 'MANA', 'SAND', 'AXS', 'FTM', 'NEAR', 'HBAR', 'EGLD', 'XMR',
    'BITCOIN', 'ETHEREUM', 'SOLANA', 'CARDANO', 'RIPPLE', 'DOGECOIN',
    'POLKADOT', 'POLYGON', 'AVALANCHE', 'CHAINLINK', 'LITECOIN'
}

COMMON_STOCKS = {
    'AAPL', 'MSFT', 'GOOGL', 'GOOG', 'AMZN', 'META', 'NVDA', 'TSLA', 'AMD',
    'INTC', 'NFLX', 'DIS', 'JPM', 'BAC', 'WMT', 'V', 'MA', 'HD', 'PG', 'JNJ',
    'UNH', 'XOM', 'CVX', 'PFE', 'ABBV', 'MRK', 'KO', 'PEP', 'COST', 'AVGO',
    'ADBE', 'CRM', 'ORCL', 'CSCO', 'ACN', 'TXN', 'QCOM', 'NOW', 'IBM', 'INTU',
    'SNOW', 'PLTR', 'COIN', 'SQ', 'PYPL', 'ROKU', 'SHOP', 'ZM', 'DDOG', 'NET',
    'CRWD', 'ZS', 'OKTA', 'MDB', 'U', 'RBLX', 'ABNB', 'UBER', 'LYFT', 'DASH',
    'SNAP', 'PINS', 'SPOT', 'SOFI', 'HOOD', 'RIVN', 'LCID', 'NIO', 'XPEV',
    'LI', 'F', 'GM', 'BABA', 'JD', 'PDD', 'BIDU', 'NKE', 'LULU', 'TGT', 'LOW',
    'SPY', 'QQQ', 'IWM', 'DIA', 'VTI', 'VOO', 'XLK', 'XLF', 'XLE', 'XLV'
}


def extract_symbols(text: str) -> list:
    """
    Extract stock/crypto symbols from text.
    
    Returns list of symbols found, prioritizing explicit tickers.
    """
    import re
    text_upper = text.upper()
    
    symbols = []
    
    # Look for explicit ticker patterns ($ prefix or common stocks)
    ticker_pattern = r'\$([A-Z]{1,5})\b'
    explicit = re.findall(ticker_pattern, text)
    symbols.extend(explicit)
    
    # Check for crypto symbols
    for crypto in CRYPTO_SYMBOLS:
        if crypto in text_upper:
            # Normalize to short form
            if crypto == 'BITCOIN':
                symbols.append('BTC')
            elif crypto == 'ETHEREUM':
                symbols.append('ETH')
            elif crypto == 'SOLANA':
                symbols.append('SOL')
            elif crypto == 'CARDANO':
                symbols.append('ADA')
            elif crypto == 'RIPPLE':
                symbols.append('XRP')
            elif crypto == 'DOGECOIN':
                symbols.append('DOGE')
            elif crypto == 'POLKADOT':
                symbols.append('DOT')
            elif crypto == 'POLYGON':
                symbols.append('MATIC')
            elif crypto == 'AVALANCHE':
                symbols.append('AVAX')
            elif crypto == 'CHAINLINK':
                symbols.append('LINK')
            elif crypto == 'LITECOIN':
                symbols.append('LTC')
            else:
                symbols.append(crypto)
    
    # Check for stock symbols (but be more careful to avoid false positives)
    words = re.findall(r'\b([A-Z]{1,5})\b', text_upper)
    for word in words:
        if word in COMMON_STOCKS and word not in symbols:
            symbols.append(word)
    
    # Remove duplicates while preserving order
    seen = set()
    unique_symbols = []
    for s in symbols:
        if s not in seen:
            seen.add(s)
            unique_symbols.append(s)
    
    return unique_symbols


def is_crypto(symbol: str) -> bool:
    """Check if a symbol is cryptocurrency."""
    return symbol.upper() in CRYPTO_SYMBOLS
