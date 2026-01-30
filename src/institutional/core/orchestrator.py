"""
Orchestrator - Central coordination for data aggregation and analysis.
Routes queries, coordinates providers, and synthesizes responses.
"""

import asyncio
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass, field
from datetime import datetime, date, timedelta
from enum import Enum
import json

from ..config import Config, load_config
from ..core.intent_engine import IntentEngine, QueryIntent
from ..core.router import DataRouter
from ..providers.base import BaseProvider, ProviderResponse, DataType
from ..providers.polygon import PolygonProvider
from ..providers.alpha_vantage import AlphaVantageProvider
from ..providers.finnhub import FinnhubProvider
from ..providers.sec_edgar import SECEdgarProvider
from ..analytics.technical import TechnicalAnalyzer
from ..analytics.sentiment import SentimentAnalyzer
from ..analytics.patterns import PatternRecognizer
from ..analytics.anomaly import AnomalyDetector


class QueryStatus(Enum):
    """Status of a query execution"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    PARTIAL = "partial"
    FAILED = "failed"


@dataclass
class QueryResult:
    """Result from an orchestrated query"""
    query: str
    intent: QueryIntent
    status: QueryStatus
    data: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    providers_used: List[str] = field(default_factory=list)
    latency_ms: float = 0
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()
    
    def to_dict(self) -> Dict:
        return {
            "query": self.query,
            "intent": {
                "primary": self.intent.primary_intent,
                "symbols": self.intent.symbols,
                "data_types": [dt.value for dt in self.intent.data_types],
            },
            "status": self.status.value,
            "data": self.data,
            "errors": self.errors,
            "providers_used": self.providers_used,
            "latency_ms": self.latency_ms,
            "timestamp": self.timestamp.isoformat()
        }


class Orchestrator:
    """
    Central orchestrator for the institutional research platform.
    
    Responsibilities:
    - Parse and understand user queries
    - Route to appropriate data providers
    - Coordinate parallel data fetching
    - Apply analytics and ML models
    - Synthesize comprehensive responses
    """
    
    def __init__(
        self,
        config: Optional[Config] = None,
        openai_api_key: Optional[str] = None
    ):
        """
        Initialize the orchestrator.
        
        Args:
            config: Configuration object (or will load from environment)
            openai_api_key: OpenAI API key for LLM synthesis
        """
        self.config = config or load_config()
        self.openai_api_key = openai_api_key or self.config.openai_api_key
        
        # Initialize core components
        self.intent_engine = IntentEngine(api_key=self.openai_api_key)
        self.router = DataRouter()
        
        # Initialize providers
        self._providers: Dict[str, BaseProvider] = {}
        self._init_providers()
        
        # Initialize analytics
        self.technical_analyzer = TechnicalAnalyzer()
        self.sentiment_analyzer = SentimentAnalyzer(use_ml=False)
        self.pattern_recognizer = PatternRecognizer(use_ml=False)
        self.anomaly_detector = AnomalyDetector()
        
        # Cache
        self._cache: Dict[str, Any] = {}
        self._cache_ttl: Dict[str, datetime] = {}
    
    def _init_providers(self):
        """Initialize data providers based on available API keys"""
        # Polygon.io
        if self.config.polygon_api_key:
            self._providers["polygon"] = PolygonProvider(
                api_key=self.config.polygon_api_key
            )
        
        # Alpha Vantage
        if self.config.alpha_vantage_api_key:
            self._providers["alpha_vantage"] = AlphaVantageProvider(
                api_key=self.config.alpha_vantage_api_key
            )
        
        # Finnhub
        if self.config.finnhub_api_key:
            self._providers["finnhub"] = FinnhubProvider(
                api_key=self.config.finnhub_api_key
            )
        
        # SEC EDGAR (no API key required)
        self._providers["sec_edgar"] = SECEdgarProvider()
    
    def get_available_providers(self) -> List[str]:
        """Get list of available providers"""
        return list(self._providers.keys())
    
    def _get_cache_key(
        self,
        symbol: str,
        data_type: DataType,
        provider: str,
        params: Optional[Dict] = None
    ) -> str:
        """Generate cache key"""
        param_str = json.dumps(params, sort_keys=True) if params else ""
        return f"{symbol}:{data_type.value}:{provider}:{param_str}"
    
    def _is_cache_valid(self, key: str, ttl_seconds: int = 60) -> bool:
        """Check if cache entry is still valid"""
        if key not in self._cache:
            return False
        expiry = self._cache_ttl.get(key)
        if not expiry:
            return False
        return datetime.now() < expiry
    
    async def _fetch_from_provider(
        self,
        provider_name: str,
        data_type: DataType,
        symbol: str,
        params: Optional[Dict] = None
    ) -> Optional[ProviderResponse]:
        """Fetch data from a specific provider"""
        provider = self._providers.get(provider_name)
        if not provider:
            return None
        
        params = params or {}
        
        try:
            if data_type == DataType.QUOTE:
                return await provider.get_quote(symbol)
            
            elif data_type == DataType.OHLCV:
                end_date = params.get("end_date", date.today())
                start_date = params.get("start_date", end_date - timedelta(days=365))
                timeframe = params.get("timeframe", "1d")
                return await provider.get_historical(symbol, start_date, end_date, timeframe)
            
            elif data_type == DataType.NEWS:
                if hasattr(provider, "get_news"):
                    return await provider.get_news(symbol)
            
            elif data_type == DataType.SENTIMENT:
                if hasattr(provider, "get_sentiment"):
                    return await provider.get_sentiment(symbol)
            
            elif data_type == DataType.OPTIONS:
                if hasattr(provider, "get_options_chain"):
                    return await provider.get_options_chain(symbol)
            
            elif data_type == DataType.FUNDAMENTALS:
                if hasattr(provider, "get_fundamentals"):
                    return await provider.get_fundamentals(symbol)
                elif hasattr(provider, "get_company_facts"):
                    return await provider.get_company_facts(symbol)
            
            elif data_type == DataType.TECHNICAL:
                if hasattr(provider, "get_technical_indicator"):
                    indicator = params.get("indicator", "RSI")
                    return await provider.get_technical_indicator(symbol, indicator)
            
            elif data_type == DataType.FILING:
                if hasattr(provider, "get_filings"):
                    form_types = params.get("form_types")
                    return await provider.get_filings(symbol, form_types)
            
            elif data_type == DataType.HOLDINGS:
                if hasattr(provider, "get_institutional_holdings"):
                    return await provider.get_institutional_holdings(symbol)
            
            elif data_type == DataType.TRANSACTIONS:
                if hasattr(provider, "get_insider_transactions"):
                    return await provider.get_insider_transactions(symbol)
            
            elif data_type == DataType.PROFILE:
                if hasattr(provider, "get_company_profile"):
                    return await provider.get_company_profile(symbol)
                elif hasattr(provider, "get_company_overview"):
                    return await provider.get_company_overview(symbol)
            
            elif data_type == DataType.EARNINGS:
                if hasattr(provider, "get_earnings_surprises"):
                    return await provider.get_earnings_surprises(symbol)
                elif hasattr(provider, "get_earnings"):
                    return await provider.get_earnings(symbol)
            
        except Exception as e:
            return ProviderResponse(
                success=False,
                data=None,
                data_type=data_type,
                provider=provider_name,
                symbol=symbol,
                error=str(e)
            )
        
        return None
    
    async def _fetch_data_parallel(
        self,
        intent: QueryIntent,
        params: Optional[Dict] = None
    ) -> Dict[str, List[ProviderResponse]]:
        """Fetch data from multiple providers in parallel"""
        results: Dict[str, List[ProviderResponse]] = {}
        tasks = []
        task_info = []  # Track what each task is fetching
        
        for symbol in intent.symbols:
            for data_type in intent.data_types:
                # Route to appropriate providers
                routing = self.router.route(
                    data_type,
                    list(self._providers.keys())
                )
                
                for provider_name in routing.providers:
                    task = self._fetch_from_provider(
                        provider_name, data_type, symbol, params
                    )
                    tasks.append(task)
                    task_info.append((symbol, data_type, provider_name))
        
        # Execute all tasks in parallel
        if tasks:
            responses = await asyncio.gather(*tasks, return_exceptions=True)
            
            for i, response in enumerate(responses):
                symbol, data_type, provider_name = task_info[i]
                
                if symbol not in results:
                    results[symbol] = []
                
                if isinstance(response, Exception):
                    results[symbol].append(ProviderResponse(
                        success=False,
                        data=None,
                        data_type=data_type,
                        provider=provider_name,
                        symbol=symbol,
                        error=str(response)
                    ))
                elif response is not None:
                    results[symbol].append(response)
        
        return results
    
    def _apply_analytics(
        self,
        data: Dict[str, List[ProviderResponse]],
        intent: QueryIntent
    ) -> Dict[str, Any]:
        """Apply analytics to fetched data"""
        analytics_results = {}
        
        for symbol, responses in data.items():
            symbol_analytics = {}
            
            # Get OHLCV data for analysis
            ohlcv_data = None
            for response in responses:
                if response.data_type == DataType.OHLCV and response.success:
                    ohlcv_data = response.data
                    break
            
            if ohlcv_data and len(ohlcv_data) > 0:
                # Extract arrays
                opens = [c.open for c in ohlcv_data]
                highs = [c.high for c in ohlcv_data]
                lows = [c.low for c in ohlcv_data]
                closes = [c.close for c in ohlcv_data]
                volumes = [c.volume for c in ohlcv_data]
                
                # Technical analysis
                if "technical" in intent.primary_intent or DataType.TECHNICAL in intent.data_types:
                    symbol_analytics["technical"] = self.technical_analyzer.analyze(
                        highs, lows, closes, volumes, symbol
                    )
                    symbol_analytics["signals"] = self.technical_analyzer.get_signals(
                        highs, lows, closes, volumes
                    )
                
                # Pattern recognition
                if "pattern" in intent.primary_intent or len(closes) >= 20:
                    symbol_analytics["patterns"] = self.pattern_recognizer.analyze(
                        opens, highs, lows, closes, volumes
                    )
                
                # Anomaly detection
                symbol_analytics["anomalies"] = self.anomaly_detector.analyze(
                    opens, highs, lows, closes, volumes, symbol
                )
            
            # Sentiment analysis on news
            news_data = None
            for response in responses:
                if response.data_type == DataType.NEWS and response.success:
                    news_data = response.data
                    break
            
            if news_data:
                articles = [
                    {
                        "title": a.title if hasattr(a, 'title') else a.get('title', ''),
                        "summary": a.summary if hasattr(a, 'summary') else a.get('summary', ''),
                        "published_at": a.published_at if hasattr(a, 'published_at') else a.get('published_at')
                    }
                    for a in news_data
                ]
                symbol_analytics["sentiment"] = self.sentiment_analyzer.analyze_news_feed(articles)
            
            if symbol_analytics:
                analytics_results[symbol] = symbol_analytics
        
        return analytics_results
    
    def _aggregate_responses(
        self,
        data: Dict[str, List[ProviderResponse]],
        analytics: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Aggregate and organize all data for response"""
        aggregated = {}
        
        for symbol, responses in data.items():
            symbol_data = {
                "symbol": symbol,
                "timestamp": datetime.now().isoformat(),
            }
            
            # Organize by data type
            for response in responses:
                if not response.success:
                    continue
                
                type_key = response.data_type.value
                
                if type_key not in symbol_data:
                    symbol_data[type_key] = response.data
                else:
                    # Merge if multiple providers returned same data type
                    existing = symbol_data[type_key]
                    if isinstance(existing, list) and isinstance(response.data, list):
                        symbol_data[type_key] = existing + response.data
            
            # Add analytics
            if symbol in analytics:
                symbol_data["analytics"] = analytics[symbol]
            
            aggregated[symbol] = symbol_data
        
        return aggregated
    
    async def query(
        self,
        query: str,
        symbols: Optional[List[str]] = None,
        params: Optional[Dict] = None
    ) -> QueryResult:
        """
        Execute a natural language query.
        
        Args:
            query: Natural language query
            symbols: Override symbols (extracted from query if not provided)
            params: Additional parameters (date range, indicators, etc.)
        
        Returns:
            QueryResult with aggregated data and analysis
        """
        start_time = datetime.now()
        errors = []
        providers_used: Set[str] = set()
        
        # Step 1: Understand the query
        intent = await self.intent_engine.analyze(query)
        
        # Override symbols if provided
        if symbols:
            intent.symbols = symbols
        
        if not intent.symbols:
            return QueryResult(
                query=query,
                intent=intent,
                status=QueryStatus.FAILED,
                errors=["No symbols detected in query. Please specify a stock symbol."],
                latency_ms=0
            )
        
        # Step 2: Fetch data from providers
        try:
            data = await self._fetch_data_parallel(intent, params)
            
            # Track which providers succeeded
            for symbol, responses in data.items():
                for response in responses:
                    if response.success:
                        providers_used.add(response.provider)
                    else:
                        errors.append(f"{response.provider}: {response.error}")
            
        except Exception as e:
            return QueryResult(
                query=query,
                intent=intent,
                status=QueryStatus.FAILED,
                errors=[str(e)],
                latency_ms=(datetime.now() - start_time).total_seconds() * 1000
            )
        
        # Step 3: Apply analytics
        analytics = self._apply_analytics(data, intent)
        
        # Step 4: Aggregate results
        aggregated = self._aggregate_responses(data, analytics)
        
        # Determine status
        if not aggregated:
            status = QueryStatus.FAILED
        elif errors:
            status = QueryStatus.PARTIAL
        else:
            status = QueryStatus.COMPLETED
        
        latency_ms = (datetime.now() - start_time).total_seconds() * 1000
        
        return QueryResult(
            query=query,
            intent=intent,
            status=status,
            data=aggregated,
            errors=errors[:5],  # Limit errors
            providers_used=list(providers_used),
            latency_ms=latency_ms
        )
    
    async def get_quote(self, symbol: str) -> QueryResult:
        """Convenience method to get a quote"""
        return await self.query(f"What is the current price of {symbol}?", [symbol])
    
    async def get_analysis(self, symbol: str) -> QueryResult:
        """Convenience method to get comprehensive analysis"""
        return await self.query(
            f"Give me a comprehensive technical and fundamental analysis of {symbol}",
            [symbol]
        )
    
    async def get_options_flow(self, symbol: str) -> QueryResult:
        """Convenience method to get options data"""
        return await self.query(
            f"Show me the options chain and unusual activity for {symbol}",
            [symbol]
        )
    
    async def get_insider_activity(self, symbol: str) -> QueryResult:
        """Convenience method to get insider transactions"""
        return await self.query(
            f"Show me recent insider transactions for {symbol}",
            [symbol]
        )
    
    async def get_sec_filings(
        self,
        symbol: str,
        form_types: Optional[List[str]] = None
    ) -> QueryResult:
        """Convenience method to get SEC filings"""
        form_str = ", ".join(form_types) if form_types else "recent"
        return await self.query(
            f"Show me {form_str} SEC filings for {symbol}",
            [symbol],
            params={"form_types": form_types}
        )
    
    async def close(self):
        """Clean up resources"""
        for provider in self._providers.values():
            if hasattr(provider, 'close'):
                await provider.close()
