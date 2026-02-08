"""
Base Provider Interface - Abstract base class for all data providers.
Defines the contract that all providers must implement.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, AsyncIterator
from dataclasses import dataclass, field
from datetime import datetime, date
from enum import Enum
import asyncio
import time

try:
    import aiohttp
    HAS_AIOHTTP = True
except ImportError:
    aiohttp = None  # type: ignore
    HAS_AIOHTTP = False


class DataType(Enum):
    """Types of data that can be returned"""
    QUOTE = "quote"
    OHLCV = "ohlcv"
    TICK = "tick"
    OPTIONS = "options"
    FUNDAMENTALS = "fundamentals"
    NEWS = "news"
    FILING = "filing"
    ANALYTICS = "analytics"
    PROFILE = "profile"
    HOLDINGS = "holdings"
    TRANSACTIONS = "transactions"


@dataclass
class Quote:
    """Real-time quote data"""
    symbol: str
    price: float
    bid: Optional[float] = None
    ask: Optional[float] = None
    bid_size: Optional[int] = None
    ask_size: Optional[int] = None
    volume: Optional[int] = None
    timestamp: Optional[datetime] = None
    change: Optional[float] = None
    change_percent: Optional[float] = None
    high: Optional[float] = None
    low: Optional[float] = None
    open: Optional[float] = None
    prev_close: Optional[float] = None
    market_cap: Optional[float] = None
    
    # Extended hours
    pre_market_price: Optional[float] = None
    after_hours_price: Optional[float] = None


@dataclass
class OHLCV:
    """OHLCV bar data"""
    symbol: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int
    vwap: Optional[float] = None
    trades: Optional[int] = None
    
    # Adjusted values
    adj_close: Optional[float] = None


@dataclass
class OptionsContract:
    """Options contract data"""
    symbol: str
    underlying: str
    contract_type: str  # call or put
    strike: float
    expiration: date
    bid: float
    ask: float
    last: float
    volume: int
    open_interest: int
    implied_volatility: Optional[float] = None
    delta: Optional[float] = None
    gamma: Optional[float] = None
    theta: Optional[float] = None
    vega: Optional[float] = None
    rho: Optional[float] = None
    in_the_money: Optional[bool] = None


@dataclass
class NewsArticle:
    """News article data"""
    id: str
    title: str
    summary: Optional[str] = None
    content: Optional[str] = None
    url: Optional[str] = None
    source: Optional[str] = None
    author: Optional[str] = None
    published_at: Optional[datetime] = None
    symbols: List[str] = field(default_factory=list)
    sentiment_score: Optional[float] = None  # -1 to 1
    sentiment_label: Optional[str] = None  # bullish, bearish, neutral
    keywords: List[str] = field(default_factory=list)


@dataclass
class Filing:
    """SEC Filing data"""
    accession_number: str
    form_type: str
    filed_date: date
    accepted_date: Optional[datetime] = None
    cik: str = ""
    company_name: str = ""
    symbol: Optional[str] = None
    url: Optional[str] = None
    description: Optional[str] = None
    
    # For 13F filings
    holdings: Optional[List[Dict]] = None
    
    # For insider filings (Form 4)
    transactions: Optional[List[Dict]] = None
    
    # Parsed content
    content: Optional[Dict] = None


@dataclass
class FinancialStatement:
    """Financial statement data (income statement, balance sheet, cash flow)"""
    symbol: str
    statement_type: str  # income_statement, balance_sheet, cash_flow
    fiscal_period: str  # Q1, Q2, Q3, Q4, FY
    fiscal_year: int
    report_date: date
    currency: str = "USD"
    
    # Common fields - will contain statement-specific line items
    data: Dict[str, Any] = field(default_factory=dict)
    
    # Statement-specific notable items
    revenue: Optional[float] = None
    net_income: Optional[float] = None
    total_assets: Optional[float] = None
    total_liabilities: Optional[float] = None
    operating_cash_flow: Optional[float] = None
    free_cash_flow: Optional[float] = None


@dataclass
class CompanyProfile:
    """Company profile data"""
    symbol: str
    name: str
    description: Optional[str] = None
    sector: Optional[str] = None
    industry: Optional[str] = None
    exchange: Optional[str] = None
    market_cap: Optional[float] = None
    employees: Optional[int] = None
    website: Optional[str] = None
    ceo: Optional[str] = None
    headquarters: Optional[str] = None
    founded: Optional[int] = None
    ipo_date: Optional[date] = None
    
    # Identifiers
    cik: Optional[str] = None
    cusip: Optional[str] = None
    isin: Optional[str] = None


@dataclass
class InstitutionalHolding:
    """Institutional holding data"""
    institution_name: str
    cik: str
    symbol: str
    shares: int
    value: float
    weight_percent: Optional[float] = None
    change_shares: Optional[int] = None
    change_percent: Optional[float] = None
    report_date: date = None
    filing_date: Optional[date] = None


@dataclass
class InsiderTransaction:
    """Insider transaction data"""
    symbol: str
    insider_name: str
    title: Optional[str] = None
    transaction_type: str = ""  # buy, sell, gift, etc.
    shares: int = 0
    price: Optional[float] = None
    value: Optional[float] = None
    shares_owned_after: Optional[int] = None
    transaction_date: date = None
    filing_date: Optional[date] = None


@dataclass
class ProviderResponse:
    """Standard response wrapper from providers"""
    success: bool
    data: Any
    data_type: DataType
    provider: str
    symbol: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)
    cached: bool = False
    latency_ms: int = 0
    error: Optional[str] = None
    
    # Rate limit info
    rate_limit_remaining: Optional[int] = None
    rate_limit_reset: Optional[datetime] = None


class RateLimiter:
    """Token bucket rate limiter"""
    
    def __init__(self, calls_per_minute: int):
        self.calls_per_minute = calls_per_minute
        self.tokens = calls_per_minute
        self.last_update = time.time()
        self._lock = asyncio.Lock()
    
    async def acquire(self):
        """Acquire a token, waiting if necessary"""
        async with self._lock:
            now = time.time()
            elapsed = now - self.last_update
            
            # Refill tokens
            self.tokens = min(
                self.calls_per_minute,
                self.tokens + elapsed * (self.calls_per_minute / 60)
            )
            self.last_update = now
            
            if self.tokens < 1:
                # Wait for token
                wait_time = (1 - self.tokens) * (60 / self.calls_per_minute)
                await asyncio.sleep(wait_time)
                self.tokens = 1
            
            self.tokens -= 1


class BaseProvider(ABC):
    """
    Abstract base class for all data providers.
    Provides common functionality for HTTP requests, rate limiting, and error handling.
    """
    
    def __init__(self, api_key: Optional[str] = None, **kwargs):
        self.api_key = api_key
        self.base_url = kwargs.get("base_url", "")
        self.timeout = kwargs.get("timeout", 30)
        self.rate_limiter = RateLimiter(kwargs.get("rate_limit", 60))
        self._session = None  # aiohttp.ClientSession, created lazily
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Provider name"""
        pass
    
    @property
    @abstractmethod
    def supported_data_types(self) -> List[DataType]:
        """List of data types this provider supports"""
        pass
    
    async def _get_session(self):
        """Get or create aiohttp session"""
        if not HAS_AIOHTTP:
            raise ImportError("aiohttp is required for async HTTP requests. Install with: pip install aiohttp")
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=self.timeout)
            self._session = aiohttp.ClientSession(timeout=timeout)
        return self._session
    
    async def close(self):
        """Close the session"""
        if self._session and not self._session.closed:
            await self._session.close()
    
    async def _request(
        self, 
        method: str,
        endpoint: str,
        params: Optional[Dict] = None,
        headers: Optional[Dict] = None,
        json_data: Optional[Dict] = None
    ) -> Dict:
        """Make an HTTP request with rate limiting and error handling"""
        await self.rate_limiter.acquire()
        
        session = await self._get_session()
        url = f"{self.base_url}{endpoint}" if not endpoint.startswith("http") else endpoint
        
        default_headers = self._get_default_headers()
        if headers:
            default_headers.update(headers)
        
        start_time = time.time()
        
        try:
            async with session.request(
                method,
                url,
                params=params,
                headers=default_headers,
                json=json_data
            ) as response:
                latency = int((time.time() - start_time) * 1000)
                
                if response.status == 429:
                    # Rate limited
                    retry_after = int(response.headers.get("Retry-After", 60))
                    await asyncio.sleep(retry_after)
                    return await self._request(method, endpoint, params, headers, json_data)
                
                response.raise_for_status()
                data = await response.json()
                
                return {
                    "data": data,
                    "latency_ms": latency,
                    "rate_limit_remaining": response.headers.get("X-RateLimit-Remaining"),
                }
                
        except (aiohttp.ClientError if HAS_AIOHTTP else Exception) as e:
            return {
                "error": str(e),
                "latency_ms": int((time.time() - start_time) * 1000)
            }
    
    def _get_default_headers(self) -> Dict[str, str]:
        """Get default headers for requests"""
        return {
            "User-Agent": "InstitutionalResearchPlatform/1.0",
            "Accept": "application/json"
        }
    
    # Abstract methods that providers must implement
    
    @abstractmethod
    async def get_quote(self, symbol: str) -> ProviderResponse:
        """Get real-time quote for a symbol"""
        pass
    
    @abstractmethod
    async def get_historical(
        self, 
        symbol: str, 
        start_date: date,
        end_date: date,
        timeframe: str = "1d"
    ) -> ProviderResponse:
        """Get historical OHLCV data"""
        pass
    
    # Optional methods with default implementations
    
    async def get_options_chain(
        self, 
        symbol: str,
        expiration: Optional[date] = None
    ) -> ProviderResponse:
        """Get options chain - not all providers support this"""
        return ProviderResponse(
            success=False,
            data=None,
            data_type=DataType.OPTIONS,
            provider=self.name,
            error="Options chain not supported by this provider"
        )
    
    async def get_news(
        self, 
        symbol: Optional[str] = None,
        limit: int = 50
    ) -> ProviderResponse:
        """Get news articles"""
        return ProviderResponse(
            success=False,
            data=None,
            data_type=DataType.NEWS,
            provider=self.name,
            error="News not supported by this provider"
        )
    
    async def get_company_profile(self, symbol: str) -> ProviderResponse:
        """Get company profile"""
        return ProviderResponse(
            success=False,
            data=None,
            data_type=DataType.PROFILE,
            provider=self.name,
            error="Company profile not supported by this provider"
        )
    
    async def get_fundamentals(
        self, 
        symbol: str,
        statement_type: str = "income"
    ) -> ProviderResponse:
        """Get fundamental data (income statement, balance sheet, cash flow)"""
        return ProviderResponse(
            success=False,
            data=None,
            data_type=DataType.FUNDAMENTALS,
            provider=self.name,
            error="Fundamentals not supported by this provider"
        )
    
    async def get_institutional_holdings(self, symbol: str) -> ProviderResponse:
        """Get institutional holdings"""
        return ProviderResponse(
            success=False,
            data=None,
            data_type=DataType.HOLDINGS,
            provider=self.name,
            error="Institutional holdings not supported by this provider"
        )
    
    async def get_insider_transactions(
        self, 
        symbol: str,
        limit: int = 100
    ) -> ProviderResponse:
        """Get insider transactions"""
        return ProviderResponse(
            success=False,
            data=None,
            data_type=DataType.TRANSACTIONS,
            provider=self.name,
            error="Insider transactions not supported by this provider"
        )
    
    async def get_filings(
        self, 
        symbol: str,
        form_types: Optional[List[str]] = None,
        limit: int = 10
    ) -> ProviderResponse:
        """Get SEC filings"""
        return ProviderResponse(
            success=False,
            data=None,
            data_type=DataType.FILING,
            provider=self.name,
            error="Filings not supported by this provider"
        )
    
    async def health_check(self) -> bool:
        """Check if provider is healthy and accessible"""
        try:
            # Try a simple request
            response = await self.get_quote("AAPL")
            return response.success
        except Exception:
            return False
