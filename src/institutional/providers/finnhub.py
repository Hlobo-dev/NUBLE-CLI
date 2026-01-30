"""
Finnhub Provider - Real-time data, sentiment, and alternative data.
Best for: News sentiment, insider transactions, supply chain, social metrics.
"""

from typing import Dict, List, Optional, Any
from datetime import datetime, date, timedelta
from dataclasses import dataclass

from .base import (
    BaseProvider, ProviderResponse, DataType,
    Quote, OHLCV, NewsArticle, InsiderTransaction, 
    InstitutionalHolding, CompanyProfile
)


@dataclass
class SentimentData:
    """News and social sentiment data"""
    symbol: str
    timestamp: datetime
    buzz_score: float
    news_score: float
    sentiment_score: float  # -1 to 1
    bullish_percent: float
    bearish_percent: float
    articles_in_week: int
    
    
@dataclass
class EarningsSurprise:
    """Earnings surprise data"""
    symbol: str
    period: date
    actual: float
    estimate: float
    surprise: float
    surprise_percent: float


@dataclass
class SupplyChainRelation:
    """Supply chain relationship"""
    symbol: str
    related_symbol: str
    relationship: str  # supplier, customer, partner
    


class FinnhubProvider(BaseProvider):
    """
    Finnhub data provider.
    
    Features:
    - Real-time US stock prices
    - Company news with sentiment
    - Insider transactions (Form 4)
    - Institutional holdings (13F)
    - Supply chain data
    - Social sentiment
    - Earnings calendar and surprises
    - Recommendation trends
    - Price targets
    
    Rate limits:
    - Free: 60 calls/minute
    - Premium: Higher based on plan
    """
    
    BASE_URL = "https://finnhub.io/api/v1"
    
    def __init__(self, api_key: str, **kwargs):
        super().__init__(
            api_key=api_key,
            base_url=self.BASE_URL,
            rate_limit=60,  # 60 calls per minute for free tier
            **kwargs
        )
    
    @property
    def name(self) -> str:
        return "finnhub"
    
    @property
    def supported_data_types(self) -> List[DataType]:
        return [
            DataType.QUOTE,
            DataType.OHLCV,
            DataType.NEWS,
            DataType.SENTIMENT,
            DataType.TRANSACTIONS,
            DataType.HOLDINGS,
            DataType.EARNINGS,
            DataType.PROFILE,
        ]
    
    def _get_default_headers(self) -> Dict[str, str]:
        return {
            "X-Finnhub-Token": self.api_key,
        }
    
    async def get_quote(self, symbol: str) -> ProviderResponse:
        """Get real-time quote"""
        try:
            result = await self._request(
                "GET",
                "/quote",
                params={"symbol": symbol}
            )
            
            if "error" in result:
                return ProviderResponse(
                    success=False,
                    data=None,
                    data_type=DataType.QUOTE,
                    provider=self.name,
                    symbol=symbol,
                    error=result["error"]
                )
            
            data = result["data"]
            
            # Check if valid data returned
            if data.get("c", 0) == 0 and data.get("pc", 0) == 0:
                return ProviderResponse(
                    success=False,
                    data=None,
                    data_type=DataType.QUOTE,
                    provider=self.name,
                    symbol=symbol,
                    error="No quote data available"
                )
            
            quote = Quote(
                symbol=symbol,
                price=data.get("c", 0),  # Current price
                bid=None,
                ask=None,
                bid_size=None,
                ask_size=None,
                volume=None,  # Volume from separate endpoint
                timestamp=datetime.fromtimestamp(data.get("t", 0)),
                change=data.get("d", 0),  # Change
                change_percent=data.get("dp", 0),  # Change percent
                high=data.get("h", 0),  # Day high
                low=data.get("l", 0),  # Day low
                open=data.get("o", 0),  # Open
                previous_close=data.get("pc", 0),  # Previous close
            )
            
            return ProviderResponse(
                success=True,
                data=quote,
                data_type=DataType.QUOTE,
                provider=self.name,
                symbol=symbol,
                latency_ms=result.get("latency_ms", 0)
            )
            
        except Exception as e:
            return ProviderResponse(
                success=False,
                data=None,
                data_type=DataType.QUOTE,
                provider=self.name,
                symbol=symbol,
                error=str(e)
            )
    
    async def get_historical(
        self,
        symbol: str,
        start_date: date,
        end_date: date,
        timeframe: str = "D"
    ) -> ProviderResponse:
        """Get historical candle data"""
        try:
            # Map timeframe to Finnhub resolution
            resolution_map = {
                "1m": "1", "5m": "5", "15m": "15", "30m": "30",
                "1h": "60", "1d": "D", "D": "D",
                "1w": "W", "W": "W",
                "1M": "M", "M": "M"
            }
            resolution = resolution_map.get(timeframe, "D")
            
            # Convert dates to Unix timestamps
            start_ts = int(datetime.combine(start_date, datetime.min.time()).timestamp())
            end_ts = int(datetime.combine(end_date, datetime.max.time()).timestamp())
            
            result = await self._request(
                "GET",
                "/stock/candle",
                params={
                    "symbol": symbol,
                    "resolution": resolution,
                    "from": start_ts,
                    "to": end_ts
                }
            )
            
            if "error" in result:
                return ProviderResponse(
                    success=False,
                    data=None,
                    data_type=DataType.OHLCV,
                    provider=self.name,
                    symbol=symbol,
                    error=result["error"]
                )
            
            data = result["data"]
            
            if data.get("s") == "no_data":
                return ProviderResponse(
                    success=False,
                    data=None,
                    data_type=DataType.OHLCV,
                    provider=self.name,
                    symbol=symbol,
                    error="No historical data available"
                )
            
            candles = []
            timestamps = data.get("t", [])
            opens = data.get("o", [])
            highs = data.get("h", [])
            lows = data.get("l", [])
            closes = data.get("c", [])
            volumes = data.get("v", [])
            
            for i in range(len(timestamps)):
                candles.append(OHLCV(
                    symbol=symbol,
                    timestamp=datetime.fromtimestamp(timestamps[i]),
                    open=opens[i],
                    high=highs[i],
                    low=lows[i],
                    close=closes[i],
                    volume=int(volumes[i]),
                ))
            
            return ProviderResponse(
                success=True,
                data=candles,
                data_type=DataType.OHLCV,
                provider=self.name,
                symbol=symbol,
                latency_ms=result.get("latency_ms", 0)
            )
            
        except Exception as e:
            return ProviderResponse(
                success=False,
                data=None,
                data_type=DataType.OHLCV,
                provider=self.name,
                symbol=symbol,
                error=str(e)
            )
    
    async def get_news(
        self,
        symbol: str,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None
    ) -> ProviderResponse:
        """Get company news"""
        try:
            if not start_date:
                start_date = date.today() - timedelta(days=7)
            if not end_date:
                end_date = date.today()
            
            result = await self._request(
                "GET",
                "/company-news",
                params={
                    "symbol": symbol,
                    "from": start_date.isoformat(),
                    "to": end_date.isoformat()
                }
            )
            
            if "error" in result:
                return ProviderResponse(
                    success=False,
                    data=None,
                    data_type=DataType.NEWS,
                    provider=self.name,
                    symbol=symbol,
                    error=result["error"]
                )
            
            articles = []
            for item in result["data"]:
                articles.append(NewsArticle(
                    title=item.get("headline", ""),
                    url=item.get("url", ""),
                    source=item.get("source", ""),
                    published_at=datetime.fromtimestamp(item.get("datetime", 0)),
                    summary=item.get("summary", ""),
                    symbols=[symbol],
                    image_url=item.get("image"),
                    sentiment_score=None,  # Calculated separately
                ))
            
            return ProviderResponse(
                success=True,
                data=articles,
                data_type=DataType.NEWS,
                provider=self.name,
                symbol=symbol,
                latency_ms=result.get("latency_ms", 0)
            )
            
        except Exception as e:
            return ProviderResponse(
                success=False,
                data=None,
                data_type=DataType.NEWS,
                provider=self.name,
                symbol=symbol,
                error=str(e)
            )
    
    async def get_sentiment(self, symbol: str) -> ProviderResponse:
        """Get news sentiment and social sentiment"""
        try:
            result = await self._request(
                "GET",
                "/news-sentiment",
                params={"symbol": symbol}
            )
            
            if "error" in result:
                return ProviderResponse(
                    success=False,
                    data=None,
                    data_type=DataType.SENTIMENT,
                    provider=self.name,
                    symbol=symbol,
                    error=result["error"]
                )
            
            data = result["data"]
            
            sentiment = SentimentData(
                symbol=symbol,
                timestamp=datetime.now(),
                buzz_score=data.get("buzz", {}).get("buzz", 0),
                news_score=data.get("companyNewsScore", 0),
                sentiment_score=data.get("sentiment", {}).get("bearishPercent", 0) * -1 + 
                               data.get("sentiment", {}).get("bullishPercent", 0),
                bullish_percent=data.get("sentiment", {}).get("bullishPercent", 0),
                bearish_percent=data.get("sentiment", {}).get("bearishPercent", 0),
                articles_in_week=data.get("buzz", {}).get("articlesInLastWeek", 0),
            )
            
            return ProviderResponse(
                success=True,
                data=sentiment,
                data_type=DataType.SENTIMENT,
                provider=self.name,
                symbol=symbol,
                latency_ms=result.get("latency_ms", 0)
            )
            
        except Exception as e:
            return ProviderResponse(
                success=False,
                data=None,
                data_type=DataType.SENTIMENT,
                provider=self.name,
                symbol=symbol,
                error=str(e)
            )
    
    async def get_insider_transactions(
        self,
        symbol: str,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None
    ) -> ProviderResponse:
        """Get insider transactions (Form 4)"""
        try:
            if not start_date:
                start_date = date.today() - timedelta(days=365)
            if not end_date:
                end_date = date.today()
            
            result = await self._request(
                "GET",
                "/stock/insider-transactions",
                params={
                    "symbol": symbol,
                    "from": start_date.isoformat(),
                    "to": end_date.isoformat()
                }
            )
            
            if "error" in result:
                return ProviderResponse(
                    success=False,
                    data=None,
                    data_type=DataType.TRANSACTIONS,
                    provider=self.name,
                    symbol=symbol,
                    error=result["error"]
                )
            
            transactions = []
            for item in result["data"].get("data", []):
                transactions.append(InsiderTransaction(
                    symbol=symbol,
                    insider_name=item.get("name", ""),
                    insider_title=item.get("position", ""),
                    transaction_type=item.get("transactionCode", ""),
                    shares=int(item.get("change", 0)),
                    price=item.get("transactionPrice"),
                    value=item.get("transactionPrice", 0) * abs(item.get("change", 0)),
                    transaction_date=datetime.strptime(
                        item.get("transactionDate", "2000-01-01"), 
                        "%Y-%m-%d"
                    ).date(),
                    filing_date=datetime.strptime(
                        item.get("filingDate", "2000-01-01"),
                        "%Y-%m-%d"
                    ).date(),
                ))
            
            return ProviderResponse(
                success=True,
                data=transactions,
                data_type=DataType.TRANSACTIONS,
                provider=self.name,
                symbol=symbol,
                latency_ms=result.get("latency_ms", 0)
            )
            
        except Exception as e:
            return ProviderResponse(
                success=False,
                data=None,
                data_type=DataType.TRANSACTIONS,
                provider=self.name,
                symbol=symbol,
                error=str(e)
            )
    
    async def get_institutional_holdings(self, symbol: str) -> ProviderResponse:
        """Get institutional ownership (13F holders)"""
        try:
            result = await self._request(
                "GET",
                "/institutional-ownership",
                params={"symbol": symbol}
            )
            
            if "error" in result:
                return ProviderResponse(
                    success=False,
                    data=None,
                    data_type=DataType.HOLDINGS,
                    provider=self.name,
                    symbol=symbol,
                    error=result["error"]
                )
            
            holdings = []
            for item in result["data"].get("data", []):
                holdings.append(InstitutionalHolding(
                    symbol=symbol,
                    holder_name=item.get("name", ""),
                    shares=int(item.get("share", 0)),
                    value=item.get("value"),
                    percent_of_portfolio=item.get("percentage"),
                    change=item.get("change"),
                    filing_date=datetime.strptime(
                        item.get("filingDate", "2000-01-01"),
                        "%Y-%m-%d"
                    ).date() if item.get("filingDate") else None,
                ))
            
            return ProviderResponse(
                success=True,
                data=holdings,
                data_type=DataType.HOLDINGS,
                provider=self.name,
                symbol=symbol,
                latency_ms=result.get("latency_ms", 0)
            )
            
        except Exception as e:
            return ProviderResponse(
                success=False,
                data=None,
                data_type=DataType.HOLDINGS,
                provider=self.name,
                symbol=symbol,
                error=str(e)
            )
    
    async def get_recommendation_trends(self, symbol: str) -> ProviderResponse:
        """Get analyst recommendation trends"""
        try:
            result = await self._request(
                "GET",
                "/stock/recommendation",
                params={"symbol": symbol}
            )
            
            if "error" in result:
                return ProviderResponse(
                    success=False,
                    data=None,
                    data_type=DataType.FUNDAMENTALS,
                    provider=self.name,
                    symbol=symbol,
                    error=result["error"]
                )
            
            return ProviderResponse(
                success=True,
                data=result["data"],
                data_type=DataType.FUNDAMENTALS,
                provider=self.name,
                symbol=symbol,
                latency_ms=result.get("latency_ms", 0)
            )
            
        except Exception as e:
            return ProviderResponse(
                success=False,
                data=None,
                data_type=DataType.FUNDAMENTALS,
                provider=self.name,
                symbol=symbol,
                error=str(e)
            )
    
    async def get_price_target(self, symbol: str) -> ProviderResponse:
        """Get analyst price targets"""
        try:
            result = await self._request(
                "GET",
                "/stock/price-target",
                params={"symbol": symbol}
            )
            
            if "error" in result:
                return ProviderResponse(
                    success=False,
                    data=None,
                    data_type=DataType.FUNDAMENTALS,
                    provider=self.name,
                    symbol=symbol,
                    error=result["error"]
                )
            
            return ProviderResponse(
                success=True,
                data=result["data"],
                data_type=DataType.FUNDAMENTALS,
                provider=self.name,
                symbol=symbol,
                latency_ms=result.get("latency_ms", 0)
            )
            
        except Exception as e:
            return ProviderResponse(
                success=False,
                data=None,
                data_type=DataType.FUNDAMENTALS,
                provider=self.name,
                symbol=symbol,
                error=str(e)
            )
    
    async def get_earnings_surprises(self, symbol: str) -> ProviderResponse:
        """Get historical earnings surprises"""
        try:
            result = await self._request(
                "GET",
                "/stock/earnings",
                params={"symbol": symbol}
            )
            
            if "error" in result:
                return ProviderResponse(
                    success=False,
                    data=None,
                    data_type=DataType.EARNINGS,
                    provider=self.name,
                    symbol=symbol,
                    error=result["error"]
                )
            
            surprises = []
            for item in result["data"]:
                surprises.append(EarningsSurprise(
                    symbol=symbol,
                    period=datetime.strptime(item.get("period", "2000-01-01"), "%Y-%m-%d").date(),
                    actual=item.get("actual", 0),
                    estimate=item.get("estimate", 0),
                    surprise=item.get("surprise", 0),
                    surprise_percent=item.get("surprisePercent", 0),
                ))
            
            return ProviderResponse(
                success=True,
                data=surprises,
                data_type=DataType.EARNINGS,
                provider=self.name,
                symbol=symbol,
                latency_ms=result.get("latency_ms", 0)
            )
            
        except Exception as e:
            return ProviderResponse(
                success=False,
                data=None,
                data_type=DataType.EARNINGS,
                provider=self.name,
                symbol=symbol,
                error=str(e)
            )
    
    async def get_supply_chain(self, symbol: str) -> ProviderResponse:
        """Get supply chain relationships"""
        try:
            # Get both suppliers and customers
            suppliers_result = await self._request(
                "GET",
                "/stock/supply-chain",
                params={"symbol": symbol}
            )
            
            if "error" in suppliers_result:
                return ProviderResponse(
                    success=False,
                    data=None,
                    data_type=DataType.FUNDAMENTALS,
                    provider=self.name,
                    symbol=symbol,
                    error=suppliers_result["error"]
                )
            
            relations = []
            for item in suppliers_result["data"].get("data", []):
                relations.append(SupplyChainRelation(
                    symbol=symbol,
                    related_symbol=item.get("symbol", ""),
                    relationship=item.get("relationship", "related"),
                ))
            
            return ProviderResponse(
                success=True,
                data=relations,
                data_type=DataType.FUNDAMENTALS,
                provider=self.name,
                symbol=symbol,
                latency_ms=suppliers_result.get("latency_ms", 0)
            )
            
        except Exception as e:
            return ProviderResponse(
                success=False,
                data=None,
                data_type=DataType.FUNDAMENTALS,
                provider=self.name,
                symbol=symbol,
                error=str(e)
            )
    
    async def get_company_profile(self, symbol: str) -> ProviderResponse:
        """Get company profile"""
        try:
            result = await self._request(
                "GET",
                "/stock/profile2",
                params={"symbol": symbol}
            )
            
            if "error" in result:
                return ProviderResponse(
                    success=False,
                    data=None,
                    data_type=DataType.PROFILE,
                    provider=self.name,
                    symbol=symbol,
                    error=result["error"]
                )
            
            data = result["data"]
            
            if not data:
                return ProviderResponse(
                    success=False,
                    data=None,
                    data_type=DataType.PROFILE,
                    provider=self.name,
                    symbol=symbol,
                    error="No profile data found"
                )
            
            profile = CompanyProfile(
                symbol=symbol,
                name=data.get("name"),
                exchange=data.get("exchange"),
                sector=data.get("finnhubIndustry"),
                industry=data.get("finnhubIndustry"),
                market_cap=data.get("marketCapitalization"),
                shares_outstanding=data.get("shareOutstanding"),
                logo_url=data.get("logo"),
                website=data.get("weburl"),
                country=data.get("country"),
                currency=data.get("currency"),
                ipo_date=data.get("ipo"),
            )
            
            return ProviderResponse(
                success=True,
                data=profile,
                data_type=DataType.PROFILE,
                provider=self.name,
                symbol=symbol,
                latency_ms=result.get("latency_ms", 0)
            )
            
        except Exception as e:
            return ProviderResponse(
                success=False,
                data=None,
                data_type=DataType.PROFILE,
                provider=self.name,
                symbol=symbol,
                error=str(e)
            )
    
    async def get_social_sentiment(
        self,
        symbol: str,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None
    ) -> ProviderResponse:
        """Get social media sentiment (Reddit, Twitter)"""
        try:
            if not start_date:
                start_date = date.today() - timedelta(days=7)
            if not end_date:
                end_date = date.today()
            
            result = await self._request(
                "GET",
                "/stock/social-sentiment",
                params={
                    "symbol": symbol,
                    "from": start_date.isoformat(),
                    "to": end_date.isoformat()
                }
            )
            
            if "error" in result:
                return ProviderResponse(
                    success=False,
                    data=None,
                    data_type=DataType.SENTIMENT,
                    provider=self.name,
                    symbol=symbol,
                    error=result["error"]
                )
            
            return ProviderResponse(
                success=True,
                data=result["data"],
                data_type=DataType.SENTIMENT,
                provider=self.name,
                symbol=symbol,
                latency_ms=result.get("latency_ms", 0)
            )
            
        except Exception as e:
            return ProviderResponse(
                success=False,
                data=None,
                data_type=DataType.SENTIMENT,
                provider=self.name,
                symbol=symbol,
                error=str(e)
            )
    
    async def get_earnings_calendar(
        self,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
        symbol: Optional[str] = None
    ) -> ProviderResponse:
        """Get earnings calendar"""
        try:
            if not start_date:
                start_date = date.today()
            if not end_date:
                end_date = start_date + timedelta(days=14)
            
            params = {
                "from": start_date.isoformat(),
                "to": end_date.isoformat()
            }
            
            if symbol:
                params["symbol"] = symbol
            
            result = await self._request(
                "GET",
                "/calendar/earnings",
                params=params
            )
            
            if "error" in result:
                return ProviderResponse(
                    success=False,
                    data=None,
                    data_type=DataType.EARNINGS,
                    provider=self.name,
                    symbol=symbol or "CALENDAR",
                    error=result["error"]
                )
            
            return ProviderResponse(
                success=True,
                data=result["data"],
                data_type=DataType.EARNINGS,
                provider=self.name,
                symbol=symbol or "CALENDAR",
                latency_ms=result.get("latency_ms", 0)
            )
            
        except Exception as e:
            return ProviderResponse(
                success=False,
                data=None,
                data_type=DataType.EARNINGS,
                provider=self.name,
                symbol=symbol or "CALENDAR",
                error=str(e)
            )
