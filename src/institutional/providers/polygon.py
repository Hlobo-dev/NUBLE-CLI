"""
Polygon.io Provider - Premium real-time and historical market data.
Supports stocks, options, forex, and crypto.
"""

from typing import Dict, List, Optional, Any
from datetime import datetime, date, timedelta
import asyncio

from .base import (
    BaseProvider, ProviderResponse, DataType,
    Quote, OHLCV, OptionsContract, NewsArticle, CompanyProfile
)


class PolygonProvider(BaseProvider):
    """
    Polygon.io data provider.
    
    Premium features:
    - Real-time quotes (sub-second latency)
    - Full options chain with Greeks
    - Tick-level historical data
    - WebSocket streaming
    """
    
    BASE_URL = "https://api.polygon.io"
    
    def __init__(self, api_key: str, **kwargs):
        super().__init__(
            api_key=api_key,
            base_url=self.BASE_URL,
            rate_limit=kwargs.get("rate_limit", 100),
            **kwargs
        )
    
    @property
    def name(self) -> str:
        return "polygon"
    
    @property
    def supported_data_types(self) -> List[DataType]:
        return [
            DataType.QUOTE,
            DataType.OHLCV,
            DataType.TICK,
            DataType.OPTIONS,
            DataType.NEWS,
            DataType.PROFILE,
        ]
    
    def _get_default_headers(self) -> Dict[str, str]:
        headers = super()._get_default_headers()
        headers["Authorization"] = f"Bearer {self.api_key}"
        return headers
    
    async def get_quote(self, symbol: str) -> ProviderResponse:
        """Get real-time quote from Polygon"""
        try:
            # Get last trade
            trade_result = await self._request(
                "GET",
                f"/v2/last/trade/{symbol}"
            )
            
            # Get last quote (bid/ask)
            quote_result = await self._request(
                "GET",
                f"/v2/last/nbbo/{symbol}"
            )
            
            # Get previous close for change calculation
            prev_result = await self._request(
                "GET",
                f"/v2/aggs/ticker/{symbol}/prev"
            )
            
            if "error" in trade_result:
                return ProviderResponse(
                    success=False,
                    data=None,
                    data_type=DataType.QUOTE,
                    provider=self.name,
                    symbol=symbol,
                    error=trade_result["error"]
                )
            
            trade_data = trade_result["data"].get("results", {})
            quote_data = quote_result.get("data", {}).get("results", {})
            prev_data = prev_result.get("data", {}).get("results", [{}])[0] if prev_result.get("data") else {}
            
            price = trade_data.get("p", 0)
            prev_close = prev_data.get("c", price)
            change = price - prev_close
            change_percent = (change / prev_close * 100) if prev_close else 0
            
            quote = Quote(
                symbol=symbol,
                price=price,
                bid=quote_data.get("p"),
                ask=quote_data.get("P"),
                bid_size=quote_data.get("s"),
                ask_size=quote_data.get("S"),
                volume=prev_data.get("v"),
                timestamp=datetime.fromtimestamp(trade_data.get("t", 0) / 1e9) if trade_data.get("t") else None,
                change=change,
                change_percent=change_percent,
                high=prev_data.get("h"),
                low=prev_data.get("l"),
                open=prev_data.get("o"),
                prev_close=prev_close,
            )
            
            return ProviderResponse(
                success=True,
                data=quote,
                data_type=DataType.QUOTE,
                provider=self.name,
                symbol=symbol,
                latency_ms=trade_result.get("latency_ms", 0)
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
        timeframe: str = "1d"
    ) -> ProviderResponse:
        """Get historical OHLCV data"""
        try:
            # Map timeframe to Polygon format
            tf_map = {
                "1m": ("minute", 1),
                "5m": ("minute", 5),
                "15m": ("minute", 15),
                "30m": ("minute", 30),
                "1h": ("hour", 1),
                "4h": ("hour", 4),
                "1d": ("day", 1),
                "1w": ("week", 1),
                "1M": ("month", 1),
            }
            
            multiplier_unit = tf_map.get(timeframe, ("day", 1))
            
            result = await self._request(
                "GET",
                f"/v2/aggs/ticker/{symbol}/range/{multiplier_unit[1]}/{multiplier_unit[0]}/{start_date}/{end_date}",
                params={
                    "adjusted": "true",
                    "sort": "asc",
                    "limit": 50000
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
            
            bars = []
            for bar in result["data"].get("results", []):
                bars.append(OHLCV(
                    symbol=symbol,
                    timestamp=datetime.fromtimestamp(bar["t"] / 1000),
                    open=bar["o"],
                    high=bar["h"],
                    low=bar["l"],
                    close=bar["c"],
                    volume=bar["v"],
                    vwap=bar.get("vw"),
                    trades=bar.get("n")
                ))
            
            return ProviderResponse(
                success=True,
                data=bars,
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
    
    async def get_options_chain(
        self,
        symbol: str,
        expiration: Optional[date] = None
    ) -> ProviderResponse:
        """Get full options chain with Greeks"""
        try:
            params = {
                "underlying_ticker": symbol,
                "limit": 250,
                "order": "asc",
                "sort": "expiration_date"
            }
            
            if expiration:
                params["expiration_date"] = expiration.isoformat()
            
            result = await self._request(
                "GET",
                "/v3/snapshot/options/" + symbol,
                params=params
            )
            
            if "error" in result:
                return ProviderResponse(
                    success=False,
                    data=None,
                    data_type=DataType.OPTIONS,
                    provider=self.name,
                    symbol=symbol,
                    error=result["error"]
                )
            
            contracts = []
            for item in result["data"].get("results", []):
                details = item.get("details", {})
                greeks = item.get("greeks", {})
                day = item.get("day", {})
                
                contracts.append(OptionsContract(
                    symbol=details.get("ticker", ""),
                    underlying=symbol,
                    contract_type=details.get("contract_type", "").lower(),
                    strike=details.get("strike_price", 0),
                    expiration=datetime.strptime(
                        details.get("expiration_date", "2099-01-01"),
                        "%Y-%m-%d"
                    ).date(),
                    bid=item.get("last_quote", {}).get("bid", 0),
                    ask=item.get("last_quote", {}).get("ask", 0),
                    last=item.get("last_trade", {}).get("price", 0),
                    volume=day.get("volume", 0),
                    open_interest=item.get("open_interest", 0),
                    implied_volatility=item.get("implied_volatility"),
                    delta=greeks.get("delta"),
                    gamma=greeks.get("gamma"),
                    theta=greeks.get("theta"),
                    vega=greeks.get("vega"),
                ))
            
            return ProviderResponse(
                success=True,
                data=contracts,
                data_type=DataType.OPTIONS,
                provider=self.name,
                symbol=symbol,
                latency_ms=result.get("latency_ms", 0)
            )
            
        except Exception as e:
            return ProviderResponse(
                success=False,
                data=None,
                data_type=DataType.OPTIONS,
                provider=self.name,
                symbol=symbol,
                error=str(e)
            )
    
    async def get_news(
        self,
        symbol: Optional[str] = None,
        limit: int = 50
    ) -> ProviderResponse:
        """Get news articles"""
        try:
            params = {
                "limit": min(limit, 1000),
                "order": "desc",
                "sort": "published_utc"
            }
            
            if symbol:
                params["ticker"] = symbol
            
            result = await self._request(
                "GET",
                "/v2/reference/news",
                params=params
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
            for item in result["data"].get("results", []):
                articles.append(NewsArticle(
                    id=item.get("id", ""),
                    title=item.get("title", ""),
                    summary=item.get("description"),
                    url=item.get("article_url"),
                    source=item.get("publisher", {}).get("name"),
                    author=item.get("author"),
                    published_at=datetime.fromisoformat(
                        item.get("published_utc", "").replace("Z", "+00:00")
                    ) if item.get("published_utc") else None,
                    symbols=item.get("tickers", []),
                    keywords=item.get("keywords", [])
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
    
    async def get_company_profile(self, symbol: str) -> ProviderResponse:
        """Get company details"""
        try:
            result = await self._request(
                "GET",
                f"/v3/reference/tickers/{symbol}"
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
            
            data = result["data"].get("results", {})
            
            profile = CompanyProfile(
                symbol=symbol,
                name=data.get("name", ""),
                description=data.get("description"),
                sector=data.get("sector"),
                industry=data.get("sic_description"),
                exchange=data.get("primary_exchange"),
                market_cap=data.get("market_cap"),
                employees=data.get("total_employees"),
                website=data.get("homepage_url"),
                headquarters=f"{data.get('address', {}).get('city', '')}, {data.get('address', {}).get('state', '')}",
                cik=data.get("cik"),
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
    
    async def get_dividends(
        self,
        symbol: str,
        limit: int = 50
    ) -> ProviderResponse:
        """Get dividend history"""
        try:
            result = await self._request(
                "GET",
                "/v3/reference/dividends",
                params={
                    "ticker": symbol,
                    "limit": limit,
                    "order": "desc"
                }
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
            
            dividends = result["data"].get("results", [])
            
            return ProviderResponse(
                success=True,
                data=dividends,
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
    
    async def get_unusual_options_activity(
        self,
        symbol: Optional[str] = None,
        min_volume: int = 1000
    ) -> ProviderResponse:
        """
        Get unusual options activity by analyzing volume spikes.
        This is a computed metric based on current vs average volume.
        """
        try:
            # Get current options snapshot
            if symbol:
                result = await self._request(
                    "GET",
                    f"/v3/snapshot/options/{symbol}"
                )
            else:
                # Get market-wide options activity
                result = await self._request(
                    "GET",
                    "/v3/snapshot/options/tickers"
                )
            
            if "error" in result:
                return ProviderResponse(
                    success=False,
                    data=None,
                    data_type=DataType.OPTIONS,
                    provider=self.name,
                    symbol=symbol,
                    error=result["error"]
                )
            
            unusual = []
            for item in result["data"].get("results", []):
                day = item.get("day", {})
                volume = day.get("volume", 0)
                open_interest = item.get("open_interest", 0)
                
                # Flag as unusual if volume > open interest (indicates new interest)
                # or if volume is exceptionally high
                if volume > min_volume and (volume > open_interest or volume > 10000):
                    unusual.append({
                        "contract": item.get("details", {}).get("ticker"),
                        "underlying": item.get("underlying_asset", {}).get("ticker"),
                        "type": item.get("details", {}).get("contract_type"),
                        "strike": item.get("details", {}).get("strike_price"),
                        "expiration": item.get("details", {}).get("expiration_date"),
                        "volume": volume,
                        "open_interest": open_interest,
                        "volume_oi_ratio": volume / open_interest if open_interest > 0 else float('inf'),
                        "implied_volatility": item.get("implied_volatility"),
                    })
            
            # Sort by volume
            unusual.sort(key=lambda x: x["volume"], reverse=True)
            
            return ProviderResponse(
                success=True,
                data=unusual[:100],  # Top 100 unusual
                data_type=DataType.OPTIONS,
                provider=self.name,
                symbol=symbol,
                latency_ms=result.get("latency_ms", 0)
            )
            
        except Exception as e:
            return ProviderResponse(
                success=False,
                data=None,
                data_type=DataType.OPTIONS,
                provider=self.name,
                symbol=symbol,
                error=str(e)
            )
