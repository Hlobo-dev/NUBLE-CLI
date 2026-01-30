"""
Alpha Vantage Provider - Technical indicators and fundamental data.
Best for: Technical analysis, forex, crypto, and fundamental metrics.
"""

from typing import Dict, List, Optional, Any
from datetime import datetime, date, timedelta
from dataclasses import dataclass

from .base import (
    BaseProvider, ProviderResponse, DataType,
    Quote, OHLCV, FinancialStatement, CompanyProfile
)


@dataclass
class TechnicalIndicator:
    """Technical indicator data point"""
    timestamp: datetime
    value: float
    indicator: str
    symbol: str
    metadata: Optional[Dict[str, Any]] = None


class AlphaVantageProvider(BaseProvider):
    """
    Alpha Vantage data provider.
    
    Features:
    - 50+ technical indicators (SMA, EMA, RSI, MACD, Bollinger, etc.)
    - Fundamental data (income, balance sheet, cash flow)
    - Forex and crypto data
    - Economic indicators
    - Earnings data
    
    Rate limits:
    - Free: 5 calls/min, 500/day
    - Premium: Higher limits based on plan
    """
    
    BASE_URL = "https://www.alphavantage.co"
    
    # Timeframe mapping
    INTERVAL_MAP = {
        "1m": "1min",
        "5m": "5min",
        "15m": "15min",
        "30m": "30min",
        "1h": "60min",
        "1d": "daily",
        "1w": "weekly",
        "1M": "monthly"
    }
    
    def __init__(self, api_key: str, **kwargs):
        super().__init__(
            api_key=api_key,
            base_url=self.BASE_URL,
            rate_limit=5,  # 5 calls per minute for free tier
            **kwargs
        )
    
    @property
    def name(self) -> str:
        return "alpha_vantage"
    
    @property
    def supported_data_types(self) -> List[DataType]:
        return [
            DataType.QUOTE,
            DataType.OHLCV,
            DataType.TECHNICAL,
            DataType.FUNDAMENTALS,
            DataType.EARNINGS,
            DataType.PROFILE,
        ]
    
    async def get_quote(self, symbol: str) -> ProviderResponse:
        """Get real-time quote using Global Quote endpoint"""
        try:
            result = await self._request(
                "GET",
                "/query",
                params={
                    "function": "GLOBAL_QUOTE",
                    "symbol": symbol,
                    "apikey": self.api_key
                }
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
            
            data = result["data"].get("Global Quote", {})
            
            if not data:
                return ProviderResponse(
                    success=False,
                    data=None,
                    data_type=DataType.QUOTE,
                    provider=self.name,
                    symbol=symbol,
                    error="No quote data returned"
                )
            
            quote = Quote(
                symbol=symbol,
                price=float(data.get("05. price", 0)),
                bid=None,  # Not provided by Alpha Vantage
                ask=None,
                bid_size=None,
                ask_size=None,
                volume=int(float(data.get("06. volume", 0))),
                timestamp=datetime.now(),
                change=float(data.get("09. change", 0)),
                change_percent=float(data.get("10. change percent", "0%").replace("%", "")),
                high=float(data.get("03. high", 0)),
                low=float(data.get("04. low", 0)),
                open=float(data.get("02. open", 0)),
                previous_close=float(data.get("08. previous close", 0)),
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
        timeframe: str = "1d"
    ) -> ProviderResponse:
        """Get historical OHLCV data"""
        try:
            av_interval = self.INTERVAL_MAP.get(timeframe, "daily")
            
            if av_interval in ["daily", "weekly", "monthly"]:
                function = f"TIME_SERIES_{av_interval.upper()}"
                time_series_key = f"Time Series ({av_interval.capitalize()})"
            else:
                function = "TIME_SERIES_INTRADAY"
                time_series_key = f"Time Series ({av_interval})"
            
            params = {
                "function": function,
                "symbol": symbol,
                "apikey": self.api_key,
                "outputsize": "full"
            }
            
            if function == "TIME_SERIES_INTRADAY":
                params["interval"] = av_interval
            
            result = await self._request("GET", "/query", params=params)
            
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
            time_series = data.get(time_series_key, {})
            
            if not time_series:
                # Try alternate key format
                for key in data.keys():
                    if "Time Series" in key:
                        time_series = data[key]
                        break
            
            candles = []
            for date_str, values in time_series.items():
                try:
                    if ":" in date_str:
                        dt = datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S")
                    else:
                        dt = datetime.strptime(date_str, "%Y-%m-%d")
                    
                    if start_date <= dt.date() <= end_date:
                        candles.append(OHLCV(
                            symbol=symbol,
                            timestamp=dt,
                            open=float(values.get("1. open", 0)),
                            high=float(values.get("2. high", 0)),
                            low=float(values.get("3. low", 0)),
                            close=float(values.get("4. close", 0)),
                            volume=int(float(values.get("5. volume", 0))),
                        ))
                except ValueError:
                    continue
            
            # Sort by timestamp
            candles.sort(key=lambda x: x.timestamp)
            
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
    
    async def get_technical_indicator(
        self,
        symbol: str,
        indicator: str,
        interval: str = "daily",
        time_period: int = 14,
        series_type: str = "close",
        **kwargs
    ) -> ProviderResponse:
        """
        Get technical indicator data.
        
        Supported indicators:
        - SMA, EMA, WMA, DEMA, TEMA, TRIMA, KAMA, MAMA, VWAP, T3
        - RSI, STOCH, STOCHF, STOCHRSI, WILLR, CCI, MOM, ROC
        - MACD, MACDEXT, APO, PPO, AROON, AROONOSC, ADX, ADXR
        - BBANDS, SAR, TRANGE, ATR, NATR, AD, ADOSC, OBV
        """
        try:
            params = {
                "function": indicator.upper(),
                "symbol": symbol,
                "interval": self.INTERVAL_MAP.get(interval, interval),
                "apikey": self.api_key,
            }
            
            # Add indicator-specific parameters
            if indicator.upper() not in ["VWAP", "AD", "OBV", "TRANGE"]:
                params["time_period"] = time_period
            
            if indicator.upper() not in ["VWAP", "AD", "OBV", "TRANGE", "ADX", "ATR"]:
                params["series_type"] = series_type
            
            # Add any additional kwargs
            params.update(kwargs)
            
            result = await self._request("GET", "/query", params=params)
            
            if "error" in result:
                return ProviderResponse(
                    success=False,
                    data=None,
                    data_type=DataType.TECHNICAL,
                    provider=self.name,
                    symbol=symbol,
                    error=result["error"]
                )
            
            data = result["data"]
            
            # Find the time series key
            analysis_key = None
            for key in data.keys():
                if "Technical Analysis" in key or "Meta Data" not in key:
                    if key != "Meta Data":
                        analysis_key = key
                        break
            
            if not analysis_key or analysis_key not in data:
                return ProviderResponse(
                    success=False,
                    data=None,
                    data_type=DataType.TECHNICAL,
                    provider=self.name,
                    symbol=symbol,
                    error=f"No technical analysis data found for {indicator}"
                )
            
            indicators = []
            for date_str, values in data[analysis_key].items():
                try:
                    if ":" in date_str:
                        dt = datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S")
                    else:
                        dt = datetime.strptime(date_str, "%Y-%m-%d")
                    
                    # Get the primary value (first key in values)
                    value_key = list(values.keys())[0]
                    
                    indicators.append(TechnicalIndicator(
                        timestamp=dt,
                        value=float(values[value_key]),
                        indicator=indicator.upper(),
                        symbol=symbol,
                        metadata=values if len(values) > 1 else None
                    ))
                except (ValueError, KeyError):
                    continue
            
            # Sort by timestamp
            indicators.sort(key=lambda x: x.timestamp)
            
            return ProviderResponse(
                success=True,
                data=indicators,
                data_type=DataType.TECHNICAL,
                provider=self.name,
                symbol=symbol,
                latency_ms=result.get("latency_ms", 0)
            )
            
        except Exception as e:
            return ProviderResponse(
                success=False,
                data=None,
                data_type=DataType.TECHNICAL,
                provider=self.name,
                symbol=symbol,
                error=str(e)
            )
    
    async def get_rsi(
        self,
        symbol: str,
        interval: str = "daily",
        time_period: int = 14
    ) -> ProviderResponse:
        """Get RSI indicator"""
        return await self.get_technical_indicator(
            symbol=symbol,
            indicator="RSI",
            interval=interval,
            time_period=time_period
        )
    
    async def get_macd(
        self,
        symbol: str,
        interval: str = "daily",
        fast_period: int = 12,
        slow_period: int = 26,
        signal_period: int = 9
    ) -> ProviderResponse:
        """Get MACD indicator"""
        return await self.get_technical_indicator(
            symbol=symbol,
            indicator="MACD",
            interval=interval,
            fastperiod=fast_period,
            slowperiod=slow_period,
            signalperiod=signal_period
        )
    
    async def get_bollinger_bands(
        self,
        symbol: str,
        interval: str = "daily",
        time_period: int = 20,
        nbdevup: int = 2,
        nbdevdn: int = 2
    ) -> ProviderResponse:
        """Get Bollinger Bands"""
        return await self.get_technical_indicator(
            symbol=symbol,
            indicator="BBANDS",
            interval=interval,
            time_period=time_period,
            nbdevup=nbdevup,
            nbdevdn=nbdevdn
        )
    
    async def get_sma(
        self,
        symbol: str,
        interval: str = "daily",
        time_period: int = 50
    ) -> ProviderResponse:
        """Get Simple Moving Average"""
        return await self.get_technical_indicator(
            symbol=symbol,
            indicator="SMA",
            interval=interval,
            time_period=time_period
        )
    
    async def get_ema(
        self,
        symbol: str,
        interval: str = "daily",
        time_period: int = 20
    ) -> ProviderResponse:
        """Get Exponential Moving Average"""
        return await self.get_technical_indicator(
            symbol=symbol,
            indicator="EMA",
            interval=interval,
            time_period=time_period
        )
    
    async def get_fundamentals(
        self,
        symbol: str,
        statement_type: str = "income"
    ) -> ProviderResponse:
        """Get fundamental data (income, balance sheet, cash flow)"""
        try:
            function_map = {
                "income": "INCOME_STATEMENT",
                "balance": "BALANCE_SHEET",
                "cash_flow": "CASH_FLOW",
                "overview": "OVERVIEW",
                "earnings": "EARNINGS",
            }
            
            function = function_map.get(statement_type, "OVERVIEW")
            
            result = await self._request(
                "GET",
                "/query",
                params={
                    "function": function,
                    "symbol": symbol,
                    "apikey": self.api_key
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
            
            data = result["data"]
            
            if function == "OVERVIEW":
                return ProviderResponse(
                    success=True,
                    data=data,
                    data_type=DataType.FUNDAMENTALS,
                    provider=self.name,
                    symbol=symbol,
                    latency_ms=result.get("latency_ms", 0)
                )
            
            # Parse financial statements
            reports = data.get("annualReports", []) + data.get("quarterlyReports", [])
            
            statements = []
            for report in reports:
                fiscal_date = report.get("fiscalDateEnding")
                if fiscal_date:
                    statements.append(FinancialStatement(
                        symbol=symbol,
                        period=fiscal_date,
                        statement_type=statement_type,
                        data=report,
                        fiscal_year=fiscal_date[:4],
                        is_quarterly="quarterlyReports" in data
                    ))
            
            return ProviderResponse(
                success=True,
                data=statements,
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
    
    async def get_earnings(self, symbol: str) -> ProviderResponse:
        """Get earnings data"""
        return await self.get_fundamentals(symbol, "earnings")
    
    async def get_company_overview(self, symbol: str) -> ProviderResponse:
        """Get company overview/profile"""
        try:
            result = await self._request(
                "GET",
                "/query",
                params={
                    "function": "OVERVIEW",
                    "symbol": symbol,
                    "apikey": self.api_key
                }
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
            
            if not data or "Symbol" not in data:
                return ProviderResponse(
                    success=False,
                    data=None,
                    data_type=DataType.PROFILE,
                    provider=self.name,
                    symbol=symbol,
                    error="No company data found"
                )
            
            profile = CompanyProfile(
                symbol=symbol,
                name=data.get("Name"),
                description=data.get("Description"),
                exchange=data.get("Exchange"),
                sector=data.get("Sector"),
                industry=data.get("Industry"),
                market_cap=float(data.get("MarketCapitalization", 0)),
                pe_ratio=float(data.get("PERatio", 0)) if data.get("PERatio") else None,
                dividend_yield=float(data.get("DividendYield", 0)) if data.get("DividendYield") else None,
                eps=float(data.get("EPS", 0)) if data.get("EPS") else None,
                beta=float(data.get("Beta", 0)) if data.get("Beta") else None,
                week_52_high=float(data.get("52WeekHigh", 0)) if data.get("52WeekHigh") else None,
                week_52_low=float(data.get("52WeekLow", 0)) if data.get("52WeekLow") else None,
                shares_outstanding=int(float(data.get("SharesOutstanding", 0))) if data.get("SharesOutstanding") else None,
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
    
    async def get_forex_rate(
        self,
        from_currency: str,
        to_currency: str
    ) -> ProviderResponse:
        """Get forex exchange rate"""
        try:
            result = await self._request(
                "GET",
                "/query",
                params={
                    "function": "CURRENCY_EXCHANGE_RATE",
                    "from_currency": from_currency,
                    "to_currency": to_currency,
                    "apikey": self.api_key
                }
            )
            
            if "error" in result:
                return ProviderResponse(
                    success=False,
                    data=None,
                    data_type=DataType.QUOTE,
                    provider=self.name,
                    symbol=f"{from_currency}/{to_currency}",
                    error=result["error"]
                )
            
            data = result["data"].get("Realtime Currency Exchange Rate", {})
            
            return ProviderResponse(
                success=True,
                data={
                    "from": from_currency,
                    "to": to_currency,
                    "rate": float(data.get("5. Exchange Rate", 0)),
                    "bid": float(data.get("8. Bid Price", 0)),
                    "ask": float(data.get("9. Ask Price", 0)),
                    "timestamp": data.get("6. Last Refreshed"),
                },
                data_type=DataType.QUOTE,
                provider=self.name,
                symbol=f"{from_currency}/{to_currency}",
                latency_ms=result.get("latency_ms", 0)
            )
            
        except Exception as e:
            return ProviderResponse(
                success=False,
                data=None,
                data_type=DataType.QUOTE,
                provider=self.name,
                symbol=f"{from_currency}/{to_currency}",
                error=str(e)
            )
    
    async def get_crypto_rating(self, symbol: str) -> ProviderResponse:
        """Get crypto fundamental rating"""
        try:
            result = await self._request(
                "GET",
                "/query",
                params={
                    "function": "CRYPTO_RATING",
                    "symbol": symbol,
                    "apikey": self.api_key
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
