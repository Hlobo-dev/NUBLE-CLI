"""
CoinDesk Premium Data API Client for KYPERIAN
Professional-grade crypto market data with OHLCV+, WebSocket streaming, and indices

API Key: 78b5a8d834762d6baf867a6a465d8bbf401fbee0bbe4384940572b9cb1404f6c

Endpoints:
- Historical OHLCV+ (daily, hourly, minute)
- Real-time WebSocket streaming
- CADLI (Real-Time Adaptive Methodology) indices
- CCIX (Direct Trading Methodology) indices

Based on CoinDesk Data API documentation (uses CryptoCompare infrastructure)
"""

import requests
import json
import time
import threading
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Any, Callable
from dataclasses import dataclass
from enum import Enum


class IndexType(Enum):
    """CoinDesk Index Types"""
    CADLI = "cadli"  # Real-Time Adaptive Methodology
    CCIX = "ccix"    # Direct Trading Methodology


class TimeUnit(Enum):
    """Time units for OHLCV data"""
    MINUTE = "minute"
    HOUR = "hour"
    DAY = "day"


@dataclass
class OHLCVBar:
    """Single OHLCV+ candle"""
    timestamp: int
    open: float
    high: float
    low: float
    close: float
    volume: float
    quote_volume: float
    total_updates: int
    unit: str
    
    @property
    def datetime(self) -> datetime:
        return datetime.utcfromtimestamp(self.timestamp)
    
    def to_dict(self) -> Dict:
        return {
            "timestamp": self.timestamp,
            "datetime": self.datetime.isoformat(),
            "open": self.open,
            "high": self.high,
            "low": self.low,
            "close": self.close,
            "volume": self.volume,
            "quote_volume": self.quote_volume,
            "total_updates": self.total_updates
        }


class CoinDeskClient:
    """
    CoinDesk Premium Data API Client
    
    Features:
    - Historical OHLCV+ data (daily/hourly/minute)
    - Real-time WebSocket streaming
    - CADLI and CCIX index data
    - Full premium API access
    
    API Docs: https://developers.coindesk.com/documentation/data-api/
    """
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key or "78b5a8d834762d6baf867a6a465d8bbf401fbee0bbe4384940572b9cb1404f6c"
        
        # CoinDesk Data API base URL (uses CryptoCompare infrastructure)
        self.base_url = "https://data-api.cryptocompare.com"
        self.min_api_url = "https://min-api.cryptocompare.com/data"
        
        # WebSocket URL
        self.ws_url = f"wss://data-streamer.coindesk.com/?api_key={self.api_key}"
        
        # Headers for authentication
        self.headers = {
            "Authorization": f"Apikey {self.api_key}",
            "Accept": "application/json"
        }
        
        # Rate limiting
        self.last_request_time = 0
        self.min_request_interval = 0.1  # 100ms between requests
        
        # WebSocket connection
        self.ws = None
        self.ws_connected = False
        self.ws_callbacks = {}
        
        # Symbol mappings for common cryptos
        self.instruments = {
            "BTC": "BTC-USD",
            "ETH": "ETH-USD",
            "SOL": "SOL-USD",
            "XRP": "XRP-USD",
            "ADA": "ADA-USD",
            "DOGE": "DOGE-USD",
            "DOT": "DOT-USD",
            "AVAX": "AVAX-USD",
            "MATIC": "MATIC-USD",
            "LINK": "LINK-USD",
            "LTC": "LTC-USD",
            "UNI": "UNI-USD",
            "ATOM": "ATOM-USD",
            "XLM": "XLM-USD",
            "ALGO": "ALGO-USD"
        }
    
    def _rate_limit(self):
        """Enforce rate limiting"""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.min_request_interval:
            time.sleep(self.min_request_interval - elapsed)
        self.last_request_time = time.time()
    
    def _get_instrument(self, symbol: str) -> str:
        """Convert symbol to instrument format"""
        symbol = symbol.upper().replace("-USD", "").replace("USD", "")
        return self.instruments.get(symbol, f"{symbol}-USD")
    
    # ========================================
    # REST API - Historical OHLCV+ Data
    # ========================================
    
    def get_historical_daily(
        self,
        symbol: str = "BTC",
        limit: int = 30,
        to_ts: int = None,
        market: str = "cadli",
        fill: bool = True
    ) -> Optional[List[OHLCVBar]]:
        """
        Get historical daily OHLCV+ candlestick data
        
        Args:
            symbol: Crypto symbol (BTC, ETH, etc.)
            limit: Number of data points (max 2000)
            to_ts: End timestamp (default: now)
            market: Index market (cadli or ccix)
            fill: Fill gaps with previous close
            
        Returns:
            List of OHLCVBar objects
        """
        self._rate_limit()
        
        instrument = self._get_instrument(symbol)
        symbol_clean = symbol.upper().replace("-USD", "").replace("USD", "")
        
        # Try CoinDesk Index endpoint first
        url = f"{self.base_url}/index/cc/v1/historical/days"
        
        params = {
            "market": market,
            "instrument": instrument,
            "limit": min(limit, 2000),
            "fill": "true" if fill else "false"
        }
        
        if to_ts:
            params["to_ts"] = to_ts
        
        try:
            response = requests.get(url, headers=self.headers, params=params, timeout=15)
            
            if response.status_code == 200:
                data = response.json()
                bars = self._parse_ohlcv_response(data)
                if bars:
                    return bars
            
            # Fallback to histoday endpoint
            return self._get_historical_fallback(symbol_clean, limit, "day")
            
        except Exception as e:
            print(f"Error fetching daily data: {e}")
            return self._get_historical_fallback(symbol_clean, limit, "day")
    
    def get_historical_hourly(
        self,
        symbol: str = "BTC",
        limit: int = 168,  # 7 days
        to_ts: int = None,
        market: str = "cadli"
    ) -> Optional[List[OHLCVBar]]:
        """
        Get historical hourly OHLCV+ data
        
        Args:
            symbol: Crypto symbol
            limit: Number of hours (max 2000)
            to_ts: End timestamp
            market: Index market
            
        Returns:
            List of OHLCVBar objects
        """
        self._rate_limit()
        
        instrument = self._get_instrument(symbol)
        symbol_clean = symbol.upper().replace("-USD", "").replace("USD", "")
        
        url = f"{self.base_url}/index/cc/v1/historical/hours"
        
        params = {
            "market": market,
            "instrument": instrument,
            "limit": min(limit, 2000)
        }
        
        if to_ts:
            params["to_ts"] = to_ts
        
        try:
            response = requests.get(url, headers=self.headers, params=params, timeout=15)
            
            if response.status_code == 200:
                data = response.json()
                bars = self._parse_ohlcv_response(data)
                if bars:
                    return bars
            
            return self._get_historical_fallback(symbol_clean, limit, "hour")
            
        except Exception as e:
            print(f"Error fetching hourly data: {e}")
            return self._get_historical_fallback(symbol_clean, limit, "hour")
    
    def get_historical_minute(
        self,
        symbol: str = "BTC",
        limit: int = 1440,  # 24 hours
        to_ts: int = None,
        market: str = "cadli"
    ) -> Optional[List[OHLCVBar]]:
        """
        Get historical minute OHLCV+ data
        
        Args:
            symbol: Crypto symbol
            limit: Number of minutes (max 2000)
            to_ts: End timestamp
            market: Index market
            
        Returns:
            List of OHLCVBar objects
        """
        self._rate_limit()
        
        instrument = self._get_instrument(symbol)
        symbol_clean = symbol.upper().replace("-USD", "").replace("USD", "")
        
        url = f"{self.base_url}/index/cc/v1/historical/minutes"
        
        params = {
            "market": market,
            "instrument": instrument,
            "limit": min(limit, 2000)
        }
        
        if to_ts:
            params["to_ts"] = to_ts
        
        try:
            response = requests.get(url, headers=self.headers, params=params, timeout=15)
            
            if response.status_code == 200:
                data = response.json()
                bars = self._parse_ohlcv_response(data)
                if bars:
                    return bars
            
            return self._get_historical_fallback(symbol_clean, limit, "minute")
            
        except Exception as e:
            print(f"Error fetching minute data: {e}")
            return self._get_historical_fallback(symbol_clean, limit, "minute")
    
    def _parse_ohlcv_response(self, data: Dict) -> List[OHLCVBar]:
        """Parse OHLCV+ response from API"""
        bars = []
        
        if "Data" in data:
            items = data["Data"] if isinstance(data["Data"], list) else data["Data"].get("Data", [])
            
            for item in items:
                try:
                    # Handle both CoinDesk and CryptoCompare formats
                    bar = OHLCVBar(
                        timestamp=item.get("TIMESTAMP", item.get("time", 0)),
                        open=item.get("OPEN", item.get("open", 0)),
                        high=item.get("HIGH", item.get("high", 0)),
                        low=item.get("LOW", item.get("low", 0)),
                        close=item.get("CLOSE", item.get("close", 0)),
                        volume=item.get("VOLUME", item.get("volumefrom", 0)),
                        quote_volume=item.get("QUOTE_VOLUME", item.get("volumeto", 0)),
                        total_updates=item.get("TOTAL_INDEX_UPDATES", 0),
                        unit=item.get("UNIT", "DAY")
                    )
                    if bar.close > 0:  # Only add valid bars
                        bars.append(bar)
                except Exception as e:
                    continue
        
        return bars
    
    def _get_historical_fallback(
        self,
        symbol: str,
        limit: int,
        unit: str
    ) -> Optional[List[OHLCVBar]]:
        """Fallback to CryptoCompare API"""
        
        endpoint_map = {
            "day": "v2/histoday",
            "hour": "v2/histohour",
            "minute": "v2/histominute"
        }
        
        url = f"{self.min_api_url}/{endpoint_map.get(unit, 'v2/histoday')}"
        
        params = {
            "fsym": symbol.upper(),
            "tsym": "USD",
            "limit": limit
        }
        
        try:
            response = requests.get(
                url,
                headers={"Authorization": f"Apikey {self.api_key}"},
                params=params,
                timeout=15
            )
            
            if response.status_code == 200:
                data = response.json()
                
                if data.get("Response") == "Success" and "Data" in data:
                    bars = []
                    for item in data["Data"].get("Data", []):
                        bar = OHLCVBar(
                            timestamp=item.get("time", 0),
                            open=item.get("open", 0),
                            high=item.get("high", 0),
                            low=item.get("low", 0),
                            close=item.get("close", 0),
                            volume=item.get("volumefrom", 0),
                            quote_volume=item.get("volumeto", 0),
                            total_updates=0,
                            unit=unit.upper()
                        )
                        if bar.close > 0:
                            bars.append(bar)
                    return bars
                    
        except Exception as e:
            print(f"Fallback API error: {e}")
        
        return None
    
    # ========================================
    # Current Price / Latest Data
    # ========================================
    
    def get_current_price(self, symbol: str = "BTC") -> Optional[Dict]:
        """
        Get current price for a cryptocurrency
        
        Args:
            symbol: Crypto symbol
            
        Returns:
            Current price data
        """
        self._rate_limit()
        
        symbol = symbol.upper().replace("-USD", "").replace("USD", "")
        
        # Use CryptoCompare price endpoint
        url = f"{self.min_api_url}/price"
        
        params = {
            "fsym": symbol,
            "tsyms": "USD,EUR,GBP"
        }
        
        try:
            response = requests.get(
                url,
                headers={"Authorization": f"Apikey {self.api_key}"},
                params=params,
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                
                if "USD" in data:
                    return {
                        "symbol": symbol,
                        "price": data.get("USD"),
                        "USD": data.get("USD"),
                        "EUR": data.get("EUR"),
                        "GBP": data.get("GBP"),
                        "timestamp": datetime.now().isoformat(),
                        "source": "coindesk_premium"
                    }
                
        except Exception as e:
            print(f"Error fetching current price: {e}")
        
        return None
    
    def get_multi_price(self, symbols: List[str] = None) -> Dict[str, Dict]:
        """
        Get current prices for multiple cryptocurrencies
        
        Args:
            symbols: List of crypto symbols
            
        Returns:
            Dict of prices by symbol
        """
        if symbols is None:
            symbols = ["BTC", "ETH", "SOL", "XRP", "ADA"]
        
        self._rate_limit()
        
        # Clean symbols
        symbols = [s.upper().replace("-USD", "").replace("USD", "") for s in symbols]
        
        url = f"{self.min_api_url}/pricemulti"
        
        params = {
            "fsyms": ",".join(symbols),
            "tsyms": "USD"
        }
        
        try:
            response = requests.get(
                url,
                headers={"Authorization": f"Apikey {self.api_key}"},
                params=params,
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                
                result = {}
                for symbol, prices in data.items():
                    result[symbol] = {
                        "price": prices.get("USD"),
                        "symbol": symbol,
                        "currency": "USD"
                    }
                
                return result
                
        except Exception as e:
            print(f"Error fetching multi prices: {e}")
        
        return {}
    
    def get_bitcoin_index(self) -> Optional[Dict]:
        """
        Get Bitcoin price index (BPI) - compatibility method
        
        Returns:
            BPI data
        """
        price = self.get_current_price("BTC")
        if price:
            return {
                "updated": datetime.now().isoformat(),
                "bpi": {
                    "USD": {"rate_float": price.get("USD", 0)},
                    "EUR": {"rate_float": price.get("EUR", 0)},
                    "GBP": {"rate_float": price.get("GBP", 0)}
                },
                "source": "coindesk_premium"
            }
        return None
    
    # ========================================
    # WebSocket Streaming
    # ========================================
    
    def connect_websocket(
        self,
        on_message: Callable = None,
        on_error: Callable = None,
        on_close: Callable = None
    ):
        """
        Connect to CoinDesk WebSocket for real-time data
        
        Args:
            on_message: Callback for incoming messages
            on_error: Callback for errors
            on_close: Callback for connection close
        """
        try:
            import websocket
        except ImportError:
            print("websocket-client required for WebSocket streaming")
            print("Install with: pip install websocket-client")
            return False
        
        def _on_message(ws, message):
            try:
                data = json.loads(message)
                msg_type = data.get("TYPE", "")
                
                # Handle different message types
                if msg_type == "4000":  # Session Welcome
                    print(f"✅ WebSocket Connected - Session ID: {data.get('SOCKET_ID')}")
                    self.ws_connected = True
                    
                elif msg_type == "4013":  # Heartbeat
                    pass  # Silently handle heartbeats
                    
                elif msg_type == "4005":  # Subscription Accepted
                    print(f"✅ Subscription accepted")
                    
                elif msg_type in ["4001", "4002", "4003", "4004", "4006"]:  # Errors
                    print(f"⚠️ WebSocket Error: {data.get('MESSAGE', msg_type)}")
                    
                else:
                    # Data message
                    if on_message:
                        on_message(data)
                    else:
                        print(f"📊 Data: {data}")
                        
            except json.JSONDecodeError:
                print(f"Invalid JSON: {message[:100]}")
        
        def _on_error(ws, error):
            print(f"❌ WebSocket Error: {error}")
            if on_error:
                on_error(error)
        
        def _on_close(ws, close_status_code, close_msg):
            print(f"🔌 WebSocket Closed: {close_status_code} - {close_msg}")
            self.ws_connected = False
            if on_close:
                on_close(close_status_code, close_msg)
        
        def _on_open(ws):
            print("🔌 WebSocket Connection Opened...")
        
        # Create WebSocket connection
        self.ws = websocket.WebSocketApp(
            self.ws_url,
            on_open=_on_open,
            on_message=_on_message,
            on_error=_on_error,
            on_close=_on_close
        )
        
        # Run in background thread
        ws_thread = threading.Thread(target=self.ws.run_forever)
        ws_thread.daemon = True
        ws_thread.start()
        
        # Wait for connection
        timeout = 10
        start = time.time()
        while not self.ws_connected and time.time() - start < timeout:
            time.sleep(0.1)
        
        return self.ws_connected
    
    def subscribe(
        self,
        symbol: str = "BTC",
        market: str = "cadli",
        groups: List[str] = None
    ):
        """
        Subscribe to real-time updates for a symbol
        
        Args:
            symbol: Crypto symbol
            market: Index market (cadli or ccix)
            groups: Data groups (VALUE, CURRENT_HOUR, etc.)
        """
        if not self.ws or not self.ws_connected:
            print("❌ WebSocket not connected")
            return False
        
        if groups is None:
            groups = ["VALUE", "CURRENT_HOUR"]
        
        instrument = self._get_instrument(symbol)
        
        message = {
            "action": "SUB_ADD",
            "type": "1101",  # CADLI Tick updates
            "groups": groups,
            "subscriptions": [{"market": market, "instrument": instrument}]
        }
        
        try:
            self.ws.send(json.dumps(message))
            print(f"📡 Subscribed to {symbol} ({instrument})")
            return True
        except Exception as e:
            print(f"❌ Subscribe error: {e}")
            return False
    
    def unsubscribe(self, symbol: str = "BTC", market: str = "cadli"):
        """Unsubscribe from a symbol"""
        if not self.ws or not self.ws_connected:
            return False
        
        instrument = self._get_instrument(symbol)
        
        message = {
            "action": "SUB_REMOVE",
            "type": "1101",
            "subscriptions": [{"market": market, "instrument": instrument}]
        }
        
        try:
            self.ws.send(json.dumps(message))
            return True
        except:
            return False
    
    def disconnect_websocket(self):
        """Close WebSocket connection"""
        if self.ws:
            self.ws.close()
            self.ws_connected = False
    
    # ========================================
    # Convenience Methods
    # ========================================
    
    def get_ohlcv_dataframe(
        self,
        symbol: str = "BTC",
        days: int = 365,
        unit: str = "day"
    ):
        """
        Get OHLCV data as pandas DataFrame
        
        Args:
            symbol: Crypto symbol
            days: Number of days of data
            unit: Time unit (day, hour, minute)
            
        Returns:
            pandas DataFrame with OHLCV data
        """
        try:
            import pandas as pd
        except ImportError:
            print("pandas required for DataFrame output")
            return None
        
        if unit == "day":
            bars = self.get_historical_daily(symbol, limit=days)
        elif unit == "hour":
            bars = self.get_historical_hourly(symbol, limit=days * 24)
        else:
            bars = self.get_historical_minute(symbol, limit=min(days * 1440, 2000))
        
        if not bars:
            return None
        
        data = [bar.to_dict() for bar in bars]
        df = pd.DataFrame(data)
        df['datetime'] = pd.to_datetime(df['datetime'])
        df.set_index('datetime', inplace=True)
        
        return df
    
    def get_historical_data(
        self,
        symbol: str = "BTC",
        start_date: str = None,
        end_date: str = None,
        currency: str = "USD"
    ) -> Optional[Dict]:
        """
        Legacy compatibility method for historical data
        
        Args:
            symbol: Crypto symbol
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            currency: Quote currency
            
        Returns:
            Historical price data
        """
        # Calculate days between dates
        if start_date and end_date:
            start = datetime.strptime(start_date, "%Y-%m-%d")
            end = datetime.strptime(end_date, "%Y-%m-%d")
            days = (end - start).days + 1
        else:
            days = 30
        
        bars = self.get_historical_daily(symbol, limit=days)
        
        if bars:
            prices = []
            for bar in bars:
                prices.append({
                    "date": bar.datetime.strftime("%Y-%m-%d"),
                    "price": bar.close,
                    "open": bar.open,
                    "high": bar.high,
                    "low": bar.low,
                    "volume": bar.volume
                })
            
            return {
                "symbol": symbol,
                "currency": currency,
                "start_date": prices[0]["date"] if prices else None,
                "end_date": prices[-1]["date"] if prices else None,
                "prices": prices,
                "data_points": len(prices),
                "source": "coindesk_premium"
            }
        
        return None


# Alias for backwards compatibility
CoinDeskPremiumClient = CoinDeskClient


def test_coindesk_premium():
    """Test CoinDesk Premium API"""
    print("=" * 70)
    print("🪙 COINDESK PREMIUM DATA API TEST")
    print("=" * 70)
    
    client = CoinDeskClient()
    passed = 0
    failed = 0
    
    # Test 1: Current Price
    print("\n💰 Test 1: Current Prices")
    print("-" * 50)
    
    prices = client.get_multi_price(["BTC", "ETH", "SOL"])
    if prices:
        for symbol, data in prices.items():
            price = data.get('price', 0)
            if price:
                print(f"   {symbol}: ${price:,.2f}")
        print("✅ Current prices: PASSED")
        passed += 1
    else:
        print("❌ Current prices: FAILED")
        failed += 1
    
    # Test 2: Single Price
    print("\n💵 Test 2: Single Price (BTC)")
    print("-" * 50)
    
    btc = client.get_current_price("BTC")
    if btc and btc.get("price"):
        print(f"   BTC: ${btc['price']:,.2f}")
        print(f"   EUR: €{btc.get('EUR', 0):,.2f}")
        print("✅ Single price: PASSED")
        passed += 1
    else:
        print("❌ Single price: FAILED")
        failed += 1
    
    # Test 3: Historical Daily
    print("\n📈 Test 3: Historical Daily OHLCV")
    print("-" * 50)
    
    daily = client.get_historical_daily("BTC", limit=7)
    if daily:
        print(f"   Retrieved {len(daily)} daily bars")
        for bar in daily[-3:]:
            print(f"   {bar.datetime.strftime('%Y-%m-%d')}: O=${bar.open:,.0f} H=${bar.high:,.0f} L=${bar.low:,.0f} C=${bar.close:,.0f}")
        print("✅ Historical daily: PASSED")
        passed += 1
    else:
        print("❌ Historical daily: FAILED")
        failed += 1
    
    # Test 4: Historical Hourly
    print("\n📊 Test 4: Historical Hourly OHLCV")
    print("-" * 50)
    
    hourly = client.get_historical_hourly("ETH", limit=24)
    if hourly:
        print(f"   Retrieved {len(hourly)} hourly bars")
        for bar in hourly[-3:]:
            print(f"   {bar.datetime.strftime('%Y-%m-%d %H:%M')}: Close=${bar.close:,.2f}")
        print("✅ Historical hourly: PASSED")
        passed += 1
    else:
        print("❌ Historical hourly: FAILED")
        failed += 1
    
    # Test 5: Bitcoin Index
    print("\n🏆 Test 5: Bitcoin Price Index")
    print("-" * 50)
    
    bpi = client.get_bitcoin_index()
    if bpi and bpi.get("bpi"):
        usd = bpi["bpi"]["USD"]["rate_float"]
        print(f"   BPI USD: ${usd:,.2f}")
        print("✅ Bitcoin index: PASSED")
        passed += 1
    else:
        print("❌ Bitcoin index: FAILED")
        failed += 1
    
    # Test 6: DataFrame Export
    print("\n📋 Test 6: DataFrame Export")
    print("-" * 50)
    
    try:
        df = client.get_ohlcv_dataframe("BTC", days=30)
        if df is not None and len(df) > 0:
            print(f"   DataFrame shape: {df.shape}")
            print(f"   Columns: {list(df.columns)}")
            print(f"   Date range: {df.index.min()} to {df.index.max()}")
            print("✅ DataFrame export: PASSED")
            passed += 1
        else:
            print("⚠️ DataFrame export: No data")
            failed += 1
    except Exception as e:
        print(f"❌ DataFrame export: {e}")
        failed += 1
    
    # Summary
    print("\n" + "=" * 70)
    print(f"📊 RESULTS: {passed}/{passed + failed} tests passed")
    
    if failed == 0:
        print("🎉 ALL COINDESK PREMIUM API TESTS PASSED!")
    else:
        print(f"⚠️ {failed} test(s) failed")
    
    print("=" * 70)


if __name__ == "__main__":
    test_coindesk_premium()
