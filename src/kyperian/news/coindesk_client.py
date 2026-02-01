"""
CoinDesk Premium Data API Client for KYPERIAN
"""

import requests
import time
from datetime import datetime
from typing import Optional, Dict, List
from dataclasses import dataclass


@dataclass
class OHLCVBar:
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
        return datetime.fromtimestamp(self.timestamp)
    
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
    """CoinDesk Premium Data API Client"""
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key or "78b5a8d834762d6baf867a6a465d8bbf401fbee0bbe4384940572b9cb1404f6c"
        self.min_api_url = "https://min-api.cryptocompare.com/data"
        self.headers = {"Authorization": f"Apikey {self.api_key}"}
        self.last_request_time = 0
    
    def _rate_limit(self):
        elapsed = time.time() - self.last_request_time
        if elapsed < 0.1:
            time.sleep(0.1 - elapsed)
        self.last_request_time = time.time()
    
    def get_current_price(self, symbol: str = "BTC") -> Optional[Dict]:
        self._rate_limit()
        symbol = symbol.upper().replace("-USD", "").replace("USD", "")
        url = f"{self.min_api_url}/price"
        params = {"fsym": symbol, "tsyms": "USD,EUR,GBP"}
        try:
            resp = requests.get(url, headers=self.headers, params=params, timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                if "USD" in data:
                    return {"symbol": symbol, "price": data["USD"], "USD": data["USD"], 
                            "EUR": data.get("EUR"), "GBP": data.get("GBP")}
        except Exception as e:
            print(f"Error: {e}")
        return None
    
    def get_multi_price(self, symbols: List[str] = None) -> Dict[str, Dict]:
        if symbols is None:
            symbols = ["BTC", "ETH", "SOL"]
        self._rate_limit()
        symbols = [s.upper().replace("-USD", "").replace("USD", "") for s in symbols]
        url = f"{self.min_api_url}/pricemulti"
        params = {"fsyms": ",".join(symbols), "tsyms": "USD"}
        try:
            resp = requests.get(url, headers=self.headers, params=params, timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                return {sym: {"price": prices["USD"], "symbol": sym} for sym, prices in data.items()}
        except Exception as e:
            print(f"Error: {e}")
        return {}
    
    def get_historical_daily(self, symbol: str = "BTC", limit: int = 30) -> Optional[List[OHLCVBar]]:
        self._rate_limit()
        symbol = symbol.upper().replace("-USD", "").replace("USD", "")
        url = f"{self.min_api_url}/v2/histoday"
        params = {"fsym": symbol, "tsym": "USD", "limit": limit}
        try:
            resp = requests.get(url, headers=self.headers, params=params, timeout=15)
            if resp.status_code == 200:
                data = resp.json()
                if data.get("Response") == "Success":
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
                            unit="DAY"
                        )
                        if bar.close > 0:
                            bars.append(bar)
                    return bars
        except Exception as e:
            print(f"Error: {e}")
        return None
    
    def get_historical_hourly(self, symbol: str = "BTC", limit: int = 168) -> Optional[List[OHLCVBar]]:
        self._rate_limit()
        symbol = symbol.upper().replace("-USD", "").replace("USD", "")
        url = f"{self.min_api_url}/v2/histohour"
        params = {"fsym": symbol, "tsym": "USD", "limit": limit}
        try:
            resp = requests.get(url, headers=self.headers, params=params, timeout=15)
            if resp.status_code == 200:
                data = resp.json()
                if data.get("Response") == "Success":
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
                            unit="HOUR"
                        )
                        if bar.close > 0:
                            bars.append(bar)
                    return bars
        except Exception as e:
            print(f"Error: {e}")
        return None
    
    def get_bitcoin_index(self) -> Optional[Dict]:
        price = self.get_current_price("BTC")
        if price:
            return {
                "updated": datetime.now().isoformat(),
                "bpi": {"USD": {"rate_float": price["USD"]}, "EUR": {"rate_float": price.get("EUR", 0)}}
            }
        return None
    
    def get_ohlcv_dataframe(self, symbol: str = "BTC", days: int = 365):
        try:
            import pandas as pd
        except ImportError:
            return None
        bars = self.get_historical_daily(symbol, limit=days)
        if not bars:
            return None
        data = [bar.to_dict() for bar in bars]
        df = pd.DataFrame(data)
        df['datetime'] = pd.to_datetime(df['datetime'])
        df.set_index('datetime', inplace=True)
        return df


CoinDeskPremiumClient = CoinDeskClient


if __name__ == "__main__":
    client = CoinDeskClient()
    btc = client.get_current_price("BTC")
    print(f"BTC: ${btc['price']:,.2f}" if btc else "Error")
