"""
POLYGON.IO DATA DOWNLOADER
===========================
Professional data acquisition using your paid Polygon.io subscription.

This provides:
- Full historical data back to 2015
- Adjusted OHLCV with splits/dividends handled
- Higher quality than free Yahoo Finance data
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import time
from typing import Optional, List, Dict
import logging

from .config import CONFIG

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PolygonDataDownloader:
    """
    Professional data downloader using Polygon.io API.
    
    Your subscription tier provides:
    - Unlimited API calls
    - Full historical data
    - Real-time data (not used here)
    - Corporate actions (splits, dividends)
    """
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or CONFIG.polygon_api_key
        self.base_url = "https://api.polygon.io"
        
        if not self.api_key:
            raise ValueError(
                "Polygon API key required. Set POLYGON_API_KEY environment variable "
                "or pass api_key parameter."
            )
    
    def get_daily_bars(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        adjusted: bool = True
    ) -> pd.DataFrame:
        """
        Fetch daily OHLCV bars for a symbol.
        
        Parameters:
        -----------
        symbol : str
            Ticker symbol (e.g., "AAPL")
        start_date : str
            Start date YYYY-MM-DD
        end_date : str
            End date YYYY-MM-DD
        adjusted : bool
            Whether to adjust for splits/dividends
            
        Returns:
        --------
        pd.DataFrame with columns: open, high, low, close, volume, vwap
        """
        url = f"{self.base_url}/v2/aggs/ticker/{symbol}/range/1/day/{start_date}/{end_date}"
        params = {
            "apiKey": self.api_key,
            "adjusted": str(adjusted).lower(),
            "sort": "asc",
            "limit": 50000
        }
        
        response = requests.get(url, params=params)
        data = response.json()
        
        if data.get("status") != "OK":
            error_msg = data.get("message", data.get("error", "Unknown error"))
            raise ValueError(f"Polygon API error for {symbol}: {error_msg}")
        
        if "results" not in data or len(data["results"]) == 0:
            raise ValueError(f"No data returned for {symbol}")
        
        df = pd.DataFrame(data["results"])
        df["date"] = pd.to_datetime(df["t"], unit="ms")
        df = df.set_index("date")
        df.index = df.index.tz_localize(None)  # Remove timezone
        
        df = df.rename(columns={
            "o": "open",
            "h": "high",
            "l": "low",
            "c": "close",
            "v": "volume",
            "vw": "vwap",
            "n": "trades"
        })
        
        # Select and order columns
        columns = ["open", "high", "low", "close", "volume"]
        if "vwap" in df.columns:
            columns.append("vwap")
        if "trades" in df.columns:
            columns.append("trades")
        
        return df[columns]
    
    def download_all_symbols(
        self,
        symbols: Optional[List[str]] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        output_dir: Optional[Path] = None,
        delay: float = 0.1  # With paid tier, can be fast
    ) -> Dict[str, pd.DataFrame]:
        """
        Download data for all symbols.
        
        Parameters:
        -----------
        symbols : List[str], optional
            Symbols to download. Defaults to CONFIG.symbols
        start_date : str, optional
            Start date. Defaults to CONFIG.train_start
        end_date : str, optional
            End date. Defaults to CONFIG.test_end
        output_dir : Path, optional
            Directory to save files. Defaults to CONFIG.historical_dir
        delay : float
            Delay between API calls (can be low with paid tier)
        
        Returns:
        --------
        Dict[str, pd.DataFrame]
        """
        symbols = symbols or CONFIG.symbols
        start_date = start_date or CONFIG.train_start
        end_date = end_date or CONFIG.test_end
        output_dir = output_dir or CONFIG.historical_dir
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        data = {}
        failed = []
        
        logger.info(f"Downloading {len(symbols)} symbols from {start_date} to {end_date}")
        
        for i, symbol in enumerate(symbols):
            logger.info(f"[{i+1}/{len(symbols)}] Downloading {symbol}...")
            
            try:
                df = self.get_daily_bars(symbol, start_date, end_date)
                
                if len(df) < 100:
                    logger.warning(f"  {symbol}: Only {len(df)} rows - skipping")
                    failed.append(symbol)
                    continue
                
                # Save to CSV
                filepath = output_dir / f"{symbol}.csv"
                df.to_csv(filepath)
                
                data[symbol] = df
                logger.info(f"  {symbol}: {len(df)} rows ({df.index[0].date()} to {df.index[-1].date()})")
                
                time.sleep(delay)
                
            except Exception as e:
                logger.error(f"  {symbol}: Failed - {e}")
                failed.append(symbol)
        
        logger.info(f"\nDownloaded: {len(data)} symbols")
        if failed:
            logger.warning(f"Failed: {failed}")
        
        return data
    
    def split_train_test(
        self,
        data: Optional[Dict[str, pd.DataFrame]] = None,
        train_end: Optional[str] = None,
        test_start: Optional[str] = None
    ) -> tuple:
        """
        Split data into training and test sets.
        
        CRITICAL: Test data must NEVER be used during development!
        
        Returns:
        --------
        tuple : (train_data dict, test_data dict)
        """
        # Load from files if not provided
        if data is None:
            data = {}
            for filepath in CONFIG.historical_dir.glob("*.csv"):
                symbol = filepath.stem
                data[symbol] = pd.read_csv(filepath, index_col=0, parse_dates=True)
        
        train_end = train_end or CONFIG.train_end
        test_start = test_start or CONFIG.test_start
        
        train_end_dt = pd.Timestamp(train_end)
        test_start_dt = pd.Timestamp(test_start)
        
        train_data = {}
        test_data = {}
        
        logger.info(f"\nSplitting data: Train <= {train_end}, Test >= {test_start}")
        
        for symbol, df in data.items():
            train_df = df[df.index <= train_end_dt].copy()
            test_df = df[df.index >= test_start_dt].copy()
            
            if len(train_df) > 0:
                train_data[symbol] = train_df
                train_df.to_csv(CONFIG.train_dir / f"{symbol}.csv")
            
            if len(test_df) > 0:
                test_data[symbol] = test_df
                test_df.to_csv(CONFIG.test_dir / f"{symbol}.csv")
            
            logger.info(f"  {symbol}: Train={len(train_df)} rows, Test={len(test_df)} rows")
        
        logger.info("\n⚠️  TEST DATA SAVED - DO NOT TOUCH UNTIL FINAL VALIDATION!")
        
        return train_data, test_data


def download_with_yfinance_fallback(
    symbols: Optional[List[str]] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    polygon_api_key: Optional[str] = None
) -> Dict[str, pd.DataFrame]:
    """
    Try Polygon.io first, fall back to Yahoo Finance if needed.
    
    This ensures data acquisition even if there are API issues.
    """
    symbols = symbols or CONFIG.symbols
    start_date = start_date or CONFIG.train_start
    end_date = end_date or CONFIG.test_end
    
    data = {}
    
    # Try Polygon first
    if polygon_api_key:
        try:
            downloader = PolygonDataDownloader(polygon_api_key)
            data = downloader.download_all_symbols(symbols, start_date, end_date)
            return data
        except Exception as e:
            logger.warning(f"Polygon.io failed: {e}. Falling back to Yahoo Finance.")
    
    # Fallback to Yahoo Finance
    import yfinance as yf
    
    for symbol in symbols:
        try:
            logger.info(f"[YF] Downloading {symbol}...")
            ticker = yf.Ticker(symbol)
            df = ticker.history(start=start_date, end=end_date, auto_adjust=False)
            
            if len(df) < 100:
                logger.warning(f"  {symbol}: Only {len(df)} rows - skipping")
                continue
            
            df = df.rename(columns={
                'Open': 'open',
                'High': 'high',
                'Low': 'low',
                'Close': 'close',
                'Volume': 'volume'
            })
            
            # Save
            filepath = CONFIG.historical_dir / f"{symbol}.csv"
            df.to_csv(filepath)
            
            data[symbol] = df
            logger.info(f"  {symbol}: {len(df)} rows")
            
            time.sleep(0.5)
            
        except Exception as e:
            logger.error(f"  {symbol}: Failed - {e}")
    
    return data


if __name__ == "__main__":
    import os
    
    # Get API key from environment
    api_key = os.getenv("POLYGON_API_KEY")
    
    if api_key:
        print("Using Polygon.io (paid subscription)")
        downloader = PolygonDataDownloader(api_key)
        data = downloader.download_all_symbols()
        train_data, test_data = downloader.split_train_test(data)
    else:
        print("No Polygon API key found. Using Yahoo Finance fallback.")
        data = download_with_yfinance_fallback()
        
        # Manual split
        downloader = PolygonDataDownloader.__new__(PolygonDataDownloader)
        train_data, test_data = downloader.split_train_test(data)
    
    print(f"\n✓ Downloaded {len(data)} symbols")
    print(f"✓ Train: {len(train_data)} symbols")
    print(f"✓ Test: {len(test_data)} symbols")
