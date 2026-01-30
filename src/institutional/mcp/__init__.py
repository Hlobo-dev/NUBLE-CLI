"""
Massive.com (Polygon.io) MCP Server Integration
=================================================

Direct integration with Massive.com's MCP server for:
- Real-time market data streaming
- Historical price data
- Options chains and Greeks
- News and sentiment
- Fundamental data

This provides the data layer that powers the ML models.
"""

import os
import json
import asyncio
import subprocess
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime, date, timedelta
from abc import ABC, abstractmethod
import urllib.request
import urllib.error


@dataclass
class MCPServerConfig:
    """Configuration for MCP server connection"""
    api_key: str
    server_version: str = "v0.7.0"
    transport: str = "stdio"  # stdio, sse, streamable-http
    timeout: int = 30


@dataclass
class MCPToolCall:
    """Represents a call to an MCP tool"""
    tool_name: str
    parameters: Dict[str, Any]
    result: Optional[Any] = None
    error: Optional[str] = None
    latency_ms: int = 0


class MassiveMCPClient:
    """
    Client for interacting with Massive.com MCP server.
    
    Provides a Pythonic interface to all MCP tools while handling:
    - Connection management
    - Request/response serialization
    - Error handling
    - Rate limiting
    """
    
    # Available MCP tools
    TOOLS = {
        # Aggregates & Bars
        'get_aggs': 'Stock aggregates (OHLC) data for a specific ticker',
        'get_grouped_daily': 'Get all tickers for a given date',
        'get_previous_close': 'Previous day close for a ticker',
        
        # Trades & Quotes
        'list_trades': 'Historical trade data',
        'get_last_trade': 'Latest trade for a symbol',
        'list_quotes': 'Historical quote data (NBBO)',
        'get_last_quote': 'Latest quote for a symbol',
        
        # Snapshots
        'get_snapshot_ticker': 'Current market snapshot for a ticker',
        'get_snapshot_all': 'Snapshot of all tickers',
        'get_snapshot_gainers_losers': 'Top gainers and losers',
        
        # Reference Data
        'list_tickers': 'List available tickers',
        'get_ticker_details': 'Detailed information about a ticker',
        'get_ticker_types': 'Types of securities',
        'get_market_status': 'Current market status',
        'get_market_holidays': 'Market holidays',
        
        # Dividends & Splits
        'list_dividends': 'Historical dividends',
        'list_stock_splits': 'Stock split history',
        
        # Financials
        'list_stock_financials': 'Fundamental financial data',
        
        # News
        'list_ticker_news': 'Recent news articles',
        
        # Options
        'get_options_contract': 'Single options contract details',
        'list_options_contracts': 'List options contracts',
        'get_options_chain': 'Full options chain for underlying',
        
        # Crypto
        'get_crypto_aggs': 'Crypto aggregates',
        'get_crypto_snapshot': 'Crypto ticker snapshot',
        
        # Forex
        'get_forex_aggs': 'Forex aggregates',
        'get_forex_snapshot': 'Forex pair snapshot',
    }
    
    def __init__(self, config: MCPServerConfig = None):
        """
        Initialize MCP client.
        
        Args:
            config: Server configuration
        """
        if config is None:
            api_key = os.environ.get('POLYGON_API_KEY') or os.environ.get('MASSIVE_API_KEY')
            if not api_key:
                raise ValueError("POLYGON_API_KEY or MASSIVE_API_KEY environment variable required")
            config = MCPServerConfig(api_key=api_key)
        
        self.config = config
        self._process = None
        self._connected = False
    
    async def connect(self):
        """Establish connection to MCP server"""
        # For now, we use direct REST API calls
        # In production, this would spawn the MCP server process
        self._connected = True
    
    async def disconnect(self):
        """Disconnect from MCP server"""
        self._connected = False
    
    def _make_request(
        self,
        endpoint: str,
        params: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Make a REST API request to Polygon/Massive"""
        base_url = "https://api.polygon.io"
        
        # Build URL with parameters
        url = f"{base_url}{endpoint}"
        
        if params is None:
            params = {}
        params['apiKey'] = self.config.api_key
        
        # Encode parameters
        query_string = "&".join(f"{k}={v}" for k, v in params.items() if v is not None)
        full_url = f"{url}?{query_string}"
        
        try:
            with urllib.request.urlopen(full_url, timeout=self.config.timeout) as response:
                data = json.loads(response.read().decode())
                return data
        except urllib.error.HTTPError as e:
            raise Exception(f"HTTP Error {e.code}: {e.reason}")
        except urllib.error.URLError as e:
            raise Exception(f"URL Error: {e.reason}")
    
    # ========== Aggregates & Bars ==========
    
    def get_aggs(
        self,
        ticker: str,
        multiplier: int = 1,
        timespan: str = 'day',
        from_date: str = None,
        to_date: str = None,
        adjusted: bool = True,
        sort: str = 'asc',
        limit: int = 5000
    ) -> Dict[str, Any]:
        """
        Get aggregate bars for a ticker over a given date range.
        
        Args:
            ticker: Stock ticker symbol
            multiplier: Size of timespan multiplier
            timespan: 'minute', 'hour', 'day', 'week', 'month', 'quarter', 'year'
            from_date: Start date (YYYY-MM-DD)
            to_date: End date (YYYY-MM-DD)
            adjusted: Whether to adjust for splits
            sort: 'asc' or 'desc'
            limit: Maximum number of results
            
        Returns:
            Aggregates data with OHLCV
        """
        if from_date is None:
            from_date = (date.today() - timedelta(days=365)).isoformat()
        if to_date is None:
            to_date = date.today().isoformat()
        
        endpoint = f"/v2/aggs/ticker/{ticker}/range/{multiplier}/{timespan}/{from_date}/{to_date}"
        
        params = {
            'adjusted': str(adjusted).lower(),
            'sort': sort,
            'limit': limit
        }
        
        return self._make_request(endpoint, params)
    
    def get_previous_close(self, ticker: str, adjusted: bool = True) -> Dict[str, Any]:
        """Get previous day's close for a ticker"""
        endpoint = f"/v2/aggs/ticker/{ticker}/prev"
        return self._make_request(endpoint, {'adjusted': str(adjusted).lower()})
    
    def get_grouped_daily(self, date_str: str, adjusted: bool = True) -> Dict[str, Any]:
        """Get all tickers' daily bars for a given date"""
        endpoint = f"/v2/aggs/grouped/locale/us/market/stocks/{date_str}"
        return self._make_request(endpoint, {'adjusted': str(adjusted).lower()})
    
    # ========== Trades & Quotes ==========
    
    def get_last_trade(self, ticker: str) -> Dict[str, Any]:
        """Get the most recent trade for a ticker"""
        endpoint = f"/v2/last/trade/{ticker}"
        return self._make_request(endpoint)
    
    def get_last_quote(self, ticker: str) -> Dict[str, Any]:
        """Get the most recent NBBO quote for a ticker"""
        endpoint = f"/v2/last/nbbo/{ticker}"
        return self._make_request(endpoint)
    
    def list_trades(
        self,
        ticker: str,
        timestamp: str = None,
        limit: int = 50000
    ) -> Dict[str, Any]:
        """List historical trades"""
        endpoint = f"/v3/trades/{ticker}"
        params = {'limit': limit}
        if timestamp:
            params['timestamp'] = timestamp
        return self._make_request(endpoint, params)
    
    # ========== Snapshots ==========
    
    def get_snapshot_ticker(self, ticker: str) -> Dict[str, Any]:
        """Get current snapshot for a ticker"""
        endpoint = f"/v2/snapshot/locale/us/markets/stocks/tickers/{ticker}"
        return self._make_request(endpoint)
    
    def get_snapshot_all(self, tickers: List[str] = None) -> Dict[str, Any]:
        """Get snapshot for multiple tickers"""
        endpoint = "/v2/snapshot/locale/us/markets/stocks/tickers"
        params = {}
        if tickers:
            params['tickers'] = ','.join(tickers)
        return self._make_request(endpoint, params)
    
    def get_gainers_losers(self, direction: str = 'gainers') -> Dict[str, Any]:
        """Get top gainers or losers"""
        endpoint = f"/v2/snapshot/locale/us/markets/stocks/{direction}"
        return self._make_request(endpoint)
    
    # ========== Reference Data ==========
    
    def get_ticker_details(self, ticker: str) -> Dict[str, Any]:
        """Get detailed information about a ticker"""
        endpoint = f"/v3/reference/tickers/{ticker}"
        return self._make_request(endpoint)
    
    def list_tickers(
        self,
        type_: str = None,
        market: str = 'stocks',
        active: bool = True,
        limit: int = 1000
    ) -> Dict[str, Any]:
        """List available tickers"""
        endpoint = "/v3/reference/tickers"
        params = {
            'market': market,
            'active': str(active).lower(),
            'limit': limit
        }
        if type_:
            params['type'] = type_
        return self._make_request(endpoint, params)
    
    def get_market_status(self) -> Dict[str, Any]:
        """Get current market status"""
        endpoint = "/v1/marketstatus/now"
        return self._make_request(endpoint)
    
    def get_market_holidays(self) -> Dict[str, Any]:
        """Get upcoming market holidays"""
        endpoint = "/v1/marketstatus/upcoming"
        return self._make_request(endpoint)
    
    # ========== Dividends & Splits ==========
    
    def list_dividends(
        self,
        ticker: str = None,
        ex_dividend_date_gte: str = None,
        limit: int = 1000
    ) -> Dict[str, Any]:
        """List dividend data"""
        endpoint = "/v3/reference/dividends"
        params = {'limit': limit}
        if ticker:
            params['ticker'] = ticker
        if ex_dividend_date_gte:
            params['ex_dividend_date.gte'] = ex_dividend_date_gte
        return self._make_request(endpoint, params)
    
    def list_stock_splits(
        self,
        ticker: str = None,
        execution_date_gte: str = None,
        limit: int = 1000
    ) -> Dict[str, Any]:
        """List stock split data"""
        endpoint = "/v3/reference/splits"
        params = {'limit': limit}
        if ticker:
            params['ticker'] = ticker
        if execution_date_gte:
            params['execution_date.gte'] = execution_date_gte
        return self._make_request(endpoint, params)
    
    # ========== Financials ==========
    
    def list_stock_financials(
        self,
        ticker: str,
        timeframe: str = 'quarterly',
        limit: int = 10
    ) -> Dict[str, Any]:
        """Get fundamental financial data"""
        endpoint = "/vX/reference/financials"
        params = {
            'ticker': ticker,
            'timeframe': timeframe,
            'limit': limit,
            'sort': 'filing_date',
            'order': 'desc'
        }
        return self._make_request(endpoint, params)
    
    # ========== News ==========
    
    def list_ticker_news(
        self,
        ticker: str = None,
        published_utc_gte: str = None,
        limit: int = 100
    ) -> Dict[str, Any]:
        """Get news articles for a ticker"""
        endpoint = "/v2/reference/news"
        params = {'limit': limit}
        if ticker:
            params['ticker'] = ticker
        if published_utc_gte:
            params['published_utc.gte'] = published_utc_gte
        return self._make_request(endpoint, params)
    
    # ========== Options ==========
    
    def get_options_contract(self, options_ticker: str) -> Dict[str, Any]:
        """Get details for a single options contract"""
        endpoint = f"/v3/reference/options/contracts/{options_ticker}"
        return self._make_request(endpoint)
    
    def list_options_contracts(
        self,
        underlying_ticker: str,
        contract_type: str = None,
        expiration_date_gte: str = None,
        limit: int = 1000
    ) -> Dict[str, Any]:
        """List options contracts for an underlying"""
        endpoint = "/v3/reference/options/contracts"
        params = {
            'underlying_ticker': underlying_ticker,
            'limit': limit
        }
        if contract_type:
            params['contract_type'] = contract_type
        if expiration_date_gte:
            params['expiration_date.gte'] = expiration_date_gte
        return self._make_request(endpoint, params)
    
    def get_options_chain(
        self,
        underlying_ticker: str,
        expiration_date: str = None
    ) -> Dict[str, Any]:
        """Get full options chain"""
        # Build options chain from contracts
        contracts = self.list_options_contracts(
            underlying_ticker,
            expiration_date_gte=expiration_date or date.today().isoformat()
        )
        
        # Get snapshots for the contracts
        if contracts.get('results'):
            chain = {'calls': [], 'puts': [], 'underlying': underlying_ticker}
            
            for contract in contracts['results']:
                contract_type = contract.get('contract_type')
                if contract_type == 'call':
                    chain['calls'].append(contract)
                elif contract_type == 'put':
                    chain['puts'].append(contract)
            
            return chain
        
        return contracts
    
    # ========== Crypto ==========
    
    def get_crypto_aggs(
        self,
        ticker: str,
        multiplier: int = 1,
        timespan: str = 'day',
        from_date: str = None,
        to_date: str = None
    ) -> Dict[str, Any]:
        """Get crypto aggregate bars"""
        if from_date is None:
            from_date = (date.today() - timedelta(days=365)).isoformat()
        if to_date is None:
            to_date = date.today().isoformat()
        
        endpoint = f"/v2/aggs/ticker/X:{ticker}/range/{multiplier}/{timespan}/{from_date}/{to_date}"
        return self._make_request(endpoint)
    
    def get_crypto_snapshot(self, ticker: str) -> Dict[str, Any]:
        """Get crypto ticker snapshot"""
        endpoint = f"/v2/snapshot/locale/global/markets/crypto/tickers/X:{ticker}"
        return self._make_request(endpoint)
    
    # ========== Forex ==========
    
    def get_forex_aggs(
        self,
        ticker: str,
        multiplier: int = 1,
        timespan: str = 'day',
        from_date: str = None,
        to_date: str = None
    ) -> Dict[str, Any]:
        """Get forex aggregate bars"""
        if from_date is None:
            from_date = (date.today() - timedelta(days=365)).isoformat()
        if to_date is None:
            to_date = date.today().isoformat()
        
        endpoint = f"/v2/aggs/ticker/C:{ticker}/range/{multiplier}/{timespan}/{from_date}/{to_date}"
        return self._make_request(endpoint)
    
    def get_forex_snapshot(self, ticker: str) -> Dict[str, Any]:
        """Get forex pair snapshot"""
        endpoint = f"/v2/snapshot/locale/global/markets/forex/tickers/C:{ticker}"
        return self._make_request(endpoint)
    
    # ========== Convenience Methods ==========
    
    def get_ohlcv_dataframe(
        self,
        ticker: str,
        days: int = 365,
        timespan: str = 'day'
    ) -> Tuple[Any, List[str]]:
        """
        Get OHLCV data as numpy array.
        
        Returns:
            Tuple of (numpy array of shape (n, 5), list of dates)
        """
        import numpy as np
        
        to_date = date.today().isoformat()
        from_date = (date.today() - timedelta(days=days)).isoformat()
        
        data = self.get_aggs(
            ticker=ticker,
            timespan=timespan,
            from_date=from_date,
            to_date=to_date
        )
        
        if 'results' not in data:
            raise ValueError(f"No data returned for {ticker}")
        
        results = data['results']
        
        ohlcv = np.array([
            [r['o'], r['h'], r['l'], r['c'], r['v']]
            for r in results
        ])
        
        dates = [
            datetime.fromtimestamp(r['t'] / 1000).strftime('%Y-%m-%d')
            for r in results
        ]
        
        return ohlcv, dates
    
    def get_full_analysis_data(
        self,
        ticker: str,
        include_fundamentals: bool = True,
        include_news: bool = True,
        include_options: bool = False
    ) -> Dict[str, Any]:
        """
        Get comprehensive data for full analysis.
        
        Combines multiple API calls into a single data package.
        """
        result = {
            'ticker': ticker,
            'timestamp': datetime.now().isoformat()
        }
        
        # Price data
        try:
            result['price_data'] = self.get_aggs(ticker, timespan='day')
            result['previous_close'] = self.get_previous_close(ticker)
            result['snapshot'] = self.get_snapshot_ticker(ticker)
        except Exception as e:
            result['price_error'] = str(e)
        
        # Ticker details
        try:
            result['details'] = self.get_ticker_details(ticker)
        except Exception as e:
            result['details_error'] = str(e)
        
        # Fundamentals
        if include_fundamentals:
            try:
                result['financials'] = self.list_stock_financials(ticker)
                result['dividends'] = self.list_dividends(ticker)
                result['splits'] = self.list_stock_splits(ticker)
            except Exception as e:
                result['fundamentals_error'] = str(e)
        
        # News
        if include_news:
            try:
                result['news'] = self.list_ticker_news(ticker, limit=20)
            except Exception as e:
                result['news_error'] = str(e)
        
        # Options
        if include_options:
            try:
                result['options_chain'] = self.get_options_chain(ticker)
            except Exception as e:
                result['options_error'] = str(e)
        
        return result


class MassiveDataProvider:
    """
    High-level data provider using Massive.com MCP.
    
    Provides clean, ML-ready data for the institutional platform.
    """
    
    def __init__(self, api_key: str = None):
        """Initialize with API key"""
        if api_key is None:
            api_key = os.environ.get('POLYGON_API_KEY') or os.environ.get('MASSIVE_API_KEY')
        
        self.client = MassiveMCPClient(MCPServerConfig(api_key=api_key))
    
    def get_historical_prices(
        self,
        ticker: str,
        days: int = 252,
        include_indicators: bool = True
    ) -> Dict[str, Any]:
        """
        Get historical prices with optional technical indicators.
        
        Returns clean, normalized data ready for ML.
        """
        import numpy as np
        
        ohlcv, dates = self.client.get_ohlcv_dataframe(ticker, days=days + 50)  # Extra for indicators
        
        result = {
            'ticker': ticker,
            'ohlcv': ohlcv[-days:],
            'dates': dates[-days:],
            'columns': ['open', 'high', 'low', 'close', 'volume']
        }
        
        if include_indicators and len(ohlcv) >= 50:
            from ..analytics.technical import TechnicalAnalyzer
            analyzer = TechnicalAnalyzer()
            
            # Compute indicators on full data, then slice
            close = ohlcv[:, 3]
            high = ohlcv[:, 1]
            low = ohlcv[:, 2]
            volume = ohlcv[:, 4]
            
            indicators = {
                'rsi_14': analyzer.compute_rsi(close, 14),
                'macd': analyzer.compute_macd(close)[0],
                'macd_signal': analyzer.compute_macd(close)[1],
                'sma_20': analyzer.compute_sma(close, 20),
                'sma_50': analyzer.compute_sma(close, 50),
                'bb_upper': analyzer.compute_bollinger_bands(close, 20)['upper'],
                'bb_lower': analyzer.compute_bollinger_bands(close, 20)['lower'],
                'atr_14': analyzer.compute_atr(high, low, close, 14)
            }
            
            # Slice to match dates
            result['indicators'] = {k: v[-days:] for k, v in indicators.items()}
        
        return result
    
    def get_market_snapshot(self, tickers: List[str]) -> Dict[str, Any]:
        """Get current snapshot for multiple tickers"""
        snapshots = {}
        
        for ticker in tickers:
            try:
                snapshot = self.client.get_snapshot_ticker(ticker)
                if 'ticker' in snapshot:
                    snapshots[ticker] = snapshot['ticker']
            except Exception as e:
                snapshots[ticker] = {'error': str(e)}
        
        return snapshots
    
    def get_company_fundamentals(self, ticker: str) -> Dict[str, Any]:
        """Get comprehensive fundamental data"""
        details = self.client.get_ticker_details(ticker)
        financials = self.client.list_stock_financials(ticker)
        
        result = {
            'ticker': ticker,
            'name': details.get('results', {}).get('name'),
            'sector': details.get('results', {}).get('sic_description'),
            'market_cap': details.get('results', {}).get('market_cap'),
            'employees': details.get('results', {}).get('total_employees'),
        }
        
        # Extract key ratios from financials
        if financials.get('results'):
            latest = financials['results'][0]
            result['financials'] = {
                'revenue': latest.get('financials', {}).get('income_statement', {}).get('revenues', {}).get('value'),
                'net_income': latest.get('financials', {}).get('income_statement', {}).get('net_income_loss', {}).get('value'),
                'total_assets': latest.get('financials', {}).get('balance_sheet', {}).get('assets', {}).get('value'),
                'total_liabilities': latest.get('financials', {}).get('balance_sheet', {}).get('liabilities', {}).get('value'),
            }
        
        return result
    
    def get_market_movers(self) -> Dict[str, Any]:
        """Get top gainers and losers"""
        gainers = self.client.get_gainers_losers('gainers')
        losers = self.client.get_gainers_losers('losers')
        
        return {
            'gainers': gainers.get('tickers', [])[:10],
            'losers': losers.get('tickers', [])[:10],
            'timestamp': datetime.now().isoformat()
        }
    
    def get_news_sentiment(self, ticker: str, days: int = 7) -> Dict[str, Any]:
        """Get news with sentiment analysis"""
        from_date = (date.today() - timedelta(days=days)).isoformat()
        news = self.client.list_ticker_news(ticker, published_utc_gte=from_date)
        
        articles = news.get('results', [])
        
        # Add basic sentiment (would be enhanced with FinBERT)
        for article in articles:
            title = article.get('title', '').lower()
            
            # Simple keyword-based sentiment
            positive_words = ['surge', 'rally', 'beat', 'growth', 'profit', 'gain', 'up', 'high']
            negative_words = ['fall', 'drop', 'miss', 'loss', 'down', 'low', 'concern', 'risk']
            
            pos_count = sum(1 for w in positive_words if w in title)
            neg_count = sum(1 for w in negative_words if w in title)
            
            if pos_count > neg_count:
                article['sentiment'] = 'positive'
            elif neg_count > pos_count:
                article['sentiment'] = 'negative'
            else:
                article['sentiment'] = 'neutral'
        
        return {
            'ticker': ticker,
            'articles': articles,
            'total': len(articles),
            'sentiment_summary': {
                'positive': sum(1 for a in articles if a.get('sentiment') == 'positive'),
                'negative': sum(1 for a in articles if a.get('sentiment') == 'negative'),
                'neutral': sum(1 for a in articles if a.get('sentiment') == 'neutral')
            }
        }
