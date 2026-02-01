"""
Asset Class Detection

Automatically detects asset class from symbol.
Supports: Stocks, ETFs, Crypto, Commodities, Forex
"""

from enum import Enum
from typing import Optional, Dict, Any
import re


class AssetClass(Enum):
    """Supported asset classes."""
    STOCK = "stock"
    ETF = "etf"
    CRYPTO = "crypto"
    COMMODITY = "commodity"
    FOREX = "forex"
    UNKNOWN = "unknown"


class AssetDetector:
    """
    Detects asset class from symbol.
    
    Usage:
        detector = AssetDetector()
        asset_class = detector.detect('AAPL')  # AssetClass.STOCK
        asset_class = detector.detect('BTC')   # AssetClass.CRYPTO
        asset_class = detector.detect('SPY')   # AssetClass.ETF
    """
    
    # Known crypto symbols
    CRYPTO_SYMBOLS = {
        # Major cryptocurrencies
        'BTC', 'ETH', 'XRP', 'LTC', 'BCH', 'ADA', 'DOT', 'LINK', 'BNB',
        'SOL', 'AVAX', 'MATIC', 'ATOM', 'UNI', 'AAVE', 'DOGE', 'SHIB',
        'XLM', 'ALGO', 'VET', 'FIL', 'THETA', 'EOS', 'TRX', 'XMR',
        'NEO', 'IOTA', 'DASH', 'ZEC', 'ETC', 'XTZ', 'MKR', 'COMP',
        'SNX', 'YFI', 'SUSHI', 'CRV', 'RUNE', 'NEAR', 'FTM', 'ONE',
        'HBAR', 'EGLD', 'KSM', 'FLOW', 'AR', 'MINA', 'ICP', 'APE',
        'OP', 'ARB', 'SUI', 'SEI', 'TIA', 'JUP', 'WIF', 'PEPE',
        # Stablecoins
        'USDT', 'USDC', 'DAI', 'BUSD', 'UST', 'FRAX', 'TUSD',
    }
    
    # Crypto trading pairs (ends with USD, USDT, BTC, ETH)
    CRYPTO_PAIR_SUFFIXES = ['USD', 'USDT', 'USDC', 'BTC', 'ETH', 'BUSD']
    
    # Known ETF symbols
    ETF_SYMBOLS = {
        # Major index ETFs
        'SPY', 'QQQ', 'IWM', 'DIA', 'VTI', 'VOO', 'IVV', 'VEA', 'VWO',
        'EEM', 'EFA', 'AGG', 'BND', 'LQD', 'TLT', 'IEF', 'SHY', 'TIP',
        # Sector ETFs
        'XLK', 'XLF', 'XLE', 'XLV', 'XLI', 'XLP', 'XLY', 'XLB', 'XLU', 'XLRE',
        'VGT', 'VHT', 'VNQ', 'VFH', 'VDE', 'VIS', 'VAW', 'VCR', 'VDC', 'VPU',
        # Thematic ETFs
        'ARKK', 'ARKG', 'ARKW', 'ARKF', 'ARKQ', 'ARKX',
        'SMH', 'SOXX', 'IBB', 'XBI', 'HACK', 'ROBO', 'BOTZ',
        # Leveraged ETFs
        'TQQQ', 'SQQQ', 'SPXL', 'SPXS', 'UPRO', 'SDOW', 'UDOW',
        # Commodity ETFs
        'GLD', 'SLV', 'USO', 'UNG', 'DBA', 'DBC', 'PDBC', 'CORN', 'WEAT', 'SOYB',
        # Volatility ETFs
        'VXX', 'UVXY', 'SVXY', 'VIXY',
        # Bond ETFs
        'HYG', 'JNK', 'EMB', 'MUB', 'VCSH', 'VCIT', 'VCLT',
        # International ETFs
        'FXI', 'INDA', 'EWJ', 'EWZ', 'EWG', 'EWU', 'EWC', 'EWA',
    }
    
    # Commodity ETFs (subset of ETFs that track commodities)
    COMMODITY_ETFS = {
        'GLD', 'SLV', 'USO', 'UNG', 'DBA', 'DBC', 'PDBC', 'CORN', 'WEAT', 'SOYB',
        'GDX', 'GDXJ', 'SIL', 'PPLT', 'PALL', 'CPER', 'JJC', 'JJN',
    }
    
    # Forex pairs
    FOREX_CURRENCIES = ['EUR', 'GBP', 'USD', 'JPY', 'CHF', 'AUD', 'NZD', 'CAD']
    
    def __init__(self):
        """Initialize detector with lookup tables."""
        self._crypto_set = self.CRYPTO_SYMBOLS.copy()
        self._etf_set = self.ETF_SYMBOLS.copy()
        self._commodity_set = self.COMMODITY_ETFS.copy()
    
    def detect(self, symbol: str) -> AssetClass:
        """
        Detect asset class from symbol.
        
        Args:
            symbol: Ticker/symbol to classify
            
        Returns:
            AssetClass enum value
        """
        symbol = symbol.upper().strip()
        
        # Check crypto first
        if self._is_crypto(symbol):
            return AssetClass.CRYPTO
        
        # Check forex
        if self._is_forex(symbol):
            return AssetClass.FOREX
        
        # Check commodity ETFs
        if symbol in self._commodity_set:
            return AssetClass.COMMODITY
        
        # Check ETFs
        if symbol in self._etf_set:
            return AssetClass.ETF
        
        # Default to stock (most common case)
        return AssetClass.STOCK
    
    def _is_crypto(self, symbol: str) -> bool:
        """Check if symbol is a cryptocurrency."""
        # Direct match
        if symbol in self._crypto_set:
            return True
        
        # Check for trading pair format
        for suffix in self.CRYPTO_PAIR_SUFFIXES:
            if symbol.endswith(suffix) and len(symbol) > len(suffix):
                base = symbol[:-len(suffix)]
                if base in self._crypto_set or len(base) >= 2:
                    # If base is known crypto or looks like a crypto symbol
                    return True
        
        # Check for common patterns
        if symbol.endswith('PERP') or symbol.endswith('SWAP'):
            return True
        
        return False
    
    def _is_forex(self, symbol: str) -> bool:
        """Check if symbol is a forex pair."""
        if len(symbol) != 6:
            return False
        
        base = symbol[:3]
        quote = symbol[3:]
        
        return base in self.FOREX_CURRENCIES and quote in self.FOREX_CURRENCIES
    
    def get_config(self, symbol: str) -> Dict[str, Any]:
        """
        Get configuration for an asset.
        
        Args:
            symbol: Ticker/symbol
            
        Returns:
            Dict with data sources, features, model type, etc.
        """
        asset_class = self.detect(symbol)
        
        configs = {
            AssetClass.STOCK: {
                'data_source': 'polygon',
                'news_source': 'stocknews_api',
                'features': ['technicals', 'fundamentals', 'sentiment', 'options_flow'],
                'model_type': 'ensemble_classifier',
                'trading_hours': 'market_hours',
                'min_volume': 1000000,
            },
            AssetClass.ETF: {
                'data_source': 'polygon',
                'news_source': 'stocknews_api',
                'features': ['technicals', 'flows', 'holdings', 'sector_sentiment'],
                'model_type': 'regime_aware_classifier',
                'trading_hours': 'market_hours',
                'min_volume': 5000000,
            },
            AssetClass.CRYPTO: {
                'data_source': 'polygon_crypto',
                'news_source': 'cryptonews_api',
                'features': ['technicals', 'on_chain', 'social_sentiment', 'funding_rates'],
                'model_type': 'high_frequency_ensemble',
                'trading_hours': '24/7',
                'min_volume': 10000000,
            },
            AssetClass.COMMODITY: {
                'data_source': 'polygon',
                'news_source': 'stocknews_api',
                'features': ['technicals', 'seasonality', 'inventory', 'macro'],
                'model_type': 'regime_aware_classifier',
                'trading_hours': 'market_hours',
                'min_volume': 1000000,
            },
            AssetClass.FOREX: {
                'data_source': 'polygon_forex',
                'news_source': 'macro_news',
                'features': ['technicals', 'interest_rate_diff', 'macro_indicators'],
                'model_type': 'carry_momentum_hybrid',
                'trading_hours': '24/5',
                'min_volume': 0,  # Forex always liquid
            },
            AssetClass.UNKNOWN: {
                'data_source': 'polygon',
                'news_source': 'stocknews_api',
                'features': ['technicals'],
                'model_type': 'ensemble_classifier',
                'trading_hours': 'market_hours',
                'min_volume': 100000,
            },
        }
        
        config = configs.get(asset_class, configs[AssetClass.UNKNOWN])
        config['asset_class'] = asset_class
        config['symbol'] = symbol
        
        return config
    
    def add_crypto(self, symbol: str):
        """Add a crypto symbol to the known list."""
        self._crypto_set.add(symbol.upper())
    
    def add_etf(self, symbol: str):
        """Add an ETF symbol to the known list."""
        self._etf_set.add(symbol.upper())
    
    def add_commodity(self, symbol: str):
        """Add a commodity symbol to the known list."""
        self._commodity_set.add(symbol.upper())


# Convenience function
def detect_asset_class(symbol: str) -> AssetClass:
    """Quick detection of asset class."""
    return AssetDetector().detect(symbol)


def get_asset_config(symbol: str) -> Dict[str, Any]:
    """Quick config lookup for asset."""
    return AssetDetector().get_config(symbol)


# Test
if __name__ == "__main__":
    detector = AssetDetector()
    
    test_symbols = [
        'AAPL', 'MSFT', 'NVDA',  # Stocks
        'SPY', 'QQQ', 'ARKK',    # ETFs
        'BTC', 'ETH', 'SOL',     # Crypto
        'BTCUSD', 'ETHUSD',      # Crypto pairs
        'GLD', 'SLV', 'USO',     # Commodities
        'EURUSD', 'GBPJPY',      # Forex
    ]
    
    print("Asset Detection Test")
    print("=" * 40)
    
    for symbol in test_symbols:
        asset_class = detector.detect(symbol)
        print(f"  {symbol:10} -> {asset_class.value}")
