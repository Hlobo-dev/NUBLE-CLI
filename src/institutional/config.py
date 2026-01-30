"""
Configuration management for the Institutional Research Platform.
Handles API keys, rate limits, caching settings, and provider configurations.
"""

import os
import json
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Dict, Any
from enum import Enum


class DataTier(Enum):
    """Data quality/latency tiers"""
    REALTIME = "realtime"      # Sub-second, exchange direct
    DELAYED = "delayed"        # 15-min delayed
    END_OF_DAY = "eod"         # Daily snapshots
    HISTORICAL = "historical"  # Archived data


class ProviderPriority(Enum):
    """Provider fallback priority"""
    PRIMARY = 1
    SECONDARY = 2
    TERTIARY = 3
    BACKUP = 4


@dataclass
class ProviderConfig:
    """Configuration for a single data provider"""
    name: str
    api_key: Optional[str] = None
    base_url: str = ""
    rate_limit_per_minute: int = 60
    rate_limit_per_day: int = 10000
    timeout_seconds: int = 30
    retry_attempts: int = 3
    priority: ProviderPriority = ProviderPriority.PRIMARY
    enabled: bool = True
    tier: DataTier = DataTier.DELAYED
    
    # Provider-specific settings
    extra_settings: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CacheConfig:
    """Caching configuration"""
    enabled: bool = True
    memory_cache_size_mb: int = 512
    disk_cache_enabled: bool = True
    disk_cache_path: str = "~/.institutional_cache"
    
    # TTL settings by data type (in seconds)
    ttl_quotes: int = 5              # Real-time quotes
    ttl_trades: int = 1              # Trade data
    ttl_daily_bars: int = 3600       # Daily OHLCV
    ttl_fundamentals: int = 86400    # Fundamentals (1 day)
    ttl_news: int = 300              # News (5 min)
    ttl_filings: int = 86400         # SEC filings (1 day)
    ttl_analytics: int = 60          # Computed analytics


@dataclass
class MLConfig:
    """Machine Learning configuration"""
    enabled: bool = True
    device: str = "auto"  # auto, cpu, cuda, mps
    
    # Model paths
    sentiment_model: str = "ProsusAI/finbert"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    
    # Pattern recognition
    pattern_detection_enabled: bool = True
    min_pattern_confidence: float = 0.75
    
    # Anomaly detection
    anomaly_detection_enabled: bool = True
    anomaly_sensitivity: float = 0.95
    
    # Predictive models
    forecasting_enabled: bool = True
    forecast_horizon_days: int = 5


@dataclass
class LLMConfig:
    """LLM configuration for synthesis"""
    provider: str = "anthropic"  # openai, anthropic, local
    model: str = "claude-opus-4-5-20250514"
    api_key: Optional[str] = None
    temperature: float = 0.3
    max_tokens: int = 4096
    
    # Anthropic models
    claude_opus: str = "claude-opus-4-5-20250514"
    claude_sonnet: str = "claude-sonnet-4-5-20250929"
    claude_haiku: str = "claude-haiku-4-5-20251001"
    
    # OpenAI fallback models
    openai_model: str = "gpt-4o"
    fallback_model: str = "gpt-4o-mini"
    
    # Cost tracking
    track_usage: bool = True
    monthly_budget_usd: float = 100.0


class Config:
    """Main configuration manager"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = Path(config_path or "~/.institutional/config.json").expanduser()
        self._load_config()
    
    # Convenience properties for easy API key access
    @property
    def openai_api_key(self) -> Optional[str]:
        """Get OpenAI API key"""
        return self.llm.api_key if hasattr(self, 'llm') else os.getenv("OPENAI_API_KEY")
    
    @property
    def anthropic_api_key(self) -> Optional[str]:
        """Get Anthropic/Claude API key"""
        return os.getenv("ANTHROPIC_API_KEY")
    
    @property
    def polygon_api_key(self) -> Optional[str]:
        """Get Polygon.io API key"""
        if "polygon" in self.providers:
            return self.providers["polygon"].api_key
        return os.getenv("POLYGON_API_KEY")
    
    @property
    def alpha_vantage_api_key(self) -> Optional[str]:
        """Get Alpha Vantage API key"""
        if "alpha_vantage" in self.providers:
            return self.providers["alpha_vantage"].api_key
        return os.getenv("ALPHA_VANTAGE_API_KEY")
    
    @property
    def finnhub_api_key(self) -> Optional[str]:
        """Get Finnhub API key"""
        if "finnhub" in self.providers:
            return self.providers["finnhub"].api_key
        return os.getenv("FINNHUB_API_KEY")
    
    def _load_config(self):
        """Load configuration from file and environment"""
        
        # Default provider configurations
        self.providers = {
            "polygon": ProviderConfig(
                name="polygon",
                api_key=os.getenv("POLYGON_API_KEY"),
                base_url="https://api.polygon.io",
                rate_limit_per_minute=100,
                priority=ProviderPriority.PRIMARY,
                tier=DataTier.REALTIME,
                extra_settings={
                    "websocket_url": "wss://socket.polygon.io",
                    "feed_type": "delayed"  # basic, delayed, realtime
                }
            ),
            "alpha_vantage": ProviderConfig(
                name="alpha_vantage",
                api_key=os.getenv("ALPHA_VANTAGE_API_KEY"),
                base_url="https://www.alphavantage.co/query",
                rate_limit_per_minute=75,
                priority=ProviderPriority.PRIMARY,
                tier=DataTier.DELAYED,
                extra_settings={
                    "premium": False
                }
            ),
            "finnhub": ProviderConfig(
                name="finnhub",
                api_key=os.getenv("FINNHUB_API_KEY"),
                base_url="https://finnhub.io/api/v1",
                rate_limit_per_minute=60,
                priority=ProviderPriority.PRIMARY,
                tier=DataTier.REALTIME,
                extra_settings={
                    "websocket_url": "wss://ws.finnhub.io"
                }
            ),
            "sec_edgar": ProviderConfig(
                name="sec_edgar",
                api_key=None,  # No key required
                base_url="https://data.sec.gov",
                rate_limit_per_minute=10,  # SEC fair use policy
                priority=ProviderPriority.PRIMARY,
                tier=DataTier.END_OF_DAY,
                extra_settings={
                    "user_agent": os.getenv("SEC_USER_AGENT", "InstitutionalResearch/1.0"),
                    "company_tickers_url": "https://www.sec.gov/files/company_tickers.json"
                }
            ),
            "yfinance": ProviderConfig(
                name="yfinance",
                api_key=None,  # No key required
                base_url="",
                rate_limit_per_minute=100,
                priority=ProviderPriority.BACKUP,
                tier=DataTier.DELAYED,
                extra_settings={}
            )
        }
        
        # Cache configuration
        self.cache = CacheConfig()
        
        # ML configuration
        self.ml = MLConfig()
        
        # LLM configuration
        self.llm = LLMConfig(
            api_key=os.getenv("OPENAI_API_KEY")
        )
        
        # Load custom config from file if exists
        if self.config_path.exists():
            self._merge_file_config()
    
    def _merge_file_config(self):
        """Merge configuration from file"""
        try:
            with open(self.config_path) as f:
                file_config = json.load(f)
            
            # Merge provider configs
            if "providers" in file_config:
                for name, settings in file_config["providers"].items():
                    if name in self.providers:
                        for key, value in settings.items():
                            if hasattr(self.providers[name], key):
                                setattr(self.providers[name], key, value)
            
            # Merge cache config
            if "cache" in file_config:
                for key, value in file_config["cache"].items():
                    if hasattr(self.cache, key):
                        setattr(self.cache, key, value)
            
            # Merge ML config
            if "ml" in file_config:
                for key, value in file_config["ml"].items():
                    if hasattr(self.ml, key):
                        setattr(self.ml, key, value)
                        
        except Exception as e:
            print(f"Warning: Could not load config file: {e}")
    
    def save_config(self):
        """Save current configuration to file"""
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        
        config_dict = {
            "providers": {
                name: {
                    k: v for k, v in vars(cfg).items() 
                    if k != "api_key" and not k.startswith("_")
                }
                for name, cfg in self.providers.items()
            },
            "cache": {k: v for k, v in vars(self.cache).items()},
            "ml": {k: v for k, v in vars(self.ml).items() if k != "api_key"}
        }
        
        with open(self.config_path, "w") as f:
            json.dump(config_dict, f, indent=2, default=str)
    
    def get_provider(self, name: str) -> Optional[ProviderConfig]:
        """Get provider configuration by name"""
        return self.providers.get(name)
    
    def get_enabled_providers(self) -> Dict[str, ProviderConfig]:
        """Get all enabled providers"""
        return {
            name: cfg for name, cfg in self.providers.items() 
            if cfg.enabled and cfg.api_key is not None
        }
    
    def validate(self) -> Dict[str, bool]:
        """Validate configuration and return status for each provider"""
        status = {}
        
        for name, cfg in self.providers.items():
            if cfg.name == "sec_edgar" or cfg.name == "yfinance":
                # These don't require API keys
                status[name] = cfg.enabled
            else:
                status[name] = cfg.enabled and cfg.api_key is not None
        
        status["llm"] = self.llm.api_key is not None
        
        return status


# Global config instance
_config: Optional[Config] = None


def get_config() -> Config:
    """Get global configuration instance"""
    global _config
    if _config is None:
        _config = Config()
    return _config


def load_config(config_path: Optional[str] = None) -> Config:
    """Load configuration (alias for get_config with optional path)"""
    global _config
    if config_path:
        _config = Config(config_path)
    elif _config is None:
        _config = Config()
    return _config


def init_config(config_path: Optional[str] = None) -> Config:
    """Initialize configuration with custom path"""
    global _config
    _config = Config(config_path)
    return _config
