"""
SharedDataLayer — Async-native shared data cache for all agents.

All agents read from this layer instead of making their own HTTP calls.
Data is fetched ONCE per query, shared across all agents.
Uses aiohttp for truly async HTTP (non-blocking).
"""
import asyncio
import aiohttp
import logging
import time
from typing import Dict, Any, Optional, List, Set
from datetime import datetime, timedelta
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

POLYGON_BASE = "https://api.polygon.io"
STOCKNEWS_BASE = "https://stocknewsapi.com/api/v1"
CRYPTONEWS_BASE = "https://cryptonews-api.com/api/v1"
ALTERNATIVE_ME = "https://api.alternative.me"
COINGECKO_BASE = "https://api.coingecko.com/api/v3"


@dataclass
class CacheEntry:
    data: Any
    fetched_at: float
    ttl_seconds: float = 60.0

    @property
    def is_fresh(self) -> bool:
        return (time.time() - self.fetched_at) < self.ttl_seconds


class SharedDataLayer:
    """
    Shared async data layer for all agents.

    Usage in orchestrator:
        shared = SharedDataLayer(polygon_key, stocknews_key, cryptonews_key)
        await shared.prefetch(symbols=["TSLA"], agent_types=["market_analyst", "news_analyst", ...])
        # Now pass `shared` to each agent — they read from cache, zero HTTP calls
    """

    def __init__(self, polygon_api_key: str, stocknews_api_key: str, cryptonews_api_key: str):
        self.polygon_key = polygon_api_key
        self.stocknews_key = stocknews_api_key
        self.cryptonews_key = cryptonews_api_key
        self._cache: Dict[str, CacheEntry] = {}
        self._locks: Dict[str, asyncio.Lock] = {}
        self._global_lock = asyncio.Lock()
        self._session: Optional[aiohttp.ClientSession] = None
        self._fetch_count = 0
        self._cache_hit_count = 0

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=15, connect=5)
            connector = aiohttp.TCPConnector(limit=20, limit_per_host=10)
            self._session = aiohttp.ClientSession(timeout=timeout, connector=connector)
        return self._session

    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()

    def _cache_key(self, category: str, identifier: str) -> str:
        return f"{category}:{identifier}"

    async def _get_lock(self, key: str) -> asyncio.Lock:
        async with self._global_lock:
            if key not in self._locks:
                self._locks[key] = asyncio.Lock()
            return self._locks[key]

    async def _fetch_with_cache(self, cache_key: str, fetch_coro, ttl: float = 60.0) -> Any:
        """Fetch data with deduplication. If two agents request the same data, only one HTTP call is made."""
        # Check cache first
        entry = self._cache.get(cache_key)
        if entry and entry.is_fresh:
            self._cache_hit_count += 1
            return entry.data

        # Acquire per-key lock to prevent duplicate fetches
        lock = await self._get_lock(cache_key)
        async with lock:
            # Double-check after acquiring lock
            entry = self._cache.get(cache_key)
            if entry and entry.is_fresh:
                self._cache_hit_count += 1
                return entry.data

            # Actually fetch
            try:
                self._fetch_count += 1
                data = await fetch_coro()
                self._cache[cache_key] = CacheEntry(data=data, fetched_at=time.time(), ttl_seconds=ttl)
                return data
            except Exception as e:
                logger.warning(f"Fetch failed for {cache_key}: {e}")
                # Return stale cache if available
                if entry:
                    return entry.data
                return None

    async def _http_get(self, url: str, params: Optional[Dict] = None) -> Optional[Dict]:
        """Truly async HTTP GET."""
        session = await self._get_session()
        try:
            async with session.get(url, params=params) as resp:
                if resp.status == 200:
                    return await resp.json()
                else:
                    logger.warning(f"HTTP {resp.status} from {url}")
                    return None
        except Exception as e:
            logger.warning(f"HTTP error for {url}: {e}")
            return None

    # ═══════════════════════════════════════════════════════════
    # POLYGON DATA
    # ═══════════════════════════════════════════════════════════

    async def get_quote(self, symbol: str) -> Optional[Dict]:
        """Polygon prev close. TTL: 30s."""
        key = self._cache_key("polygon_quote", symbol)
        return await self._fetch_with_cache(key, lambda: self._http_get(
            f"{POLYGON_BASE}/v2/aggs/ticker/{symbol}/prev",
            {"apiKey": self.polygon_key}
        ), ttl=30.0)

    async def get_historical(self, symbol: str, days: int = 90) -> Optional[Dict]:
        """Polygon historical OHLCV. TTL: 300s (5 min)."""
        end = datetime.now()
        start = end - timedelta(days=days)
        key = self._cache_key("polygon_historical", f"{symbol}_{days}d")
        return await self._fetch_with_cache(key, lambda: self._http_get(
            f"{POLYGON_BASE}/v2/aggs/ticker/{symbol}/range/1/day/{start.strftime('%Y-%m-%d')}/{end.strftime('%Y-%m-%d')}",
            {"apiKey": self.polygon_key, "adjusted": "true", "sort": "asc", "limit": "5000"}
        ), ttl=300.0)

    async def get_sma(self, symbol: str, window: int = 50) -> Optional[Dict]:
        """Polygon SMA indicator. TTL: 300s."""
        key = self._cache_key("polygon_sma", f"{symbol}_{window}")
        return await self._fetch_with_cache(key, lambda: self._http_get(
            f"{POLYGON_BASE}/v1/indicators/sma/{symbol}",
            {"apiKey": self.polygon_key, "timespan": "day", "window": str(window), "series_type": "close", "order": "desc", "limit": "1"}
        ), ttl=300.0)

    async def get_rsi(self, symbol: str, window: int = 14) -> Optional[Dict]:
        """Polygon RSI. TTL: 300s."""
        key = self._cache_key("polygon_rsi", f"{symbol}_{window}")
        return await self._fetch_with_cache(key, lambda: self._http_get(
            f"{POLYGON_BASE}/v1/indicators/rsi/{symbol}",
            {"apiKey": self.polygon_key, "timespan": "day", "window": str(window), "series_type": "close", "order": "desc", "limit": "1"}
        ), ttl=300.0)

    async def get_ema(self, symbol: str, window: int = 12) -> Optional[Dict]:
        """Polygon EMA. TTL: 300s."""
        key = self._cache_key("polygon_ema", f"{symbol}_{window}")
        return await self._fetch_with_cache(key, lambda: self._http_get(
            f"{POLYGON_BASE}/v1/indicators/ema/{symbol}",
            {"apiKey": self.polygon_key, "timespan": "day", "window": str(window), "series_type": "close", "order": "desc", "limit": "1"}
        ), ttl=300.0)

    async def get_macd(self, symbol: str) -> Optional[Dict]:
        """Polygon MACD. TTL: 300s."""
        key = self._cache_key("polygon_macd", symbol)
        return await self._fetch_with_cache(key, lambda: self._http_get(
            f"{POLYGON_BASE}/v1/indicators/macd/{symbol}",
            {"apiKey": self.polygon_key, "timespan": "day", "short_window": "12", "long_window": "26", "signal_window": "9", "series_type": "close", "order": "desc", "limit": "1"}
        ), ttl=300.0)

    async def get_company_info(self, symbol: str) -> Optional[Dict]:
        """Polygon ticker details. TTL: 3600s (1 hour)."""
        key = self._cache_key("polygon_company", symbol)
        return await self._fetch_with_cache(key, lambda: self._http_get(
            f"{POLYGON_BASE}/v3/reference/tickers/{symbol}",
            {"apiKey": self.polygon_key}
        ), ttl=3600.0)

    async def get_financials(self, symbol: str, timeframe: str = "quarterly", limit: int = 5) -> Optional[Dict]:
        """Polygon financials. TTL: 3600s."""
        key = self._cache_key("polygon_financials", f"{symbol}_{timeframe}_{limit}")
        return await self._fetch_with_cache(key, lambda: self._http_get(
            f"{POLYGON_BASE}/vX/reference/financials",
            {"apiKey": self.polygon_key, "ticker": symbol, "timeframe": timeframe, "limit": str(limit), "sort": "filing_date", "order": "desc"}
        ), ttl=3600.0)

    async def get_dividends(self, symbol: str) -> Optional[Dict]:
        """Polygon dividends. TTL: 3600s."""
        key = self._cache_key("polygon_dividends", symbol)
        return await self._fetch_with_cache(key, lambda: self._http_get(
            f"{POLYGON_BASE}/v3/reference/dividends",
            {"apiKey": self.polygon_key, "ticker": symbol, "limit": "12", "order": "desc"}
        ), ttl=3600.0)

    async def get_polygon_news(self, symbol: str, limit: int = 10) -> Optional[Dict]:
        """Polygon news. TTL: 120s."""
        key = self._cache_key("polygon_news", f"{symbol}_{limit}")
        return await self._fetch_with_cache(key, lambda: self._http_get(
            f"{POLYGON_BASE}/v2/reference/news",
            {"apiKey": self.polygon_key, "ticker": symbol, "limit": str(limit), "order": "desc"}
        ), ttl=120.0)

    # ═══════════════════════════════════════════════════════════
    # STOCKNEWS DATA
    # ═══════════════════════════════════════════════════════════

    async def get_stocknews(self, symbol: str, items: int = 10) -> Optional[Dict]:
        """StockNews ticker news. TTL: 120s."""
        key = self._cache_key("stocknews_ticker", f"{symbol}_{items}")
        return await self._fetch_with_cache(key, lambda: self._http_get(
            STOCKNEWS_BASE,
            {"tickers": symbol, "items": str(items), "sortby": "rank",
             "extra-fields": "id,eventid,rankscore", "token": self.stocknews_key}
        ), ttl=120.0)

    async def get_stocknews_negative(self, symbol: str, items: int = 5) -> Optional[Dict]:
        """StockNews negative sentiment filter. TTL: 120s."""
        key = self._cache_key("stocknews_negative", f"{symbol}_{items}")
        return await self._fetch_with_cache(key, lambda: self._http_get(
            STOCKNEWS_BASE,
            {"tickers": symbol, "items": str(items), "sentiment": "negative",
             "extra-fields": "id,eventid,rankscore", "token": self.stocknews_key}
        ), ttl=120.0)

    async def get_stocknews_stat(self, symbol: str, period: str = "last7days") -> Optional[Dict]:
        """StockNews sentiment stats. TTL: 300s."""
        key = self._cache_key("stocknews_stat", f"{symbol}_{period}")
        return await self._fetch_with_cache(key, lambda: self._http_get(
            f"{STOCKNEWS_BASE}/stat",
            {"tickers": symbol, "date": period, "page": "1", "token": self.stocknews_key}
        ), ttl=300.0)

    async def get_stocknews_ratings(self) -> Optional[Dict]:
        """StockNews analyst ratings. TTL: 600s."""
        key = self._cache_key("stocknews_ratings", "global")
        return await self._fetch_with_cache(key, lambda: self._http_get(
            f"{STOCKNEWS_BASE}/ratings",
            {"items": "20", "page": "1", "token": self.stocknews_key}
        ), ttl=600.0)

    async def get_stocknews_earnings(self) -> Optional[Dict]:
        """StockNews earnings calendar. TTL: 600s."""
        key = self._cache_key("stocknews_earnings", "global")
        return await self._fetch_with_cache(key, lambda: self._http_get(
            f"{STOCKNEWS_BASE}/earnings-calendar",
            {"page": "1", "items": "50", "token": self.stocknews_key}
        ), ttl=600.0)

    async def get_stocknews_events(self) -> Optional[Dict]:
        """StockNews events. TTL: 300s."""
        key = self._cache_key("stocknews_events", "global")
        return await self._fetch_with_cache(key, lambda: self._http_get(
            f"{STOCKNEWS_BASE}/events",
            {"page": "1", "token": self.stocknews_key}
        ), ttl=300.0)

    async def get_stocknews_trending(self) -> Optional[Dict]:
        """StockNews trending headlines. TTL: 120s."""
        key = self._cache_key("stocknews_trending", "global")
        return await self._fetch_with_cache(key, lambda: self._http_get(
            f"{STOCKNEWS_BASE}/trending-headlines",
            {"page": "1", "token": self.stocknews_key}
        ), ttl=120.0)

    async def get_stocknews_top_mentioned(self) -> Optional[Dict]:
        """StockNews top mentioned. TTL: 300s."""
        key = self._cache_key("stocknews_top_mentioned", "global")
        return await self._fetch_with_cache(key, lambda: self._http_get(
            f"{STOCKNEWS_BASE}/top-mention",
            {"date": "last7days", "token": self.stocknews_key}
        ), ttl=300.0)

    async def get_stocknews_sundown(self) -> Optional[Dict]:
        """StockNews sundown digest. TTL: 600s."""
        key = self._cache_key("stocknews_sundown", "global")
        return await self._fetch_with_cache(key, lambda: self._http_get(
            f"{STOCKNEWS_BASE}/sundown-digest",
            {"page": "1", "token": self.stocknews_key}
        ), ttl=600.0)

    async def get_stocknews_category(self, section: str = "general") -> Optional[Dict]:
        """StockNews category news. TTL: 120s."""
        key = self._cache_key("stocknews_category", section)
        return await self._fetch_with_cache(key, lambda: self._http_get(
            f"{STOCKNEWS_BASE}/category",
            {"section": section, "items": "10", "sortby": "rank",
             "extra-fields": "id,eventid,rankscore", "page": "1", "token": self.stocknews_key}
        ), ttl=120.0)

    async def get_stocknews_topic(self, symbol: str, topic: str, items: int = 5) -> Optional[Dict]:
        """StockNews ticker news filtered by topic. TTL: 120s."""
        key = self._cache_key("stocknews_topic", f"{symbol}_{topic}_{items}")
        return await self._fetch_with_cache(key, lambda: self._http_get(
            STOCKNEWS_BASE,
            {"tickers": symbol, "items": str(items), "topic": topic,
             "extra-fields": "id,eventid,rankscore", "token": self.stocknews_key}
        ), ttl=120.0)

    # ═══════════════════════════════════════════════════════════
    # CRYPTO DATA
    # ═══════════════════════════════════════════════════════════

    async def get_cryptonews(self, symbol: str, items: int = 10) -> Optional[Dict]:
        """CryptoNews ticker news. TTL: 120s."""
        key = self._cache_key("cryptonews_ticker", f"{symbol}_{items}")
        return await self._fetch_with_cache(key, lambda: self._http_get(
            CRYPTONEWS_BASE,
            {"tickers": symbol, "items": str(items), "sortby": "rank",
             "extra-fields": "id,eventid,rankscore", "token": self.cryptonews_key}
        ), ttl=120.0)

    async def get_cryptonews_stat(self, symbol: str, period: str = "last30days") -> Optional[Dict]:
        """CryptoNews sentiment stats. TTL: 300s."""
        key = self._cache_key("cryptonews_stat", f"{symbol}_{period}")
        return await self._fetch_with_cache(key, lambda: self._http_get(
            f"{CRYPTONEWS_BASE}/stat",
            {"tickers": symbol, "date": period, "page": "1", "token": self.cryptonews_key}
        ), ttl=300.0)

    async def get_cryptonews_trending(self) -> Optional[Dict]:
        key = self._cache_key("cryptonews_trending", "global")
        return await self._fetch_with_cache(key, lambda: self._http_get(
            f"{CRYPTONEWS_BASE}/trending-headlines",
            {"page": "1", "token": self.cryptonews_key}
        ), ttl=120.0)

    async def get_cryptonews_top_mentioned(self) -> Optional[Dict]:
        key = self._cache_key("cryptonews_top_mentioned", "global")
        return await self._fetch_with_cache(key, lambda: self._http_get(
            f"{CRYPTONEWS_BASE}/top-mention",
            {"date": "last7days", "token": self.cryptonews_key}
        ), ttl=300.0)

    async def get_cryptonews_events(self) -> Optional[Dict]:
        key = self._cache_key("cryptonews_events", "global")
        return await self._fetch_with_cache(key, lambda: self._http_get(
            f"{CRYPTONEWS_BASE}/events",
            {"page": "1", "token": self.cryptonews_key}
        ), ttl=300.0)

    async def get_cryptonews_category(self, section: str = "general") -> Optional[Dict]:
        key = self._cache_key("cryptonews_category", section)
        return await self._fetch_with_cache(key, lambda: self._http_get(
            f"{CRYPTONEWS_BASE}/category",
            {"section": section, "items": "10", "sortby": "rank",
             "extra-fields": "id,eventid,rankscore", "page": "1", "token": self.cryptonews_key}
        ), ttl=120.0)

    async def get_cryptonews_sundown(self) -> Optional[Dict]:
        key = self._cache_key("cryptonews_sundown", "global")
        return await self._fetch_with_cache(key, lambda: self._http_get(
            f"{CRYPTONEWS_BASE}/sundown-digest",
            {"page": "1", "token": self.cryptonews_key}
        ), ttl=600.0)

    # ═══════════════════════════════════════════════════════════
    # MARKET-WIDE DATA
    # ═══════════════════════════════════════════════════════════

    async def get_fear_greed(self) -> Optional[Dict]:
        """Alternative.me Fear & Greed Index. TTL: 300s."""
        key = self._cache_key("fear_greed", "global")
        return await self._fetch_with_cache(key, lambda: self._http_get(
            f"{ALTERNATIVE_ME}/fng/",
            {"limit": "7"}
        ), ttl=300.0)

    async def get_vix(self) -> Optional[Dict]:
        """VIX via Polygon prev close. TTL: 60s."""
        return await self.get_quote("VIX")

    async def get_coingecko_global(self) -> Optional[Dict]:
        """CoinGecko global crypto data. TTL: 300s."""
        key = self._cache_key("coingecko_global", "market")
        return await self._fetch_with_cache(key, lambda: self._http_get(
            f"{COINGECKO_BASE}/global"
        ), ttl=300.0)

    async def get_coingecko_coin(self, coin_id: str) -> Optional[Dict]:
        """CoinGecko coin details. TTL: 300s."""
        key = self._cache_key("coingecko_coin", coin_id)
        return await self._fetch_with_cache(key, lambda: self._http_get(
            f"{COINGECKO_BASE}/coins/{coin_id}",
            {"localization": "false", "tickers": "false", "community_data": "false", "developer_data": "false"}
        ), ttl=300.0)

    async def get_coingecko_defi(self) -> Optional[Dict]:
        """CoinGecko DeFi overview. TTL: 300s."""
        key = self._cache_key("coingecko_defi", "global")
        return await self._fetch_with_cache(key, lambda: self._http_get(
            f"{COINGECKO_BASE}/global/decentralized_finance_defi"
        ), ttl=300.0)

    # ═══════════════════════════════════════════════════════════
    # PREFETCH — The key method
    # ═══════════════════════════════════════════════════════════

    async def prefetch(self, symbols: List[str], agent_types: List[str]):
        """
        Prefetch ALL data needed by the specified agents, in parallel.
        After this completes, agents can read from cache with zero HTTP calls.

        This is the key optimization — instead of 7 agents each making 10-15 calls
        sequentially, we make ~25 unique calls in parallel via aiohttp.
        """
        tasks = []

        # Always fetch for all symbols
        for symbol in symbols:
            tasks.append(self.get_quote(symbol))
            tasks.append(self.get_historical(symbol, 90))

        # Market-wide data (needed by most agents)
        tasks.append(self.get_vix())
        tasks.append(self.get_fear_greed())

        # Agent-specific prefetch
        needs_stocknews = any(a in agent_types for a in [
            'market_analyst', 'news_analyst', 'risk_manager',
            'fundamental_analyst', 'quant_analyst', 'macro_analyst', 'portfolio_optimizer'
        ])
        needs_crypto = 'crypto_specialist' in agent_types
        needs_macro = 'macro_analyst' in agent_types
        needs_fundamental = 'fundamental_analyst' in agent_types

        for symbol in symbols:
            if needs_stocknews:
                tasks.append(self.get_stocknews(symbol))
                tasks.append(self.get_stocknews_stat(symbol))
                tasks.append(self.get_stocknews_stat(symbol, "last30days"))
                tasks.append(self.get_stocknews_ratings())
                tasks.append(self.get_stocknews_earnings())
                tasks.append(self.get_stocknews_events())
                tasks.append(self.get_stocknews_negative(symbol))
                tasks.append(self.get_stocknews_topic(symbol, "earnings"))

            if needs_fundamental:
                tasks.append(self.get_company_info(symbol))
                tasks.append(self.get_financials(symbol, "quarterly", 5))
                tasks.append(self.get_financials(symbol, "annual", 2))
                tasks.append(self.get_dividends(symbol))
                tasks.append(self.get_polygon_news(symbol))

            # Technical indicators
            tasks.append(self.get_rsi(symbol))
            tasks.append(self.get_sma(symbol, 20))
            tasks.append(self.get_sma(symbol, 50))
            tasks.append(self.get_sma(symbol, 200))
            tasks.append(self.get_ema(symbol, 12))
            tasks.append(self.get_ema(symbol, 26))
            tasks.append(self.get_macd(symbol))

        # Market-wide StockNews (deduplicated — called once, not per agent)
        if needs_stocknews:
            tasks.append(self.get_stocknews_trending())
            tasks.append(self.get_stocknews_top_mentioned())
            tasks.append(self.get_stocknews_sundown())
            tasks.append(self.get_stocknews_category("general"))

        # Macro data
        if needs_macro:
            macro_etfs = ['SPY', 'QQQ', 'IWM', 'DIA', 'XLK', 'XLF', 'XLE', 'XLV',
                          'XLI', 'XLY', 'XLP', 'XLU', 'XLRE', 'XLC', 'XLB',
                          'SHY', 'IEF', 'TLT',
                          'GLD', 'USO', 'UUP']
            for etf in macro_etfs:
                tasks.append(self.get_quote(etf))
                tasks.append(self.get_sma(etf, 50))

        # Crypto data
        if needs_crypto:
            tasks.append(self.get_coingecko_global())
            tasks.append(self.get_coingecko_defi())
            tasks.append(self.get_cryptonews_trending())
            tasks.append(self.get_cryptonews_top_mentioned())
            tasks.append(self.get_cryptonews_events())
            tasks.append(self.get_cryptonews_category("general"))
            tasks.append(self.get_cryptonews_sundown())
            for symbol in symbols:
                tasks.append(self.get_cryptonews(symbol))
                tasks.append(self.get_cryptonews_stat(symbol))

        # Fire ALL requests in parallel via aiohttp
        logger.info(f"SharedDataLayer: prefetching {len(tasks)} data points for {symbols}")
        start = time.time()
        await asyncio.gather(*tasks, return_exceptions=True)
        elapsed = time.time() - start
        logger.info(f"SharedDataLayer: prefetch complete in {elapsed:.1f}s "
                     f"({self._fetch_count} fetches, {self._cache_hit_count} cache hits)")

    def get_stats(self) -> Dict[str, Any]:
        return {
            "total_fetches": self._fetch_count,
            "cache_hits": self._cache_hit_count,
            "cache_entries": len(self._cache),
            "hit_rate": self._cache_hit_count / max(1, self._fetch_count + self._cache_hit_count),
        }


__all__ = ['SharedDataLayer']
