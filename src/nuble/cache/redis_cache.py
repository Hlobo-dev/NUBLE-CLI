"""
NUBLE ELITE - Redis Cache Layer
====================================
Ultra-low latency caching for signals and decisions.
Target: <1ms cache operations

Features:
- Signal caching with TTL by timeframe
- Decision caching for repeated queries
- Connection pooling for high throughput
- Graceful fallback if Redis unavailable

Author: NUBLE ELITE System
Version: 1.0.0
"""

import os
import json
import asyncio
import logging
from typing import Optional, Dict, Any, List
from datetime import datetime, timezone
from dataclasses import dataclass, asdict
from contextlib import asynccontextmanager

logger = logging.getLogger(__name__)

# Try to import redis, gracefully degrade if not available
try:
    import redis.asyncio as aioredis
    from redis.asyncio import ConnectionPool
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    logger.warning("Redis not available - caching disabled")


@dataclass
class CacheConfig:
    """Redis cache configuration."""
    host: str = "localhost"
    port: int = 6379
    password: Optional[str] = None
    db: int = 0
    max_connections: int = 20
    socket_timeout: float = 1.0
    socket_connect_timeout: float = 1.0
    decode_responses: bool = True
    # TTL settings (seconds)
    signal_ttl_weekly: int = 604800  # 7 days
    signal_ttl_daily: int = 86400    # 24 hours
    signal_ttl_4h: int = 28800       # 8 hours
    signal_ttl_1h: int = 7200        # 2 hours
    decision_ttl: int = 300          # 5 minutes
    alignment_ttl: int = 60          # 1 minute


class RedisCache:
    """
    High-performance Redis cache for NUBLE signals.
    
    Features:
    - Async operations for non-blocking I/O
    - Connection pooling
    - Automatic serialization/deserialization
    - Graceful degradation if Redis unavailable
    """
    
    def __init__(self, config: Optional[CacheConfig] = None):
        """Initialize Redis cache with configuration."""
        self.config = config or CacheConfig(
            host=os.getenv("REDIS_HOST", "localhost"),
            port=int(os.getenv("REDIS_PORT", "6379")),
            password=os.getenv("REDIS_PASSWORD"),
        )
        self._pool: Optional[ConnectionPool] = None
        self._redis: Optional[aioredis.Redis] = None
        self._connected = False
        
        # Key prefixes
        self.PREFIX_SIGNAL = "nuble:signal"
        self.PREFIX_DECISION = "nuble:decision"
        self.PREFIX_ALIGNMENT = "nuble:alignment"
        self.PREFIX_VETO = "nuble:veto"
    
    async def connect(self) -> bool:
        """Establish Redis connection."""
        if not REDIS_AVAILABLE:
            logger.warning("Redis not available - operating in memory-only mode")
            return False
        
        try:
            self._pool = ConnectionPool(
                host=self.config.host,
                port=self.config.port,
                password=self.config.password,
                db=self.config.db,
                max_connections=self.config.max_connections,
                socket_timeout=self.config.socket_timeout,
                socket_connect_timeout=self.config.socket_connect_timeout,
                decode_responses=self.config.decode_responses,
            )
            self._redis = aioredis.Redis(connection_pool=self._pool)
            
            # Test connection
            await self._redis.ping()
            self._connected = True
            logger.info(f"Connected to Redis at {self.config.host}:{self.config.port}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            self._connected = False
            return False
    
    async def disconnect(self) -> None:
        """Close Redis connection."""
        if self._redis:
            await self._redis.close()
        if self._pool:
            await self._pool.disconnect()
        self._connected = False
        logger.info("Disconnected from Redis")
    
    @asynccontextmanager
    async def connection(self):
        """Context manager for Redis connection."""
        await self.connect()
        try:
            yield self
        finally:
            await self.disconnect()
    
    def _get_signal_ttl(self, timeframe: str) -> int:
        """Get TTL for signal based on timeframe."""
        ttl_map = {
            "1w": self.config.signal_ttl_weekly,
            "1d": self.config.signal_ttl_daily,
            "4h": self.config.signal_ttl_4h,
            "1h": self.config.signal_ttl_1h,
        }
        return ttl_map.get(timeframe.lower(), self.config.signal_ttl_4h)
    
    def _make_signal_key(self, symbol: str, timeframe: str, source: str) -> str:
        """Generate cache key for signal."""
        return f"{self.PREFIX_SIGNAL}:{symbol}:{timeframe}:{source}"
    
    def _make_decision_key(self, symbol: str) -> str:
        """Generate cache key for decision."""
        return f"{self.PREFIX_DECISION}:{symbol}"
    
    def _make_alignment_key(self, symbol: str) -> str:
        """Generate cache key for alignment."""
        return f"{self.PREFIX_ALIGNMENT}:{symbol}"
    
    async def cache_signal(
        self,
        symbol: str,
        timeframe: str,
        source: str,
        signal_data: Dict[str, Any]
    ) -> bool:
        """
        Cache a signal with appropriate TTL.
        
        Args:
            symbol: Trading symbol
            timeframe: Chart timeframe
            source: Signal source
            signal_data: Signal data to cache
        
        Returns:
            True if cached successfully
        """
        if not self._connected:
            return False
        
        try:
            key = self._make_signal_key(symbol, timeframe, source)
            ttl = self._get_signal_ttl(timeframe)
            
            # Add timestamp if not present
            if "cached_at" not in signal_data:
                signal_data["cached_at"] = datetime.now(timezone.utc).isoformat()
            
            await self._redis.setex(key, ttl, json.dumps(signal_data))
            logger.debug(f"Cached signal: {key} (TTL: {ttl}s)")
            return True
            
        except Exception as e:
            logger.error(f"Failed to cache signal: {e}")
            return False
    
    async def get_signal(
        self,
        symbol: str,
        timeframe: str,
        source: str
    ) -> Optional[Dict[str, Any]]:
        """
        Retrieve cached signal.
        
        Args:
            symbol: Trading symbol
            timeframe: Chart timeframe
            source: Signal source
        
        Returns:
            Signal data if found, None otherwise
        """
        if not self._connected:
            return None
        
        try:
            key = self._make_signal_key(symbol, timeframe, source)
            data = await self._redis.get(key)
            
            if data:
                logger.debug(f"Cache hit: {key}")
                return json.loads(data)
            
            logger.debug(f"Cache miss: {key}")
            return None
            
        except Exception as e:
            logger.error(f"Failed to get signal: {e}")
            return None
    
    async def get_all_signals(self, symbol: str) -> Dict[str, Dict[str, Any]]:
        """
        Get all cached signals for a symbol across timeframes.
        
        Args:
            symbol: Trading symbol
        
        Returns:
            Dict mapping "timeframe:source" to signal data
        """
        if not self._connected:
            return {}
        
        try:
            pattern = f"{self.PREFIX_SIGNAL}:{symbol}:*"
            signals = {}
            
            async for key in self._redis.scan_iter(pattern):
                data = await self._redis.get(key)
                if data:
                    # Extract timeframe:source from key
                    parts = key.split(":")
                    if len(parts) >= 4:
                        tf_source = f"{parts[3]}:{parts[4]}"
                        signals[tf_source] = json.loads(data)
            
            return signals
            
        except Exception as e:
            logger.error(f"Failed to get all signals: {e}")
            return {}
    
    async def cache_decision(
        self,
        symbol: str,
        decision_data: Dict[str, Any]
    ) -> bool:
        """
        Cache a trading decision.
        
        Args:
            symbol: Trading symbol
            decision_data: Decision data to cache
        
        Returns:
            True if cached successfully
        """
        if not self._connected:
            return False
        
        try:
            key = self._make_decision_key(symbol)
            decision_data["cached_at"] = datetime.now(timezone.utc).isoformat()
            
            await self._redis.setex(
                key,
                self.config.decision_ttl,
                json.dumps(decision_data, default=str)
            )
            logger.debug(f"Cached decision: {key}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to cache decision: {e}")
            return False
    
    async def get_decision(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Get cached decision for a symbol.
        
        Args:
            symbol: Trading symbol
        
        Returns:
            Decision data if found, None otherwise
        """
        if not self._connected:
            return None
        
        try:
            key = self._make_decision_key(symbol)
            data = await self._redis.get(key)
            
            if data:
                return json.loads(data)
            return None
            
        except Exception as e:
            logger.error(f"Failed to get decision: {e}")
            return None
    
    async def cache_alignment(
        self,
        symbol: str,
        alignment_data: Dict[str, Any]
    ) -> bool:
        """Cache alignment status for a symbol."""
        if not self._connected:
            return False
        
        try:
            key = self._make_alignment_key(symbol)
            alignment_data["cached_at"] = datetime.now(timezone.utc).isoformat()
            
            await self._redis.setex(
                key,
                self.config.alignment_ttl,
                json.dumps(alignment_data, default=str)
            )
            return True
            
        except Exception as e:
            logger.error(f"Failed to cache alignment: {e}")
            return False
    
    async def get_alignment(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get cached alignment for a symbol."""
        if not self._connected:
            return None
        
        try:
            key = self._make_alignment_key(symbol)
            data = await self._redis.get(key)
            
            if data:
                return json.loads(data)
            return None
            
        except Exception as e:
            logger.error(f"Failed to get alignment: {e}")
            return None
    
    async def invalidate_symbol(self, symbol: str) -> int:
        """
        Invalidate all cache entries for a symbol.
        
        Args:
            symbol: Trading symbol
        
        Returns:
            Number of keys deleted
        """
        if not self._connected:
            return 0
        
        try:
            # Find all keys for this symbol
            patterns = [
                f"{self.PREFIX_SIGNAL}:{symbol}:*",
                f"{self.PREFIX_DECISION}:{symbol}",
                f"{self.PREFIX_ALIGNMENT}:{symbol}",
                f"{self.PREFIX_VETO}:{symbol}",
            ]
            
            deleted = 0
            for pattern in patterns:
                async for key in self._redis.scan_iter(pattern):
                    await self._redis.delete(key)
                    deleted += 1
            
            logger.info(f"Invalidated {deleted} cache entries for {symbol}")
            return deleted
            
        except Exception as e:
            logger.error(f"Failed to invalidate cache: {e}")
            return 0
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        if not self._connected:
            return {"connected": False}
        
        try:
            info = await self._redis.info()
            
            return {
                "connected": True,
                "used_memory": info.get("used_memory_human"),
                "connected_clients": info.get("connected_clients"),
                "total_commands_processed": info.get("total_commands_processed"),
                "keyspace_hits": info.get("keyspace_hits"),
                "keyspace_misses": info.get("keyspace_misses"),
                "hit_rate": round(
                    info.get("keyspace_hits", 0) / 
                    max(1, info.get("keyspace_hits", 0) + info.get("keyspace_misses", 0)) * 100,
                    2
                ),
            }
            
        except Exception as e:
            logger.error(f"Failed to get stats: {e}")
            return {"connected": False, "error": str(e)}
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on Redis connection."""
        if not self._connected:
            return {
                "status": "disconnected",
                "latency_ms": None,
            }
        
        try:
            import time
            start = time.perf_counter()
            await self._redis.ping()
            latency = (time.perf_counter() - start) * 1000
            
            return {
                "status": "healthy",
                "latency_ms": round(latency, 2),
            }
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
            }


# Global cache instance
_cache: Optional[RedisCache] = None


async def get_cache() -> RedisCache:
    """Get or create global cache instance."""
    global _cache
    
    if _cache is None:
        _cache = RedisCache()
        await _cache.connect()
    
    return _cache


async def close_cache() -> None:
    """Close global cache instance."""
    global _cache
    
    if _cache is not None:
        await _cache.disconnect()
        _cache = None


# Example usage
if __name__ == "__main__":
    async def main():
        cache = RedisCache()
        
        async with cache.connection():
            # Test signal caching
            signal = {
                "action": "BUY",
                "confidence": 85,
                "price": 175.50,
            }
            
            await cache.cache_signal("AAPL", "4h", "luxalgo", signal)
            
            # Retrieve signal
            cached = await cache.get_signal("AAPL", "4h", "luxalgo")
            print(f"Cached signal: {cached}")
            
            # Get stats
            stats = await cache.get_stats()
            print(f"Cache stats: {stats}")
            
            # Health check
            health = await cache.health_check()
            print(f"Health: {health}")
    
    asyncio.run(main())
