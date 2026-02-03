"""
NUBLE ELITE - Cache Module
==============================
High-performance caching infrastructure.
"""

from .redis_cache import (
    RedisCache,
    CacheConfig,
    get_cache,
    close_cache,
    REDIS_AVAILABLE,
)

__all__ = [
    "RedisCache",
    "CacheConfig",
    "get_cache",
    "close_cache",
    "REDIS_AVAILABLE",
]
