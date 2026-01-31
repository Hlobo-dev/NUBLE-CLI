"""
Real-Time Streaming Infrastructure
===================================

Production-grade WebSocket streaming for live market data.

Modules:
- realtime: WebSocket managers and stream processors
- signal_engine: Real-time trading signal generation
"""

from .realtime import (
    MarketDataStream,
    SyncMarketDataStream,
    StreamProcessor,
    StreamConfig,
    MarketTick,
    MessageType,
    StreamState,
    PolygonStream,
    AlpacaStream,
)

__all__ = [
    'MarketDataStream',
    'SyncMarketDataStream', 
    'StreamProcessor',
    'StreamConfig',
    'MarketTick',
    'MessageType',
    'StreamState',
    'PolygonStream',
    'AlpacaStream',
]
