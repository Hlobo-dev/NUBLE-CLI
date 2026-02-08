"""
Background task that resolves outstanding predictions by checking current prices.
Runs every hour, checks predictions from 1d, 5d, and 20d ago.

Started by the API server on startup. Uses Polygon to fetch current prices.
"""
import asyncio
import logging
import os
from datetime import datetime
from typing import Optional
import requests

logger = logging.getLogger(__name__)

POLYGON_BASE = "https://api.polygon.io"


class PredictionResolver:
    """
    Background async task that periodically resolves outstanding predictions
    by fetching current prices and comparing against recorded entry prices.
    """

    def __init__(self, learning_hub, polygon_api_key: str = None):
        self.hub = learning_hub
        self.api_key = polygon_api_key or os.environ.get(
            'POLYGON_API_KEY', 'JHKwAdyIOeExkYOxh3LwTopmqqVVFeBY'
        )
        self._running = False
        self._task: Optional[asyncio.Task] = None

    def start(self):
        """Start the background resolver loop."""
        if not self._running:
            self._running = True
            self._task = asyncio.create_task(self._resolve_loop())
            logger.info("PredictionResolver started (runs every 3600s)")

    def stop(self):
        """Stop the background resolver."""
        self._running = False
        if self._task:
            self._task.cancel()
            logger.info("PredictionResolver stopped")

    async def _resolve_loop(self):
        """Main loop: resolve predictions, then sleep for an hour."""
        while self._running:
            try:
                await self._resolve_all()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"PredictionResolver error: {e}")

            try:
                await asyncio.sleep(3600)  # Run every hour
            except asyncio.CancelledError:
                break

    async def _resolve_all(self):
        """Get all unresolved predictions and check current prices."""
        unresolved = self.hub.get_unresolved()
        if not unresolved:
            return

        symbols = set(p.get('symbol', '') for p in unresolved if p.get('symbol'))
        logger.info(f"PredictionResolver: resolving {len(unresolved)} predictions across {len(symbols)} symbols")

        resolved_count = 0
        for symbol in symbols:
            try:
                price = await self._get_price(symbol)
                if price and price > 0:
                    self.hub.resolve_predictions(symbol, price)
                    resolved_count += 1
            except Exception as e:
                logger.warning(f"Failed to resolve {symbol}: {e}")

        if resolved_count > 0:
            logger.info(f"PredictionResolver: resolved predictions for {resolved_count} symbols")

    async def _get_price(self, symbol: str) -> Optional[float]:
        """Fetch current price from Polygon (runs in executor to avoid blocking)."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._fetch_price_sync, symbol)

    def _fetch_price_sync(self, symbol: str) -> Optional[float]:
        """Synchronous price fetch from Polygon."""
        try:
            url = f"{POLYGON_BASE}/v2/aggs/ticker/{symbol}/prev"
            resp = requests.get(url, params={'apiKey': self.api_key}, timeout=10)
            if resp.ok:
                data = resp.json()
                results = data.get('results', [])
                if results:
                    return results[0].get('c')
        except Exception as e:
            logger.warning(f"Price fetch failed for {symbol}: {e}")
        return None
