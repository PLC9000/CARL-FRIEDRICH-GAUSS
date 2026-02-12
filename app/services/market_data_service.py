"""Pre-fetch additional Binance market data for strategies that need it.

Currently used by Strategy M (Claude AI) which needs order book depth
and 24hr ticker stats beyond the standard OHLCV candles.
All endpoints are public (no API key required).
"""

from __future__ import annotations

import asyncio
import logging

from app.config import get_settings
from app.services.http_client import get_client

logger = logging.getLogger(__name__)


async def fetch_order_book(symbol: str, limit: int = 20) -> dict | None:
    """GET /api/v3/depth — order book snapshot.

    Returns raw JSON or None on failure (never raises).
    """
    settings = get_settings()
    client = get_client()
    try:
        resp = await client.get(
            f"{settings.binance_base_url}/api/v3/depth",
            params={"symbol": symbol, "limit": limit},
        )
        if resp.status_code != 200:
            logger.warning("Order book %s returned %s", symbol, resp.status_code)
            return None
        return resp.json()
    except Exception as exc:
        logger.warning("Order book fetch failed for %s: %s", symbol, exc)
        return None


async def fetch_ticker_24hr(symbol: str) -> dict | None:
    """GET /api/v3/ticker/24hr — 24-hour rolling stats.

    Returns raw JSON or None on failure (never raises).
    """
    settings = get_settings()
    client = get_client()
    try:
        resp = await client.get(
            f"{settings.binance_base_url}/api/v3/ticker/24hr",
            params={"symbol": symbol},
        )
        if resp.status_code != 200:
            logger.warning("24hr ticker %s returned %s", symbol, resp.status_code)
            return None
        return resp.json()
    except Exception as exc:
        logger.warning("24hr ticker fetch failed for %s: %s", symbol, exc)
        return None


async def prefetch_market_data(symbol: str) -> dict:
    """Fetch all extra market data for Strategy M.

    Returns a dict ready for ``params["_market_data"]``.
    Individual failures are graceful (keys become None).
    """
    results = await asyncio.gather(
        fetch_order_book(symbol),
        fetch_ticker_24hr(symbol),
        return_exceptions=True,
    )
    depth = results[0] if not isinstance(results[0], BaseException) else None
    ticker = results[1] if not isinstance(results[1], BaseException) else None
    return {
        "symbol": symbol,
        "depth": depth,
        "ticker_24hr": ticker,
    }
