"""Thin async client for Binance public market-data endpoints."""

import datetime
import logging

import httpx
import numpy as np

from app.config import get_settings
from app.utils.retry import retry_request

logger = logging.getLogger(__name__)

# Map our interval strings to Binance-accepted values (they're the same).
VALID_INTERVALS = {"1m", "5m", "15m", "1h", "4h", "1d"}


def _dt_to_ms(dt: datetime.datetime) -> int:
    return int(dt.timestamp() * 1000)


async def fetch_candles(
    symbol: str,
    interval: str,
    start_dt: datetime.datetime,
    end_dt: datetime.datetime,
) -> np.ndarray:
    """
    Fetch OHLCV candles from Binance.

    Returns an ndarray of shape (N, 6): [open_time_ms, open, high, low, close, volume].
    Handles pagination (Binance caps at 1000 per request).
    """
    settings = get_settings()
    if interval not in VALID_INTERVALS:
        raise ValueError(f"Invalid interval: {interval}")

    all_candles: list[list[float]] = []
    current_start = _dt_to_ms(start_dt)
    end_ms = _dt_to_ms(end_dt)

    async with httpx.AsyncClient(timeout=30.0) as client:
        while current_start < end_ms:
            params = {
                "symbol": symbol,
                "interval": interval,
                "startTime": current_start,
                "endTime": end_ms,
                "limit": settings.binance_max_candles,
            }
            resp = await retry_request(
                client,
                "GET",
                f"{settings.binance_base_url}/api/v3/klines",
                params=params,
                max_attempts=settings.binance_retry_attempts,
                backoff=settings.binance_retry_backoff,
            )
            resp.raise_for_status()
            data = resp.json()
            if not data:
                break
            for row in data:
                # row: [open_time, open, high, low, close, volume, close_time, ...]
                all_candles.append([
                    float(row[0]),   # open_time ms
                    float(row[1]),   # open
                    float(row[2]),   # high
                    float(row[3]),   # low
                    float(row[4]),   # close
                    float(row[5]),   # volume
                ])
            # advance past last candle
            current_start = int(data[-1][0]) + 1
            if len(data) < settings.binance_max_candles:
                break

    if not all_candles:
        raise ValueError(f"No candle data returned for {symbol} {interval}")

    logger.info("Fetched %d candles for %s %s", len(all_candles), symbol, interval)
    return np.array(all_candles)


async def validate_symbol(symbol: str) -> bool:
    """Check whether a symbol exists on Binance."""
    settings = get_settings()
    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await retry_request(
            client,
            "GET",
            f"{settings.binance_base_url}/api/v3/exchangeInfo",
            params={"symbol": symbol},
            max_attempts=2,
            backoff=0.5,
        )
        if resp.status_code != 200:
            return False
        info = resp.json()
        symbols = [s["symbol"] for s in info.get("symbols", [])]
        return symbol in symbols
