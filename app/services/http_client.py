"""Shared httpx.AsyncClient singleton for all Binance API calls.

Avoids creating a new TCP/TLS connection per request.  Started in the
FastAPI lifespan and closed on shutdown.
"""

import httpx

_client: httpx.AsyncClient | None = None


def get_client() -> httpx.AsyncClient:
    """Return the shared client.  Raises if called before startup."""
    if _client is None:
        raise RuntimeError("HTTP client not initialised — call start_client() first")
    return _client


async def start_client() -> None:
    global _client
    _client = httpx.AsyncClient(
        timeout=30.0,
        limits=httpx.Limits(max_connections=20, max_keepalive_connections=10),
    )


async def stop_client() -> None:
    global _client
    if _client is not None:
        await _client.aclose()
        _client = None
