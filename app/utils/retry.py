import asyncio
import logging

import httpx

logger = logging.getLogger(__name__)


async def retry_request(
    client: httpx.AsyncClient,
    method: str,
    url: str,
    *,
    max_attempts: int = 3,
    backoff: float = 1.0,
    **kwargs,
) -> httpx.Response:
    """Execute an HTTP request with exponential-backoff retry on transient errors."""
    last_exc: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            resp = await client.request(method, url, **kwargs)
            # Binance returns 429 for rate limits, 418 for IP bans
            if resp.status_code in (429, 418, 500, 502, 503):
                wait = backoff * (2 ** (attempt - 1))
                logger.warning("Binance %s (attempt %d/%d), retrying in %.1fs", resp.status_code, attempt, max_attempts, wait)
                await asyncio.sleep(wait)
                continue
            return resp
        except (httpx.ConnectError, httpx.ReadTimeout, httpx.WriteTimeout) as exc:
            last_exc = exc
            wait = backoff * (2 ** (attempt - 1))
            logger.warning("Network error (attempt %d/%d): %s, retrying in %.1fs", attempt, max_attempts, exc, wait)
            await asyncio.sleep(wait)
    raise httpx.HTTPError(f"Request failed after {max_attempts} attempts") from last_exc
