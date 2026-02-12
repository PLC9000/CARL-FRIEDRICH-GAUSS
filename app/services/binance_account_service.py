"""Binance account info: API key permissions / restrictions."""

import logging
import time

import httpx

from app.auth.encryption import decrypt
from app.config import get_settings
from app.services.binance_sign import signed_query as _signed_query

logger = logging.getLogger(__name__)


async def get_spot_balances(
    api_key_enc: str, api_secret_enc: str, assets: list[str]
) -> dict[str, dict]:
    """Fetch free/locked balances for specific assets from Binance Spot.

    Uses GET /api/v3/account (signed).  Returns a dict keyed by asset, e.g.
    ``{"BTC": {"free": 0.045, "locked": 0.0}, "USDT": {"free": 120.5, "locked": 0.0}}``.
    Assets not found in the account are returned with 0/0.
    """
    api_key = decrypt(api_key_enc)
    api_secret = decrypt(api_secret_enc)

    params = {"timestamp": int(time.time() * 1000)}
    settings = get_settings()
    url = f"{settings.binance_base_url}/api/v3/account"
    qs = _signed_query(params, api_secret)

    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await client.get(
            f"{url}?{qs}",
            headers={"X-MBX-APIKEY": api_key},
        )
        data = resp.json()
        if resp.status_code != 200:
            msg = data.get("msg", str(data))
            code = data.get("code", resp.status_code)
            raise RuntimeError(f"Binance error {code}: {msg}")

    want = {a.upper() for a in assets}
    balances_list = data.get("balances", [])
    result: dict[str, dict] = {}
    for b in balances_list:
        asset = b["asset"]
        if asset in want:
            result[asset] = {
                "free": float(b["free"]),
                "locked": float(b["locked"]),
            }
    # Fill missing assets with zeros
    for a in want:
        if a not in result:
            result[a] = {"free": 0.0, "locked": 0.0}
    return result


async def get_api_permissions(api_key_enc: str, api_secret_enc: str) -> dict:
    """Fetch API key restrictions from Binance (GET /sapi/v1/account/apiRestrictions)."""
    api_key = decrypt(api_key_enc)
    api_secret = decrypt(api_secret_enc)

    params = {"timestamp": int(time.time() * 1000)}
    settings = get_settings()
    url = f"{settings.binance_base_url}/sapi/v1/account/apiRestrictions"
    qs = _signed_query(params, api_secret)

    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await client.get(
            f"{url}?{qs}",
            headers={"X-MBX-APIKEY": api_key},
        )
        data = resp.json()
        if resp.status_code != 200:
            msg = data.get("msg", str(data))
            code = data.get("code", resp.status_code)
            raise RuntimeError(f"Binance error {code}: {msg}")
        return data
