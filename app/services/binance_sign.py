"""Shared Binance HMAC-SHA256 signing for signed endpoints."""

import hashlib
import hmac
from urllib.parse import urlencode


def signed_query(params: dict, secret: str) -> str:
    """Build a query-string with an appended HMAC-SHA256 signature."""
    query = urlencode(params)
    sig = hmac.new(secret.encode(), query.encode(), hashlib.sha256).hexdigest()
    return query + "&signature=" + sig
