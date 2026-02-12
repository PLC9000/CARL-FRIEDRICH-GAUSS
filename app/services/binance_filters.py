"""Binance symbol filter validation and automatic order adjustment.

Fetches exchange filters (LOT_SIZE, PRICE_FILTER, NOTIONAL) from Binance,
caches them per symbol, and provides ``validate_and_adjust_order()`` which
every order function must call before sending to the API.

Key behaviours
--------------
* quantity is always rounded **down** to the nearest stepSize.
* price is rounded to the nearest tickSize.
* If notional (price × qty) < minNotional after rounding:
  - BUY MARKET  → switch to ``quoteOrderQty`` (≥ minNotional × 1.02).
  - SELL         → raise ``ValueError("NO-TRADE: …")`` — never force a sell.
  - BUY LIMIT    → bump qty up to the next valid step; fail if balance < needed.
* BUY orders never use more than 99 % of the available quote balance (98 % if
  commissions are paid in the quote asset).
"""

from __future__ import annotations

import logging
import math
import time
from decimal import Decimal, ROUND_DOWN, ROUND_HALF_UP
from typing import Any

import httpx

from app.config import get_settings

logger = logging.getLogger(__name__)

# ── In-memory cache (symbol → filters) with TTL ─────────────────────────────

_cache: dict[str, tuple[float, dict]] = {}   # symbol → (expiry_ts, filters)
_CACHE_TTL = 300  # 5 minutes


# ── Public helpers ───────────────────────────────────────────────────────────

def round_step(value: float, step: str, mode: str = "down") -> Decimal:
    """Round *value* to the nearest multiple of *step*, truncating toward zero.

    >>> round_step(0.12345, "0.001")
    Decimal('0.123')
    >>> round_step(1.999, "0.01")
    Decimal('1.99')
    """
    d_val = Decimal(str(value))
    d_step = Decimal(step)
    if d_step == 0:
        return d_val
    if mode == "down":
        return (d_val / d_step).to_integral_value(rounding=ROUND_DOWN) * d_step
    return (d_val / d_step).to_integral_value(rounding=ROUND_HALF_UP) * d_step


def round_tick(value: float, tick: str) -> Decimal:
    """Round *value* to the nearest tick (half-up)."""
    d_val = Decimal(str(value))
    d_tick = Decimal(tick)
    if d_tick == 0:
        return d_val
    return (d_val / d_tick).to_integral_value(rounding=ROUND_HALF_UP) * d_tick


def step_decimals(step: str) -> int:
    """Return the number of decimal places implied by a step string."""
    p = step.rstrip("0")
    if "." in p:
        return len(p.split(".")[1])
    return 0


# ── Fetch & cache ────────────────────────────────────────────────────────────

async def get_symbol_filters(symbol: str) -> dict:
    """Return parsed filters for *symbol*, hitting the cache first.

    Returned dict keys::

        step_size, min_qty, max_qty,          (LOT_SIZE)
        tick_size, min_price, max_price,      (PRICE_FILTER)
        min_notional, apply_to_market,        (NOTIONAL / MIN_NOTIONAL)
        avg_price_mins,                       (for market notional check)
    """
    now = time.time()
    if symbol in _cache:
        expiry, cached = _cache[symbol]
        if now < expiry:
            return cached

    settings = get_settings()
    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await client.get(
            f"{settings.binance_base_url}/api/v3/exchangeInfo",
            params={"symbol": symbol},
        )
        if resp.status_code != 200:
            raise RuntimeError(f"No se pudo obtener filtros para {symbol}")
        data = resp.json()

    sym_info = None
    for s in data.get("symbols", []):
        if s["symbol"] == symbol:
            sym_info = s
            break
    if sym_info is None:
        raise RuntimeError(f"Símbolo {symbol} no encontrado en Binance")

    filters = _parse_all_filters(sym_info)
    _cache[symbol] = (now + _CACHE_TTL, filters)
    return filters


def _parse_all_filters(sym_info: dict) -> dict:
    """Extract every filter we care about from the symbol info."""
    result: dict[str, Any] = {
        "step_size": "0.00001",
        "min_qty": "0.00001",
        "max_qty": "99999999",
        "tick_size": "0.01",
        "min_price": "0.01",
        "max_price": "99999999",
        "min_notional": "0",
        "apply_to_market": True,
        "avg_price_mins": 5,
    }
    for f in sym_info.get("filters", []):
        ft = f["filterType"]
        if ft == "LOT_SIZE":
            result["step_size"] = f["stepSize"]
            result["min_qty"] = f["minQty"]
            result["max_qty"] = f["maxQty"]
        elif ft == "PRICE_FILTER":
            result["tick_size"] = f["tickSize"]
            result["min_price"] = f["minPrice"]
            result["max_price"] = f["maxPrice"]
        elif ft in ("NOTIONAL", "MIN_NOTIONAL"):
            result["min_notional"] = f.get("minNotional", "0")
            result["apply_to_market"] = f.get("applyToMarket", True)
            result["avg_price_mins"] = f.get("avgPriceMins", 5)
    return result


# ── Price helper (for market notional check) ─────────────────────────────────

async def _get_current_price(symbol: str) -> float:
    """GET /api/v3/ticker/price for the symbol."""
    settings = get_settings()
    async with httpx.AsyncClient(timeout=8.0) as client:
        resp = await client.get(
            f"{settings.binance_base_url}/api/v3/ticker/price",
            params={"symbol": symbol},
        )
        if resp.status_code != 200:
            raise RuntimeError(f"No se pudo obtener precio de {symbol}")
        return float(resp.json()["price"])


# ── Core validation ──────────────────────────────────────────────────────────

async def validate_and_adjust_order(
    symbol: str,
    side: str,          # BUY | SELL
    order_type: str,    # MARKET | LIMIT | LIMIT_MAKER | STOP_LOSS_LIMIT
    quantity: float | None,
    price: float | None,
    quote_order_qty: float | None = None,
    balance_free: float | None = None,
) -> dict:
    """Validate and adjust order params to comply with Binance filters.

    Returns a dict ready for the order params::

        {
            "quantity": "0.123",        # or absent if quoteOrderQty used
            "quoteOrderQty": "15.00",   # only for MARKET BUY fallback
            "price": "42000.50",        # absent for MARKET orders
            "adjustments": [...],       # list of human-readable adjustments
            "original": {...},          # snapshot of the original values
        }

    Raises ``ValueError`` when the order cannot be placed at all.
    """
    filters = await get_symbol_filters(symbol)
    adjustments: list[str] = []
    original = {
        "quantity": quantity,
        "price": price,
        "quote_order_qty": quote_order_qty,
    }

    is_market = order_type == "MARKET"
    min_notional = Decimal(filters["min_notional"])
    step = filters["step_size"]
    tick = filters["tick_size"]
    min_qty = Decimal(filters["min_qty"])
    max_qty = Decimal(filters["max_qty"])

    # ── Determine effective price for notional checks ────────────────────
    if is_market:
        eff_price = await _get_current_price(symbol)
    else:
        if price is None:
            raise ValueError("Se requiere precio para órdenes LIMIT")
        eff_price = price

    # ── Round price (LIMIT orders only) ──────────────────────────────────
    adj_price: Decimal | None = None
    if not is_market and price is not None:
        adj_price = round_tick(price, tick)
        if adj_price != Decimal(str(price)):
            adjustments.append(
                f"PRICE_FILTER: precio {price} → {adj_price} (tickSize={tick})"
            )
        # Clamp to min/max
        d_min_price = Decimal(filters["min_price"])
        d_max_price = Decimal(filters["max_price"])
        if d_min_price > 0 and adj_price < d_min_price:
            adj_price = d_min_price
            adjustments.append(f"PRICE_FILTER: precio ajustado al mínimo {d_min_price}")
        if d_max_price > 0 and adj_price > d_max_price:
            raise ValueError(
                f"Precio {adj_price} supera el máximo permitido ({d_max_price})"
            )
        eff_price = float(adj_price)

    # ── Round quantity (LOT_SIZE) ────────────────────────────────────────
    use_quote_qty = False  # flag: switch to quoteOrderQty for MARKET BUY

    if quantity is not None:
        adj_qty = round_step(quantity, step, mode="down")
        if adj_qty != Decimal(str(quantity)):
            adjustments.append(
                f"LOT_SIZE: cantidad {quantity} → {adj_qty} (stepSize={step})"
            )
        # Clamp to min/max
        if adj_qty < min_qty:
            if is_market and side == "BUY":
                adj_qty = min_qty
                adjustments.append(
                    f"LOT_SIZE: cantidad ajustada al mínimo {min_qty}"
                )
            else:
                raise ValueError(
                    f"Cantidad {adj_qty} está por debajo del mínimo ({min_qty})"
                )
        if adj_qty > max_qty:
            raise ValueError(
                f"Cantidad {adj_qty} supera el máximo permitido ({max_qty})"
            )

        # ── Notional check ───────────────────────────────────────────────
        notional = Decimal(str(eff_price)) * adj_qty
        if min_notional > 0 and notional < min_notional:
            if is_market and side == "BUY":
                # Switch to quoteOrderQty
                quote_needed = float(min_notional) * 1.02
                use_quote_qty = True
                adjustments.append(
                    f"NOTIONAL: precio×cantidad={notional} < minNotional={min_notional}. "
                    f"Usando quoteOrderQty={quote_needed:.4f}"
                )
            elif side == "SELL":
                raise ValueError(
                    f"NO-TRADE: posición ({adj_qty} × {eff_price} = {notional}) "
                    f"por debajo del mínimo permitido ({min_notional})"
                )
            else:
                # LIMIT BUY — try to bump qty up
                needed_qty = (min_notional / Decimal(str(eff_price)))
                needed_qty = round_step(float(needed_qty), step, mode="down")
                # one more step to ensure we're >= minNotional
                needed_qty += Decimal(step)
                if balance_free is not None:
                    max_affordable = Decimal(str(balance_free)) / Decimal(str(eff_price))
                    max_affordable = round_step(float(max_affordable), step, mode="down")
                    if needed_qty > max_affordable:
                        raise ValueError(
                            f"Balance insuficiente. Se necesitan al menos "
                            f"{needed_qty} unidades para cumplir minNotional "
                            f"({min_notional}), pero solo alcanza para {max_affordable}"
                        )
                adj_qty = needed_qty
                adjustments.append(
                    f"NOTIONAL: cantidad aumentada a {adj_qty} para cumplir "
                    f"minNotional={min_notional}"
                )

        # ── Balance cap for BUY (99% / 98%) ─────────────────────────────
        if side == "BUY" and balance_free is not None and not use_quote_qty:
            cap_pct = Decimal("0.98") if is_market else Decimal("0.99")
            max_spend = Decimal(str(balance_free)) * cap_pct
            max_qty_afford = round_step(
                float(max_spend / Decimal(str(eff_price))), step, mode="down",
            )
            if adj_qty > max_qty_afford:
                adj_qty = max_qty_afford
                adjustments.append(
                    f"BALANCE: cantidad limitada a {adj_qty} ({cap_pct*100}% del balance libre)"
                )
                if adj_qty < min_qty:
                    raise ValueError(
                        f"Balance insuficiente: después de limitar al {cap_pct*100}% "
                        f"la cantidad ({adj_qty}) queda por debajo del mínimo ({min_qty})"
                    )
    else:
        adj_qty = None

    # ── Build result ─────────────────────────────────────────────────────
    result: dict[str, Any] = {
        "adjustments": adjustments,
        "original": original,
    }
    if use_quote_qty:
        quote_val = float(min_notional) * 1.02
        dec = step_decimals(tick)
        result["quoteOrderQty"] = f"{quote_val:.{dec}f}"
    elif adj_qty is not None:
        dec = step_decimals(step)
        result["quantity"] = f"{float(adj_qty):.{dec}f}"

    if adj_price is not None:
        dec = step_decimals(tick)
        result["price"] = f"{float(adj_price):.{dec}f}"

    # Log adjustments
    if adjustments:
        logger.info(
            "Order adjustments for %s %s %s: %s",
            symbol, side, order_type, " | ".join(adjustments),
        )
    else:
        logger.debug("Order for %s %s %s — no adjustments needed", symbol, side, order_type)

    return result


def clear_cache():
    """Clear the filter cache (useful in tests)."""
    _cache.clear()
