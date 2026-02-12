"""Binance signed-request service: market orders + OCO orders.

Uses the current Binance API (2024+):
  - Market: POST /api/v3/order
  - OCO:    POST /api/v3/orderList/oco  (new endpoint with above/below params)

Every order goes through ``validate_and_adjust_order()`` from
``binance_filters`` before being sent so that LOT_SIZE, PRICE_FILTER,
and NOTIONAL constraints are always satisfied.
"""

import logging
import time

from app.auth.encryption import decrypt
from app.config import get_settings
from app.services.http_client import get_client
from app.services.binance_filters import (
    get_symbol_filters,
    round_step,
    round_tick,
    step_decimals,
    validate_and_adjust_order,
)
from app.services.binance_sign import signed_query as _signed_params

logger = logging.getLogger(__name__)


def _base_headers(api_key: str) -> dict:
    return {"X-MBX-APIKEY": api_key}


# ── Signed POST helper ──────────────────────────────────────────────────────

async def _signed_post(api_key: str, api_secret: str, endpoint: str, params: dict) -> dict:
    """Send a signed POST request to Binance and return parsed JSON."""
    settings = get_settings()
    url = f"{settings.binance_base_url}{endpoint}"
    qs = _signed_params(params, api_secret)
    full_url = f"{url}?{qs}"

    client = get_client()
    resp = await client.post(full_url, headers=_base_headers(api_key))
    data = resp.json()
    if resp.status_code != 200:
        logger.error("Binance %s error %d: %s", endpoint, resp.status_code, data)
        msg = data.get("msg", str(data))
        code = data.get("code", resp.status_code)
        raise RuntimeError(f"Binance error {code}: {msg}")
    return data


# ── Market Order ─────────────────────────────────────────────────────────────

async def place_market_order(
    api_key_enc: str,
    api_secret_enc: str,
    symbol: str,
    side: str,
    quantity: float,
) -> dict:
    """Place a simple MARKET order on Binance Spot.

    Validates against exchange filters before sending.  If quantity × price
    falls below minNotional for a BUY, automatically switches to
    ``quoteOrderQty``.
    """
    api_key = decrypt(api_key_enc)
    api_secret = decrypt(api_secret_enc)

    adj = await validate_and_adjust_order(
        symbol=symbol,
        side=side,
        order_type="MARKET",
        quantity=quantity,
        price=None,
    )

    params: dict = {
        "symbol": symbol,
        "side": side,
        "type": "MARKET",
        "newOrderRespType": "FULL",
        "timestamp": int(time.time() * 1000),
    }

    # quoteOrderQty fallback (MARKET BUY below minNotional)
    if "quoteOrderQty" in adj:
        params["quoteOrderQty"] = adj["quoteOrderQty"]
    else:
        params["quantity"] = adj["quantity"]

    logger.info(
        "Sending MARKET %s %s | original_qty=%.8f | adjusted=%s | adjustments=%s",
        side, symbol, quantity, adj.get("quantity", adj.get("quoteOrderQty")),
        adj["adjustments"] or "none",
    )

    data = await _signed_post(api_key, api_secret, "/api/v3/order", params)
    data["_filter_adjustments"] = adj["adjustments"]
    return data


# ── Limit Order ──────────────────────────────────────────────────────────────

async def place_limit_order(
    api_key_enc: str,
    api_secret_enc: str,
    symbol: str,
    side: str,
    quantity: float,
    price: float,
) -> dict:
    """Place a LIMIT GTC order on Binance Spot.

    Validates against exchange filters before sending.
    """
    api_key = decrypt(api_key_enc)
    api_secret = decrypt(api_secret_enc)

    adj = await validate_and_adjust_order(
        symbol=symbol,
        side=side,
        order_type="LIMIT",
        quantity=quantity,
        price=price,
    )

    filters = await get_symbol_filters(symbol)
    tick = filters["tick_size"]
    tick_dec = step_decimals(tick)
    price_str = f"{float(round_tick(price, tick)):.{tick_dec}f}"

    params: dict = {
        "symbol": symbol,
        "side": side,
        "type": "LIMIT",
        "timeInForce": "GTC",
        "quantity": adj["quantity"],
        "price": price_str,
        "newOrderRespType": "FULL",
        "timestamp": int(time.time() * 1000),
    }

    logger.info(
        "Sending LIMIT %s %s qty=%s price=%s | adjustments=%s",
        side, symbol, adj["quantity"], price_str,
        adj["adjustments"] or "none",
    )

    data = await _signed_post(api_key, api_secret, "/api/v3/order", params)
    data["_filter_adjustments"] = adj["adjustments"]
    return data


# ── Signed GET helper ─────────────────────────────────────────────────────

async def _signed_get(api_key: str, api_secret: str, endpoint: str, params: dict) -> dict:
    """Send a signed GET request to Binance and return parsed JSON."""
    settings = get_settings()
    url = f"{settings.binance_base_url}{endpoint}"
    qs = _signed_params(params, api_secret)
    full_url = f"{url}?{qs}"

    client = get_client()
    resp = await client.get(full_url, headers=_base_headers(api_key))
    data = resp.json()
    if resp.status_code != 200:
        logger.error("Binance GET %s error %d: %s", endpoint, resp.status_code, data)
        msg = data.get("msg", str(data))
        code = data.get("code", resp.status_code)
        raise RuntimeError(f"Binance error {code}: {msg}")
    return data


# ── OCO Order (new endpoint: /api/v3/orderList/oco) ─────────────────────────

async def place_oco_order(
    api_key_enc: str,
    api_secret_enc: str,
    symbol: str,
    side: str,
    quantity: float,
    take_profit: float,
    stop_loss: float,
) -> dict:
    """Place an OCO order via the new Binance endpoint.

    Validates quantity, take-profit, and stop-loss against exchange filters.
    """
    api_key = decrypt(api_key_enc)
    api_secret = decrypt(api_secret_enc)

    filters = await get_symbol_filters(symbol)
    tick = filters["tick_size"]
    step = filters["step_size"]

    # Validate quantity via the filter system (using TP price for notional)
    adj = await validate_and_adjust_order(
        symbol=symbol,
        side=side,
        order_type="LIMIT",
        quantity=quantity,
        price=take_profit,
    )
    qty_str = adj["quantity"]

    # Round prices to tickSize
    tp_rounded = round_tick(take_profit, tick)
    sl_rounded = round_tick(stop_loss, tick)
    tick_dec = step_decimals(tick)
    tp_str = f"{float(tp_rounded):.{tick_dec}f}"
    sl_str = f"{float(sl_rounded):.{tick_dec}f}"

    price_adjustments = []
    if tp_rounded != round_tick(take_profit, "0.00000001"):
        price_adjustments.append(f"TP {take_profit}→{tp_str}")
    if sl_rounded != round_tick(stop_loss, "0.00000001"):
        price_adjustments.append(f"SL {stop_loss}→{sl_str}")

    all_adjustments = adj["adjustments"] + (
        [f"PRICE_FILTER(OCO): {', '.join(price_adjustments)}"]
        if price_adjustments else []
    )

    if side == "SELL":
        sl_limit = stop_loss * 0.998
        sl_limit_str = f"{float(round_tick(sl_limit, tick)):.{tick_dec}f}"
        params = {
            "symbol": symbol,
            "side": "SELL",
            "quantity": qty_str,
            "aboveType": "LIMIT_MAKER",
            "abovePrice": tp_str,
            "belowType": "STOP_LOSS_LIMIT",
            "belowStopPrice": sl_str,
            "belowPrice": sl_limit_str,
            "belowTimeInForce": "GTC",
            "newOrderRespType": "FULL",
            "timestamp": int(time.time() * 1000),
        }
    else:
        sl_limit = stop_loss * 1.002
        sl_limit_str = f"{float(round_tick(sl_limit, tick)):.{tick_dec}f}"
        params = {
            "symbol": symbol,
            "side": "BUY",
            "quantity": qty_str,
            "aboveType": "STOP_LOSS_LIMIT",
            "aboveStopPrice": sl_str,
            "abovePrice": sl_limit_str,
            "aboveTimeInForce": "GTC",
            "belowType": "LIMIT_MAKER",
            "belowPrice": tp_str,
            "newOrderRespType": "FULL",
            "timestamp": int(time.time() * 1000),
        }

    logger.info(
        "Sending OCO %s %s qty=%s TP=%s SL=%s | adjustments=%s",
        side, symbol, qty_str, tp_str, sl_str,
        all_adjustments or "none",
    )

    data = await _signed_post(api_key, api_secret, "/api/v3/orderList/oco", params)
    data["_filter_adjustments"] = all_adjustments
    return data


# ── Full OCO Flow: Market entry + OCO exit ───────────────────────────────────

async def execute_market_then_oco(
    api_key_enc: str,
    api_secret_enc: str,
    symbol: str,
    entry_side: str,
    quantity: float,
    take_profit: float,
    stop_loss: float,
) -> dict:
    """
    1. Place a MARKET order for entry (BUY or SELL).
    2. Place an OCO order for exit (opposite side) with TP + SL.

    Returns combined result.
    """
    # Step 1 — Market entry
    market_result = await place_market_order(
        api_key_enc, api_secret_enc, symbol, entry_side, quantity,
    )

    # Determine filled price and quantity from market order
    fills = market_result.get("fills", [])
    if fills:
        filled_price = sum(float(f["price"]) * float(f["qty"]) for f in fills) / \
                        sum(float(f["qty"]) for f in fills)
    else:
        filled_price = float(market_result.get("price", 0))

    filled_qty = float(market_result.get("executedQty", quantity))

    # Step 2 — OCO exit (opposite side)
    exit_side = "SELL" if entry_side == "BUY" else "BUY"

    oco_result = await place_oco_order(
        api_key_enc, api_secret_enc,
        symbol, exit_side, filled_qty,
        take_profit=take_profit,
        stop_loss=stop_loss,
    )

    return {
        "market_order": market_result,
        "oco_order": oco_result,
        "filled_price": filled_price,
        "filled_quantity": filled_qty,
        "take_profit": take_profit,
        "stop_loss": stop_loss,
    }


async def _signed_delete(api_key: str, api_secret: str, endpoint: str, params: dict) -> dict:
    """Send a signed DELETE request to Binance and return parsed JSON."""
    settings = get_settings()
    url = f"{settings.binance_base_url}{endpoint}"
    qs = _signed_params(params, api_secret)
    full_url = f"{url}?{qs}"

    client = get_client()
    resp = await client.delete(full_url, headers=_base_headers(api_key))
    data = resp.json()
    if resp.status_code != 200:
        logger.error("Binance DELETE %s error %d: %s", endpoint, resp.status_code, data)
        msg = data.get("msg", str(data))
        code = data.get("code", resp.status_code)
        raise RuntimeError(f"Binance error {code}: {msg}")
    return data


async def cancel_order(
    api_key_enc: str,
    api_secret_enc: str,
    symbol: str,
    order_id: int | None = None,
    order_list_id: int | None = None,
) -> dict:
    """Cancel an open order or OCO order list on Binance.

    Returns a dict with cancel status.
    """
    api_key = decrypt(api_key_enc)
    api_secret = decrypt(api_secret_enc)

    if order_list_id:
        # Cancel OCO order list
        data = await _signed_delete(api_key, api_secret, "/api/v3/orderList", {
            "symbol": symbol,
            "orderListId": order_list_id,
            "timestamp": int(time.time() * 1000),
        })
        logger.info("Cancelled OCO orderListId=%d symbol=%s", order_list_id, symbol)
        return {"cancelled": True, "type": "oco", "detail": data}

    if order_id:
        # Cancel single order
        data = await _signed_delete(api_key, api_secret, "/api/v3/order", {
            "symbol": symbol,
            "orderId": order_id,
            "timestamp": int(time.time() * 1000),
        })
        logger.info("Cancelled order orderId=%d symbol=%s", order_id, symbol)
        return {"cancelled": True, "type": "single", "detail": data}

    raise RuntimeError("No orderId ni orderListId para cancelar")


# ── OCO Status Check ─────────────────────────────────────────────────────

async def check_oco_status(
    api_key_enc: str,
    api_secret_enc: str,
    symbol: str,
    order_list_id: int,
) -> dict:
    """Query Binance for the current status of an OCO order list.

    Returns a dict with:
      - oco_status: "EXEC_STARTED" (active) or "ALL_DONE" (one leg triggered)
      - exit_type: "TP" or "SL" (only when ALL_DONE)
      - exit_price: the fill price of the triggered leg (only when ALL_DONE)
      - exit_time: epoch ms of the fill (only when ALL_DONE)
    """
    api_key = decrypt(api_key_enc)
    api_secret = decrypt(api_secret_enc)

    oco_data = await _signed_get(api_key, api_secret, "/api/v3/orderList", {
        "orderListId": order_list_id,
        "timestamp": int(time.time() * 1000),
    })

    result: dict = {
        "oco_status": oco_data.get("listStatusType", "UNKNOWN"),
    }

    if result["oco_status"] != "ALL_DONE":
        return result

    # OCO completed — find which leg filled
    for order_info in oco_data.get("orders", []):
        order_detail = await _signed_get(api_key, api_secret, "/api/v3/order", {
            "symbol": symbol,
            "orderId": order_info["orderId"],
            "timestamp": int(time.time() * 1000),
        })

        if order_detail.get("status") == "FILLED":
            order_type = order_detail.get("type", "")
            if order_type == "LIMIT_MAKER":
                result["exit_type"] = "TP"
            elif order_type in ("STOP_LOSS_LIMIT", "STOP_LOSS"):
                result["exit_type"] = "SL"
            else:
                result["exit_type"] = "UNKNOWN"

            executed_qty = float(order_detail.get("executedQty", 0))
            if executed_qty > 0:
                cumulative_quote = float(order_detail.get("cummulativeQuoteQty", 0))
                result["exit_price"] = cumulative_quote / executed_qty
            else:
                result["exit_price"] = float(order_detail.get("price", 0))

            result["exit_time"] = order_detail.get("updateTime", order_detail.get("time"))
            break

    return result


# ── Order Validation ──────────────────────────────────────────────────────

DEAD_STATUSES = {"CANCELED", "EXPIRED", "REJECTED", "EXPIRED_IN_MATCH"}


async def validate_order_exists(
    api_key_enc: str,
    api_secret_enc: str,
    symbol: str,
    order_id: int | None = None,
    order_list_id: int | None = None,
) -> dict:
    """Check if an order/OCO still exists and is active on Binance.

    Returns:
      - valid: True if the order is active (NEW / PARTIALLY_FILLED / FILLED)
      - binance_status: the raw status from Binance
      - reason: human-readable explanation when invalid
    """
    api_key = decrypt(api_key_enc)
    api_secret = decrypt(api_secret_enc)

    # --- OCO order list ---
    if order_list_id:
        try:
            oco_data = await _signed_get(api_key, api_secret, "/api/v3/orderList", {
                "orderListId": order_list_id,
                "timestamp": int(time.time() * 1000),
            })
        except RuntimeError as exc:
            if "-2013" in str(exc) or "does not exist" in str(exc).lower():
                return {"valid": False, "binance_status": "NOT_FOUND",
                        "reason": "OCO no existe en Binance"}
            raise

        list_status = oco_data.get("listStatusType", "UNKNOWN")
        if list_status == "ALL_DONE":
            orders = oco_data.get("orders", [])
            any_filled = False
            all_cancelled = True
            for o in orders:
                try:
                    detail = await _signed_get(api_key, api_secret, "/api/v3/order", {
                        "symbol": symbol,
                        "orderId": o["orderId"],
                        "timestamp": int(time.time() * 1000),
                    })
                    st = detail.get("status", "")
                    if st == "FILLED":
                        any_filled = True
                        all_cancelled = False
                    elif st not in DEAD_STATUSES:
                        all_cancelled = False
                except RuntimeError:
                    pass
            if all_cancelled and not any_filled:
                return {"valid": False, "binance_status": "ALL_CANCELED",
                        "reason": "Todas las ordenes OCO fueron canceladas"}
            return {"valid": True, "binance_status": "ALL_DONE",
                    "reason": "OCO completada"}
        elif list_status == "EXEC_STARTED":
            return {"valid": True, "binance_status": "EXEC_STARTED",
                    "reason": "OCO activa"}
        else:
            return {"valid": False, "binance_status": list_status,
                    "reason": f"Estado OCO inesperado: {list_status}"}

    # --- Single order ---
    if order_id:
        try:
            detail = await _signed_get(api_key, api_secret, "/api/v3/order", {
                "symbol": symbol,
                "orderId": order_id,
                "timestamp": int(time.time() * 1000),
            })
        except RuntimeError as exc:
            if "-2013" in str(exc) or "does not exist" in str(exc).lower():
                return {"valid": False, "binance_status": "NOT_FOUND",
                        "reason": "Orden no existe en Binance"}
            raise

        status = detail.get("status", "UNKNOWN")
        if status in ("NEW", "PARTIALLY_FILLED"):
            return {"valid": True, "binance_status": status, "reason": "Orden activa"}
        if status == "FILLED":
            return {"valid": True, "binance_status": "FILLED", "reason": "Orden ejecutada"}
        if status in DEAD_STATUSES:
            return {"valid": False, "binance_status": status,
                    "reason": f"Orden {status.lower()} en Binance"}
        return {"valid": False, "binance_status": status,
                "reason": f"Estado inesperado: {status}"}

    return {"valid": False, "binance_status": "NO_IDS",
            "reason": "Sin orderId ni orderListId para validar"}
