"""On-demand trade tracking: fetch current price, compute P&L, evaluate."""

import datetime

from app.models import TradeExecution
from app.services.binance_filters import _get_current_price


def _format_elapsed(delta: datetime.timedelta) -> str:
    total_seconds = int(delta.total_seconds())
    if total_seconds < 0:
        return "0m"
    days = total_seconds // 86400
    hours = (total_seconds % 86400) // 3600
    minutes = (total_seconds % 3600) // 60
    if days > 0:
        return f"{days}d {hours}h"
    if hours > 0:
        return f"{hours}h {minutes}m"
    return f"{minutes}m"


async def track_trade(trade: TradeExecution) -> dict:
    """Fetch current price and compute tracking metrics for a filled trade."""
    current_price = await _get_current_price(trade.symbol)
    entry = trade.price or 0.0
    qty = trade.quantity or 0.0
    side = trade.side  # BUY or SELL

    # Check if OCO has completed (exit info stored in result)
    result_data = trade.result or {}
    oco_completed = result_data.get("oco_status") == "ALL_DONE"
    exit_price = result_data.get("exit_price")
    exit_type = result_data.get("exit_type")

    # Use actual exit price for P&L when OCO completed
    effective_price = exit_price if (oco_completed and exit_price) else current_price

    # P&L
    if entry > 0:
        if side == "BUY":
            pnl_pct = round((effective_price - entry) / entry * 100, 4)
        else:
            pnl_pct = round((entry - effective_price) / entry * 100, 4)
    else:
        pnl_pct = 0.0

    pnl_abs = round(pnl_pct / 100 * entry * qty, 4)

    # Distances to SL / TP
    sl = trade.stop_loss
    tp = trade.take_profit

    sl_distance_pct = None
    tp_distance_pct = None
    tp_progress_pct = None

    if sl and entry > 0:
        if side == "BUY":
            sl_distance_pct = round((current_price - sl) / entry * 100, 2)
        else:
            sl_distance_pct = round((sl - current_price) / entry * 100, 2)

    if tp and entry > 0:
        if side == "BUY":
            tp_distance_pct = round((tp - current_price) / entry * 100, 2)
        else:
            tp_distance_pct = round((current_price - tp) / entry * 100, 2)

    # Progress toward TP (0 = at entry, 100 = at TP, negative = moving away)
    if tp and entry > 0 and tp != entry:
        if side == "BUY":
            tp_progress_pct = round((current_price - entry) / (tp - entry) * 100, 1)
        else:
            tp_progress_pct = round((entry - current_price) / (entry - tp) * 100, 1)

    # Evaluation
    if oco_completed and exit_type:
        if exit_type == "TP":
            evaluation = "TP ALCANZADO"
        elif exit_type == "SL":
            evaluation = "SL ALCANZADO"
        else:
            evaluation = "OCO COMPLETADA"
    elif sl and side == "BUY" and current_price <= sl:
        evaluation = "SL ALCANZADO"
    elif sl and side == "SELL" and current_price >= sl:
        evaluation = "SL ALCANZADO"
    elif tp and side == "BUY" and current_price >= tp:
        evaluation = "TP ALCANZADO"
    elif tp and side == "SELL" and current_price <= tp:
        evaluation = "TP ALCANZADO"
    elif pnl_pct > 0.05:
        evaluation = "EN GANANCIA"
    elif pnl_pct < -0.05:
        evaluation = "EN PERDIDA"
    else:
        evaluation = "NEUTRAL"

    # Elapsed
    now = datetime.datetime.utcnow()
    elapsed = _format_elapsed(now - trade.executed_at) if trade.executed_at else "—"

    return {
        "trade_id": trade.id,
        "symbol": trade.symbol,
        "side": side,
        "entry_price": entry,
        "current_price": current_price,
        "pnl_pct": pnl_pct,
        "pnl_abs": pnl_abs,
        "stop_loss": sl,
        "take_profit": tp,
        "sl_distance_pct": sl_distance_pct,
        "tp_distance_pct": tp_distance_pct,
        "tp_progress_pct": tp_progress_pct,
        "evaluation": evaluation,
        "elapsed": elapsed,
        "timestamp": now.isoformat(),
        "oco_status": result_data.get("oco_status"),
        "exit_type": exit_type,
        "exit_price": exit_price,
    }
