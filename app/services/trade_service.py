"""Trade execution — always live via Binance API (requires configured keys).

Supports:
  - market: simple buy/sell at market price
  - oco: market entry + OCO exit (take-profit + stop-loss)
"""

import datetime
import logging

from sqlalchemy.orm import Session

from app.models import (
    Approval, ApprovalStatus, Recommendation, TradeExecution, AuditLog, User,
)
from app.services.binance_order_service import execute_market_then_oco, place_market_order, place_limit_order

logger = logging.getLogger(__name__)


async def execute_trade(
    approval_id: int,
    admin_id: int,
    quantity: float,
    order_type: str,
    db: Session,
) -> TradeExecution:
    approval = db.query(Approval).filter(Approval.id == approval_id).first()
    if approval is None:
        raise ValueError("Registro de aprobación no encontrado")
    if approval.status != ApprovalStatus.approved:
        raise ValueError("No se puede ejecutar — la aprobación no está en estado 'approved'")

    rec: Recommendation = approval.recommendation
    if rec.recommendation == "NO-TRADE":
        raise ValueError("No se puede ejecutar una recomendación NO-TRADE")

    admin = db.query(User).filter(User.id == admin_id).first()

    if not admin.binance_api_key_enc:
        raise ValueError(
            "Necesitás configurar tus API keys de Binance antes de operar. "
            "Andá a Configuración (ícono ⚙) y guardalas."
        )

    if order_type == "oco" and (rec.stop_loss is None or rec.take_profit is None):
        raise ValueError("La recomendación no tiene stop-loss/take-profit — no se puede hacer OCO")

    if order_type == "limit" and rec.entry_price is None:
        raise ValueError("La recomendación no tiene precio de entrada — no se puede hacer orden Limit")

    result: dict
    status: str
    error_msg: str | None = None

    try:
        if order_type == "oco":
            result = await execute_market_then_oco(
                admin.binance_api_key_enc,
                admin.binance_api_secret_enc,
                rec.symbol,
                rec.recommendation,
                quantity,
                take_profit=rec.take_profit,
                stop_loss=rec.stop_loss,
            )
            result["mode"] = "LIVE"
            result["order_type"] = "oco"
            status = "filled"
        elif order_type == "limit":
            # For SELL limit: place above entry (entry + profit margin)
            # For BUY limit: place at entry (fills at market or better)
            limit_price = rec.entry_price
            if rec.recommendation == "SELL" and rec.entry_price and rec.take_profit:
                # rec.take_profit for SELL = entry*(1-tp_pct/100)
                # limit sell price = entry*(1+tp_pct/100) = entry + (entry - take_profit)
                delta = rec.entry_price - rec.take_profit
                limit_price = round(rec.entry_price + delta, 8)
            raw = await place_limit_order(
                admin.binance_api_key_enc,
                admin.binance_api_secret_enc,
                rec.symbol,
                rec.recommendation,
                quantity,
                price=limit_price,
            )
            result = {"limit_order": raw, "mode": "LIVE", "order_type": "limit",
                       "entry_price": rec.entry_price, "limit_price": limit_price}
            status = "filled"
        else:
            raw = await place_market_order(
                admin.binance_api_key_enc,
                admin.binance_api_secret_enc,
                rec.symbol,
                rec.recommendation,
                quantity,
            )
            result = {"market_order": raw, "mode": "LIVE", "order_type": "market"}
            status = "filled"
    except RuntimeError as exc:
        # Binance API errors (signature, filters, insufficient balance, etc.)
        error_msg = str(exc)
        logger.error("Binance order failed: %s", error_msg)
        result = {"error": error_msg, "mode": "LIVE_FAILED"}
        status = "failed"
    except Exception as exc:
        error_msg = str(exc)
        logger.error("Unexpected order error: %s", error_msg)
        result = {"error": error_msg, "mode": "LIVE_FAILED"}
        status = "failed"

    # For limit orders, store the actual limit price and target TP
    if order_type == "limit" and rec.recommendation == "SELL" and rec.entry_price and rec.take_profit:
        limit_tp = round(rec.entry_price + (rec.entry_price - rec.take_profit), 8)
    else:
        limit_tp = None

    trade = TradeExecution(
        approval_id=approval_id,
        executed_by=admin_id,
        is_live=True,
        order_type=order_type,
        symbol=rec.symbol,
        side=rec.recommendation,
        quantity=quantity,
        price=rec.entry_price,
        stop_loss=rec.stop_loss if order_type == "oco" else None,
        take_profit=rec.take_profit if order_type == "oco" else (limit_tp if order_type == "limit" else None),
        status=status,
        result=result,
        executed_at=datetime.datetime.utcnow(),
    )
    db.add(trade)

    # Extract filter adjustments from the result for auditing
    filter_adj = result.get("_filter_adjustments") or result.get("market_order", {}).get("_filter_adjustments", [])

    db.add(AuditLog(
        user_id=admin_id,
        action="trade.execute",
        payload={
            "approval_id": approval_id,
            "symbol": rec.symbol,
            "side": rec.recommendation,
            "quantity": quantity,
            "order_type": order_type,
            "status": status,
            "error": error_msg,
            "filter_adjustments": filter_adj,
        },
    ))
    db.commit()
    db.refresh(trade)

    logger.info(
        "Trade %d (LIVE/%s) %s: %s %s qty=%.8f @ %.8f",
        trade.id, order_type, status,
        rec.recommendation, rec.symbol, quantity, rec.entry_price or 0,
    )
    return trade
