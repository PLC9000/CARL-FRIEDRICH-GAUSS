from typing import List

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session, joinedload

from app.database import get_db
from app.models import User, TradeExecution, AuditLog, Approval, Recommendation
from app.schemas import (
    TradeExecuteRequest, TradeExecutionOut, TradeOperationEnrichedOut,
    TradeTrackingOut, OCOStatusOut, AnalyticsOut, TradePricePathOut,
)
from app.auth.dependencies import require_admin, get_current_user
from app.services.trade_service import execute_trade
from app.services.tracking_service import track_trade
from app.services.binance_order_service import check_oco_status, validate_order_exists, cancel_order
from app.services.binance_account_service import get_spot_balances
from app.services.analytics_service import compute_analytics

router = APIRouter(prefix="/trade", tags=["Operaciones"])


@router.get(
    "/balance/{symbol}",
    summary="Saldo disponible en Binance para un par",
)
async def get_balance_for_symbol(
    symbol: str,
    user: User = Depends(get_current_user),
):
    """Devuelve el saldo libre (free) del activo base y quote de un par.

    Ejemplo: para BTCUSDT devuelve balances de BTC y USDT.
    """
    if not user.binance_api_key_enc:
        raise HTTPException(
            status_code=400,
            detail="API keys de Binance no configuradas",
        )

    sym = symbol.upper().strip()
    if sym.endswith("FDUSD"):
        quote = "FDUSD"
        base = sym[: -len("FDUSD")]
    elif sym.endswith("USDT"):
        quote = "USDT"
        base = sym[: -len("USDT")]
    elif sym.endswith("BUSD"):
        quote = "BUSD"
        base = sym[: -len("BUSD")]
    elif sym.endswith("BTC"):
        quote = "BTC"
        base = sym[: -len("BTC")]
    else:
        quote = sym[-3:]
        base = sym[: -3]

    try:
        balances = await get_spot_balances(
            user.binance_api_key_enc,
            user.binance_api_secret_enc,
            [base, quote],
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=f"Error consultando Binance: {exc}")

    return {
        "symbol": sym,
        "base": base,
        "quote": quote,
        "balances": balances,
    }


@router.post(
    "/execute",
    response_model=TradeExecutionOut,
    summary="Ejecutar un trade",
)
async def trade_execute(
    body: TradeExecuteRequest,
    admin: User = Depends(require_admin),
    db: Session = Depends(get_db),
):
    """Ejecuta un trade real en Binance basado en una recomendación aprobada (solo admin).

    Requiere API keys de Binance configuradas (desde el ícono ⚙ en la interfaz).

    **Tipos de orden:**
    - **market**: compra o venta simple a precio de mercado.
    - **oco**: compra a mercado + orden OCO automática. La OCO coloca
      simultáneamente un take-profit (venta límite) y un stop-loss
      (venta stop-limit). Cuando una se ejecuta, la otra se cancela.
    """
    try:
        return await execute_trade(body.approval_id, admin.id, body.quantity, body.order_type, db)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


def _enrich_trade(trade: TradeExecution, rec) -> dict:
    """Transform a TradeExecution + its Recommendation into enriched output."""
    result = trade.result or {}

    # -- Filled price from fills array --
    filled_price = None
    total_commission = None
    commission_asset = None
    for order_key in ("market_order", "limit_order"):
        order_data = result.get(order_key, {})
        if not isinstance(order_data, dict):
            continue
        fills = order_data.get("fills", [])
        if fills:
            total_qty = sum(float(f.get("qty", 0)) for f in fills)
            if total_qty > 0:
                filled_price = sum(
                    float(f.get("price", 0)) * float(f.get("qty", 0)) for f in fills
                ) / total_qty
            total_commission = sum(float(f.get("commission", 0)) for f in fills)
            commission_asset = fills[0].get("commissionAsset") if fills else None
            break
    # OCO has pre-computed filled_price
    if filled_price is None and "filled_price" in result:
        filled_price = result["filled_price"]

    # -- Recipe info from recommendation --
    recipe_name = None
    recipe_id = None
    confidence = None
    signal_price = None
    if rec:
        metrics = getattr(rec, "metrics", None) or {}
        recipe_name = metrics.get("recipe_name")
        recipe_id = metrics.get("recipe_id")
        confidence = getattr(rec, "confidence", None)
        signal_price = getattr(rec, "entry_price", None)

    # -- Status extraction from result JSON --
    oco_status = result.get("oco_status")
    exit_type = result.get("exit_type")
    exit_price = result.get("exit_price")
    exit_time = result.get("exit_time")
    is_closed = (oco_status == "ALL_DONE") or result.get("closed_manually", False)
    is_active = (trade.status == "filled") and not is_closed

    # -- Binance IDs --
    oco_order = result.get("oco_order", {}) or {}
    market_order = result.get("market_order", {}) or {}
    limit_order = result.get("limit_order", {}) or {}
    binance_order_id = market_order.get("orderId") or limit_order.get("orderId")
    binance_order_list_id = oco_order.get("orderListId")

    # -- Validation --
    validation = result.get("validation", {}) or {}

    # -- Filter adjustments --
    filter_adj = (
        result.get("_filter_adjustments")
        or market_order.get("_filter_adjustments")
        or limit_order.get("_filter_adjustments")
    )

    return {
        "id": trade.id,
        "approval_id": trade.approval_id,
        "order_type": trade.order_type,
        "symbol": trade.symbol,
        "side": trade.side,
        "quantity": trade.quantity,
        "price": trade.price,
        "stop_loss": trade.stop_loss,
        "take_profit": trade.take_profit,
        "status": trade.status,
        "result": result,
        "executed_at": trade.executed_at,
        "signal_price": signal_price,
        "filled_price": round(filled_price, 8) if filled_price else None,
        "total_commission": round(total_commission, 8) if total_commission else None,
        "commission_asset": commission_asset,
        "recipe_name": recipe_name,
        "recipe_id": recipe_id,
        "confidence": confidence,
        "oco_status": oco_status,
        "exit_type": exit_type,
        "exit_price": exit_price,
        "exit_time": exit_time,
        "pnl_pct": result.get("pnl_pct"),
        "net_pnl_pct": result.get("net_pnl_pct"),
        "net_pnl_abs": result.get("net_pnl_abs"),
        "is_closed": is_closed,
        "is_active": is_active,
        "closed_manually": result.get("closed_manually", False),
        "binance_order_id": binance_order_id,
        "binance_order_list_id": binance_order_list_id,
        "validation_status": validation.get("binance_status"),
        "validation_reason": validation.get("reason"),
        "error_message": result.get("error"),
        "filter_adjustments": filter_adj if filter_adj else None,
    }


@router.get(
    "/operations/enriched",
    response_model=List[TradeOperationEnrichedOut],
    summary="Operaciones enriquecidas",
)
def list_operations_enriched(
    limit: int = 50,
    offset: int = 0,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Operaciones con datos pre-computados: precio de señal, precio de
    ejecución real, receta, P&L, estado OCO, IDs de Binance, etc."""
    trades = (
        db.query(TradeExecution)
        .filter(TradeExecution.executed_by == user.id)
        .order_by(TradeExecution.executed_at.desc())
        .offset(offset)
        .limit(min(limit, 200))
        .all()
    )
    approval_ids = [t.approval_id for t in trades]
    approvals = (
        db.query(Approval)
        .options(joinedload(Approval.recommendation))
        .filter(Approval.id.in_(approval_ids))
        .all()
    ) if approval_ids else []
    approval_map = {a.id: a for a in approvals}

    enriched = []
    for trade in trades:
        appr = approval_map.get(trade.approval_id)
        rec = appr.recommendation if appr else None
        enriched.append(_enrich_trade(trade, rec))
    return enriched


@router.get(
    "/operations",
    response_model=List[TradeExecutionOut],
    summary="Ver operaciones",
)
def list_operations(
    limit: int = 50,
    offset: int = 0,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Devuelve las operaciones ejecutadas por el usuario actual,
    ordenadas de la más reciente a la más antigua."""
    trades = (
        db.query(TradeExecution)
        .filter(TradeExecution.executed_by == user.id)
        .order_by(TradeExecution.executed_at.desc())
        .offset(offset)
        .limit(min(limit, 200))
        .all()
    )
    return trades


@router.get(
    "/operations/{trade_id}/track",
    response_model=TradeTrackingOut,
    summary="Seguimiento de una operación",
)
async def track_operation(
    trade_id: int,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Obtiene el precio actual de Binance y calcula P&L, distancia a SL/TP,
    evaluación y tiempo transcurrido para una operación ejecutada."""
    trade = (
        db.query(TradeExecution)
        .filter(TradeExecution.id == trade_id, TradeExecution.executed_by == user.id)
        .first()
    )
    if not trade:
        raise HTTPException(status_code=404, detail="Operación no encontrada")
    if trade.status != "filled":
        raise HTTPException(status_code=400, detail="Solo se puede hacer seguimiento de operaciones ejecutadas (filled)")
    try:
        return await track_trade(trade)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Error al obtener precio de Binance: {exc}")


@router.get(
    "/operations/{trade_id}/price-path",
    response_model=TradePricePathOut,
    summary="Historial de precios para graficar una operación cerrada",
)
async def get_trade_price_path(
    trade_id: int,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Obtiene velas históricas entre la entrada y salida de una operación
    cerrada para dibujar el gráfico de la operación."""
    trade = (
        db.query(TradeExecution)
        .filter(TradeExecution.id == trade_id, TradeExecution.executed_by == user.id)
        .first()
    )
    if not trade:
        raise HTTPException(status_code=404, detail="Operación no encontrada")

    result_data = trade.result or {}
    exit_time_str = result_data.get("exit_time")
    if not exit_time_str:
        raise HTTPException(status_code=400, detail="Operación sin tiempo de salida")

    import datetime as _dt
    from app.binance_client import fetch_candles

    try:
        exit_dt = _dt.datetime.fromisoformat(exit_time_str)
    except (ValueError, TypeError):
        raise HTTPException(status_code=400, detail="Formato de fecha de salida inválido")

    start_dt = trade.executed_at
    if start_dt.tzinfo is None:
        start_dt = start_dt.replace(tzinfo=_dt.timezone.utc)
    if exit_dt.tzinfo is None:
        exit_dt = exit_dt.replace(tzinfo=_dt.timezone.utc)

    duration = (exit_dt - start_dt).total_seconds()
    if duration < 3600:
        interval = "1m"
    elif duration < 21600:
        interval = "5m"
    elif duration < 86400:
        interval = "15m"
    elif duration < 604800:
        interval = "1h"
    else:
        interval = "4h"

    pad = _dt.timedelta(seconds=max(duration * 0.05, 60))

    try:
        candles = await fetch_candles(trade.symbol, interval, start_dt - pad, exit_dt + pad)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Error obteniendo velas: {exc}")

    closes = [round(float(c[4]), 8) for c in candles] if len(candles) > 0 else []
    times = [int(c[0]) for c in candles] if len(candles) > 0 else []

    return {
        "trade_id": trade.id,
        "closes": closes,
        "times": times,
        "entry_price": trade.price,
        "exit_price": result_data.get("exit_price"),
        "stop_loss": trade.stop_loss,
        "take_profit": trade.take_profit,
        "entry_time": trade.executed_at.isoformat(),
        "exit_time": exit_time_str,
        "exit_type": result_data.get("exit_type"),
        "side": trade.side,
    }


@router.delete(
    "/operations/{trade_id}",
    summary="Eliminar una operación del historial",
)
def delete_operation(
    trade_id: int,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Elimina una operación del historial.

    Se puede eliminar cualquier operación excepto órdenes OCO con seguimiento
    activo (status=filled y oco_status != ALL_DONE).
    """
    trade = (
        db.query(TradeExecution)
        .filter(TradeExecution.id == trade_id, TradeExecution.executed_by == user.id)
        .first()
    )
    if not trade:
        raise HTTPException(status_code=404, detail="Operación no encontrada")

    # Warn: deleting active OCO is tracked in audit log
    db.add(AuditLog(
        user_id=user.id,
        action="trade.delete",
        payload={
            "trade_id": trade.id,
            "symbol": trade.symbol,
            "side": trade.side,
            "order_type": trade.order_type,
            "error": trade.result.get("error") if trade.result else None,
        },
    ))
    db.delete(trade)
    db.commit()
    return {"detail": "Operación eliminada"}


@router.post(
    "/operations/{trade_id}/check-status",
    response_model=OCOStatusOut,
    summary="Verificar estado de orden OCO en Binance",
)
async def check_oco_operation(
    trade_id: int,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Consulta a Binance si la orden OCO de salida (TP/SL) ya se ejecutó.

    Solo funciona para operaciones tipo 'oco' con estado 'filled'.
    Si la OCO se completó, actualiza la operación con el precio y tipo de salida."""
    trade = (
        db.query(TradeExecution)
        .filter(TradeExecution.id == trade_id, TradeExecution.executed_by == user.id)
        .first()
    )
    if not trade:
        raise HTTPException(status_code=404, detail="Operación no encontrada")
    if trade.order_type != "oco":
        raise HTTPException(status_code=400, detail="Solo aplica para operaciones tipo OCO")
    if trade.status != "filled":
        raise HTTPException(status_code=400, detail="Solo se puede verificar operaciones ejecutadas (filled)")

    result_data = trade.result or {}
    oco_order = result_data.get("oco_order", {})
    order_list_id = oco_order.get("orderListId")
    if not order_list_id:
        raise HTTPException(status_code=400, detail="No se encontró orderListId en el resultado de la orden")

    if not user.binance_api_key_enc:
        raise HTTPException(status_code=400, detail="API keys de Binance no configuradas")

    try:
        oco_result = await check_oco_status(
            user.binance_api_key_enc,
            user.binance_api_secret_enc,
            trade.symbol,
            order_list_id,
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=f"Error al consultar Binance: {exc}")

    response = {
        "trade_id": trade.id,
        "oco_status": oco_result["oco_status"],
        "exit_type": oco_result.get("exit_type"),
        "exit_price": oco_result.get("exit_price"),
        "exit_time": None,
    }

    if oco_result.get("exit_time"):
        import datetime
        ts = oco_result["exit_time"]
        if isinstance(ts, (int, float)):
            response["exit_time"] = datetime.datetime.utcfromtimestamp(ts / 1000).isoformat()
        else:
            response["exit_time"] = str(ts)

    if oco_result["oco_status"] == "ALL_DONE":
        updated_result = dict(result_data)
        updated_result["oco_status"] = "ALL_DONE"
        updated_result["exit_type"] = oco_result.get("exit_type")
        updated_result["exit_price"] = oco_result.get("exit_price")
        updated_result["exit_time"] = response["exit_time"]

        # Compute and persist P&L
        exit_px = oco_result.get("exit_price") or 0.0
        entry_px = trade.price or 0.0
        qty = trade.quantity or 0.0
        if entry_px > 0 and exit_px > 0:
            if trade.side == "BUY":
                pnl_pct = round((exit_px - entry_px) / entry_px * 100, 4)
            else:
                pnl_pct = round((entry_px - exit_px) / entry_px * 100, 4)
            pnl_abs = round(pnl_pct / 100 * entry_px * qty, 4)
            from app.config import get_settings as _gs
            fee = _gs().commission_pct
            net_pnl_pct = round(pnl_pct - fee * 2, 4)
            net_pnl_abs = round(net_pnl_pct / 100 * entry_px * qty, 4)
            updated_result["pnl_pct"] = pnl_pct
            updated_result["pnl_abs"] = pnl_abs
            updated_result["commission_pct"] = fee
            updated_result["net_pnl_pct"] = net_pnl_pct
            updated_result["net_pnl_abs"] = net_pnl_abs

        trade.result = updated_result
        db.commit()
        db.refresh(trade)

    return response


@router.post(
    "/operations/{trade_id}/close",
    summary="Cerrar/cancelar una orden en Binance",
)
async def close_operation(
    trade_id: int,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Cancela una orden abierta en Binance (limit u OCO activa).

    Marca la operación como 'closed' en el sistema local."""
    trade = (
        db.query(TradeExecution)
        .filter(TradeExecution.id == trade_id, TradeExecution.executed_by == user.id)
        .first()
    )
    if not trade:
        raise HTTPException(status_code=404, detail="Operación no encontrada")
    if trade.status != "filled":
        raise HTTPException(status_code=400, detail="Solo se pueden cerrar operaciones ejecutadas")

    if not user.binance_api_key_enc:
        raise HTTPException(status_code=400, detail="API keys de Binance no configuradas")

    result_data = trade.result or {}
    oco_order = result_data.get("oco_order", {})
    order_list_id = oco_order.get("orderListId") if oco_order else None
    market_order = result_data.get("market_order", {})
    limit_order = result_data.get("limit_order", {})
    order_id = limit_order.get("orderId") or market_order.get("orderId")

    if not order_list_id and not order_id:
        raise HTTPException(status_code=400, detail="No se encontraron IDs de orden para cancelar")

    try:
        cancel_result = await cancel_order(
            user.binance_api_key_enc,
            user.binance_api_secret_enc,
            trade.symbol,
            order_id=order_id if not order_list_id else None,
            order_list_id=order_list_id,
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=f"Error cancelando en Binance: {exc}")

    # Update local state
    updated_result = dict(result_data)
    updated_result["closed_manually"] = True
    updated_result["cancel_detail"] = cancel_result
    if order_list_id:
        updated_result["oco_status"] = "ALL_DONE"
        updated_result["exit_type"] = "MANUAL"
    trade.result = updated_result

    import datetime
    db.add(AuditLog(
        user_id=user.id,
        action="trade.close",
        payload={
            "trade_id": trade.id,
            "symbol": trade.symbol,
            "order_type": trade.order_type,
            "cancel_type": cancel_result.get("type"),
        },
    ))
    db.commit()
    db.refresh(trade)

    return {"detail": "Orden cerrada en Binance", "trade_id": trade.id}


@router.post(
    "/operations/{trade_id}/validate",
    summary="Validar orden contra Binance",
)
async def validate_operation(
    trade_id: int,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Verifica si la orden existe y está activa en Binance.

    Si la orden fue cancelada, expiró o no existe, la marca como 'invalid'
    para que pueda ser eliminada del sistema."""
    trade = (
        db.query(TradeExecution)
        .filter(TradeExecution.id == trade_id, TradeExecution.executed_by == user.id)
        .first()
    )
    if not trade:
        raise HTTPException(status_code=404, detail="Operación no encontrada")
    if trade.status not in ("filled",):
        raise HTTPException(status_code=400, detail="Solo se validan operaciones con estado 'filled'")

    if not user.binance_api_key_enc:
        raise HTTPException(status_code=400, detail="API keys de Binance no configuradas")

    result_data = trade.result or {}

    # Determine IDs to check
    oco_order = result_data.get("oco_order", {})
    order_list_id = oco_order.get("orderListId") if oco_order else None
    market_order = result_data.get("market_order", {})
    order_id = market_order.get("orderId") if market_order else None

    if not order_list_id and not order_id:
        raise HTTPException(
            status_code=400,
            detail="No se encontraron IDs de orden para validar contra Binance",
        )

    try:
        # For OCO trades, validate the OCO exit order
        if trade.order_type == "oco" and order_list_id:
            validation = await validate_order_exists(
                user.binance_api_key_enc,
                user.binance_api_secret_enc,
                trade.symbol,
                order_list_id=order_list_id,
            )
        else:
            validation = await validate_order_exists(
                user.binance_api_key_enc,
                user.binance_api_secret_enc,
                trade.symbol,
                order_id=order_id,
            )
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=f"Error consultando Binance: {exc}")

    if not validation["valid"]:
        trade.status = "invalid"
        updated_result = dict(result_data)
        updated_result["validation"] = {
            "binance_status": validation["binance_status"],
            "reason": validation["reason"],
        }
        trade.result = updated_result

        db.add(AuditLog(
            user_id=user.id,
            action="trade.invalidate",
            payload={
                "trade_id": trade.id,
                "symbol": trade.symbol,
                "binance_status": validation["binance_status"],
                "reason": validation["reason"],
            },
        ))
        db.commit()
        db.refresh(trade)

    return {
        "trade_id": trade.id,
        "valid": validation["valid"],
        "binance_status": validation["binance_status"],
        "reason": validation["reason"],
        "new_status": trade.status,
    }


@router.get(
    "/analytics",
    response_model=AnalyticsOut,
    summary="Analíticas de P&L",
)
def get_analytics(
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Retorna resumen de P&L, desglose diario y mensual de operaciones cerradas."""
    return compute_analytics(user.id, db)


@router.post(
    "/operations/sync",
    summary="Sincronizar todas las operaciones activas con Binance",
)
async def sync_operations(
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Verifica TODAS las operaciones activas contra Binance en lote.

    Para cada operación activa:
    - OCO: consulta si TP o SL se ejecutó → guarda exit_type, exit_price, P&L
    - Limit/Market: valida si la orden sigue activa o fue cancelada/expirada
    - Marca como inválida las que ya no existen en Binance

    Retorna un resumen de cambios detectados.
    """
    if not user.binance_api_key_enc:
        raise HTTPException(status_code=400, detail="API keys de Binance no configuradas")

    # Get all active trades (filled, not closed)
    trades = (
        db.query(TradeExecution)
        .filter(
            TradeExecution.executed_by == user.id,
            TradeExecution.status == "filled",
        )
        .order_by(TradeExecution.executed_at.desc())
        .all()
    )

    # Filter only truly active (not already closed)
    active_trades = []
    for t in trades:
        rd = t.result or {}
        if rd.get("oco_status") == "ALL_DONE" or rd.get("closed_manually"):
            continue
        active_trades.append(t)

    import datetime as _dt
    from app.config import get_settings as _gs

    changes = []
    errors = []

    for trade in active_trades:
        result_data = trade.result or {}
        try:
            if trade.order_type == "oco":
                # Check OCO status
                oco_order = result_data.get("oco_order", {})
                order_list_id = oco_order.get("orderListId")
                if not order_list_id:
                    continue

                oco_result = await check_oco_status(
                    user.binance_api_key_enc,
                    user.binance_api_secret_enc,
                    trade.symbol,
                    order_list_id,
                )

                if oco_result["oco_status"] == "ALL_DONE":
                    # OCO completed — persist exit info + P&L
                    updated = dict(result_data)
                    updated["oco_status"] = "ALL_DONE"
                    updated["exit_type"] = oco_result.get("exit_type")
                    updated["exit_price"] = oco_result.get("exit_price")

                    exit_time = oco_result.get("exit_time")
                    if exit_time and isinstance(exit_time, (int, float)):
                        updated["exit_time"] = _dt.datetime.utcfromtimestamp(exit_time / 1000).isoformat()
                    elif exit_time:
                        updated["exit_time"] = str(exit_time)

                    # Compute P&L
                    exit_px = oco_result.get("exit_price") or 0.0
                    entry_px = trade.price or 0.0
                    qty = trade.quantity or 0.0
                    if entry_px > 0 and exit_px > 0:
                        if trade.side == "BUY":
                            pnl_pct = round((exit_px - entry_px) / entry_px * 100, 4)
                        else:
                            pnl_pct = round((entry_px - exit_px) / entry_px * 100, 4)
                        pnl_abs = round(pnl_pct / 100 * entry_px * qty, 4)
                        fee = _gs().commission_pct
                        net_pnl_pct = round(pnl_pct - fee * 2, 4)
                        net_pnl_abs = round(net_pnl_pct / 100 * entry_px * qty, 4)
                        updated["pnl_pct"] = pnl_pct
                        updated["pnl_abs"] = pnl_abs
                        updated["commission_pct"] = fee
                        updated["net_pnl_pct"] = net_pnl_pct
                        updated["net_pnl_abs"] = net_pnl_abs

                    trade.result = updated
                    changes.append({
                        "trade_id": trade.id,
                        "symbol": trade.symbol,
                        "action": "oco_completed",
                        "exit_type": oco_result.get("exit_type"),
                        "exit_price": oco_result.get("exit_price"),
                    })
                # else: still active, no change

            else:
                # Limit or market — validate existence
                market_order = result_data.get("market_order", {}) or {}
                limit_order = result_data.get("limit_order", {}) or {}
                oco_order = result_data.get("oco_order", {}) or {}
                order_id = limit_order.get("orderId") or market_order.get("orderId")
                order_list_id = oco_order.get("orderListId")

                if not order_id and not order_list_id:
                    continue

                validation = await validate_order_exists(
                    user.binance_api_key_enc,
                    user.binance_api_secret_enc,
                    trade.symbol,
                    order_id=order_id,
                    order_list_id=order_list_id,
                )

                if not validation["valid"]:
                    trade.status = "invalid"
                    updated = dict(result_data)
                    updated["validation"] = {
                        "binance_status": validation["binance_status"],
                        "reason": validation["reason"],
                    }
                    trade.result = updated
                    changes.append({
                        "trade_id": trade.id,
                        "symbol": trade.symbol,
                        "action": "invalidated",
                        "reason": validation["reason"],
                    })

        except Exception as exc:
            errors.append({
                "trade_id": trade.id,
                "symbol": trade.symbol,
                "error": str(exc),
            })

    if changes:
        db.add(AuditLog(
            user_id=user.id,
            action="trade.sync",
            payload={"changes": changes, "errors": errors},
        ))
        db.commit()

    return {
        "synced": len(active_trades),
        "changes": changes,
        "errors": errors,
    }
