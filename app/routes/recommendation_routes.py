from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.config import get_settings
from app.database import get_db
from app.models import User, AuditLog
from app.schemas import RecommendationRequest, RecommendationOut, LevelsOut, ReturnsOut
from app.auth.dependencies import get_current_user
from app.services.analysis_service import generate_recommendation

router = APIRouter(tags=["Recomendaciones"])


def _calc_returns(entry: float | None, sl: float | None, tp: float | None, side: str, fee_pct: float) -> ReturnsOut:
    """Calculate potential returns accounting for round-trip commissions.

    fee_pct is per-trade (e.g. 0.1 for 0.1%).  Round-trip = 2 * fee_pct.
    For BUY: profit = (tp - entry)/entry, loss = (sl - entry)/entry  (loss is negative)
    For SELL: profit = (entry - tp)/entry, loss = (entry - sl)/entry  (loss is negative)
    """
    total_fees = round(fee_pct * 2, 4)

    if entry is None or entry == 0:
        return ReturnsOut(
            commission_pct=fee_pct,
            total_fees_pct=total_fees,
            gross_profit_pct=None,
            net_profit_pct=None,
            gross_loss_pct=None,
            net_loss_pct=None,
            risk_reward=None,
        )

    if side == "BUY":
        gross_profit = ((tp - entry) / entry * 100) if tp else None
        gross_loss = ((sl - entry) / entry * 100) if sl else None
    elif side == "SELL":
        gross_profit = ((entry - tp) / entry * 100) if tp else None
        gross_loss = ((entry - sl) / entry * 100) if sl else None
    else:
        return ReturnsOut(
            commission_pct=fee_pct,
            total_fees_pct=total_fees,
            gross_profit_pct=None,
            net_profit_pct=None,
            gross_loss_pct=None,
            net_loss_pct=None,
            risk_reward=None,
        )

    net_profit = round(gross_profit - total_fees, 4) if gross_profit is not None else None
    net_loss = round(gross_loss - total_fees, 4) if gross_loss is not None else None

    gross_profit = round(gross_profit, 4) if gross_profit is not None else None
    gross_loss = round(gross_loss, 4) if gross_loss is not None else None

    if net_profit is not None and net_loss is not None and net_loss != 0:
        risk_reward = round(abs(net_profit / net_loss), 2)
    else:
        risk_reward = None

    return ReturnsOut(
        commission_pct=fee_pct,
        total_fees_pct=total_fees,
        gross_profit_pct=gross_profit,
        net_profit_pct=net_profit,
        gross_loss_pct=gross_loss,
        net_loss_pct=net_loss,
        risk_reward=risk_reward,
    )


@router.post(
    "/recommendation",
    response_model=RecommendationOut,
    summary="Solicitar recomendación de trading",
)
async def create_recommendation(
    body: RecommendationRequest,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Analiza un par de Binance con la estrategia elegida y devuelve una recomendación.

    **Cualquier usuario autenticado** puede usar este endpoint (no hace falta ser admin).

    El sistema:
    1. Valida que el símbolo exista en Binance
    2. Descarga las velas históricas del período indicado
    3. Corre la estrategia seleccionada (A, B o C)
    4. Devuelve la recomendación con niveles de precio, métricas y **retornos potenciales
       descontando las comisiones de Binance** (0.1% por operación = 0.2% ida y vuelta)
    5. Crea automáticamente un registro de aprobación en estado **pendiente**
    """
    db.add(AuditLog(
        user_id=user.id,
        action="recommendation.request",
        payload=body.model_dump(),
    ))
    db.commit()

    try:
        rec = await generate_recommendation(
            user_id=user.id,
            symbol=body.symbol,
            interval=body.interval,
            strategy=body.strategy,
            last_n_days=body.last_n_days,
            start=body.start,
            end=body.end,
            db=db,
            params=body.params,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Error de conexión con Binance: {exc}")

    settings = get_settings()
    returns = _calc_returns(
        entry=rec.entry_price,
        sl=rec.stop_loss,
        tp=rec.take_profit,
        side=rec.recommendation,
        fee_pct=settings.commission_pct,
    )

    return RecommendationOut(
        id=rec.id,
        symbol=rec.symbol,
        interval=rec.interval,
        period={"start": rec.period_start, "end": rec.period_end},
        strategy=rec.strategy,
        metrics=rec.metrics,
        recommendation=rec.recommendation,
        confidence=rec.confidence,
        levels=LevelsOut(
            entry=rec.entry_price,
            stop_loss=rec.stop_loss,
            take_profit=rec.take_profit,
        ),
        returns=returns,
        explanation=rec.explanation,
    )
