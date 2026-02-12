from typing import List

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session, joinedload

from app.database import get_db
from app.models import Approval, ApprovalStatus, Recommendation, RecipeEvaluation, TradeExecution, User
from app.schemas import ApprovalAction, ApprovalHistoryOut, ApprovalOut
from app.auth.dependencies import get_current_user, require_admin
from app.services.approval_service import approve, reject

router = APIRouter(prefix="/approval", tags=["Aprobaciones"])


@router.get(
    "/history",
    response_model=List[ApprovalHistoryOut],
    summary="Historial de aprobaciones",
)
def list_approval_history(
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Lista todas las aprobaciones (aprobadas, rechazadas, pendientes) con datos enriquecidos."""
    approvals = (
        db.query(Approval)
        .options(joinedload(Approval.recommendation))
        .order_by(Approval.created_at.desc())
        .limit(200)
        .all()
    )

    results = []
    for appr in approvals:
        rec: Recommendation = appr.recommendation
        if rec is None:
            continue
        metrics = rec.metrics or {}

        results.append(ApprovalHistoryOut(
            approval_id=appr.id,
            recommendation_id=rec.id,
            symbol=rec.symbol,
            strategies_used=rec.strategy,
            final_score=metrics.get("final_score"),
            recommendation=rec.recommendation,
            confidence=rec.confidence,
            entry_price=rec.entry_price,
            stop_loss=rec.stop_loss,
            take_profit=rec.take_profit,
            explanation=rec.explanation,
            recipe_name=metrics.get("recipe_name"),
            order_type=metrics.get("order_type"),
            created_at=appr.created_at,
            status=appr.status.value if hasattr(appr.status, "value") else str(appr.status),
            review_reason=appr.review_reason,
            reviewed_at=appr.reviewed_at,
        ))

    return results


@router.delete(
    "/{approval_id}",
    summary="Eliminar una aprobación y su recomendación",
)
def delete_approval(
    approval_id: int,
    admin: User = Depends(require_admin),
    db: Session = Depends(get_db),
):
    """Elimina una aprobación y la recomendación asociada (solo admin)."""
    appr = db.query(Approval).filter(Approval.id == approval_id).first()
    if appr is None:
        raise HTTPException(status_code=404, detail="Aprobación no encontrada")

    # Delete trade executions linked to this approval
    db.query(TradeExecution).filter(TradeExecution.approval_id == appr.id).delete()

    rec = appr.recommendation
    if rec is not None:
        # Nullify evaluation FK before deleting recommendation
        db.query(RecipeEvaluation).filter(
            RecipeEvaluation.recommendation_id == rec.id
        ).update({"recommendation_id": None})

    db.delete(appr)
    if rec is not None:
        db.delete(rec)
    db.commit()
    return {"ok": True, "deleted_approval_id": approval_id}


@router.post(
    "/{approval_id}/approve",
    response_model=ApprovalOut,
    summary="Aprobar una recomendación",
)
def approve_recommendation(
    approval_id: int,
    body: ApprovalAction,
    admin: User = Depends(require_admin),
    db: Session = Depends(get_db),
):
    """Marca una recomendación pendiente como **aprobada** (solo admin).

    Una vez aprobada, se habilita la posibilidad de ejecutar el trade
    desde el endpoint `/trade/execute`.

    Es buena práctica dejar un motivo en el campo `reason` — queda
    registrado en el log de auditoría y sirve para trazabilidad.
    """
    try:
        return approve(approval_id, admin.id, body.reason, db)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.post(
    "/{approval_id}/reject",
    response_model=ApprovalOut,
    summary="Rechazar una recomendación",
)
def reject_recommendation(
    approval_id: int,
    body: ApprovalAction,
    admin: User = Depends(require_admin),
    db: Session = Depends(get_db),
):
    """Marca una recomendación pendiente como **rechazada** (solo admin).

    Una recomendación rechazada no se puede ejecutar.
    Dejá el motivo del rechazo en `reason` para que quede documentado.
    """
    try:
        return reject(approval_id, admin.id, body.reason, db)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
