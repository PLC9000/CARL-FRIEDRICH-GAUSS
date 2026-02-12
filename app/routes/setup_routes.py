"""Setup endpoints — strategy enable/disable management."""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.database import get_db
from app.auth.dependencies import get_current_user, require_admin
from app.models import User
from app.services.setup_service import list_strategy_configs, toggle_strategy

router = APIRouter(prefix="/setup", tags=["Setup"])


@router.get(
    "/strategies",
    summary="Listar estado de estrategias (enabled/disabled)",
)
def list_strategies_config(
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Returns all strategies with their enabled/disabled state."""
    return list_strategy_configs(db)


@router.post(
    "/strategies/{strategy_key}/toggle",
    summary="Toggle enabled/disabled de una estrategia",
)
def toggle_strategy_config(
    strategy_key: str,
    admin: User = Depends(require_admin),
    db: Session = Depends(get_db),
):
    """Toggle a strategy's enabled state. Admin only."""
    key = strategy_key.upper()
    try:
        return toggle_strategy(key, admin.id, db)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
