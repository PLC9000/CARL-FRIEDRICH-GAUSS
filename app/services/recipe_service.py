"""Recipe CRUD operations."""

import logging

from sqlalchemy.orm import Session

from app.models import (
    Recipe, RecipeStatus, RecipeEvaluation, AuditLog,
    Recommendation, Approval, ApprovalStatus,
)
from app.strategies import STRATEGY_MAP
from app.services.setup_service import get_enabled_strategy_keys

logger = logging.getLogger(__name__)

VALID_STRATEGIES = set(STRATEGY_MAP.keys())


def create_recipe(
    user_id: int,
    name: str,
    symbol: str,
    strategies: list[dict],
    interval: str,
    lookback_days: int,
    buy_threshold: float,
    sell_threshold: float,
    auto_threshold: float,
    max_order_pct: float,
    stop_loss_pct: float,
    take_profit_pct: float,
    strategy_params: dict | None,
    db: Session,
    *,
    auto_quantity: float | None = None,
    buy_quantity: float | None = None,
    sell_quantity: float | None = None,
    buy_order_type: str = "oco",
    sell_order_type: str = "oco",
    strength_threshold: float = 0.0,
    auto_strength_threshold: float = 0.0,
    turbo_threshold: float = 0.0,
    confirmation_minutes: float = 0.0,
    confirmation_seconds: int = 0,
    mode: str | None = None,
) -> Recipe:
    """Create a new recipe (defaults to INACTIVE)."""
    enabled_keys = get_enabled_strategy_keys(db)
    for s in strategies:
        if s["strategy"] not in VALID_STRATEGIES:
            raise ValueError(f"Estrategia desconocida: {s['strategy']}")
        if s["strategy"] not in enabled_keys:
            raise ValueError(
                f"La estrategia {s['strategy']} no está habilitada. "
                "Un administrador debe habilitarla en Setup."
            )

    # Validate weights sum to 100%
    total_weight = sum(s.get("weight", 0) for s in strategies)
    if abs(total_weight - 100) > 1:
        raise ValueError(
            f"Los pesos deben sumar 100%. Actualmente suman {total_weight:.0f}%"
        )

    # Convert percentages to fractions (0-1) for the evaluation engine
    for s in strategies:
        s["weight"] = round(s.get("weight", 0) / 100, 4)

    # Validate auto_threshold constraint
    if auto_threshold > 0 and (auto_threshold < buy_threshold or auto_threshold < sell_threshold):
        raise ValueError(
            "El umbral de auto-aprobación debe ser >= umbral de compra y >= umbral de venta"
        )

    recipe = Recipe(
        user_id=user_id,
        name=name,
        symbol=symbol.upper().strip(),
        strategies=strategies,
        interval=interval,
        lookback_days=lookback_days,
        buy_threshold=buy_threshold,
        sell_threshold=sell_threshold,
        auto_threshold=auto_threshold,
        auto_quantity=auto_quantity,
        buy_quantity=buy_quantity,
        sell_quantity=sell_quantity,
        buy_order_type=buy_order_type,
        sell_order_type=sell_order_type,
        strength_threshold=strength_threshold,
        auto_strength_threshold=auto_strength_threshold,
        turbo_threshold=turbo_threshold,
        confirmation_minutes=confirmation_minutes,
        confirmation_seconds=confirmation_seconds,
        mode=mode,
        max_order_pct=max_order_pct,
        stop_loss_pct=stop_loss_pct,
        take_profit_pct=take_profit_pct,
        strategy_params=strategy_params,
        status=RecipeStatus.inactive,
    )
    db.add(recipe)

    db.add(AuditLog(
        user_id=user_id,
        action="recipe.create",
        payload={"name": name, "symbol": symbol, "strategies": strategies},
    ))

    db.commit()
    db.refresh(recipe)
    logger.info("Recipe %d created by user %d: %s", recipe.id, user_id, name)
    return recipe


def update_recipe(recipe_id: int, user_id: int, updates: dict, db: Session) -> Recipe:
    """Update recipe fields. Only the owner can update."""
    recipe = db.query(Recipe).filter(
        Recipe.id == recipe_id, Recipe.user_id == user_id
    ).first()
    if recipe is None:
        raise ValueError("Receta no encontrada")

    allowed = {
        "name", "symbol", "strategies", "interval", "lookback_days",
        "buy_threshold", "sell_threshold", "auto_threshold",
        "auto_quantity", "buy_quantity", "sell_quantity",
        "auto_order_type", "buy_order_type", "sell_order_type",
        "strength_threshold", "auto_strength_threshold", "turbo_threshold", "confirmation_minutes", "confirmation_seconds",
        "max_order_pct", "stop_loss_pct", "take_profit_pct", "strategy_params", "mode",
        "max_ops_count", "max_ops_hours",
    }

    # Validate auto_threshold cross-field constraint
    at = updates.get("auto_threshold", recipe.auto_threshold)
    bt = updates.get("buy_threshold", recipe.buy_threshold)
    st = updates.get("sell_threshold", recipe.sell_threshold)
    if at and at > 0 and (at < bt or at < st):
        raise ValueError(
            "El umbral de auto-aprobación debe ser >= umbral de compra y >= umbral de venta"
        )

    # If strategies are being updated, validate and convert weights
    if "strategies" in updates and updates["strategies"] is not None:
        strats = updates["strategies"]
        enabled_keys = get_enabled_strategy_keys(db)
        for s in strats:
            if s["strategy"] not in enabled_keys:
                raise ValueError(
                    f"La estrategia {s['strategy']} no está habilitada. "
                    "Un administrador debe habilitarla en Setup."
                )
        total_weight = sum(s.get("weight", 0) for s in strats)
        if abs(total_weight - 100) > 1:
            raise ValueError(
                f"Los pesos deben sumar 100%. Actualmente suman {total_weight:.0f}%"
            )
        for s in strats:
            s["weight"] = round(s.get("weight", 0) / 100, 4)

    if "symbol" in updates and updates["symbol"] is not None:
        updates["symbol"] = updates["symbol"].upper().strip()

    for field, value in updates.items():
        if field in allowed:
            setattr(recipe, field, value)

    db.add(AuditLog(
        user_id=user_id,
        action="recipe.update",
        payload={"recipe_id": recipe_id, "updates": list(updates.keys())},
    ))

    db.commit()
    db.refresh(recipe)
    return recipe


def toggle_recipe_status(recipe_id: int, user_id: int, db: Session) -> Recipe:
    """Toggle between ACTIVE and INACTIVE."""
    recipe = db.query(Recipe).filter(
        Recipe.id == recipe_id, Recipe.user_id == user_id
    ).first()
    if recipe is None:
        raise ValueError("Receta no encontrada")

    new_status = (RecipeStatus.inactive if recipe.status == RecipeStatus.active
                  else RecipeStatus.active)

    if new_status == RecipeStatus.active and not recipe.mode:
        raise ValueError(
            "Debe definir el modo de la receta (ponderado o roles) antes de activarla"
        )

    recipe.status = new_status

    # When deactivating, reject all pending approvals linked to this recipe
    rejected_count = 0
    if new_status == RecipeStatus.inactive:
        pending = (
            db.query(Approval)
            .join(Recommendation, Approval.recommendation_id == Recommendation.id)
            .join(RecipeEvaluation, RecipeEvaluation.recommendation_id == Recommendation.id)
            .filter(
                RecipeEvaluation.recipe_id == recipe_id,
                Approval.status.in_([ApprovalStatus.pending, ApprovalStatus.approved]),
            )
            .all()
        )
        import datetime
        now = datetime.datetime.utcnow()
        for appr in pending:
            appr.status = ApprovalStatus.rejected
            appr.review_reason = "Receta desactivada"
            appr.reviewed_at = now
            rejected_count += 1

    db.add(AuditLog(
        user_id=user_id,
        action="recipe.toggle",
        payload={
            "recipe_id": recipe_id,
            "new_status": new_status.value,
            "rejected_approvals": rejected_count,
        },
    ))

    db.commit()
    db.refresh(recipe)
    return recipe


def delete_recipe(recipe_id: int, user_id: int, db: Session):
    """Delete a recipe and its evaluations."""
    recipe = db.query(Recipe).filter(
        Recipe.id == recipe_id, Recipe.user_id == user_id
    ).first()
    if recipe is None:
        raise ValueError("Receta no encontrada")

    db.query(RecipeEvaluation).filter(
        RecipeEvaluation.recipe_id == recipe_id
    ).delete()

    db.add(AuditLog(
        user_id=user_id,
        action="recipe.delete",
        payload={"recipe_id": recipe_id, "name": recipe.name},
    ))

    db.delete(recipe)
    db.commit()


def list_recipes(user_id: int, db: Session) -> list[Recipe]:
    """List all recipes for a user."""
    return (
        db.query(Recipe)
        .filter(Recipe.user_id == user_id)
        .order_by(Recipe.created_at.desc())
        .all()
    )


def get_recipe_evaluations(
    recipe_id: int, user_id: int, db: Session, limit: int = 50
) -> list[RecipeEvaluation]:
    """Get recent evaluations for a recipe."""
    recipe = db.query(Recipe).filter(
        Recipe.id == recipe_id, Recipe.user_id == user_id
    ).first()
    if recipe is None:
        raise ValueError("Receta no encontrada")

    return (
        db.query(RecipeEvaluation)
        .filter(RecipeEvaluation.recipe_id == recipe_id)
        .order_by(RecipeEvaluation.evaluated_at.desc())
        .limit(limit)
        .all()
    )
