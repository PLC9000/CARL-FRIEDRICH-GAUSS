import datetime
import logging
from typing import List

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session, joinedload

from app.database import get_db
from sqlalchemy import or_

from app.models import User, Approval, ApprovalStatus, Recommendation, TradeExecution, RecipeEvaluation, Recipe
from app.binance_client import fetch_candles
from app.strategies import STRATEGY_MAP
from app.services.evaluation_engine import normalize_score, compute_weighted_score, compute_strength_factor, determine_signal, is_turbo, TURBO_INTERVAL, NORMAL_INTERVAL

logger = logging.getLogger(__name__)
from app.schemas import (
    RecipeCreate, RecipeUpdate, RecipeOut,
    RecipeEvaluationOut, PendingApprovalOut,
)
from app.auth.dependencies import get_current_user, require_admin
from app.services.recipe_service import (
    create_recipe, update_recipe, toggle_recipe_status,
    delete_recipe, list_recipes, get_recipe_evaluations,
)

router = APIRouter(prefix="/recipe", tags=["Recetas"])


@router.get("/", response_model=List[RecipeOut], summary="Listar mis recetas")
def list_my_recipes(
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Devuelve todas las recetas del usuario actual."""
    recipes = list_recipes(user.id, db)
    if not recipes:
        return []

    recipe_ids = [r.id for r in recipes]

    # Batch-fetch last evaluation per recipe (single query with window function)
    from sqlalchemy import func
    subq = (
        db.query(
            RecipeEvaluation.recipe_id,
            func.max(RecipeEvaluation.id).label("max_id"),
        )
        .filter(RecipeEvaluation.recipe_id.in_(recipe_ids))
        .group_by(RecipeEvaluation.recipe_id)
        .subquery()
    )
    last_evals_rows = (
        db.query(RecipeEvaluation)
        .join(subq, RecipeEvaluation.id == subq.c.max_id)
        .all()
    )
    last_evals = {ev.recipe_id: ev for ev in last_evals_rows}

    # Batch-fetch recent 50 evaluations per recipe (for confirmation progress)
    # Only for recipes that need confirmation
    confirm_recipe_ids = []
    for r in recipes:
        conf_secs = r.confirmation_seconds or 0
        conf_mins = r.confirmation_minutes or 0.0
        req = conf_secs if conf_secs > 0 else int(conf_mins * 60) if conf_mins > 0 else 0
        ev = last_evals.get(r.id)
        if req > 0 and ev and ev.signal in ("BUY", "SELL"):
            confirm_recipe_ids.append(r.id)

    recent_evals_map: dict[int, list[RecipeEvaluation]] = {}
    if confirm_recipe_ids:
        recent_rows = (
            db.query(RecipeEvaluation)
            .filter(RecipeEvaluation.recipe_id.in_(confirm_recipe_ids))
            .order_by(RecipeEvaluation.recipe_id, RecipeEvaluation.evaluated_at.desc())
            .all()
        )
        for ev in recent_rows:
            recent_evals_map.setdefault(ev.recipe_id, []).append(ev)
        # Trim to 50 per recipe
        for rid in recent_evals_map:
            recent_evals_map[rid] = recent_evals_map[rid][:50]

    # Batch-compute ops_used for recipes with limits (grouped by cutoff)
    from sqlalchemy import func as sa_func
    import datetime as _dt
    ops_used_map: dict[int, int] = {}
    limit_recipes = [r for r in recipes if (r.max_ops_count or 0) > 0]
    if limit_recipes:
        # Group recipes by max_ops_hours to minimize queries
        by_hours: dict[float, list[int]] = {}
        for r in limit_recipes:
            h = r.max_ops_hours or 24.0
            by_hours.setdefault(h, []).append(r.id)
        now = _dt.datetime.utcnow()
        for hours, rids in by_hours.items():
            cutoff = now - _dt.timedelta(hours=hours)
            rows = (
                db.query(
                    RecipeEvaluation.recipe_id,
                    sa_func.count(TradeExecution.id),
                )
                .join(Recommendation, RecipeEvaluation.recommendation_id == Recommendation.id)
                .join(Approval, Approval.recommendation_id == Recommendation.id)
                .join(TradeExecution, TradeExecution.approval_id == Approval.id)
                .filter(
                    RecipeEvaluation.recipe_id.in_(rids),
                    TradeExecution.executed_at >= cutoff,
                    TradeExecution.status.in_(["filled", "simulated"]),
                )
                .group_by(RecipeEvaluation.recipe_id)
                .all()
            )
            for rid, cnt in rows:
                ops_used_map[rid] = cnt

    results = []
    for r in recipes:
        out = RecipeOut.model_validate(r)
        out.ops_used = ops_used_map.get(r.id, 0)
        last_eval = last_evals.get(r.id)
        if last_eval:
            out.last_score = last_eval.final_score
            out.last_signal = last_eval.signal
            out.last_strategy_results = last_eval.strategy_results
            out.last_strength_factor = last_eval.strength_value
            out.last_direction_status = last_eval.direction_status
            out.last_strength_status = last_eval.strength_status
            out.last_triggered = last_eval.triggered

            conf_secs = r.confirmation_seconds or 0
            conf_mins = r.confirmation_minutes or 0.0
            req_secs = conf_secs if conf_secs > 0 else int(conf_mins * 60) if conf_mins > 0 else 0
            out.confirmation_required_secs = req_secs

            if req_secs > 0 and last_eval.signal in ("BUY", "SELL"):
                recent = recent_evals_map.get(r.id, [])
                earliest = last_eval.evaluated_at
                for ev in recent[1:] if len(recent) > 1 else []:
                    if ev.signal != last_eval.signal:
                        break
                    if ev.direction_status and ev.direction_status != last_eval.direction_status:
                        break
                    if ev.strength_status and ev.strength_status == "FAIL":
                        break
                    earliest = ev.evaluated_at
                elapsed = (last_eval.evaluated_at - earliest).total_seconds()
                out.confirmation_elapsed_secs = round(elapsed, 1)
        results.append(out)
    return results


@router.post("/", response_model=RecipeOut, summary="Crear receta",
             status_code=201)
def create_new_recipe(
    body: RecipeCreate,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Crea una nueva receta de trading (inactiva por defecto)."""
    try:
        return create_recipe(
            user_id=user.id,
            name=body.name,
            symbol=body.symbol,
            strategies=[s.model_dump() for s in body.strategies],
            interval=body.interval,
            lookback_days=body.lookback_days,
            buy_threshold=body.buy_threshold,
            sell_threshold=body.sell_threshold,
            auto_threshold=body.auto_threshold,
            max_order_pct=body.max_order_pct,
            stop_loss_pct=body.stop_loss_pct,
            take_profit_pct=body.take_profit_pct,
            strategy_params=body.strategy_params,
            db=db,
            auto_quantity=body.auto_quantity,
            buy_quantity=body.buy_quantity,
            sell_quantity=body.sell_quantity,
            buy_order_type=body.buy_order_type,
            sell_order_type=body.sell_order_type,
            strength_threshold=body.strength_threshold,
            auto_strength_threshold=body.auto_strength_threshold,
            turbo_threshold=body.turbo_threshold,
            confirmation_minutes=body.confirmation_minutes,
            confirmation_seconds=body.confirmation_seconds,
            mode=body.mode,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.patch("/{recipe_id}", response_model=RecipeOut,
              summary="Actualizar receta")
def update_existing_recipe(
    recipe_id: int,
    body: RecipeUpdate,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Actualiza campos de una receta existente."""
    updates = body.model_dump(exclude_unset=True)
    if "strategies" in updates and updates["strategies"] is not None:
        updates["strategies"] = [
            s.model_dump() if hasattr(s, "model_dump") else s
            for s in updates["strategies"]
        ]
    try:
        return update_recipe(recipe_id, user.id, updates, db)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.post("/{recipe_id}/toggle", response_model=RecipeOut,
             summary="Activar/desactivar receta")
def toggle_recipe(
    recipe_id: int,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Alterna el estado de la receta entre activa e inactiva."""
    try:
        return toggle_recipe_status(recipe_id, user.id, db)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.delete("/{recipe_id}", summary="Eliminar receta")
def remove_recipe(
    recipe_id: int,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Elimina una receta y todo su historial de evaluaciones."""
    try:
        delete_recipe(recipe_id, user.id, db)
        return {"detail": "Receta eliminada"}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.get("/{recipe_id}/evaluations",
            response_model=List[RecipeEvaluationOut],
            summary="Historial de evaluaciones")
def get_evaluations(
    recipe_id: int,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Devuelve las últimas evaluaciones de una receta."""
    try:
        return get_recipe_evaluations(recipe_id, user.id, db)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


# ── Simulación de receta (flujo de cálculo) ───────────────────────────

@router.get("/{recipe_id}/simulate", summary="Simular receta en vivo")
async def simulate_recipe(
    recipe_id: int,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Run all strategies in a recipe and return the full scoring breakdown."""
    recipe = db.query(Recipe).filter(
        Recipe.id == recipe_id, Recipe.user_id == user.id
    ).first()
    if not recipe:
        raise HTTPException(status_code=404, detail="Receta no encontrada")

    now = datetime.datetime.utcnow()
    start_dt = now - datetime.timedelta(days=recipe.lookback_days)

    try:
        candles = await fetch_candles(recipe.symbol, recipe.interval, start_dt, now)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Error obteniendo velas: {exc}")

    if candles is None or len(candles) == 0:
        raise HTTPException(status_code=400, detail="No se obtuvieron velas")

    last_close = round(float(candles[-1, 4]), 8)

    strategy_details = []
    for sc in recipe.strategies:
        skey = sc["strategy"]
        weight = sc.get("weight", 1.0)
        role = sc.get("role", "direction")
        fn = STRATEGY_MAP.get(skey)
        if fn is None:
            strategy_details.append({
                "strategy": skey, "weight": weight, "role": role,
                "error": f"Estrategia {skey} no encontrada",
            })
            continue

        params = None
        if recipe.strategy_params and skey in recipe.strategy_params:
            params = recipe.strategy_params[skey]

        try:
            result = fn(candles, params=params)
        except Exception as exc:
            strategy_details.append({
                "strategy": skey, "weight": weight, "role": role,
                "error": str(exc),
            })
            continue

        score = normalize_score(result["recommendation"], result["confidence"])
        force = result.get("force", result["confidence"] / 100.0) if role == "strength" else None
        raw_force = result.get("force")

        detail = {
            "strategy": skey,
            "weight": weight,
            "role": role,
            "recommendation": result["recommendation"],
            "confidence": round(result["confidence"], 2),
            "score": round(score, 4),
            "entry": result.get("entry"),
            "stop_loss": result.get("stop_loss"),
            "take_profit": result.get("take_profit"),
            "explanation": result.get("explanation"),
            "metrics": result.get("metrics"),
        }
        if force is not None:
            detail["force"] = round(force, 4)
        elif raw_force is not None:
            detail["force"] = round(raw_force, 4)
        strategy_details.append(detail)

    # ── Mode-conditional scoring ────────────────────────────────────
    valid = [d for d in strategy_details if "error" not in d]
    recipe_mode = getattr(recipe, "mode", None) or "roles"

    if recipe_mode == "weighted":
        # Weighted mode: ALL strategies contribute to a single weighted score
        total_w = sum(d["weight"] for d in valid)
        final_score = (sum(d["score"] * d["weight"] for d in valid) / total_w) if total_w else 0.0
        final_score = max(-1.0, min(1.0, final_score))
        strength_factor = 1.0
        strength_th = 0.0
    else:
        # Roles mode: direction + strength dual-trigger
        final_score = compute_weighted_score(valid) if valid else 0.0
        strength_factor = compute_strength_factor(valid) if valid else 1.0
        strength_th = getattr(recipe, "strength_threshold", 0.0) or 0.0

    signal = determine_signal(
        final_score, recipe.buy_threshold, recipe.sell_threshold,
        strength_factor=strength_factor, strength_threshold=strength_th,
    )

    # ── Breakdown of the combination ─────────────────────────────
    if recipe_mode == "weighted":
        # Weighted: single contribution list with all strategies
        total_w = sum(d["weight"] for d in valid) or 1
        dir_contributions = []
        for d in valid:
            rel_weight = d["weight"] / total_w
            contribution = d["score"] * rel_weight
            dir_contributions.append({
                "strategy": d["strategy"],
                "score_pct": round(d["score"] * 100, 1),
                "weight_pct": round(d["weight"] * 100, 1),
                "relative_weight_pct": round(rel_weight * 100, 1),
                "contribution_pct": round(contribution * 100, 1),
            })
        str_contributions = []
        dir_score = final_score
        str_factor = 1.0
        str_results = []
    else:
        # Roles: separate direction and strength
        dir_results = [d for d in valid if d.get("role", "direction") == "direction"]
        str_results = [d for d in valid if d.get("role") == "strength"]

        dir_w = sum(d["weight"] for d in dir_results) if dir_results else 0
        dir_score = (sum(d["score"] * d["weight"] for d in dir_results) / dir_w) if dir_w else 0.0
        str_w = sum(d["weight"] for d in str_results) if str_results else 0
        str_factor = (sum(d.get("force", 0) * d["weight"] for d in str_results) / str_w) if str_w else 1.0

        dir_contributions = []
        for d in dir_results:
            rel_weight = d["weight"] / dir_w if dir_w else 0
            contribution = d["score"] * rel_weight
            dir_contributions.append({
                "strategy": d["strategy"],
                "score_pct": round(d["score"] * 100, 1),
                "weight_pct": round(d["weight"] * 100, 1),
                "relative_weight_pct": round(rel_weight * 100, 1),
                "contribution_pct": round(contribution * 100, 1),
            })

        str_contributions = []
        for d in str_results:
            rel_weight = d["weight"] / str_w if str_w else 0
            force_val = d.get("force", 0)
            contribution = force_val * rel_weight
            str_contributions.append({
                "strategy": d["strategy"],
                "force_pct": round(force_val * 100, 1),
                "weight_pct": round(d["weight"] * 100, 1),
                "relative_weight_pct": round(rel_weight * 100, 1),
                "contribution_pct": round(contribution * 100, 1),
            })

    # Per-strategy gate statuses for confirmation info
    direction_signal = "HOLD"
    if final_score >= recipe.buy_threshold:
        direction_signal = "BUY"
    elif final_score <= -recipe.sell_threshold:
        direction_signal = "SELL"

    strength_gate_ok = strength_factor >= strength_th if strength_th > 0 else True

    conf_secs = getattr(recipe, "confirmation_seconds", 0) or 0
    conf_mins = getattr(recipe, "confirmation_minutes", 0.0) or 0.0
    effective_conf_secs = conf_secs if conf_secs > 0 else (conf_mins * 60 if conf_mins > 0 else 0)

    result = {
        "recipe_id": recipe.id,
        "recipe_name": recipe.name,
        "mode": getattr(recipe, "mode", None),
        "symbol": recipe.symbol,
        "interval": recipe.interval,
        "lookback_days": recipe.lookback_days,
        "candle_count": len(candles),
        "last_close": last_close,
        "strategies": strategy_details,
        "confirmation": {
            "enabled": effective_conf_secs > 0,
            "seconds": int(effective_conf_secs),
            "display": f"{int(effective_conf_secs)}s" if effective_conf_secs > 0 else "off",
            "direction_status": direction_signal,
            "strength_status": "PASS" if strength_gate_ok else "FAIL",
        },
        "combination": {
            "direction_contributions": dir_contributions,
            "direction_score": round(dir_score, 4),
            "direction_pct": round(dir_score * 100, 1),
            "strength_contributions": str_contributions,
            "strength_factor": round(str_factor, 4),
            "strength_pct": round(str_factor * 100, 1),
            "has_strength": len(str_results) > 0,
            "strength_gate_ok": strength_gate_ok,
            "final_score": round(final_score, 4),
            "final_pct": round(final_score * 100, 1),
            "formula": (
                f"promedio ponderado = {round(final_score*100,1)}%"
                if recipe_mode == "weighted" else
                (
                    f"dirección ({round(dir_score*100,1)}%) AND fuerza ({round(str_factor*100,1)}%) ≥ {round(strength_th*100,1)}%"
                    if str_results else
                    f"promedio ponderado = {round(final_score*100,1)}%"
                )
            ),
        },
        "thresholds": {
            "buy": recipe.buy_threshold,
            "buy_pct": round(recipe.buy_threshold * 100, 1),
            "sell": recipe.sell_threshold,
            "sell_pct": round(recipe.sell_threshold * 100, 1),
            "auto": recipe.auto_threshold,
            "auto_pct": round(recipe.auto_threshold * 100, 1),
            "strength": strength_th,
            "strength_pct": round(strength_th * 100, 1),
            "auto_strength": getattr(recipe, "auto_strength_threshold", 0.0) or 0.0,
            "auto_strength_pct": round((getattr(recipe, "auto_strength_threshold", 0.0) or 0.0) * 100, 1),
        },
        "signal": signal,
        "execution": {
            "buy_quantity": getattr(recipe, "buy_quantity", None),
            "sell_quantity": getattr(recipe, "sell_quantity", None),
            "auto_quantity": getattr(recipe, "auto_quantity", None),
            "buy_order_type": getattr(recipe, "buy_order_type", "oco"),
            "sell_order_type": getattr(recipe, "sell_order_type", "oco"),
            "stop_loss_pct": recipe.stop_loss_pct,
            "take_profit_pct": recipe.take_profit_pct,
            "max_order_pct": recipe.max_order_pct,
            "turbo_threshold": getattr(recipe, "turbo_threshold", 0) or 0,
            "turbo_pct": round((getattr(recipe, "turbo_threshold", 0) or 0) * 100, 1),
        },
        "entry": None,
        "stop_loss": None,
        "take_profit": None,
    }

    # Compute entry/SL/TP from primary strategy
    candidates = [d for d in strategy_details if "error" not in d and d.get("entry")]
    if candidates:
        primary = max(candidates, key=lambda r: abs(r["score"] * r["weight"]))
        entry_price = primary.get("entry")
        if entry_price and entry_price > 0:
            result["entry"] = entry_price
            if signal == "BUY":
                result["stop_loss"] = round(entry_price * (1 - recipe.stop_loss_pct / 100), 8)
                result["take_profit"] = round(entry_price * (1 + recipe.take_profit_pct / 100), 8)
            elif signal == "SELL":
                result["stop_loss"] = round(entry_price * (1 + recipe.stop_loss_pct / 100), 8)
                result["take_profit"] = round(entry_price * (1 - recipe.take_profit_pct / 100), 8)

    return result


# ── Aprobaciones pendientes (para la solapa Aprobaciones) ─────────────

@router.get("/pending-approvals", response_model=List[PendingApprovalOut],
            summary="Aprobaciones pendientes")
def list_pending_approvals(
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Lista aprobaciones pendientes + auto-aprobadas sin ejecutar."""
    # Auto-expire: reject pending + approved-not-executed approvals > 30 min
    cutoff = datetime.datetime.utcnow() - datetime.timedelta(minutes=30)

    # IDs of approvals that already have a trade execution
    executed_ids = {
        row[0]
        for row in db.query(TradeExecution.approval_id).all()
    }

    expired = (
        db.query(Approval)
        .filter(
            or_(
                Approval.status == ApprovalStatus.pending,
                Approval.status == ApprovalStatus.approved,
            ),
            Approval.created_at < cutoff,
        )
        .all()
    )
    expired_count = 0
    for exp in expired:
        if exp.status == ApprovalStatus.approved and exp.id in executed_ids:
            continue
        exp.status = ApprovalStatus.rejected
        exp.review_reason = "Expirada (>30 min sin acción)"
        exp.reviewed_at = datetime.datetime.utcnow()
        expired_count += 1
    if expired_count:
        db.commit()

    # Pending approvals + approved-not-executed (auto-approved waiting action)
    approvals = (
        db.query(Approval)
        .options(joinedload(Approval.recommendation))
        .filter(
            or_(
                Approval.status == ApprovalStatus.pending,
                Approval.status == ApprovalStatus.approved,
            )
        )
        .order_by(Approval.created_at.desc())
        .all()
    )

    results = []
    for appr in approvals:
        # Skip approved items that already have a trade
        if appr.status == ApprovalStatus.approved and appr.id in executed_ids:
            continue

        rec: Recommendation = appr.recommendation
        if rec is None:
            continue
        metrics = rec.metrics or {}

        status_val = appr.status.value if hasattr(appr.status, "value") else str(appr.status)

        results.append(PendingApprovalOut(
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
            status=status_val,
            created_at=appr.created_at,
        ))

    return results


@router.get("/engine-status", summary="Estado del motor de evaluación")
def engine_status(user: User = Depends(get_current_user)):
    """Devuelve si el engine está en modo turbo."""
    turbo = is_turbo()
    return {
        "turbo": turbo,
        "interval": TURBO_INTERVAL if turbo else NORMAL_INTERVAL,
    }


# ── Test de ejecución automática ──────────────────────────────────────

@router.post("/{recipe_id}/test-execute/{side}",
             summary="Simular ejecución automática (test)")
async def test_execute(
    recipe_id: int,
    side: str,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Simula que se alcanzó el umbral y ejecuta el flujo completo:
    Recommendation → Approval (auto-aprobado) → Trade execution.
    Solo para testing — usa los parámetros configurados en la receta.
    """
    if side.upper() not in ("BUY", "SELL"):
        raise HTTPException(status_code=400, detail="side debe ser BUY o SELL")
    signal = side.upper()

    recipe = db.query(Recipe).filter(
        Recipe.id == recipe_id, Recipe.user_id == user.id,
    ).first()
    if not recipe:
        raise HTTPException(status_code=404, detail="Receta no encontrada")

    # Determine quantity and order type
    if signal == "BUY":
        qty = recipe.buy_quantity or recipe.auto_quantity or 0
        order_type = recipe.buy_order_type or "oco"
    else:
        qty = recipe.sell_quantity or recipe.auto_quantity or 0
        order_type = recipe.sell_order_type or "oco"

    if not qty or qty <= 0:
        raise HTTPException(
            status_code=400,
            detail=f"No hay cantidad de {'compra' if signal == 'BUY' else 'venta'} configurada",
        )

    # Get current price from Binance
    from app.services.binance_filters import _get_current_price
    try:
        entry = await _get_current_price(recipe.symbol)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Error obteniendo precio: {exc}")

    # Compute SL/TP from recipe percentages
    if signal == "BUY":
        stop_loss = round(entry * (1 - recipe.stop_loss_pct / 100), 8)
        take_profit = round(entry * (1 + recipe.take_profit_pct / 100), 8)
    else:
        stop_loss = round(entry * (1 + recipe.stop_loss_pct / 100), 8)
        take_profit = round(entry * (1 - recipe.take_profit_pct / 100), 8)

    now = datetime.datetime.utcnow()

    # Create Recommendation
    rec = Recommendation(
        user_id=user.id,
        symbol=recipe.symbol,
        interval=recipe.interval,
        period_start=now.isoformat(),
        period_end=now.isoformat(),
        strategy="TEST",
        metrics={
            "recipe_id": recipe.id,
            "recipe_name": recipe.name,
            "final_score": 1.0,
            "order_type": order_type,
            "test_mode": True,
        },
        recommendation=signal,
        confidence=100.0,
        entry_price=entry,
        stop_loss=stop_loss,
        take_profit=take_profit,
        explanation=f"TEST: Ejecución manual de {signal} para receta '{recipe.name}'",
    )
    db.add(rec)
    db.flush()

    # Create auto-approved Approval
    approval = Approval(
        recommendation_id=rec.id,
        status=ApprovalStatus.approved,
        reviewed_by=user.id,
        review_reason="Test de ejecución automática",
        reviewed_at=now,
    )
    db.add(approval)
    from app.models import AuditLog
    db.add(AuditLog(
        user_id=user.id,
        action="recipe.test_execute",
        payload={
            "recipe_id": recipe.id,
            "signal": signal,
            "quantity": qty,
            "order_type": order_type,
            "entry": entry,
        },
    ))
    db.commit()
    db.refresh(approval)

    # Execute trade via normal flow
    from app.services.trade_service import execute_trade
    try:
        trade = await execute_trade(
            approval_id=approval.id,
            admin_id=user.id,
            quantity=qty,
            order_type=order_type,
            db=db,
        )
        return {
            "status": trade.status,
            "trade_id": trade.id,
            "signal": signal,
            "quantity": qty,
            "order_type": order_type,
            "entry": entry,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "result": trade.result,
        }
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Error ejecutando trade: {exc}")
