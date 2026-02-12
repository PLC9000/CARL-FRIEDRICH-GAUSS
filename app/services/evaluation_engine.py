"""Background evaluation engine for active recipes.

Runs as an asyncio task started/stopped via FastAPI lifespan.
Every 30 seconds it checks which active recipes are due for evaluation,
fetches candles, runs each strategy, computes a combined score, and
triggers a Recommendation + Approval when the score crosses thresholds.
"""

import asyncio
import datetime
import logging

from sqlalchemy.orm import Session

from app.database import SessionLocal
from sqlalchemy import or_

from app.models import (
    Recipe, RecipeStatus, RecipeEvaluation,
    Recommendation, Approval, ApprovalStatus, AuditLog, TradeExecution,
    User,
)
from app.binance_client import fetch_candles
from app.strategies import STRATEGY_MAP
from app.services.setup_service import get_enabled_strategy_keys
from app.services.market_data_service import prefetch_market_data
from app.services.claude_outcome_service import (
    backfill_outcomes,
    get_recent_outcomes,
    record_prediction,
)

logger = logging.getLogger(__name__)

INTERVAL_SECONDS = {
    "1m": 60,
    "5m": 300,
    "15m": 900,
    "1h": 3600,
    "4h": 14400,
    "1d": 86400,
}

_engine_task: asyncio.Task | None = None
_stop_event = asyncio.Event()
_turbo_mode: bool = False
NORMAL_INTERVAL = 30.0
TURBO_INTERVAL = 15.0
SYNC_INTERVAL = 30.0  # seconds between Binance sync checks
_last_sync_time: float = 0.0


# ── Public API ─────────────────────────────────────────────────────────

async def start_engine():
    """Start the background evaluation loop."""
    global _engine_task
    _stop_event.clear()
    _engine_task = asyncio.create_task(_evaluation_loop())
    logger.info("Evaluation engine started")


def is_turbo() -> bool:
    """Return whether the engine is in turbo mode."""
    return _turbo_mode


async def stop_engine():
    """Stop the background evaluation loop."""
    global _engine_task
    _stop_event.set()
    if _engine_task:
        _engine_task.cancel()
        try:
            await _engine_task
        except asyncio.CancelledError:
            pass
        _engine_task = None
    logger.info("Evaluation engine stopped")


# ── Score helpers (exported for tests) ─────────────────────────────────

def normalize_score(recommendation: str, confidence: float) -> float:
    """Convert strategy output to normalized score in [-1, +1]."""
    if recommendation == "BUY":
        return confidence / 100.0
    elif recommendation == "SELL":
        return -(confidence / 100.0)
    return 0.0


def compute_weighted_score(strategy_results: list[dict]) -> float:
    """Compute direction-only score from direction-role strategies.

    Returns the weighted average of direction strategies in [-1, +1].
    Strength strategies are NOT included in this score — they act as
    an independent gate via compute_strength_factor().
    """
    dir_results = [r for r in strategy_results if r.get("role", "direction") == "direction"]

    if dir_results:
        dir_w = sum(r["weight"] for r in dir_results)
        dir_score = sum(r["score"] * r["weight"] for r in dir_results) / dir_w if dir_w else 0.0
    else:
        dir_score = 0.0

    return max(-1.0, min(1.0, dir_score))


def compute_strength_factor(strategy_results: list[dict]) -> float:
    """Compute the strength factor from strength-role strategies.

    Returns the weighted average of force values in [0, 1].
    If there are no strength strategies, returns 1.0 (no gate).
    """
    str_results = [r for r in strategy_results if r.get("role") == "strength"]

    if str_results:
        str_w = sum(r["weight"] for r in str_results)
        str_factor = sum(r["force"] * r["weight"] for r in str_results) / str_w if str_w else 1.0
        return max(0.0, min(1.0, str_factor))
    return 1.0


def determine_signal(
    score: float,
    buy_threshold: float,
    sell_threshold: float,
    strength_factor: float = 1.0,
    strength_threshold: float = 0.0,
) -> str:
    """Map direction score to signal using dual-trigger logic.

    1. Direction gate: score must cross buy/sell threshold.
    2. Strength gate: if strength_threshold > 0, strength_factor must
       be >= strength_threshold for the signal to fire.

    Both conditions must be met simultaneously.
    """
    # Check direction gate first
    if score >= buy_threshold:
        direction = "BUY"
    elif score <= -sell_threshold:
        direction = "SELL"
    else:
        return "HOLD"

    # Check strength gate
    if strength_threshold > 0 and strength_factor < strength_threshold:
        return "HOLD"

    return direction


# ── Main loop ──────────────────────────────────────────────────────────

async def _evaluation_loop():
    """Check active recipes with dynamic interval (turbo/normal)."""
    while not _stop_event.is_set():
        try:
            await _evaluate_due_recipes()
        except Exception:
            logger.exception("Error in evaluation loop")

        # Periodic Binance sync for active trades
        try:
            await _maybe_sync_active_trades()
        except Exception:
            logger.exception("Error in trade sync")

        sleep = TURBO_INTERVAL if _turbo_mode else NORMAL_INTERVAL
        try:
            await asyncio.wait_for(_stop_event.wait(), timeout=sleep)
            break
        except asyncio.TimeoutError:
            pass


async def _evaluate_due_recipes():
    """Find and evaluate active recipes that are due."""
    global _turbo_mode
    db: Session = SessionLocal()
    try:
        now = datetime.datetime.utcnow()
        recipes = (
            db.query(Recipe)
            .filter(Recipe.status == RecipeStatus.active)
            .all()
        )

        enabled_keys = get_enabled_strategy_keys(db)

        # Compute executed_ids ONCE per tick for reuse
        executed_ids = {
            row[0]
            for row in db.query(TradeExecution.approval_id).all()
        }

        # Expire stale approvals ONCE per tick, reusing executed_ids
        _expire_stale_approvals(db, executed_ids)

        # Batch-fetch auto_only flags for recipe owners
        user_ids = {r.user_id for r in recipes}
        auto_only_flags = {}
        if user_ids:
            rows = db.query(User.id, User.auto_only).filter(User.id.in_(user_ids)).all()
            auto_only_flags = {uid: bool(ao) for uid, ao in rows}

        any_turbo = False
        for recipe in recipes:
            if not recipe.mode:
                continue  # skip recipes without mode defined
            interval_secs = INTERVAL_SECONDS.get(recipe.interval, 3600)
            turbo_th = recipe.turbo_threshold or 0

            # In turbo mode, use 15s min interval instead of recipe interval
            effective_interval = min(interval_secs, TURBO_INTERVAL) if _turbo_mode else interval_secs

            if recipe.last_evaluated_at:
                elapsed = (now - recipe.last_evaluated_at).total_seconds()
                if elapsed < effective_interval:
                    # Still check if this recipe should activate turbo
                    if turbo_th > 0:
                        last_eval = (
                            db.query(RecipeEvaluation)
                            .filter(RecipeEvaluation.recipe_id == recipe.id)
                            .order_by(RecipeEvaluation.evaluated_at.desc())
                            .first()
                        )
                        if last_eval and abs(last_eval.final_score) >= turbo_th:
                            any_turbo = True
                    continue

            try:
                score = await _evaluate_recipe(recipe, db, enabled_keys, executed_ids, auto_only=auto_only_flags.get(recipe.user_id, False))
                # Check if this evaluation activates turbo
                if turbo_th > 0 and score is not None and abs(score) >= turbo_th:
                    any_turbo = True
            except Exception:
                logger.exception(
                    "Error evaluating recipe %d (%s)", recipe.id, recipe.name
                )

        if any_turbo != _turbo_mode:
            _turbo_mode = any_turbo
            logger.info("Engine turbo mode: %s", "ON (15s)" if _turbo_mode else "OFF (30s)")
    finally:
        db.close()


async def _evaluate_recipe(recipe: Recipe, db: Session, enabled_keys: set[str] | None = None, executed_ids: set | None = None, *, auto_only: bool = False) -> float | None:
    """Run all strategies in a recipe and compute combined score.

    Returns the final_score (for turbo tracking) or None on failure.
    """
    # Early return if no strategies configured
    if not recipe.strategies:
        return None

    now = datetime.datetime.utcnow()

    start_dt = now - datetime.timedelta(days=recipe.lookback_days)
    end_dt = now

    try:
        candles = await fetch_candles(
            recipe.symbol, recipe.interval, start_dt, end_dt
        )
    except Exception as exc:
        logger.warning("Candle fetch failed for recipe %d: %s", recipe.id, exc)
        return None

    # ── Pre-fetch extra data for Strategy M (Claude AI) ─────────────
    _has_m = any(s["strategy"] == "M" for s in recipe.strategies)
    _market_data: dict | None = None
    if _has_m:
        try:
            _market_data = await prefetch_market_data(recipe.symbol)
            # Back-fill past prediction outcomes & gather learning history
            current_px = float(candles[-1][4]) if candles else 0.0
            if current_px > 0:
                backfill_outcomes(db, recipe.symbol, current_px)
            _market_data["past_outcomes"] = get_recent_outcomes(db, recipe.symbol)
            # Read Anthropic key: prefer per-user encrypted key, fallback to env var
            _anthropic_key = ""
            try:
                from app.auth.encryption import decrypt
                from app.models import User as _User
                _owner = db.get(_User, recipe.user_id)
                enc = getattr(_owner, "anthropic_api_key_enc", "") or "" if _owner else ""
                if enc:
                    _anthropic_key = decrypt(enc)
            except Exception as _e:
                logger.debug("Could not read user Anthropic key: %s", _e)
            if not _anthropic_key:
                from app.config import get_settings as _gs
                _anthropic_key = _gs().anthropic_api_key
            _market_data["anthropic_api_key"] = _anthropic_key
        except Exception as exc:
            logger.warning("Pre-fetch for Strategy M failed: %s", exc)
            _market_data = None

    strategy_results = []
    for strat_config in recipe.strategies:
        strat_key = strat_config["strategy"]
        weight = strat_config.get("weight", 1.0)
        role = strat_config.get("role", "direction")

        # Skip disabled strategies
        if enabled_keys is not None and strat_key not in enabled_keys:
            logger.info("Skipping disabled strategy %s in recipe %d", strat_key, recipe.id)
            continue

        strategy_fn = STRATEGY_MAP.get(strat_key)
        if strategy_fn is None:
            logger.warning("No function for strategy %s", strat_key)
            continue

        params = None
        if recipe.strategy_params and strat_key in recipe.strategy_params:
            params = dict(recipe.strategy_params[strat_key])
        # Inject market data for Strategy M
        if strat_key == "M" and _market_data:
            params = params or {}
            params["_market_data"] = _market_data

        try:
            result = strategy_fn(candles, params=params)
        except Exception as exc:
            logger.warning(
                "Strategy %s failed for recipe %d: %s", strat_key, recipe.id, exc
            )
            continue

        # Record Claude AI prediction for learning loop
        if strat_key == "M" and result["recommendation"] != "NO-TRADE":
            try:
                metrics = result.get("metrics") or {}
                record_prediction(
                    db,
                    recipe_id=recipe.id,
                    symbol=recipe.symbol,
                    interval=recipe.interval,
                    direction=metrics.get("direction", 0),
                    intensity=metrics.get("intensity", 0),
                    confidence=metrics.get("ai_confidence", 0),
                    entry_price=result.get("entry") or 0.0,
                    reasoning=result.get("explanation", ""),
                )
            except Exception as exc:
                logger.warning("Failed to record Claude prediction: %s", exc)

        score = normalize_score(result["recommendation"], result["confidence"])

        # For strength-role strategies, use the force field if available,
        # otherwise fall back to confidence/100.
        force = result.get("force", result["confidence"] / 100.0) if role == "strength" else None

        # Always capture raw force from strategy (e.g. ADX always returns
        # force 0-1 regardless of configured role).
        raw_force = result.get("force")

        entry = {
            "strategy": strat_key,
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
        # Include force for strength calc
        if force is not None:
            entry["force"] = round(force, 4)
        # Always include raw force for display (ADX = 0-100 bar)
        elif raw_force is not None:
            entry["force"] = round(raw_force, 4)

        strategy_results.append(entry)

    if not strategy_results:
        logger.warning("No strategy produced results for recipe %d", recipe.id)
        return None

    recipe_mode = recipe.mode or "roles"

    if recipe_mode == "weighted":
        # Weighted mode: ALL strategies contribute to a single weighted score
        total_w = sum(r["weight"] for r in strategy_results)
        final_score = (sum(r["score"] * r["weight"] for r in strategy_results) / total_w) if total_w else 0.0
        final_score = max(-1.0, min(1.0, final_score))
        strength_factor = 1.0  # no strength gate in weighted mode
        strength_th = 0.0
    else:
        # Roles mode: direction + strength dual-trigger
        final_score = compute_weighted_score(strategy_results)
        strength_factor = compute_strength_factor(strategy_results)
        strength_th = recipe.strength_threshold or 0.0

    signal = determine_signal(
        final_score, recipe.buy_threshold, recipe.sell_threshold,
        strength_factor=strength_factor, strength_threshold=strength_th,
    )

    # ── Per-strategy gate statuses ───────────────────────────────────
    direction_signal = "HOLD"
    if final_score >= recipe.buy_threshold:
        direction_signal = "BUY"
    elif final_score <= -recipe.sell_threshold:
        direction_signal = "SELL"

    strength_ok = True
    if strength_th > 0:
        strength_ok = strength_factor >= strength_th

    # ── Confirmation window check (seconds priority over minutes) ────
    conf_secs = recipe.confirmation_seconds or 0
    conf_mins = recipe.confirmation_minutes or 0.0
    effective_conf_secs = conf_secs if conf_secs > 0 else (conf_mins * 60 if conf_mins > 0 else 0)

    if signal in ("BUY", "SELL") and effective_conf_secs > 0:
        # Fetch last N evaluations (enough to cover the confirmation window
        # even if the engine interval > confirmation_seconds).
        recent_evals = (
            db.query(RecipeEvaluation)
            .filter(RecipeEvaluation.recipe_id == recipe.id)
            .order_by(RecipeEvaluation.evaluated_at.desc())
            .limit(20)
            .all()
        )

        # Walk backwards to find how long the current direction has persisted
        first_matching_at = now
        confirmation_broken_by = None
        for ev in recent_evals:
            ev_dir = ev.direction_status or ev.signal
            if ev_dir != signal:
                confirmation_broken_by = f"direction={ev_dir}"
                break
            if ev.strength_status and ev.strength_status == "FAIL":
                confirmation_broken_by = f"strength=FAIL"
                break
            first_matching_at = ev.evaluated_at

        if confirmation_broken_by or not recent_evals:
            # Direction changed or no history — check how long the
            # consistent streak has been going.
            consistent_secs = (now - first_matching_at).total_seconds()
            if consistent_secs < effective_conf_secs:
                logger.info(
                    "Recipe %d: signal %s not confirmed (%.0fs < %ds, broken by %s)",
                    recipe.id, signal, consistent_secs,
                    effective_conf_secs, confirmation_broken_by or "no history",
                )
                signal = "HOLD"
        else:
            # All recent evaluations match — check the time span
            consistent_secs = (now - first_matching_at).total_seconds()
            if consistent_secs < effective_conf_secs:
                logger.info(
                    "Recipe %d: signal %s consistent but only %.0fs (need %ds)",
                    recipe.id, signal, consistent_secs, effective_conf_secs,
                )
                signal = "HOLD"

    triggered = signal in ("BUY", "SELL")
    recommendation_id = None
    auto_approved = False
    approval_id = None

    # Auto-only gate: skip signals that would create pending (manual) approvals
    if triggered and auto_only:
        auto_th = recipe.auto_threshold or 0
        would_auto = auto_th > 0 and abs(final_score) >= auto_th
        if would_auto and (recipe.mode or "roles") == "roles":
            auto_str_th = recipe.auto_strength_threshold or 0
            if auto_str_th > 0 and strength_factor < auto_str_th:
                would_auto = False
        if not would_auto:
            triggered = False
            logger.info(
                "Recipe %d: signal %s skipped (auto-only mode, would be pending)",
                recipe.id, signal,
            )

    if triggered:
        # Anti-duplication: skip if there's already a pending approval or recent trade
        cooldown = INTERVAL_SECONDS.get(recipe.interval, 300)
        if _has_pending_approval(
            recipe.id, db, executed_ids, cooldown_secs=cooldown,
            max_ops_count=getattr(recipe, 'max_ops_count', 0) or 0,
            max_ops_hours=getattr(recipe, 'max_ops_hours', 24.0) or 24.0,
        ):
            triggered = False
        else:
            recommendation_id, auto_approved, approval_id = (
                _create_recommendation_from_recipe(
                    recipe, strategy_results, final_score, strength_factor, signal, db
                )
            )

    evaluation = RecipeEvaluation(
        recipe_id=recipe.id,
        strategy_results=strategy_results,
        final_score=round(final_score, 6),
        signal=signal,
        direction_status=direction_signal,
        strength_status="PASS" if strength_ok else "FAIL",
        direction_value=round(final_score, 6),
        strength_value=round(strength_factor, 4) if strength_th > 0 else None,
        recommendation_id=recommendation_id,
        triggered=triggered,
        evaluated_at=now,
    )
    db.add(evaluation)

    recipe.last_evaluated_at = now
    db.commit()

    logger.info(
        "Recipe %d evaluated: score=%.4f signal=%s triggered=%s",
        recipe.id, final_score, signal, triggered,
    )

    # Auto-execute if auto-approved and buy_quantity/sell_quantity is configured
    if auto_approved and approval_id:
        if signal == "BUY":
            auto_qty = recipe.buy_quantity or recipe.auto_quantity
            order_type = recipe.buy_order_type or "oco"
        else:
            auto_qty = recipe.sell_quantity or recipe.auto_quantity
            order_type = recipe.sell_order_type or "oco"
        if auto_qty and auto_qty > 0:
            try:
                from app.services.trade_service import execute_trade
                trade = await execute_trade(
                    approval_id=approval_id,
                    admin_id=recipe.user_id,
                    quantity=auto_qty,
                    order_type=order_type,
                    db=db,
                )
                logger.info(
                    "Auto-executed trade %d for recipe %d: %s %s qty=%.8f status=%s",
                    trade.id, recipe.id, signal, recipe.symbol, auto_qty, trade.status,
                )
            except Exception as exc:
                logger.error(
                    "Auto-execute failed for recipe %d: %s", recipe.id, exc
                )
                # Create a failed TradeExecution so the approval doesn't
                # permanently block future signals for this recipe.
                failed_trade = TradeExecution(
                    approval_id=approval_id,
                    executed_by=recipe.user_id,
                    is_live=True,
                    order_type=order_type,
                    symbol=recipe.symbol,
                    side=signal,
                    quantity=auto_qty,
                    price=None,
                    status="failed",
                    result={"error": str(exc), "mode": "AUTO_FAILED"},
                    executed_at=datetime.datetime.utcnow(),
                )
                db.add(failed_trade)
                db.commit()

    return final_score


def _expire_stale_approvals(db: Session, executed_ids: set | None = None):
    """Auto-reject pending and approved-not-executed approvals older than 30 min."""
    cutoff = datetime.datetime.utcnow() - datetime.timedelta(minutes=30)

    if executed_ids is None:
        executed_ids = {
            row[0]
            for row in db.query(TradeExecution.approval_id).all()
        }

    stale = (
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
    for appr in stale:
        # Skip approved approvals that already have a trade execution
        if appr.status == ApprovalStatus.approved and appr.id in executed_ids:
            continue
        appr.status = ApprovalStatus.rejected
        appr.review_reason = "Expirada (>30 min sin acción)"
        appr.reviewed_at = datetime.datetime.utcnow()
        expired_count += 1
    if expired_count:
        db.commit()
        logger.info("Expired %d stale approvals", expired_count)


def _has_pending_approval(
    recipe_id: int, db: Session, executed_ids: set | None = None,
    cooldown_secs: int = 300,
    max_ops_count: int = 0, max_ops_hours: float = 24.0,
) -> bool:
    """Check if an actionable approval already exists for this recipe.

    Blocks new recommendations when:
    1. There is a pending OR approved-but-not-executed approval, OR
    2. A trade was recently executed for this recipe (within cooldown_secs),
       preventing duplicate signals in rapid evaluation cycles (turbo mode), OR
    3. The recipe has reached its max_ops_count within the last max_ops_hours.
    """
    if executed_ids is None:
        executed_ids = {
            row[0]
            for row in db.query(TradeExecution.approval_id).all()
        }

    existing = (
        db.query(Approval)
        .join(Recommendation, Approval.recommendation_id == Recommendation.id)
        .join(RecipeEvaluation, RecipeEvaluation.recommendation_id == Recommendation.id)
        .filter(
            RecipeEvaluation.recipe_id == recipe_id,
            RecipeEvaluation.triggered.is_(True),
            or_(
                Approval.status == ApprovalStatus.pending,
                Approval.status == ApprovalStatus.approved,
            ),
        )
        .all()
    )

    for appr in existing:
        if appr.status == ApprovalStatus.pending:
            return True
        if appr.status == ApprovalStatus.approved and appr.id not in executed_ids:
            return True

    # Cooldown: block if a trade was executed recently for this recipe
    if cooldown_secs > 0:
        cooldown_cutoff = datetime.datetime.utcnow() - datetime.timedelta(seconds=cooldown_secs)
        recent_trade = (
            db.query(TradeExecution)
            .join(Approval, TradeExecution.approval_id == Approval.id)
            .join(Recommendation, Approval.recommendation_id == Recommendation.id)
            .join(RecipeEvaluation, RecipeEvaluation.recommendation_id == Recommendation.id)
            .filter(
                RecipeEvaluation.recipe_id == recipe_id,
                TradeExecution.executed_at >= cooldown_cutoff,
                TradeExecution.status.in_(["filled", "simulated"]),
            )
            .first()
        )
        if recent_trade:
            logger.info(
                "Recipe %d: cooldown active (trade %d executed at %s)",
                recipe_id, recent_trade.id, recent_trade.executed_at,
            )
            return True

    # Ops limit: block if recipe reached max operations within the time window
    if max_ops_count > 0:
        ops_cutoff = datetime.datetime.utcnow() - datetime.timedelta(hours=max_ops_hours)
        ops_count = (
            db.query(TradeExecution)
            .join(Approval, TradeExecution.approval_id == Approval.id)
            .join(Recommendation, Approval.recommendation_id == Recommendation.id)
            .join(RecipeEvaluation, RecipeEvaluation.recommendation_id == Recommendation.id)
            .filter(
                RecipeEvaluation.recipe_id == recipe_id,
                TradeExecution.executed_at >= ops_cutoff,
                TradeExecution.status.in_(["filled", "simulated"]),
            )
            .count()
        )
        if ops_count >= max_ops_count:
            logger.info(
                "Recipe %d: ops limit reached (%d/%d in last %.1fh)",
                recipe_id, ops_count, max_ops_count, max_ops_hours,
            )
            return True

    return False


def _create_recommendation_from_recipe(
    recipe: Recipe,
    strategy_results: list[dict],
    final_score: float,
    strength_factor: float,
    signal: str,
    db: Session,
) -> int:
    """Create Recommendation + pending Approval from a triggered evaluation."""
    # Pick primary strategy (highest |score * weight|) for entry price
    candidates = [r for r in strategy_results if r.get("entry")]
    primary = max(
        candidates if candidates else strategy_results,
        key=lambda r: abs(r["score"] * r["weight"]),
    )

    entry = primary.get("entry")

    # Apply recipe risk percentages for SL/TP
    if entry and entry > 0:
        if signal == "BUY":
            stop_loss = round(entry * (1 - recipe.stop_loss_pct / 100), 8)
            take_profit = round(entry * (1 + recipe.take_profit_pct / 100), 8)
        else:
            stop_loss = round(entry * (1 + recipe.stop_loss_pct / 100), 8)
            take_profit = round(entry * (1 - recipe.take_profit_pct / 100), 8)
    else:
        stop_loss = primary.get("stop_loss")
        take_profit = primary.get("take_profit")

    strategy_names = "+".join(r["strategy"] for r in strategy_results)

    parts = []
    for r in strategy_results:
        parts.append(
            f"Estrategia {r['strategy']}: {r['recommendation']} "
            f"(conf={r['confidence']:.1f}%, score={r['score']:.3f}, "
            f"peso={r['weight']:.2f})"
        )
    explanation = (
        f"Receta '{recipe.name}' disparó {signal}. "
        f"Score final: {final_score:.4f}. "
        + " | ".join(parts)
    )

    now = datetime.datetime.utcnow()

    rec = Recommendation(
        user_id=recipe.user_id,
        symbol=recipe.symbol,
        interval=recipe.interval,
        period_start=(now - datetime.timedelta(days=recipe.lookback_days)).isoformat(),
        period_end=now.isoformat(),
        strategy=strategy_names,
        metrics={
            "recipe_id": recipe.id,
            "recipe_name": recipe.name,
            "final_score": round(final_score, 4),
            "strategy_results": strategy_results,
            "order_type": (
                getattr(recipe, "buy_order_type", None)
                if signal == "BUY"
                else getattr(recipe, "sell_order_type", None)
            ) or getattr(recipe, "auto_order_type", "oco") or "oco",
        },
        recommendation=signal,
        confidence=round(abs(final_score) * 100, 1),
        entry_price=entry,
        stop_loss=stop_loss,
        take_profit=take_profit,
        explanation=explanation,
    )
    db.add(rec)
    db.flush()

    # Auto-approve if score crosses auto_threshold (+ strength gate in roles mode)
    auto_th = recipe.auto_threshold or 0
    should_auto = auto_th > 0 and abs(final_score) >= auto_th
    # In roles mode, also require strength >= auto_strength_threshold
    if should_auto and (recipe.mode or "roles") == "roles":
        auto_str_th = recipe.auto_strength_threshold or 0
        if auto_str_th > 0 and strength_factor < auto_str_th:
            should_auto = False

    if should_auto:
        approval = Approval(
            recommendation_id=rec.id,
            status=ApprovalStatus.approved,
            reviewed_by=recipe.user_id,
            review_reason="Auto-aprobado (score superó umbral automático)",
            reviewed_at=datetime.datetime.utcnow(),
        )
        db.add(approval)
        db.add(AuditLog(
            user_id=recipe.user_id,
            action="approval.auto_approve",
            payload={
                "recipe_id": recipe.id,
                "recipe_name": recipe.name,
                "signal": signal,
                "final_score": round(final_score, 4),
                "auto_threshold": recipe.auto_threshold,
                "recommendation_id": rec.id,
            },
        ))
    else:
        approval = Approval(
            recommendation_id=rec.id,
            status=ApprovalStatus.pending,
        )
        db.add(approval)

    db.add(AuditLog(
        user_id=recipe.user_id,
        action="recipe.trigger",
        payload={
            "recipe_id": recipe.id,
            "recipe_name": recipe.name,
            "signal": signal,
            "final_score": round(final_score, 4),
            "recommendation_id": rec.id,
            "auto_approved": should_auto,
        },
    ))

    db.flush()
    return rec.id, should_auto, approval.id


# ── Server-side Binance sync ──────────────────────────────────────────

async def _maybe_sync_active_trades():
    """Sync active trades against Binance at SYNC_INTERVAL frequency.

    Runs inside the evaluation loop so it works even when the browser
    is closed or the screen is off.
    """
    import time
    global _last_sync_time

    now = time.time()
    if now - _last_sync_time < SYNC_INTERVAL:
        return
    _last_sync_time = now

    from app.services.binance_order_service import check_oco_status, validate_order_exists
    from app.config import get_settings

    db: Session = SessionLocal()
    try:
        # Get all users that have active trades and API keys
        active_trades = (
            db.query(TradeExecution)
            .filter(TradeExecution.status == "filled")
            .all()
        )
        if not active_trades:
            return

        # Filter truly active (not closed)
        pending = []
        for t in active_trades:
            rd = t.result or {}
            if rd.get("oco_status") == "ALL_DONE" or rd.get("closed_manually"):
                continue
            pending.append(t)

        if not pending:
            return

        # Group by user for API key lookup
        user_ids = {t.executed_by for t in pending}
        users = {u.id: u for u in db.query(User).filter(User.id.in_(user_ids)).all()}

        changes = 0
        for trade in pending:
            user = users.get(trade.executed_by)
            if not user or not user.binance_api_key_enc:
                continue

            result_data = trade.result or {}
            try:
                if trade.order_type == "oco":
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
                        updated = dict(result_data)
                        updated["oco_status"] = "ALL_DONE"
                        updated["exit_type"] = oco_result.get("exit_type")
                        updated["exit_price"] = oco_result.get("exit_price")

                        exit_time = oco_result.get("exit_time")
                        if exit_time and isinstance(exit_time, (int, float)):
                            updated["exit_time"] = datetime.datetime.utcfromtimestamp(
                                exit_time / 1000
                            ).isoformat()
                        elif exit_time:
                            updated["exit_time"] = str(exit_time)

                        # Compute P&L
                        exit_px = oco_result.get("exit_price") or 0.0
                        entry_px = trade.price or 0.0
                        qty = trade.quantity or 0.0
                        if entry_px > 0 and exit_px > 0:
                            if trade.side == "BUY":
                                pnl_pct = round(
                                    (exit_px - entry_px) / entry_px * 100, 4
                                )
                            else:
                                pnl_pct = round(
                                    (entry_px - exit_px) / entry_px * 100, 4
                                )
                            pnl_abs = round(pnl_pct / 100 * entry_px * qty, 4)
                            fee = get_settings().commission_pct
                            net_pnl_pct = round(pnl_pct - fee * 2, 4)
                            net_pnl_abs = round(
                                net_pnl_pct / 100 * entry_px * qty, 4
                            )
                            updated["pnl_pct"] = pnl_pct
                            updated["pnl_abs"] = pnl_abs
                            updated["commission_pct"] = fee
                            updated["net_pnl_pct"] = net_pnl_pct
                            updated["net_pnl_abs"] = net_pnl_abs

                        trade.result = updated
                        changes += 1
                        exit_label = (
                            "TP" if oco_result.get("exit_type") == "TP" else "SL"
                        )
                        logger.info(
                            "Sync: trade %d (%s) OCO %s at %s",
                            trade.id,
                            trade.symbol,
                            exit_label,
                            oco_result.get("exit_price"),
                        )

                else:
                    # Non-OCO: validate existence
                    market_order = result_data.get("market_order", {}) or {}
                    limit_order = result_data.get("limit_order", {}) or {}
                    oco_order = result_data.get("oco_order", {}) or {}
                    order_id = (
                        limit_order.get("orderId") or market_order.get("orderId")
                    )
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
                        changes += 1
                        logger.info(
                            "Sync: trade %d (%s) invalidated: %s",
                            trade.id,
                            trade.symbol,
                            validation["reason"],
                        )

            except Exception as exc:
                logger.warning(
                    "Sync error for trade %d: %s", trade.id, str(exc)
                )

        if changes:
            db.commit()
            logger.info("Sync: %d trade(s) updated", changes)
    finally:
        db.close()
