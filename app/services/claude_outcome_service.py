"""Track Claude AI predictions and outcomes for in-context learning.

Each time Strategy M produces a signal, the prediction is recorded.
On subsequent evaluation cycles the actual market outcome is back-filled
so Claude can see its own track record in the prompt.
"""

from __future__ import annotations

import datetime
import logging

from sqlalchemy.orm import Session

from app.models import ClaudeOutcome

logger = logging.getLogger(__name__)

# Minimum age (seconds) before we consider a prediction "mature" enough
# to back-fill with the actual outcome.
_MATURITY_SECS = 1800  # 30 minutes


def record_prediction(
    db: Session,
    recipe_id: int,
    symbol: str,
    interval: str,
    direction: int,
    intensity: int,
    confidence: int,
    entry_price: float,
    reasoning: str,
) -> int:
    """Persist a new Claude prediction.  Returns the row id."""
    outcome = ClaudeOutcome(
        recipe_id=recipe_id,
        symbol=symbol,
        interval=interval,
        direction=direction,
        intensity=intensity,
        confidence=confidence,
        entry_price=entry_price,
        reasoning=(reasoning or "")[:500],
        predicted_at=datetime.datetime.utcnow(),
    )
    db.add(outcome)
    db.flush()
    return outcome.id


def backfill_outcomes(db: Session, symbol: str, current_price: float) -> int:
    """Fill actual return % for mature predictions.  Returns count updated."""
    cutoff = datetime.datetime.utcnow() - datetime.timedelta(seconds=_MATURITY_SECS)
    pending = (
        db.query(ClaudeOutcome)
        .filter(
            ClaudeOutcome.symbol == symbol,
            ClaudeOutcome.actual_return_pct.is_(None),
            ClaudeOutcome.predicted_at < cutoff,
        )
        .limit(50)
        .all()
    )
    count = 0
    for p in pending:
        if not p.entry_price or p.entry_price <= 0:
            continue
        ret_pct = (current_price - p.entry_price) / p.entry_price * 100
        p.actual_return_pct = round(ret_pct, 4)
        if p.direction > 0:
            p.outcome = "CORRECT" if ret_pct > 0 else "WRONG"
        elif p.direction < 0:
            p.outcome = "CORRECT" if ret_pct < 0 else "WRONG"
        else:
            p.outcome = "NEUTRAL"
        count += 1
    if count:
        db.commit()
        logger.debug("Back-filled %d Claude outcomes for %s", count, symbol)
    return count


def get_recent_outcomes(db: Session, symbol: str, limit: int = 10) -> list[dict]:
    """Return the most recent completed predictions (with outcomes)."""
    rows = (
        db.query(ClaudeOutcome)
        .filter(
            ClaudeOutcome.symbol == symbol,
            ClaudeOutcome.actual_return_pct.isnot(None),
        )
        .order_by(ClaudeOutcome.predicted_at.desc())
        .limit(limit)
        .all()
    )
    return [
        {
            "timestamp": r.predicted_at.strftime("%Y-%m-%dT%H:%M"),
            "direction": r.direction,
            "intensity": r.intensity,
            "confidence": r.confidence,
            "outcome": r.outcome,
            "return_pct": r.actual_return_pct,
            "horizon": r.interval,
        }
        for r in reversed(rows)  # chronological order
    ]
