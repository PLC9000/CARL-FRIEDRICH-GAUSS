"""Orchestrates: fetch candles -> run strategy -> persist recommendation."""

import logging

from sqlalchemy.orm import Session

from app.binance_client import fetch_candles, validate_symbol
from app.models import Recommendation, Approval, ApprovalStatus
from app.strategies import STRATEGY_MAP
from app.utils.validation import parse_period

logger = logging.getLogger(__name__)


async def generate_recommendation(
    user_id: int,
    symbol: str,
    interval: str,
    strategy: str,
    last_n_days: int | None,
    start: str | None,
    end: str | None,
    db: Session,
    params: dict | None = None,
) -> Recommendation:
    # Validate symbol on Binance
    if not await validate_symbol(symbol):
        raise ValueError(f"Symbol '{symbol}' not found on Binance")

    # Parse time period
    start_dt, end_dt = parse_period(last_n_days, start, end)

    # Fetch candles
    candles = await fetch_candles(symbol, interval, start_dt, end_dt)

    # Run strategy
    strategy_fn = STRATEGY_MAP.get(strategy)
    if strategy_fn is None:
        raise ValueError(f"Unknown strategy: {strategy}")

    result = strategy_fn(candles, params=params)

    # Persist
    rec = Recommendation(
        user_id=user_id,
        symbol=symbol,
        interval=interval,
        period_start=start_dt.isoformat(),
        period_end=end_dt.isoformat(),
        strategy=strategy,
        metrics=result.get("metrics"),
        recommendation=result["recommendation"],
        confidence=result["confidence"],
        entry_price=result.get("entry"),
        stop_loss=result.get("stop_loss"),
        take_profit=result.get("take_profit"),
        explanation=result.get("explanation"),
    )
    db.add(rec)
    db.flush()

    # Auto-create a pending approval record
    approval = Approval(
        recommendation_id=rec.id,
        status=ApprovalStatus.pending,
    )
    db.add(approval)
    db.commit()
    db.refresh(rec)

    logger.info(
        "Recommendation %d created: %s %s %s -> %s (%.1f%%)",
        rec.id, symbol, interval, strategy, rec.recommendation, rec.confidence,
    )
    return rec
