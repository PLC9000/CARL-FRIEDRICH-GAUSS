"""Strategy enabled/disabled configuration service."""

import logging

from sqlalchemy.orm import Session

from app.models import StrategyConfig, AuditLog
from app.strategies import STRATEGY_REGISTRY

logger = logging.getLogger(__name__)


def seed_strategy_configs(db: Session):
    """Ensure every strategy in STRATEGY_REGISTRY has a StrategyConfig row.

    New strategies default to disabled. Existing rows are not touched.
    """
    existing_keys = {
        row.strategy_key
        for row in db.query(StrategyConfig.strategy_key).all()
    }
    added = 0
    for key in STRATEGY_REGISTRY:
        if key not in existing_keys:
            db.add(StrategyConfig(strategy_key=key, enabled=False))
            added += 1
    if added:
        db.commit()
        logger.info("Seeded %d strategy config rows", added)


def list_strategy_configs(db: Session) -> list[dict]:
    """Return all strategies with their enabled/disabled state."""
    configs = {
        row.strategy_key: row.enabled
        for row in db.query(StrategyConfig).all()
    }
    result = []
    for key, info in STRATEGY_REGISTRY.items():
        result.append({
            "key": key,
            "name": info["name"],
            "icon": info["icon"],
            "category": info.get("category", ""),
            "short_desc": info["short_desc"],
            "enabled": configs.get(key, False),
        })
    return result


def toggle_strategy(strategy_key: str, admin_id: int, db: Session) -> dict:
    """Toggle a strategy's enabled state. Returns the new state."""
    config = db.query(StrategyConfig).filter(
        StrategyConfig.strategy_key == strategy_key,
    ).first()
    if config is None:
        raise ValueError(f"Estrategia '{strategy_key}' no encontrada")

    config.enabled = not config.enabled
    db.add(AuditLog(
        user_id=admin_id,
        action="strategy.toggle",
        payload={
            "strategy_key": strategy_key,
            "enabled": config.enabled,
        },
    ))
    db.commit()
    db.refresh(config)
    return {"key": strategy_key, "enabled": config.enabled}


def get_enabled_strategy_keys(db: Session) -> set[str]:
    """Return set of enabled strategy keys."""
    return {
        row.strategy_key
        for row in db.query(StrategyConfig.strategy_key).filter(
            StrategyConfig.enabled.is_(True),
        ).all()
    }
