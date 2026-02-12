"""Strategy registry endpoints — list, detail, metadata."""

import datetime
import logging

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from app.binance_client import fetch_candles
from app.database import get_db
from app.strategies import STRATEGY_MAP, STRATEGY_REGISTRY
from app.services.setup_service import get_enabled_strategy_keys

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/strategies", tags=["Estrategias"])


@router.get("/", summary="Listar todas las estrategias disponibles")
def list_strategies(
    include_disabled: bool = Query(False, description="Include disabled strategies"),
    db: Session = Depends(get_db),
):
    """Devuelve las estrategias disponibles. Por defecto solo las habilitadas."""
    enabled_keys = get_enabled_strategy_keys(db)
    result = []
    for info in STRATEGY_REGISTRY.values():
        is_enabled = info["key"] in enabled_keys
        if not include_disabled and not is_enabled:
            continue
        result.append({
            "key": info["key"],
            "name": info["name"],
            "icon": info["icon"],
            "category": info.get("category", ""),
            "short_desc": info["short_desc"],
            "when_works": info["when_works"],
            "when_fails": info["when_fails"],
            "example": info.get("example", ""),
            "param_count": len(info["default_params"]),
            "enabled": is_enabled,
        })
    return result


@router.get("/{strategy_key}/run", summary="Ejecutar estrategia en vivo")
async def run_strategy_live(
    strategy_key: str,
    symbol: str = "BTCUSDT",
    interval: str = "1h",
    lookback_days: int = 7,
):
    """Fetch candles from Binance and run the strategy, returning full results."""
    key = strategy_key.upper()
    fn = STRATEGY_MAP.get(key)
    if fn is None:
        raise HTTPException(status_code=404, detail=f"Estrategia '{key}' no encontrada")

    end_dt = datetime.datetime.utcnow()
    start_dt = end_dt - datetime.timedelta(days=lookback_days)

    try:
        candles = await fetch_candles(symbol.upper(), interval, start_dt, end_dt)
    except Exception as exc:
        logger.warning("Failed to fetch candles for %s: %s", symbol, exc)
        raise HTTPException(status_code=400, detail=f"Error obteniendo velas: {exc}")

    if candles is None or len(candles) == 0:
        raise HTTPException(status_code=400, detail="No se obtuvieron velas de Binance")

    try:
        result = fn(candles)
    except Exception as exc:
        logger.warning("Strategy %s failed: %s", key, exc)
        raise HTTPException(status_code=500, detail=f"Error ejecutando estrategia: {exc}")

    return {
        "symbol": symbol.upper(),
        "interval": interval,
        "candle_count": len(candles),
        "last_close": round(float(candles[-1, 4]), 8),
        **result,
    }


@router.get("/{strategy_key}", summary="Detalle de una estrategia")
def get_strategy(strategy_key: str):
    """Devuelve detalle completo de una estrategia con todos sus parámetros."""
    key = strategy_key.upper()
    info = STRATEGY_REGISTRY.get(key)
    if info is None:
        raise HTTPException(status_code=404, detail=f"Estrategia '{key}' no encontrada")
    return info
