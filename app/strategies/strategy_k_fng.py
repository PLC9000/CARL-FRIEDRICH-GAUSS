"""
Strategy K: Fear & Greed Index (Sentimiento de mercado).

Uses the Alternative.me Crypto Fear & Greed Index as a contrarian
indicator.  The index ranges from 0 (Extreme Fear) to 100 (Extreme Greed).

Signals (contrarian logic):
  BUY  — FNG below fear_threshold  (market too fearful → buying opportunity)
  SELL — FNG above greed_threshold (market too greedy → take profits)
  NO-TRADE — FNG in neutral zone
"""

import logging
import numpy as np
import httpx

from app.strategies._common import no_trade, atr as _atr, compute_sl_tp

logger = logging.getLogger(__name__)

DEFAULTS = {
    "fear_threshold": 25,       # FNG <= this → BUY
    "greed_threshold": 75,      # FNG >= this → SELL
    "trend_days": 7,            # Days of FNG history for trend analysis
    "sl_multiplier": 1.5,
    "tp_multiplier": 2.5,
}

FNG_API_URL = "https://api.alternative.me/fng/"


def _fetch_fng(limit: int = 30) -> list[dict] | None:
    """Fetch Fear & Greed history. Returns list of {value, timestamp} or None."""
    try:
        resp = httpx.get(FNG_API_URL, params={"limit": limit}, timeout=10)
        resp.raise_for_status()
        data = resp.json().get("data", [])
        return [{"value": int(d["value"]), "classification": d["value_classification"],
                 "timestamp": int(d["timestamp"])} for d in data]
    except Exception as exc:
        logger.warning("FNG API error: %s", exc)
        return None


def run_strategy_k(candles: np.ndarray, params: dict | None = None) -> dict:
    p = {**DEFAULTS, **(params or {})}
    fear_th = int(p["fear_threshold"])
    greed_th = int(p["greed_threshold"])
    trend_days = int(p["trend_days"])
    sl_mult = float(p["sl_multiplier"])
    tp_mult = float(p["tp_multiplier"])

    # Fetch Fear & Greed data
    fng_data = _fetch_fng(limit=max(trend_days, 7))
    if not fng_data:
        return no_trade("No se pudo obtener el indice Fear & Greed. Intenta de nuevo.")

    fng_now = fng_data[0]["value"]
    classification = fng_data[0]["classification"]

    # Trend: compare current vs average of last N days
    fng_values = [d["value"] for d in fng_data[:trend_days]]
    fng_avg = sum(fng_values) / len(fng_values) if fng_values else fng_now
    fng_trend = fng_now - fng_avg  # positive = greed increasing, negative = fear increasing

    # Signal
    signal = "NO-TRADE"
    if fng_now <= fear_th:
        signal = "BUY"
    elif fng_now >= greed_th:
        signal = "SELL"

    # Confidence: how far into the zone
    if signal == "BUY":
        confidence = min((fear_th - fng_now) / max(fear_th, 1) * 80 + 20, 100)
    elif signal == "SELL":
        confidence = min((fng_now - greed_th) / max(100 - greed_th, 1) * 80 + 20, 100)
    else:
        # Neutral zone: low confidence proportional to distance from center
        dist_from_center = abs(fng_now - 50)
        confidence = dist_from_center / 50 * 15
    confidence = round(max(0, min(100, confidence)), 1)

    # Price levels from candles
    closes = candles[:, 4].astype(float)
    highs = candles[:, 2].astype(float)
    lows = candles[:, 3].astype(float)
    last_close = float(closes[-1])
    atr_val = _atr(highs, lows, closes)

    entry = last_close
    stop_loss, take_profit = compute_sl_tp(signal, entry, atr_val, sl_mult, tp_mult)

    trend_label = (
        "miedo creciente" if fng_trend < -5
        else "codicia creciente" if fng_trend > 5
        else "estable"
    )

    return {
        "recommendation": signal,
        "confidence": confidence,
        "entry": round(entry, 8),
        "stop_loss": stop_loss,
        "take_profit": take_profit,
        "explanation": (
            f"Fear & Greed Index = {fng_now} ({classification}). "
            f"Promedio {trend_days}d = {fng_avg:.0f} (tendencia: {trend_label}). "
            f"Umbrales: miedo <= {fear_th}, codicia >= {greed_th}."
        ),
        "metrics": {
            "fng_value": fng_now,
            "fng_classification": classification,
            "fng_avg": round(fng_avg, 1),
            "fng_trend": round(fng_trend, 1),
            "fear_threshold": fear_th,
            "greed_threshold": greed_th,
        },
    }


