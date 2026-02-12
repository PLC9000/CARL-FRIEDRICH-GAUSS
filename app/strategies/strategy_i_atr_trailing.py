"""
Strategy I: ATR Trailing Stop.

Computes a dynamic trailing stop based on ATR.  When price crosses
above the trailing stop, the bias flips to BUY (long); when it
crosses below, the bias flips to SELL (short).  The direction
parameter limits to long-only, short-only, or both.
"""

import logging
import numpy as np

from app.strategies._common import no_trade, compute_sl_tp

logger = logging.getLogger(__name__)

DEFAULTS = {
    "atr_period": 14,
    "atr_multiplier": 3.0,
    "direction": "both",  # "long_only", "short_only", "both"
}


def _atr(candles: np.ndarray, period: int) -> np.ndarray:
    """Compute Average True Range."""
    highs = candles[:, 2].astype(float)
    lows = candles[:, 3].astype(float)
    closes = candles[:, 4].astype(float)

    tr = np.empty(len(candles))
    tr[0] = highs[0] - lows[0]
    for i in range(1, len(candles)):
        tr[i] = max(
            highs[i] - lows[i],
            abs(highs[i] - closes[i - 1]),
            abs(lows[i] - closes[i - 1]),
        )

    atr_vals = np.empty(len(candles))
    atr_vals[:] = np.nan
    atr_vals[period - 1] = np.mean(tr[:period])
    alpha = 1.0 / period
    for i in range(period, len(candles)):
        atr_vals[i] = alpha * tr[i] + (1 - alpha) * atr_vals[i - 1]
    return atr_vals


def run_strategy_i(candles: np.ndarray, params: dict | None = None) -> dict:
    p = {**DEFAULTS, **(params or {})}
    atr_period = int(p["atr_period"])
    multiplier = float(p["atr_multiplier"])
    direction = str(p["direction"]).lower()

    closes = candles[:, 4].astype(float)
    if len(candles) < atr_period + 10:
        return no_trade(f"Need >= {atr_period + 10} candles")

    atr_vals = _atr(candles, atr_period)

    # Compute trailing stop series
    n = len(closes)
    trail = np.zeros(n)
    trend = np.zeros(n, dtype=int)  # 1 = up (long), -1 = down (short)

    start = atr_period
    trail[start] = closes[start] - multiplier * atr_vals[start]
    trend[start] = 1

    for i in range(start + 1, n):
        atr_v = atr_vals[i]
        if np.isnan(atr_v):
            trail[i] = trail[i - 1]
            trend[i] = trend[i - 1]
            continue

        up_stop = closes[i] - multiplier * atr_v
        dn_stop = closes[i] + multiplier * atr_v

        if trend[i - 1] == 1:
            trail[i] = max(trail[i - 1], up_stop)
            if closes[i] < trail[i]:
                trend[i] = -1
                trail[i] = dn_stop
            else:
                trend[i] = 1
        else:
            trail[i] = min(trail[i - 1], dn_stop)
            if closes[i] > trail[i]:
                trend[i] = 1
                trail[i] = up_stop
            else:
                trend[i] = -1

    curr_trend = int(trend[-1])
    prev_trend = int(trend[-2])
    flip = curr_trend != prev_trend

    # Determine signal based on direction filter
    if flip and curr_trend == 1 and direction != "short_only":
        signal = "BUY"
    elif flip and curr_trend == -1 and direction != "long_only":
        signal = "SELL"
    else:
        signal = "NO-TRADE"

    last_close = float(closes[-1])
    last_atr = float(atr_vals[-1]) if not np.isnan(atr_vals[-1]) else 0
    trailing_stop = float(trail[-1])

    distance_pct = abs(last_close - trailing_stop) / last_close * 100 if last_close else 0
    confidence = round(min(distance_pct * 20, 100), 1) if flip else round(max(0, 20 - distance_pct * 5), 1)

    entry = last_close
    stop_loss, take_profit = compute_sl_tp(signal, entry, last_atr * multiplier, 1.5, 2.5)

    trend_label = "ALCISTA" if curr_trend == 1 else "BAJISTA"

    return {
        "recommendation": signal,
        "confidence": confidence,
        "entry": round(entry, 8),
        "stop_loss": stop_loss,
        "take_profit": take_profit,
        "explanation": (
            f"ATR Trailing({atr_period}, {multiplier}x): stop={trailing_stop:.2f}, "
            f"trend={trend_label}. Distance={distance_pct:.2f}%. "
            f"Direction: {direction}."
        ),
        "metrics": {
            "atr": round(last_atr, 8),
            "trailing_stop": round(trailing_stop, 8),
            "trend": trend_label,
            "distance_pct": round(distance_pct, 4),
            "flip": flip,
        },
    }


