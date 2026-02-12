"""
Strategy G: Bollinger Bands (volatility + mean reversion).

BUY when price touches/closes below the lower band (oversold).
SELL when price touches/closes above the upper band (overbought).
Confidence is derived from how far beyond the band price has gone.
"""

import logging
import numpy as np

from app.strategies._common import no_trade, simple_atr, compute_sl_tp

logger = logging.getLogger(__name__)

DEFAULTS = {
    "period": 20,
    "std_dev": 2.0,
    "entry_rule": "touch",  # "touch" or "close_outside"
}


def run_strategy_g(candles: np.ndarray, params: dict | None = None) -> dict:
    p = {**DEFAULTS, **(params or {})}
    period = int(p["period"])
    std_dev = float(p["std_dev"])
    entry_rule = str(p["entry_rule"]).lower()

    closes = candles[:, 4].astype(float)
    lows = candles[:, 3].astype(float)
    highs = candles[:, 2].astype(float)

    if len(closes) < period + 2:
        return no_trade(f"Need >= {period + 2} candles")

    # Compute Bollinger Bands on the last window
    window = closes[-period:]
    middle = float(np.mean(window))
    std = float(np.std(window, ddof=1))

    upper = middle + std_dev * std
    lower = middle - std_dev * std

    last_close = float(closes[-1])
    last_low = float(lows[-1])
    last_high = float(highs[-1])

    bandwidth = (upper - lower) / middle * 100 if middle else 0
    pct_b = (last_close - lower) / (upper - lower) if (upper - lower) > 0 else 0.5

    if entry_rule == "close_outside":
        # Signal only if close is outside the band
        touch_lower = last_close < lower
        touch_upper = last_close > upper
    else:
        # "touch" — low touched lower band or high touched upper band
        touch_lower = last_low <= lower
        touch_upper = last_high >= upper

    if touch_lower:
        signal = "BUY"
        penetration = (lower - last_close) / std if std > 0 else 0
        confidence = round(min(50 + abs(penetration) * 30, 100), 1)
    elif touch_upper:
        signal = "SELL"
        penetration = (last_close - upper) / std if std > 0 else 0
        confidence = round(min(50 + abs(penetration) * 30, 100), 1)
    else:
        signal = "NO-TRADE"
        # Closer to band = more confidence something might happen
        dist_to_band = min(abs(last_close - lower), abs(last_close - upper))
        confidence = round(max(0, 30 - (dist_to_band / std * 10 if std else 30)), 1)

    atr = simple_atr(candles)
    entry = last_close
    stop_loss, take_profit = compute_sl_tp(signal, entry, atr, 1.5, 2.5)

    return {
        "recommendation": signal,
        "confidence": confidence,
        "entry": round(entry, 8),
        "stop_loss": stop_loss,
        "take_profit": take_profit,
        "explanation": (
            f"Bollinger({period}, {std_dev}x): upper={upper:.2f}, "
            f"mid={middle:.2f}, lower={lower:.2f}. "
            f"Close={last_close:.2f}, %B={pct_b:.2f}. Rule: {entry_rule}."
        ),
        "metrics": {
            "upper_band": round(upper, 8),
            "middle_band": round(middle, 8),
            "lower_band": round(lower, 8),
            "bandwidth_pct": round(bandwidth, 4),
            "pct_b": round(pct_b, 4),
        },
    }


