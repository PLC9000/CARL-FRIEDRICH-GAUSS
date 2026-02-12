"""
Strategy H: Breakout / Donchian Channel.

BUY when price breaks above the highest high of the lookback period.
SELL when price breaks below the lowest low.  An optional buffer
percentage reduces false breakouts.  Volume filter optionally
requires above-average volume.
"""

import logging
import numpy as np

from app.strategies._common import no_trade, simple_atr, compute_sl_tp

logger = logging.getLogger(__name__)

DEFAULTS = {
    "lookback": 20,
    "breakout_type": "donchian",  # "donchian" or "high_low"
    "buffer_pct": 0.1,            # percent
    "volume_filter": False,
}


def run_strategy_h(candles: np.ndarray, params: dict | None = None) -> dict:
    p = {**DEFAULTS, **(params or {})}
    lookback = int(p["lookback"])
    buffer_pct = float(p["buffer_pct"]) / 100
    volume_filter = bool(p.get("volume_filter", False))

    highs = candles[:, 2].astype(float)
    lows = candles[:, 3].astype(float)
    closes = candles[:, 4].astype(float)
    volumes = candles[:, 5].astype(float)

    if len(closes) < lookback + 2:
        return no_trade(f"Need >= {lookback + 2} candles")

    # Channel from the lookback period BEFORE the current candle
    channel_highs = highs[-(lookback + 1):-1]
    channel_lows = lows[-(lookback + 1):-1]

    upper_channel = float(np.max(channel_highs))
    lower_channel = float(np.min(channel_lows))

    upper_break = upper_channel * (1 + buffer_pct)
    lower_break = lower_channel * (1 - buffer_pct)

    last_close = float(closes[-1])
    last_high = float(highs[-1])
    last_low = float(lows[-1])

    # Volume check
    if volume_filter:
        avg_vol = float(np.mean(volumes[-(lookback + 1):-1]))
        curr_vol = float(volumes[-1])
        vol_ok = curr_vol > avg_vol * 1.2
    else:
        avg_vol = 0
        curr_vol = float(volumes[-1])
        vol_ok = True

    if last_close > upper_break and vol_ok:
        signal = "BUY"
        penetration = (last_close - upper_break) / upper_break
        confidence = round(min(50 + penetration * 500, 100), 1)
    elif last_close < lower_break and vol_ok:
        signal = "SELL"
        penetration = (lower_break - last_close) / lower_break
        confidence = round(min(50 + penetration * 500, 100), 1)
    else:
        signal = "NO-TRADE"
        dist = min(abs(last_close - upper_break), abs(last_close - lower_break))
        range_size = upper_channel - lower_channel if upper_channel != lower_channel else 1
        confidence = round(max(0, 30 - dist / range_size * 30), 1)

    atr = simple_atr(candles)
    entry = last_close
    stop_loss, take_profit = compute_sl_tp(signal, entry, atr, 1.5, 2.5)

    channel_width = (upper_channel - lower_channel) / last_close * 100 if last_close else 0

    return {
        "recommendation": signal,
        "confidence": confidence,
        "entry": round(entry, 8),
        "stop_loss": stop_loss,
        "take_profit": take_profit,
        "explanation": (
            f"Breakout({lookback}): channel [{lower_channel:.2f} - {upper_channel:.2f}], "
            f"buffer={buffer_pct * 100:.1f}%. Close={last_close:.2f}. "
            f"Vol filter: {'ON' if volume_filter else 'OFF'}."
        ),
        "metrics": {
            "upper_channel": round(upper_channel, 8),
            "lower_channel": round(lower_channel, 8),
            "channel_width_pct": round(channel_width, 4),
            "buffer_pct": round(buffer_pct * 100, 2),
            "volume_ratio": round(curr_vol / avg_vol, 2) if avg_vol > 0 else None,
        },
    }


