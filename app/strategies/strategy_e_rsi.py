"""
Strategy E: RSI (Relative Strength Index) — overbought/oversold detection.

BUY when RSI drops below the oversold level, SELL when it rises above
overbought.  Confidence is proportional to how deep into the extreme zone
the RSI is.
"""

import logging
import numpy as np

from app.strategies._common import no_trade, simple_atr, compute_sl_tp

logger = logging.getLogger(__name__)

DEFAULTS = {
    "rsi_period": 14,
    "overbought": 70,
    "oversold": 30,
    "sl_multiplier": 1.5,
    "tp_multiplier": 2.5,
}


def _rsi(closes: np.ndarray, period: int) -> np.ndarray:
    """Compute RSI using exponential moving average of gains/losses."""
    deltas = np.diff(closes)
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)

    alpha = 1.0 / period
    avg_gain = np.zeros(len(deltas))
    avg_loss = np.zeros(len(deltas))

    avg_gain[period - 1] = np.mean(gains[:period])
    avg_loss[period - 1] = np.mean(losses[:period])

    for i in range(period, len(deltas)):
        avg_gain[i] = alpha * gains[i] + (1 - alpha) * avg_gain[i - 1]
        avg_loss[i] = alpha * losses[i] + (1 - alpha) * avg_loss[i - 1]

    rs = np.where(avg_loss > 0, avg_gain / avg_loss, 100.0)
    rsi = 100 - 100 / (1 + rs)
    # First period-1 values are invalid
    rsi[:period - 1] = np.nan
    return rsi


def run_strategy_e(candles: np.ndarray, params: dict | None = None) -> dict:
    p = {**DEFAULTS, **(params or {})}
    period = int(p["rsi_period"])
    overbought = float(p["overbought"])
    oversold = float(p["oversold"])
    sl_mult = float(p["sl_multiplier"])
    tp_mult = float(p["tp_multiplier"])

    closes = candles[:, 4].astype(float)
    if len(closes) < period + 5:
        return no_trade(f"Need >= {period + 5} candles")

    rsi_values = _rsi(closes, period)
    current_rsi = float(rsi_values[-1])

    if np.isnan(current_rsi):
        return no_trade("RSI calculation returned NaN")

    prev_rsi = float(rsi_values[-2]) if not np.isnan(rsi_values[-2]) else current_rsi

    if current_rsi < oversold:
        signal = "BUY"
        # Deeper below oversold = more confidence
        depth = (oversold - current_rsi) / oversold
        confidence = round(min(50 + depth * 100, 100), 1)
    elif current_rsi > overbought:
        signal = "SELL"
        depth = (current_rsi - overbought) / (100 - overbought)
        confidence = round(min(50 + depth * 100, 100), 1)
    else:
        signal = "NO-TRADE"
        confidence = round(max(0, 30 - abs(current_rsi - 50)), 1)

    last_close = float(closes[-1])
    atr = simple_atr(candles)
    entry = last_close
    stop_loss, take_profit = compute_sl_tp(signal, entry, atr, sl_mult, tp_mult)

    rsi_direction = "subiendo" if current_rsi > prev_rsi else "bajando"

    return {
        "recommendation": signal,
        "confidence": confidence,
        "entry": round(entry, 8),
        "stop_loss": stop_loss,
        "take_profit": take_profit,
        "explanation": (
            f"RSI({period}) = {current_rsi:.1f} ({rsi_direction}). "
            f"Sobreventa < {oversold}, Sobrecompra > {overbought}."
        ),
        "metrics": {
            "rsi": round(current_rsi, 2),
            "rsi_prev": round(prev_rsi, 2),
            "overbought": overbought,
            "oversold": oversold,
            "rsi_direction": rsi_direction,
        },
    }


