"""
Strategy F: MACD (Moving Average Convergence Divergence).

BUY when MACD line crosses above the signal line (or histogram turns
positive, depending on trigger mode).  SELL on the opposite cross.
Confidence is derived from histogram magnitude.
"""

import logging
import numpy as np

from app.strategies._common import no_trade, ema as _ema, simple_atr, compute_sl_tp

logger = logging.getLogger(__name__)

DEFAULTS = {
    "fast": 12,
    "slow": 26,
    "signal": 9,
    "trigger": "cross",  # "cross" or "histogram"
    "sl_multiplier": 1.5,
    "tp_multiplier": 2.5,
}


def run_strategy_f(candles: np.ndarray, params: dict | None = None) -> dict:
    p = {**DEFAULTS, **(params or {})}
    fast_p = int(p["fast"])
    slow_p = int(p["slow"])
    signal_p = int(p["signal"])
    trigger = str(p["trigger"]).lower()
    sl_mult = float(p["sl_multiplier"])
    tp_mult = float(p["tp_multiplier"])

    closes = candles[:, 4].astype(float)
    min_len = slow_p + signal_p + 2
    if len(closes) < min_len:
        return no_trade(f"Need >= {min_len} candles")

    fast_ema = _ema(closes, fast_p)
    slow_ema = _ema(closes, slow_p)
    macd_line = fast_ema - slow_ema

    # Signal line = EMA of MACD line (only on valid portion)
    valid_start = slow_p - 1
    macd_valid = macd_line[valid_start:]
    signal_line_partial = _ema(macd_valid, signal_p)

    # Build full-length signal line
    signal_line = np.full(len(closes), np.nan)
    signal_line[valid_start:] = signal_line_partial

    histogram = macd_line - signal_line

    curr_macd = float(macd_line[-1])
    curr_signal = float(signal_line[-1])
    curr_hist = float(histogram[-1])
    prev_hist = float(histogram[-2]) if not np.isnan(histogram[-2]) else 0

    if np.isnan(curr_macd) or np.isnan(curr_signal):
        return no_trade("MACD not ready — not enough data")

    if trigger == "histogram":
        # BUY when histogram crosses from negative to positive
        if prev_hist <= 0 < curr_hist:
            signal = "BUY"
        elif prev_hist >= 0 > curr_hist:
            signal = "SELL"
        else:
            signal = "NO-TRADE"
    else:
        # Cross mode: MACD crosses signal
        prev_macd = float(macd_line[-2])
        prev_signal = float(signal_line[-2]) if not np.isnan(signal_line[-2]) else prev_macd
        if prev_macd <= prev_signal and curr_macd > curr_signal:
            signal = "BUY"
        elif prev_macd >= prev_signal and curr_macd < curr_signal:
            signal = "SELL"
        else:
            signal = "NO-TRADE"

    last_close = float(closes[-1])
    hist_pct = abs(curr_hist) / last_close * 100 if last_close else 0
    confidence = round(min(hist_pct * 200, 100), 1)

    atr = simple_atr(candles)
    entry = last_close
    stop_loss, take_profit = compute_sl_tp(signal, entry, atr, sl_mult, tp_mult)

    return {
        "recommendation": signal,
        "confidence": confidence,
        "entry": round(entry, 8),
        "stop_loss": stop_loss,
        "take_profit": take_profit,
        "explanation": (
            f"MACD({fast_p},{slow_p},{signal_p}) line={curr_macd:.4f}, "
            f"signal={curr_signal:.4f}, histogram={curr_hist:.4f}. "
            f"Trigger: {trigger}."
        ),
        "metrics": {
            "macd_line": round(curr_macd, 6),
            "signal_line": round(curr_signal, 6),
            "histogram": round(curr_hist, 6),
            "trigger_mode": trigger,
        },
    }


