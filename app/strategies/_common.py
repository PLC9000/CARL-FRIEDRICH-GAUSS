"""
Shared utilities for all strategy modules.

Centralises no_trade(), EMA, ATR, true-range, and SL/TP computations
that were previously copy-pasted across every strategy file.
"""

import numpy as np


# ---------------------------------------------------------------------------
# Standard "no signal" return dict
# ---------------------------------------------------------------------------

def no_trade(reason: str) -> dict:
    return {
        "recommendation": "NO-TRADE",
        "confidence": 0,
        "entry": None,
        "stop_loss": None,
        "take_profit": None,
        "explanation": reason,
        "metrics": {},
    }


# ---------------------------------------------------------------------------
# EMA — used by strategy D (EMA crossover) and F (MACD)
# ---------------------------------------------------------------------------

def ema(series: np.ndarray, period: int) -> np.ndarray:
    """Exponential Moving Average (standard formula)."""
    out = np.empty(len(series))
    out[:] = np.nan
    alpha = 2.0 / (period + 1)
    out[period - 1] = np.mean(series[:period])
    for i in range(period, len(series)):
        out[i] = alpha * series[i] + (1 - alpha) * out[i - 1]
    return out


# ---------------------------------------------------------------------------
# True Range — vectorised
# ---------------------------------------------------------------------------

def true_range(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray) -> np.ndarray:
    """True Range for each candle (first element uses high-low only)."""
    prev_close = np.roll(closes, 1)
    prev_close[0] = closes[0]
    hl = highs - lows
    hc = np.abs(highs - prev_close)
    lc = np.abs(lows - prev_close)
    return np.maximum(hl, np.maximum(hc, lc))


# ---------------------------------------------------------------------------
# ATR helpers
# ---------------------------------------------------------------------------

def simple_atr(candles: np.ndarray, period: int = 20) -> float:
    """Average high-low range over the last *period* candles.

    This is the lightweight proxy used by strategies A-C, E-H where a
    full true-range computation is not needed.
    """
    return float(np.mean(candles[-period:, 2] - candles[-period:, 3]))


def atr(highs: np.ndarray, lows: np.ndarray,
        closes: np.ndarray, period: int = 14) -> float:
    """Mean of the last *period* true-range values.

    Used by strategies D, K that need proper ATR sizing.
    """
    tr = true_range(highs, lows, closes)
    if len(tr) < period:
        return float(np.mean(tr))
    return float(np.mean(tr[-period:]))


# ---------------------------------------------------------------------------
# SL / TP computation
# ---------------------------------------------------------------------------

def compute_sl_tp(
    signal: str,
    entry: float,
    atr_val: float,
    sl_mult: float,
    tp_mult: float,
) -> tuple[float | None, float | None]:
    """Return ``(stop_loss, take_profit)`` based on signal direction.

    Returns ``(None, None)`` when *signal* is ``NO-TRADE`` or *atr_val*
    is zero/falsy.
    """
    if not atr_val or signal not in ("BUY", "SELL"):
        return None, None
    if signal == "BUY":
        return round(entry - sl_mult * atr_val, 8), round(entry + tp_mult * atr_val, 8)
    # SELL
    return round(entry + sl_mult * atr_val, 8), round(entry - tp_mult * atr_val, 8)
