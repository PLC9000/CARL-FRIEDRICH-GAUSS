"""
Strategy J: ADX (Average Directional Index).

Measures trend STRENGTH (not direction) using the ADX line (0-100).
Direction comes from +DI and -DI crossovers.

Signals:
  BUY  — ADX > threshold AND +DI > -DI (strong bullish trend)
  SELL — ADX > threshold AND -DI > +DI (strong bearish trend)
  NO-TRADE — ADX below threshold (weak/no trend)
"""

import logging
import numpy as np

from app.strategies._common import no_trade, true_range as _true_range, compute_sl_tp

logger = logging.getLogger(__name__)

DEFAULTS = {
    "adx_period": 14,
    "adx_threshold": 25,    # ADX must be above this to signal a trend
    "di_confirm": 1,        # candles +DI must stay above -DI (or vice versa)
    "require_rising": False, # only signal if ADX is rising (strengthening)
    "sl_multiplier": 1.5,
    "tp_multiplier": 2.5,
}


def _wilder_smooth(series: np.ndarray, period: int) -> np.ndarray:
    """Wilder's smoothing (equivalent to EMA with alpha=1/period).

    Uses the average form: seed = mean(first N values),
    then smoothed[i] = (smoothed[i-1] * (N-1) + value[i]) / N.
    """
    out = np.empty(len(series))
    out[:] = np.nan
    out[period - 1] = np.mean(series[:period])
    for i in range(period, len(series)):
        out[i] = (out[i - 1] * (period - 1) + series[i]) / period
    return out


def _compute_adx(highs, lows, closes, period):
    """Return (adx, plus_di, minus_di) arrays."""
    n = len(closes)

    # Directional movement
    up_move = np.diff(highs, prepend=highs[0])
    down_move = np.diff(-lows, prepend=-lows[0])

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    tr = _true_range(highs, lows, closes)

    # Wilder-smooth TR, +DM, -DM
    atr = _wilder_smooth(tr, period)
    smooth_plus = _wilder_smooth(plus_dm, period)
    smooth_minus = _wilder_smooth(minus_dm, period)

    # +DI and -DI
    plus_di = np.where(atr > 0, 100 * smooth_plus / atr, 0.0)
    minus_di = np.where(atr > 0, 100 * smooth_minus / atr, 0.0)

    # DX
    di_sum = plus_di + minus_di
    dx = np.where(di_sum > 0, 100 * np.abs(plus_di - minus_di) / di_sum, 0.0)

    # ADX = Wilder smooth of DX
    adx = _wilder_smooth(dx, period)

    return adx, plus_di, minus_di, atr


def run_strategy_j(candles: np.ndarray, params: dict | None = None) -> dict:
    p = {**DEFAULTS, **(params or {})}
    period = int(p["adx_period"])
    threshold = float(p["adx_threshold"])
    confirm = int(p["di_confirm"])
    require_rising = bool(p["require_rising"])
    sl_mult = float(p["sl_multiplier"])
    tp_mult = float(p["tp_multiplier"])

    highs = candles[:, 2].astype(float)
    lows = candles[:, 3].astype(float)
    closes = candles[:, 4].astype(float)

    min_len = period * 2 + confirm + 2
    if len(closes) < min_len:
        return no_trade(f"Need >= {min_len} candles, got {len(closes)}")

    adx, plus_di, minus_di, atr_arr = _compute_adx(highs, lows, closes, period)

    # Current values
    adx_now = float(adx[-1])
    adx_prev = float(adx[-2])
    plus_now = float(plus_di[-1])
    minus_now = float(minus_di[-1])

    if np.isnan(adx_now) or np.isnan(plus_now):
        return no_trade("Not enough data to compute ADX")

    # Determine signal
    signal = "NO-TRADE"
    adx_rising = adx_now > adx_prev

    if adx_now >= threshold:
        if require_rising and not adx_rising:
            signal = "NO-TRADE"
        else:
            # Check DI direction with confirmation
            plus_tail = plus_di[-(confirm + 1):]
            minus_tail = minus_di[-(confirm + 1):]
            if not np.any(np.isnan(plus_tail)):
                di_diff = plus_tail - minus_tail
                if np.all(di_diff[1:] > 0):
                    signal = "BUY"
                elif np.all(di_diff[1:] < 0):
                    signal = "SELL"

    # Confidence: based on ADX strength (25=low, 50+=high)
    if adx_now >= threshold:
        denom = max(50 - threshold, 1)
        raw_conf = min((adx_now - threshold) / denom * 80 + 20, 100)
    else:
        raw_conf = adx_now / threshold * 15  # weak confidence below threshold
    confidence = round(max(0, min(100, raw_conf)), 1)

    last_close = float(closes[-1])
    atr_val = float(atr_arr[-1]) if not np.isnan(atr_arr[-1]) else 0

    entry = last_close
    stop_loss, take_profit = compute_sl_tp(signal, entry, atr_val, sl_mult, tp_mult)

    trend_str = (
        "fuerte alcista" if adx_now >= 50 and plus_now > minus_now
        else "alcista" if plus_now > minus_now
        else "fuerte bajista" if adx_now >= 50 and minus_now > plus_now
        else "bajista" if minus_now > plus_now
        else "sin direccion"
    )

    # Force: raw ADX value normalized to 0-1 for strength role
    force = round(min(adx_now / 100.0, 1.0), 4)

    return {
        "recommendation": signal,
        "confidence": confidence,
        "force": force,
        "entry": round(entry, 8),
        "stop_loss": stop_loss,
        "take_profit": take_profit,
        "explanation": (
            f"ADX({period}) = {adx_now:.1f} ({'subiendo' if adx_rising else 'bajando'}). "
            f"+DI = {plus_now:.1f}, -DI = {minus_now:.1f}. "
            f"Tendencia: {trend_str}. Umbral: {threshold}."
        ),
        "metrics": {
            "adx": round(adx_now, 2),
            "adx_prev": round(adx_prev, 2),
            "adx_rising": adx_rising,
            "plus_di": round(plus_now, 2),
            "minus_di": round(minus_now, 2),
            "trend": trend_str,
            "threshold": threshold,
            "force": force,
        },
    }


