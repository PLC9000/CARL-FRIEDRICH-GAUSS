"""
Strategy D: EMA Crossover — Scoring continuo con normalización por volatilidad.

En vez de detectar un cruce puntual, calcula un score continuo basado en la
distancia relativa entre EMA rápida y EMA lenta, normalizada por volatilidad
y mapeada a [-100, +100] mediante tanh.

Convención de señal (trend-following):
  pct > 0   → BUY   (EMA fast por encima de slow = tendencia alcista)
  pct < 0   → SELL  (EMA fast por debajo de slow = tendencia bajista)
  |pct| < dz → HOLD / NO-TRADE  (dentro de zona muerta)
"""

import logging
import math

import numpy as np

from app.strategies._common import (
    no_trade, ema as _ema, atr as _atr, compute_sl_tp,
)

logger = logging.getLogger(__name__)

DEFAULTS = {
    "ema_fast_period": 9,
    "ema_slow_period": 21,
    "volatility_period": 14,
    "k": 1.0,              # sensibilidad del tanh
    "deadzone": 3.0,       # |pct| menor a esto → HOLD
    "sl_multiplier": 1.5,
    "tp_multiplier": 2.5,
}


def run_strategy_d(candles: np.ndarray, params: dict | None = None) -> dict:
    raw = params or {}
    p = {**DEFAULTS, **raw}

    # Accept legacy param names (fast_period/slow_period) as fallback
    fast_period = int(raw.get("ema_fast_period", raw.get("fast_period", DEFAULTS["ema_fast_period"])))
    slow_period = int(raw.get("ema_slow_period", raw.get("slow_period", DEFAULTS["ema_slow_period"])))
    vol_period = int(p["volatility_period"])
    k = float(p["k"])
    deadzone = float(p["deadzone"])
    sl_mult = float(p["sl_multiplier"])
    tp_mult = float(p["tp_multiplier"])

    # --- Validaciones ---
    if fast_period >= slow_period:
        return no_trade("ema_fast_period debe ser < ema_slow_period")

    closes = candles[:, 4].astype(float)
    highs = candles[:, 2].astype(float)
    lows = candles[:, 3].astype(float)

    min_len = max(slow_period + 2, vol_period + 2)
    if len(closes) < min_len:
        return no_trade(f"Need >= {min_len} candles, got {len(closes)}")

    # --- 1-2) Calcular EMAs ---
    ema_fast = _ema(closes, fast_period)
    ema_slow = _ema(closes, slow_period)

    ema_f = float(ema_fast[-1])
    ema_s = float(ema_slow[-1])
    last_close = float(closes[-1])

    if np.isnan(ema_f) or np.isnan(ema_s):
        return no_trade("Not enough data to compute EMAs")

    # --- 3) diff = (EMA_fast - EMA_slow) / close ---
    diff = (ema_f - ema_s) / last_close if last_close else 0.0

    # --- 4) Normalizar por volatilidad (std de retornos) ---
    ret_slice = closes[-(vol_period + 1):]
    returns = np.diff(ret_slice) / ret_slice[:-1]
    volatility = float(np.std(returns)) if len(returns) > 1 else 0.0

    epsilon = 1e-10
    norm = diff / (volatility + epsilon)

    # --- 5) Mapeo tanh → [-100, +100] ---
    scaled_pct = round(100.0 * math.tanh(k * norm), 1)

    # --- 6) Reglas con deadzone ---
    if abs(scaled_pct) < deadzone:
        scaled_pct = 0.0
        signal = "NO-TRADE"
    elif scaled_pct > 0:
        signal = "BUY"
    else:
        signal = "SELL"

    confidence = round(abs(scaled_pct), 1)

    # --- Entry / SL / TP ---
    atr_val = _atr(highs, lows, closes, min(14, len(closes)))
    entry = last_close
    stop_loss, take_profit = compute_sl_tp(signal, entry, atr_val, sl_mult, tp_mult)

    return {
        "recommendation": signal,
        "confidence": confidence,
        "entry": round(entry, 8),
        "stop_loss": stop_loss,
        "take_profit": take_profit,
        "explanation": (
            f"EMA Crossover: fast({fast_period})={ema_f:.2f}, "
            f"slow({slow_period})={ema_s:.2f}. "
            f"Diff={diff * 100:.4f}%, Vol={volatility * 100:.4f}%, "
            f"Norm={norm:.4f}, Score={scaled_pct:+.1f}%."
        ),
        "metrics": {
            "ema_fast": round(ema_f, 8),
            "ema_slow": round(ema_s, 8),
            "diff": round(diff * 100, 4),
            "volatility": round(volatility * 100, 4),
            "norm": round(norm, 4),
            "scaled_pct": scaled_pct,
            "k": k,
            "deadzone": deadzone,
            "ema_fast_period": fast_period,
            "ema_slow_period": slow_period,
        },
    }


