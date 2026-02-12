"""
Strategy L: Pendiente de EMA (EMA Slope Angle).

Mide el ángulo de inclinación de la EMA rápida respecto a la horizontal
mediante regresión lineal sobre una ventana móvil. El ángulo se normaliza
a un factor continuo [-100, +100] donde:

    0   → EMA plana (lateralización)
  +100  → máxima pendiente alcista esperada
  -100  → máxima pendiente bajista esperada

El rango máximo de ángulos (±30°, ±45°, ±60°) es parametrizable.

Base matemática:
  1. Se calcula la EMA rápida sobre closes (filtra ruido de alta frecuencia).
  2. Regresión lineal sobre los últimos N puntos de la EMA → pendiente m.
  3. Se normaliza m por el nivel de precio medio: m_norm = m / mean(ema).
  4. Ángulo: θ = arctan(m_norm × slope_window) en grados.
  5. Factor = clamp(θ / max_angle, -1, 1) × 100.
  6. Suavizado opcional: se calcula una serie de ángulos desplazando la
     ventana hacia atrás y se aplica media ponderada exponencial para
     prevenir picos aislados.

¿Por qué sobre la EMA y no sobre el precio?
  La EMA ya filtra el ruido intravela. Calcular pendiente directamente
  sobre el precio crudo produce ángulos erráticos dominados por velas
  individuales. La pendiente de la EMA refleja la tendencia subyacente.
"""

import logging
import math

import numpy as np

from app.strategies._common import (
    no_trade,
    ema as _ema,
    simple_atr as _simple_atr,
    compute_sl_tp,
)

logger = logging.getLogger(__name__)

DEFAULTS = {
    "ema_period": 9,          # Período de la EMA rápida
    "slope_window": 10,       # Ventana de regresión lineal (puntos)
    "max_angle": 45.0,        # Ángulo máximo para normalización (grados)
    "smooth_period": 5,       # EMA secundaria sobre ángulos (1 = sin suavizado)
    "deadzone": 5.0,          # |factor| mínimo para generar señal
    "sl_multiplier": 1.5,
    "tp_multiplier": 2.5,
}


def _compute_angle(ema_slice: np.ndarray, slope_window: int) -> float:
    """Calcula el ángulo de inclinación de una porción de EMA.

    Usa regresión lineal sobre los últimos *slope_window* puntos,
    normaliza la pendiente por el nivel medio de precio, y retorna
    el ángulo en grados (positivo = alcista, negativo = bajista).
    """
    y = ema_slice[-slope_window:]
    x = np.arange(slope_window, dtype=float)

    # Regresión lineal: y = m*x + b
    m, _ = np.polyfit(x, y, 1)

    # Normalizar por precio medio para hacer la pendiente independiente
    # del nivel absoluto del activo (BTC ~97000 vs SOL ~200)
    mean_price = float(np.mean(y))
    if mean_price == 0:
        return 0.0
    m_norm = m / mean_price

    # El ángulo "visual" considera cuánto se movió en toda la ventana,
    # no solo en una vela. Multiplicamos por slope_window.
    angle_deg = math.degrees(math.atan(m_norm * slope_window))
    return angle_deg


def run_strategy_l(candles: np.ndarray, params: dict | None = None) -> dict:
    raw = params or {}
    p = {**DEFAULTS, **raw}

    ema_period = int(p["ema_period"])
    slope_window = int(p["slope_window"])
    max_angle = float(p["max_angle"])
    smooth_period = int(p["smooth_period"])
    deadzone = float(p["deadzone"])
    sl_mult = float(p["sl_multiplier"])
    tp_mult = float(p["tp_multiplier"])

    # --- Validaciones ---
    if max_angle <= 0:
        return no_trade("max_angle debe ser > 0")

    closes = candles[:, 4].astype(float)
    highs = candles[:, 2].astype(float)
    lows = candles[:, 3].astype(float)

    # Necesitamos al menos: ema_period + slope_window + smooth_period puntos
    min_len = ema_period + slope_window + max(smooth_period, 1)
    if len(closes) < min_len:
        return no_trade(
            f"Se necesitan >= {min_len} velas, hay {len(closes)}"
        )

    # --- 1) Calcular EMA rápida ---
    ema_fast = _ema(closes, ema_period)

    # Encontrar el primer índice no-NaN
    valid_mask = ~np.isnan(ema_fast)
    if valid_mask.sum() < slope_window + smooth_period:
        return no_trade("No hay suficientes datos de EMA válidos")

    # --- 2) Ángulo actual (sin suavizar) ---
    angle_raw = _compute_angle(ema_fast[valid_mask], slope_window)

    # --- 3) Suavizado: calcular serie de ángulos y promediar ---
    if smooth_period > 1:
        valid_ema = ema_fast[valid_mask]
        n_angles = min(smooth_period, len(valid_ema) - slope_window + 1)
        if n_angles < 1:
            n_angles = 1

        angles = []
        for i in range(n_angles):
            end_idx = len(valid_ema) - i
            if end_idx < slope_window:
                break
            a = _compute_angle(valid_ema[:end_idx], slope_window)
            angles.append(a)

        # Pesos exponenciales: más recientes pesan más
        angles.reverse()  # ahora van de más antiguo a más reciente
        if len(angles) > 1:
            alpha = 2.0 / (len(angles) + 1)
            smoothed = angles[0]
            for a in angles[1:]:
                smoothed = alpha * a + (1 - alpha) * smoothed
            angle_smoothed = smoothed
        else:
            angle_smoothed = angles[0] if angles else angle_raw
    else:
        angle_smoothed = angle_raw

    # --- 4) Normalizar a factor [-100, +100] ---
    clamped = max(-1.0, min(1.0, angle_smoothed / max_angle))
    factor = round(clamped * 100, 1)

    # --- 5) Señal y confianza ---
    if abs(factor) < deadzone:
        signal = "NO-TRADE"
        confidence = 0.0
    elif factor > 0:
        signal = "BUY"
        confidence = round(abs(factor), 1)
    else:
        signal = "SELL"
        confidence = round(abs(factor), 1)

    # --- 6) Force para rol "strength" ---
    force = round(abs(factor) / 100.0, 4)

    # --- 7) Entry / SL / TP ---
    last_close = float(closes[-1])
    atr_val = _simple_atr(candles, min(20, len(closes)))
    entry = last_close
    stop_loss, take_profit = compute_sl_tp(signal, entry, atr_val, sl_mult, tp_mult)

    ema_value = float(ema_fast[valid_mask][-1])

    # Pendiente cruda del último cálculo
    y = ema_fast[valid_mask][-slope_window:]
    slope_raw = float(np.polyfit(np.arange(slope_window), y, 1)[0])
    slope_normalized = slope_raw / float(np.mean(y)) if np.mean(y) != 0 else 0.0

    return {
        "recommendation": signal,
        "confidence": confidence,
        "force": force,
        "entry": round(entry, 8),
        "stop_loss": stop_loss,
        "take_profit": take_profit,
        "explanation": (
            f"EMA Slope: EMA({ema_period}) = {ema_value:.2f}. "
            f"Pendiente normalizada = {slope_normalized:.6f}. "
            f"Ángulo crudo = {angle_raw:+.2f}°, "
            f"suavizado = {angle_smoothed:+.2f}°. "
            f"Factor = {factor:+.1f} (max_angle = ±{max_angle}°)."
        ),
        "metrics": {
            "ema_value": round(ema_value, 8),
            "slope_raw": round(slope_raw, 8),
            "slope_normalized": round(slope_normalized, 6),
            "angle_degrees": round(angle_raw, 2),
            "angle_smoothed": round(angle_smoothed, 2),
            "factor": factor,
            "force": force,
            "max_angle": max_angle,
            "ema_period": ema_period,
            "slope_window": slope_window,
            "smooth_period": smooth_period,
            "deadzone": deadzone,
        },
    }
