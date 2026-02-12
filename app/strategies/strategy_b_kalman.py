"""
Strategy B: Kalman Filter / State-Space smoothing + trend detection.

Uses a simple Kalman filter to smooth the price series, then
derives a trend slope. Signal is based on trend strength filtered
by a volatility gate.
"""

import logging

import numpy as np

from app.strategies._common import no_trade, simple_atr, compute_sl_tp

logger = logging.getLogger(__name__)

DEFAULTS: dict[str, float] = {
    "process_noise": 0.00001,
    "measurement_noise": 0.01,
    "trend_threshold": 0.03,     # percent (0.03% per bar)
    "volatility_cap": 5.0,      # percent
    "sl_multiplier": 1.5,
    "tp_multiplier": 2.5,
}


def _kalman_smooth(prices: np.ndarray, process_noise: float, measurement_noise: float) -> tuple[np.ndarray, np.ndarray]:
    """
    1-D Kalman filter that tracks [level, slope].

    Returns
    -------
    smoothed : ndarray — filtered price level
    slopes   : ndarray — estimated slope at each step
    """
    n = len(prices)
    # State: [level, slope]
    x = np.array([prices[0], 0.0])
    P = np.eye(2) * 1.0
    F = np.array([[1.0, 1.0],
                  [0.0, 1.0]])
    H = np.array([[1.0, 0.0]])
    Q = np.eye(2) * process_noise
    R = np.array([[measurement_noise]])

    smoothed = np.zeros(n)
    slopes = np.zeros(n)

    for i in range(n):
        # Predict
        x = F @ x
        P = F @ P @ F.T + Q
        # Update
        y = prices[i] - H @ x
        S = H @ P @ H.T + R
        K = P @ H.T @ np.linalg.inv(S)
        x = x + (K @ y).flatten()
        P = (np.eye(2) - K @ H) @ P

        smoothed[i] = x[0]
        slopes[i] = x[1]

    return smoothed, slopes


def run_strategy_b(candles: np.ndarray, params: dict | None = None) -> dict:
    p = {**DEFAULTS, **(params or {})}
    process_noise = p["process_noise"]
    measurement_noise = p["measurement_noise"]
    trend_threshold = p["trend_threshold"] / 100   # convert from % to decimal
    volatility_cap = p["volatility_cap"] / 100     # convert from % to decimal
    sl_mult = p["sl_multiplier"]
    tp_mult = p["tp_multiplier"]

    closes = candles[:, 4].astype(float)

    if len(closes) < 30:
        return no_trade("Insufficient data for Kalman filter (need >= 30 candles)")

    smoothed, slopes = _kalman_smooth(closes, process_noise, measurement_noise)

    last_close = float(closes[-1])
    last_smooth = float(smoothed[-1])
    last_slope = float(slopes[-1])

    # Normalised slope (per-unit price move per candle)
    norm_slope = last_slope / last_close if last_close != 0 else 0

    # Rolling volatility (last 20 candles)
    returns = np.diff(np.log(closes[-21:]))
    rolling_vol = float(np.std(returns)) if len(returns) > 1 else 0

    # Trend strength: absolute normalised slope relative to volatility
    trend_strength = abs(norm_slope) / (rolling_vol + 1e-12)

    # Confidence from trend strength (capped at 100)
    confidence = round(min(trend_strength * 20, 100), 1)

    # Volatility filter
    if rolling_vol > volatility_cap:
        return no_trade(
            f"Volatility too high ({rolling_vol:.4f} > {volatility_cap:.4f}). "
            "Kalman trend unreliable."
        )

    # Signal
    if norm_slope > trend_threshold:
        signal = "BUY"
    elif norm_slope < -trend_threshold:
        signal = "SELL"
    else:
        signal = "NO-TRADE"

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
            f"Kalman filter normalised slope = {norm_slope:.6f} "
            f"(threshold \u00b1{trend_threshold:.6f}). "
            f"Rolling vol = {rolling_vol:.4f}. Trend strength = {trend_strength:.2f}."
        ),
        "metrics": {
            "kalman_level": round(last_smooth, 8),
            "kalman_slope": round(last_slope, 8),
            "norm_slope": round(norm_slope, 8),
            "rolling_vol": round(rolling_vol, 6),
            "trend_strength": round(trend_strength, 4),
        },
    }


