"""
Strategy A: Autoregressive time-series forecasting.

Fits an AR(p) model on differenced closing prices (equivalent to an
ARI(p,1) model), forecasts the next K candles, and derives a
BUY/SELL/NO-TRADE signal from the forecasted return vs. a threshold.

Uses numpy/scipy only — no heavy external dependencies.
"""

import logging

import numpy as np
from scipy import linalg

from app.strategies._common import no_trade, simple_atr, compute_sl_tp

logger = logging.getLogger(__name__)

DEFAULTS: dict[str, float] = {
    "ar_order": 4,
    "forecast_horizon": 5,
    "return_threshold": 0.5,    # percent
}


def _fit_ar(series: np.ndarray, order: int) -> tuple[np.ndarray, float]:
    """Fit AR(order) via Yule-Walker equations. Returns (coefficients, residual_std)."""
    n = len(series)
    if n <= order:
        raise ValueError("Series too short for requested AR order")
    mean = np.mean(series)
    centred = series - mean
    # Autocorrelation
    r = np.correlate(centred, centred, mode="full")[n - 1:]
    r = r[: order + 1]
    if r[0] == 0:
        raise ValueError("Zero-variance series")
    r_norm = r / r[0]
    # Yule-Walker: R * phi = r
    R_mat = linalg.toeplitz(r_norm[:order])
    phi = linalg.solve(R_mat, r_norm[1: order + 1])
    # Residual variance
    residuals = []
    for t in range(order, n):
        pred = mean + np.dot(phi, centred[t - order: t][::-1])
        residuals.append(series[t] - pred)
    res_std = float(np.std(residuals)) if residuals else 0.0
    return phi, res_std


def _forecast_ar(
    series: np.ndarray, phi: np.ndarray, steps: int,
) -> np.ndarray:
    """Produce multi-step forecasts from an AR model."""
    order = len(phi)
    mean = np.mean(series)
    centred = series - mean
    buf = list(centred[-order:])
    forecasts = []
    for _ in range(steps):
        pred = np.dot(phi, buf[-order:][::-1])
        forecasts.append(pred + mean)
        buf.append(pred)
    return np.array(forecasts)


def run_strategy_a(candles: np.ndarray, params: dict | None = None) -> dict:
    """
    Parameters
    ----------
    candles : ndarray of shape (N, 6) — [time, open, high, low, close, volume]
    params  : optional overrides for strategy constants

    Returns
    -------
    dict with keys: recommendation, confidence, entry, stop_loss, take_profit,
                    explanation, metrics
    """
    p = {**DEFAULTS, **(params or {})}
    ar_order = int(p["ar_order"])
    forecast_horizon = int(p["forecast_horizon"])
    return_threshold = p["return_threshold"] / 100  # convert from % to decimal

    closes = candles[:, 4].astype(float)

    if len(closes) < 30:
        return no_trade("Insufficient data for AR model (need >= 30 candles)")

    # Difference once (I(1))
    diff = np.diff(closes)

    try:
        phi, res_std = _fit_ar(diff, ar_order)
    except ValueError as exc:
        logger.warning("AR fit failed: %s", exc)
        return no_trade(f"AR fitting error: {exc}")

    # Forecast differenced series
    diff_forecast = _forecast_ar(diff, phi, forecast_horizon)

    # Reconstruct price levels from cumulative sum of forecasted diffs
    forecast_prices = closes[-1] + np.cumsum(diff_forecast - np.mean(diff))

    last_close = float(closes[-1])
    forecast_end = float(forecast_prices[-1])
    expected_return = (forecast_end - last_close) / last_close

    # Confidence from residual std relative to price
    raw_confidence = max(0, 1 - (res_std / last_close) * 10)
    confidence = round(min(raw_confidence * 100, 100), 1)

    # Signal
    if expected_return > return_threshold:
        signal = "BUY"
    elif expected_return < -return_threshold:
        signal = "SELL"
    else:
        signal = "NO-TRADE"

    # Price levels
    entry = last_close
    atr = simple_atr(candles)
    stop_loss, take_profit = compute_sl_tp(signal, entry, atr, 1.5, 2.5)

    # Approximate 95% confidence interval
    ci_half = 1.96 * res_std * np.sqrt(forecast_horizon)

    return {
        "recommendation": signal,
        "confidence": confidence,
        "entry": round(entry, 8),
        "stop_loss": stop_loss,
        "take_profit": take_profit,
        "explanation": (
            f"AR({ar_order}) on differenced prices forecasts {forecast_horizon}-step "
            f"return of {expected_return * 100:.2f}%. Threshold \u00b1{return_threshold * 100:.1f}%."
        ),
        "metrics": {
            "forecast_return_pct": round(expected_return * 100, 4),
            "forecast_end_price": round(forecast_end, 8),
            "residual_std": round(res_std, 8),
            "ar_coefficients": [round(float(c), 6) for c in phi],
            "conf_lower": round(forecast_end - ci_half, 8),
            "conf_upper": round(forecast_end + ci_half, 8),
        },
    }


