"""
Strategy C: Probabilistic mean-reversion via Ornstein–Uhlenbeck / z-score.

Computes a rolling z-score of log-returns. If the z-score falls
below -Z we signal BUY (price is stretched below the mean); above +Z
we signal SELL. A volatility-regime filter suppresses signals in
high-volatility regimes where mean-reversion is unreliable.
"""

import logging

import numpy as np

from app.strategies._common import no_trade, simple_atr, compute_sl_tp

logger = logging.getLogger(__name__)

DEFAULTS: dict[str, float] = {
    "lookback": 50,
    "z_threshold": 2.0,
    "vol_regime_multiplier": 1.5,
}


def run_strategy_c(candles: np.ndarray, params: dict | None = None) -> dict:
    p = {**DEFAULTS, **(params or {})}
    lookback = int(p["lookback"])
    z_threshold = p["z_threshold"]
    vol_mult = p["vol_regime_multiplier"]

    closes = candles[:, 4].astype(float)

    if len(closes) < lookback + 10:
        return no_trade(f"Insufficient data (need >= {lookback + 10} candles)")

    log_prices = np.log(closes)
    log_returns = np.diff(log_prices)

    # Rolling statistics over the lookback window
    window = log_returns[-lookback:]
    rolling_mean = float(np.mean(window))
    rolling_std = float(np.std(window, ddof=1))

    if rolling_std < 1e-12:
        return no_trade("Zero variance in returns \u2014 flat price series")

    # Current return z-score (last return vs rolling distribution)
    current_return = float(log_returns[-1])
    z_score = (current_return - rolling_mean) / rolling_std

    # Cumulative z-score (smoothed over last 5 returns for robustness)
    recent_z_scores = [(log_returns[-i] - rolling_mean) / rolling_std for i in range(1, 6)]
    avg_z = float(np.mean(recent_z_scores))

    # Volatility regime filter
    long_run_std = float(np.std(log_returns, ddof=1))
    vol_ratio = rolling_std / (long_run_std + 1e-12)
    high_vol_regime = vol_ratio > vol_mult

    if high_vol_regime:
        return no_trade(
            f"High-volatility regime (vol ratio {vol_ratio:.2f} > "
            f"{vol_mult}). Mean-reversion suppressed."
        )

    # Confidence: based on how far beyond the threshold the z-score is
    excess = abs(avg_z) - z_threshold
    confidence = round(min(max(excess / z_threshold * 80 + 50, 0), 100), 1) if excess > 0 else round(max(30 - abs(excess) * 20, 0), 1)

    # Signal from averaged z-score
    if avg_z < -z_threshold:
        signal = "BUY"
    elif avg_z > z_threshold:
        signal = "SELL"
    else:
        signal = "NO-TRADE"

    last_close = float(closes[-1])
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
            f"OU mean-reversion: avg z-score (5-bar) = {avg_z:.3f} "
            f"(threshold \u00b1{z_threshold}). "
            f"Vol ratio = {vol_ratio:.2f}."
        ),
        "metrics": {
            "z_score_current": round(z_score, 4),
            "z_score_avg5": round(avg_z, 4),
            "rolling_mean_return": round(rolling_mean, 8),
            "rolling_std_return": round(rolling_std, 8),
            "vol_ratio": round(vol_ratio, 4),
            "lookback": lookback,
        },
    }


