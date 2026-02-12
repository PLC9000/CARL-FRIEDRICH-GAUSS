"""Unit tests for the three trading strategies using synthetic candle data."""

import numpy as np
import pytest

from app.strategies.strategy_a_arima import run_strategy_a
from app.strategies.strategy_b_kalman import run_strategy_b
from app.strategies.strategy_c_ou_mean_reversion import run_strategy_c


def _make_candles(closes: np.ndarray) -> np.ndarray:
    """Build a minimal (N, 6) candle array from a close-price series."""
    n = len(closes)
    times = np.arange(n, dtype=float) * 60000
    opens = closes * 0.999
    highs = closes * 1.002
    lows = closes * 0.998
    volumes = np.ones(n) * 100
    return np.column_stack([times, opens, highs, lows, closes, volumes])


def _trending_up(n: int = 200, start: float = 100.0, drift: float = 0.002) -> np.ndarray:
    np.random.seed(42)
    returns = np.random.normal(drift, 0.005, n)
    prices = start * np.cumprod(1 + returns)
    return prices


def _trending_down(n: int = 200, start: float = 100.0, drift: float = -0.002) -> np.ndarray:
    np.random.seed(42)
    returns = np.random.normal(drift, 0.005, n)
    prices = start * np.cumprod(1 + returns)
    return prices


def _mean_reverting(n: int = 200, mean: float = 100.0) -> np.ndarray:
    """Ornstein–Uhlenbeck-ish series that oscillates around `mean`."""
    np.random.seed(42)
    prices = [mean]
    for _ in range(n - 1):
        prices.append(prices[-1] + 0.1 * (mean - prices[-1]) + np.random.normal(0, 0.3))
    return np.array(prices)


# ── Strategy A (ARIMA) ──────────────────────────────────────────────────────

class TestStrategyA:
    def test_trending_up_signals_buy(self):
        candles = _make_candles(_trending_up())
        result = run_strategy_a(candles)
        assert result["recommendation"] in ("BUY", "NO-TRADE")
        assert 0 <= result["confidence"] <= 100
        assert "metrics" in result

    def test_trending_down_signals_sell(self):
        candles = _make_candles(_trending_down())
        result = run_strategy_a(candles)
        assert result["recommendation"] in ("SELL", "NO-TRADE")

    def test_insufficient_data(self):
        candles = _make_candles(np.array([100.0] * 10))
        result = run_strategy_a(candles)
        assert result["recommendation"] == "NO-TRADE"
        assert "Insufficient" in result["explanation"]

    def test_output_shape(self):
        candles = _make_candles(_trending_up())
        result = run_strategy_a(candles)
        for key in ("recommendation", "confidence", "entry", "stop_loss", "take_profit", "explanation", "metrics"):
            assert key in result

    def test_custom_params_ar_order(self):
        candles = _make_candles(_trending_up())
        r_default = run_strategy_a(candles)
        r_custom = run_strategy_a(candles, params={"ar_order": 2})
        # Different AR order should produce different coefficients
        if r_default["metrics"] and r_custom["metrics"]:
            assert len(r_custom["metrics"]["ar_coefficients"]) == 2
            assert len(r_default["metrics"]["ar_coefficients"]) == 4

    def test_custom_sl_tp_multiplier(self):
        candles = _make_candles(_trending_up())
        r1 = run_strategy_a(candles, params={"sl_multiplier": 1.0, "tp_multiplier": 4.0})
        r2 = run_strategy_a(candles, params={"sl_multiplier": 2.0, "tp_multiplier": 1.5})
        if r1["stop_loss"] and r2["stop_loss"]:
            # Larger SL multiplier → further from entry
            assert abs(r2["entry"] - r2["stop_loss"]) > abs(r1["entry"] - r1["stop_loss"])

    def test_none_params_backward_compat(self):
        candles = _make_candles(_trending_up())
        r1 = run_strategy_a(candles)
        r2 = run_strategy_a(candles, params=None)
        assert r1["recommendation"] == r2["recommendation"]
        assert r1["confidence"] == r2["confidence"]


# ── Strategy B (Kalman) ─────────────────────────────────────────────────────

class TestStrategyB:
    def test_trending_up(self):
        candles = _make_candles(_trending_up())
        result = run_strategy_b(candles)
        assert result["recommendation"] in ("BUY", "NO-TRADE")

    def test_trending_down(self):
        candles = _make_candles(_trending_down())
        result = run_strategy_b(candles)
        assert result["recommendation"] in ("SELL", "NO-TRADE")

    def test_insufficient_data(self):
        candles = _make_candles(np.array([50.0] * 10))
        result = run_strategy_b(candles)
        assert result["recommendation"] == "NO-TRADE"

    def test_output_shape(self):
        candles = _make_candles(_trending_up())
        result = run_strategy_b(candles)
        for key in ("recommendation", "confidence", "entry", "explanation", "metrics"):
            assert key in result

    def test_custom_volatility_cap(self):
        """A very low volatility cap should suppress signals (NO-TRADE)."""
        candles = _make_candles(_trending_up())
        result = run_strategy_b(candles, params={"volatility_cap": 0.001})
        assert result["recommendation"] == "NO-TRADE"

    def test_none_params_backward_compat(self):
        candles = _make_candles(_trending_up())
        r1 = run_strategy_b(candles)
        r2 = run_strategy_b(candles, params=None)
        assert r1["recommendation"] == r2["recommendation"]


# ── Strategy C (OU / z-score) ───────────────────────────────────────────────

class TestStrategyC:
    def test_mean_reverting_series(self):
        candles = _make_candles(_mean_reverting())
        result = run_strategy_c(candles)
        assert result["recommendation"] in ("BUY", "SELL", "NO-TRADE")
        assert 0 <= result["confidence"] <= 100

    def test_insufficient_data(self):
        candles = _make_candles(np.array([100.0] * 20))
        result = run_strategy_c(candles)
        assert result["recommendation"] == "NO-TRADE"

    def test_output_shape(self):
        candles = _make_candles(_mean_reverting())
        result = run_strategy_c(candles)
        for key in ("recommendation", "confidence", "entry", "explanation", "metrics"):
            assert key in result

    def test_custom_lookback(self):
        candles = _make_candles(_mean_reverting())
        result = run_strategy_c(candles, params={"lookback": 30})
        assert result["metrics"].get("lookback") == 30

    def test_custom_z_threshold_high_suppresses(self):
        """Very high z-threshold should make it harder to trigger a signal."""
        candles = _make_candles(_mean_reverting())
        result = run_strategy_c(candles, params={"z_threshold": 99.0})
        assert result["recommendation"] == "NO-TRADE"

    def test_none_params_backward_compat(self):
        candles = _make_candles(_mean_reverting())
        r1 = run_strategy_c(candles)
        r2 = run_strategy_c(candles, params=None)
        assert r1["recommendation"] == r2["recommendation"]
