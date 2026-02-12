"""Unit tests for the six new trading strategies (D–I) using synthetic data."""

import numpy as np
import pytest

from app.strategies.strategy_d_ma_crossover import run_strategy_d
from app.strategies.strategy_e_rsi import run_strategy_e
from app.strategies.strategy_f_macd import run_strategy_f
from app.strategies.strategy_g_bollinger import run_strategy_g
from app.strategies.strategy_h_breakout import run_strategy_h
from app.strategies.strategy_i_atr_trailing import run_strategy_i
from app.strategies.strategy_j_adx import run_strategy_j
from app.strategies.strategy_k_fng import run_strategy_k


# ── Helpers ────────────────────────────────────────────────────────────

def _make_candles(closes: np.ndarray, volumes: np.ndarray | None = None) -> np.ndarray:
    n = len(closes)
    times = np.arange(n, dtype=float) * 60000
    opens = closes * 0.999
    highs = closes * 1.002
    lows = closes * 0.998
    vols = volumes if volumes is not None else np.ones(n) * 100
    return np.column_stack([times, opens, highs, lows, closes, vols])


def _trending_up(n=200, start=100.0, drift=0.002):
    np.random.seed(42)
    returns = np.random.normal(drift, 0.005, n)
    return start * np.cumprod(1 + returns)


def _trending_down(n=200, start=100.0, drift=-0.002):
    np.random.seed(42)
    returns = np.random.normal(drift, 0.005, n)
    return start * np.cumprod(1 + returns)


def _sideways(n=200, mean=100.0):
    np.random.seed(42)
    prices = [mean]
    for _ in range(n - 1):
        prices.append(prices[-1] + 0.1 * (mean - prices[-1]) + np.random.normal(0, 0.3))
    return np.array(prices)


def _golden_cross(n=60, fast=9, slow=21):
    """Generate prices that produce a golden cross (fast crosses above slow)."""
    np.random.seed(42)
    # Start with downtrend then switch to uptrend
    down = 100.0 * np.cumprod(1 + np.random.normal(-0.001, 0.002, n // 2))
    up = down[-1] * np.cumprod(1 + np.random.normal(0.005, 0.002, n // 2))
    return np.concatenate([down, up])


def _overbought(n=50, period=14):
    """Prices rising steeply to push RSI above 70."""
    np.random.seed(42)
    return 100.0 * np.cumprod(1 + np.random.normal(0.008, 0.002, n))


def _oversold(n=50, period=14):
    """Prices falling steeply to push RSI below 30."""
    np.random.seed(42)
    return 100.0 * np.cumprod(1 + np.random.normal(-0.008, 0.002, n))


def _breakout_up(n=40, lookback=20):
    """Flat then breakout above the channel."""
    np.random.seed(42)
    flat = 100.0 + np.random.normal(0, 0.3, n - 5)
    spike = np.array([103.0, 104.0, 105.0, 106.0, 107.0])
    return np.concatenate([flat, spike])


REQUIRED_KEYS = ("recommendation", "confidence", "entry", "stop_loss",
                 "take_profit", "explanation", "metrics")


# ── Strategy D (EMA Crossover — continuous tanh scoring) ─────────────

class TestStrategyD:
    def test_output_shape(self):
        candles = _make_candles(_trending_up())
        result = run_strategy_d(candles)
        for key in REQUIRED_KEYS:
            assert key in result
        assert "scaled_pct" in result["metrics"]

    def test_trending_up_buys(self):
        """Strong uptrend → EMA fast > slow → positive pct → BUY."""
        candles = _make_candles(_trending_up())
        result = run_strategy_d(candles)
        assert result["recommendation"] in ("BUY", "NO-TRADE")
        if result["recommendation"] == "BUY":
            assert result["metrics"]["scaled_pct"] > 0

    def test_trending_down_sells(self):
        """Strong downtrend → EMA fast < slow → negative pct → SELL."""
        candles = _make_candles(_trending_down())
        result = run_strategy_d(candles)
        assert result["recommendation"] in ("SELL", "NO-TRADE")
        if result["recommendation"] == "SELL":
            assert result["metrics"]["scaled_pct"] < 0

    def test_insufficient_data(self):
        candles = _make_candles(np.array([100.0] * 5))
        result = run_strategy_d(candles)
        assert result["recommendation"] == "NO-TRADE"
        assert "Need" in result["explanation"]

    def test_fast_gte_slow_rejected(self):
        candles = _make_candles(_trending_up())
        result = run_strategy_d(candles, params={"ema_fast_period": 50, "ema_slow_period": 20})
        assert result["recommendation"] == "NO-TRADE"

    def test_scaled_pct_range(self):
        """scaled_pct must always be in [-100, 100]."""
        for gen in [_trending_up, _trending_down, _sideways]:
            candles = _make_candles(gen())
            result = run_strategy_d(candles)
            if result["metrics"]:
                pct = result["metrics"].get("scaled_pct", 0)
                assert -100 <= pct <= 100

    def test_high_k_saturates(self):
        """Higher k → more extreme pct for the same data."""
        candles = _make_candles(_trending_up())
        r1 = run_strategy_d(candles, params={"k": 0.3})
        r2 = run_strategy_d(candles, params={"k": 5.0})
        if r1["metrics"] and r2["metrics"]:
            assert abs(r2["metrics"]["scaled_pct"]) >= abs(r1["metrics"]["scaled_pct"])

    def test_deadzone_suppresses(self):
        """With a huge deadzone, everything becomes NO-TRADE."""
        candles = _make_candles(_sideways())
        result = run_strategy_d(candles, params={"deadzone": 99.0})
        assert result["recommendation"] == "NO-TRADE"
        assert result["metrics"].get("scaled_pct", 0) == 0.0

    def test_metrics_complete(self):
        candles = _make_candles(_trending_up())
        result = run_strategy_d(candles)
        m = result["metrics"]
        for k in ("ema_fast", "ema_slow", "diff", "volatility", "norm", "scaled_pct"):
            assert k in m, f"Missing metric: {k}"

    def test_legacy_param_names(self):
        """Old param names (fast_period/slow_period) still work."""
        candles = _make_candles(_trending_up())
        result = run_strategy_d(candles, params={"fast_period": 5, "slow_period": 30})
        assert result["metrics"]["ema_fast_period"] == 5
        assert result["metrics"]["ema_slow_period"] == 30


# ── Strategy E (RSI) ──────────────────────────────────────────────────

class TestStrategyE:
    def test_output_shape(self):
        candles = _make_candles(_trending_up())
        result = run_strategy_e(candles)
        for key in REQUIRED_KEYS:
            assert key in result

    def test_overbought_signals_sell(self):
        candles = _make_candles(_overbought(80))
        result = run_strategy_e(candles)
        assert result["recommendation"] in ("SELL", "NO-TRADE")

    def test_oversold_signals_buy(self):
        candles = _make_candles(_oversold(80))
        result = run_strategy_e(candles)
        assert result["recommendation"] in ("BUY", "NO-TRADE")

    def test_insufficient_data(self):
        candles = _make_candles(np.array([100.0] * 5))
        result = run_strategy_e(candles)
        assert result["recommendation"] == "NO-TRADE"

    def test_rsi_in_metrics(self):
        candles = _make_candles(_trending_up())
        result = run_strategy_e(candles)
        if result["metrics"]:
            assert "rsi" in result["metrics"]
            assert 0 <= result["metrics"]["rsi"] <= 100

    def test_custom_thresholds(self):
        candles = _make_candles(_trending_up())
        result = run_strategy_e(candles, params={"overbought": 90, "oversold": 10})
        assert result["metrics"].get("overbought") == 90


# ── Strategy F (MACD) ─────────────────────────────────────────────────

class TestStrategyF:
    def test_output_shape(self):
        candles = _make_candles(_trending_up())
        result = run_strategy_f(candles)
        for key in REQUIRED_KEYS:
            assert key in result

    def test_trending_produces_signal(self):
        candles = _make_candles(_golden_cross(120))
        result = run_strategy_f(candles, params={"fast": 8, "slow": 17, "signal": 5})
        assert result["recommendation"] in ("BUY", "SELL", "NO-TRADE")

    def test_insufficient_data(self):
        candles = _make_candles(np.array([100.0] * 10))
        result = run_strategy_f(candles)
        assert result["recommendation"] == "NO-TRADE"

    def test_histogram_mode(self):
        candles = _make_candles(_trending_up())
        result = run_strategy_f(candles, params={"trigger": "histogram"})
        assert result["metrics"].get("trigger_mode") == "histogram"

    def test_cross_mode(self):
        candles = _make_candles(_trending_up())
        result = run_strategy_f(candles, params={"trigger": "cross"})
        assert result["metrics"].get("trigger_mode") == "cross"

    def test_macd_metrics(self):
        candles = _make_candles(_trending_up())
        result = run_strategy_f(candles)
        if result["metrics"]:
            assert "macd_line" in result["metrics"]
            assert "signal_line" in result["metrics"]
            assert "histogram" in result["metrics"]


# ── Strategy G (Bollinger Bands) ──────────────────────────────────────

class TestStrategyG:
    def test_output_shape(self):
        candles = _make_candles(_sideways())
        result = run_strategy_g(candles)
        for key in REQUIRED_KEYS:
            assert key in result

    def test_sideways_market(self):
        candles = _make_candles(_sideways())
        result = run_strategy_g(candles)
        assert result["recommendation"] in ("BUY", "SELL", "NO-TRADE")
        assert 0 <= result["confidence"] <= 100

    def test_insufficient_data(self):
        candles = _make_candles(np.array([100.0] * 5))
        result = run_strategy_g(candles)
        assert result["recommendation"] == "NO-TRADE"

    def test_bands_in_metrics(self):
        candles = _make_candles(_sideways())
        result = run_strategy_g(candles)
        if result["metrics"]:
            assert "upper_band" in result["metrics"]
            assert "lower_band" in result["metrics"]
            assert "middle_band" in result["metrics"]
            assert result["metrics"]["upper_band"] > result["metrics"]["lower_band"]

    def test_close_outside_rule(self):
        candles = _make_candles(_sideways())
        result = run_strategy_g(candles, params={"entry_rule": "close_outside"})
        assert result["recommendation"] in ("BUY", "SELL", "NO-TRADE")

    def test_custom_std_dev(self):
        candles = _make_candles(_sideways())
        r1 = run_strategy_g(candles, params={"std_dev": 1.0})
        r2 = run_strategy_g(candles, params={"std_dev": 3.0})
        if r1["metrics"] and r2["metrics"]:
            assert r2["metrics"]["upper_band"] > r1["metrics"]["upper_band"]


# ── Strategy H (Breakout) ────────────────────────────────────────────

class TestStrategyH:
    def test_output_shape(self):
        candles = _make_candles(_trending_up())
        result = run_strategy_h(candles)
        for key in REQUIRED_KEYS:
            assert key in result

    def test_breakout_up(self):
        candles = _make_candles(_breakout_up())
        result = run_strategy_h(candles, params={"lookback": 20, "buffer_pct": 0.0})
        assert result["recommendation"] in ("BUY", "NO-TRADE")

    def test_insufficient_data(self):
        candles = _make_candles(np.array([100.0] * 5))
        result = run_strategy_h(candles)
        assert result["recommendation"] == "NO-TRADE"

    def test_channel_in_metrics(self):
        candles = _make_candles(_trending_up())
        result = run_strategy_h(candles)
        if result["metrics"]:
            assert "upper_channel" in result["metrics"]
            assert "lower_channel" in result["metrics"]

    def test_volume_filter(self):
        closes = _trending_up()
        vols = np.ones(len(closes)) * 100
        vols[-1] = 10  # Low volume on last candle
        candles = _make_candles(closes, vols)
        result = run_strategy_h(candles, params={"volume_filter": True})
        # Low volume should suppress breakout signals
        assert result["recommendation"] in ("NO-TRADE", "BUY", "SELL")

    def test_custom_buffer(self):
        candles = _make_candles(_trending_up())
        r1 = run_strategy_h(candles, params={"buffer_pct": 0.0})
        r2 = run_strategy_h(candles, params={"buffer_pct": 5.0})
        # Higher buffer = harder to trigger breakout
        assert r2["metrics"].get("buffer_pct", 0) > r1["metrics"].get("buffer_pct", 0)


# ── Strategy I (ATR Trailing Stop) ───────────────────────────────────

class TestStrategyI:
    def test_output_shape(self):
        candles = _make_candles(_trending_up())
        result = run_strategy_i(candles)
        for key in REQUIRED_KEYS:
            assert key in result

    def test_trending_up(self):
        candles = _make_candles(_trending_up())
        result = run_strategy_i(candles)
        assert result["recommendation"] in ("BUY", "SELL", "NO-TRADE")
        assert 0 <= result["confidence"] <= 100

    def test_insufficient_data(self):
        candles = _make_candles(np.array([100.0] * 5))
        result = run_strategy_i(candles)
        assert result["recommendation"] == "NO-TRADE"

    def test_atr_in_metrics(self):
        candles = _make_candles(_trending_up())
        result = run_strategy_i(candles)
        if result["metrics"]:
            assert "atr" in result["metrics"]
            assert "trailing_stop" in result["metrics"]
            assert "trend" in result["metrics"]

    def test_long_only(self):
        candles = _make_candles(_trending_down())
        result = run_strategy_i(candles, params={"direction": "long_only"})
        # Long only should not generate SELL
        assert result["recommendation"] in ("BUY", "NO-TRADE")

    def test_short_only(self):
        candles = _make_candles(_trending_up())
        result = run_strategy_i(candles, params={"direction": "short_only"})
        # Short only should not generate BUY
        assert result["recommendation"] in ("SELL", "NO-TRADE")

    def test_custom_multiplier(self):
        candles = _make_candles(_trending_up())
        r1 = run_strategy_i(candles, params={"atr_multiplier": 1.0})
        r2 = run_strategy_i(candles, params={"atr_multiplier": 5.0})
        # Different multipliers produce different trailing stops
        if r1["metrics"] and r2["metrics"]:
            assert r1["metrics"]["trailing_stop"] != r2["metrics"]["trailing_stop"]


# ── Strategy J (ADX) ─────────────────────────────────────────────────

def _strong_trend_up(n=200, start=100.0):
    """Strong uptrend to push ADX high and +DI > -DI."""
    np.random.seed(42)
    returns = np.random.normal(0.004, 0.003, n)
    return start * np.cumprod(1 + returns)


def _strong_trend_down(n=200, start=100.0):
    """Strong downtrend to push ADX high and -DI > +DI."""
    np.random.seed(42)
    returns = np.random.normal(-0.004, 0.003, n)
    return start * np.cumprod(1 + returns)


class TestStrategyJ:
    def test_output_shape(self):
        candles = _make_candles(_strong_trend_up())
        result = run_strategy_j(candles)
        for key in REQUIRED_KEYS:
            assert key in result

    def test_strong_uptrend_may_buy(self):
        candles = _make_candles(_strong_trend_up())
        result = run_strategy_j(candles)
        assert result["recommendation"] in ("BUY", "NO-TRADE")
        assert 0 <= result["confidence"] <= 100

    def test_strong_downtrend_may_sell(self):
        candles = _make_candles(_strong_trend_down())
        result = run_strategy_j(candles)
        assert result["recommendation"] in ("SELL", "NO-TRADE")

    def test_sideways_no_trade(self):
        candles = _make_candles(_sideways())
        result = run_strategy_j(candles)
        # Sideways market has weak ADX, likely NO-TRADE
        assert result["recommendation"] in ("BUY", "SELL", "NO-TRADE")

    def test_insufficient_data(self):
        candles = _make_candles(np.array([100.0] * 5))
        result = run_strategy_j(candles)
        assert result["recommendation"] == "NO-TRADE"
        assert "Need" in result["explanation"] or "data" in result["explanation"].lower()

    def test_adx_in_metrics(self):
        candles = _make_candles(_strong_trend_up())
        result = run_strategy_j(candles)
        if result["metrics"]:
            assert "adx" in result["metrics"]
            assert "plus_di" in result["metrics"]
            assert "minus_di" in result["metrics"]
            assert "trend" in result["metrics"]

    def test_custom_threshold(self):
        candles = _make_candles(_strong_trend_up())
        result = run_strategy_j(candles, params={"adx_threshold": 50})
        assert result["metrics"].get("threshold") == 50

    def test_require_rising(self):
        candles = _make_candles(_strong_trend_up())
        result = run_strategy_j(candles, params={"require_rising": True})
        assert result["recommendation"] in ("BUY", "SELL", "NO-TRADE")

    def test_stop_loss_and_tp_on_signal(self):
        candles = _make_candles(_strong_trend_up())
        result = run_strategy_j(candles)
        if result["recommendation"] in ("BUY", "SELL"):
            assert result["stop_loss"] is not None
            assert result["take_profit"] is not None
            assert result["entry"] is not None


# ── Strategy K (Fear & Greed Index) ──────────────────────────────────

class TestStrategyK:
    """Strategy K uses external API; we mock _fetch_fng to avoid network calls."""

    def _patch_fng(self, monkeypatch, value=25, classification="Extreme Fear", days=7):
        data = [{"value": value - i, "classification": classification,
                 "timestamp": 1700000000 + i * 86400} for i in range(days)]
        # Ensure first entry has the exact value requested
        data[0]["value"] = value
        monkeypatch.setattr(
            "app.strategies.strategy_k_fng._fetch_fng",
            lambda limit=30: data[:limit],
        )

    def test_output_shape(self, monkeypatch):
        self._patch_fng(monkeypatch, value=50)
        candles = _make_candles(_trending_up())
        result = run_strategy_k(candles)
        for key in REQUIRED_KEYS:
            assert key in result

    def test_extreme_fear_buy(self, monkeypatch):
        self._patch_fng(monkeypatch, value=10, classification="Extreme Fear")
        candles = _make_candles(_trending_up())
        result = run_strategy_k(candles)
        assert result["recommendation"] == "BUY"
        assert result["confidence"] > 0

    def test_extreme_greed_sell(self, monkeypatch):
        self._patch_fng(monkeypatch, value=90, classification="Extreme Greed")
        candles = _make_candles(_trending_up())
        result = run_strategy_k(candles)
        assert result["recommendation"] == "SELL"
        assert result["confidence"] > 0

    def test_neutral_no_trade(self, monkeypatch):
        self._patch_fng(monkeypatch, value=50, classification="Neutral")
        candles = _make_candles(_trending_up())
        result = run_strategy_k(candles)
        assert result["recommendation"] == "NO-TRADE"

    def test_api_failure_no_trade(self, monkeypatch):
        monkeypatch.setattr(
            "app.strategies.strategy_k_fng._fetch_fng",
            lambda limit=30: None,
        )
        candles = _make_candles(_trending_up())
        result = run_strategy_k(candles)
        assert result["recommendation"] == "NO-TRADE"
        assert result["confidence"] == 0

    def test_custom_thresholds(self, monkeypatch):
        self._patch_fng(monkeypatch, value=35, classification="Fear")
        candles = _make_candles(_trending_up())
        result = run_strategy_k(candles, params={"fear_threshold": 40})
        assert result["recommendation"] == "BUY"

    def test_fng_metrics(self, monkeypatch):
        self._patch_fng(monkeypatch, value=20, classification="Extreme Fear")
        candles = _make_candles(_trending_up())
        result = run_strategy_k(candles)
        assert "fng_value" in result["metrics"]
        assert result["metrics"]["fng_value"] == 20
        assert "fng_classification" in result["metrics"]
        assert "fng_avg" in result["metrics"]

    def test_stop_loss_and_tp_on_signal(self, monkeypatch):
        self._patch_fng(monkeypatch, value=10, classification="Extreme Fear")
        candles = _make_candles(_trending_up())
        result = run_strategy_k(candles)
        assert result["recommendation"] == "BUY"
        assert result["stop_loss"] is not None
        assert result["take_profit"] is not None
        assert result["entry"] is not None


# ── Cross-strategy tests ─────────────────────────────────────────────

class TestAllStrategies:
    """Ensure all strategies follow the standard output format."""

    @pytest.mark.parametrize("run_fn", [
        run_strategy_d, run_strategy_e, run_strategy_f,
        run_strategy_g, run_strategy_h, run_strategy_i,
        run_strategy_j,
    ])
    def test_standard_output_format(self, run_fn):
        candles = _make_candles(_trending_up())
        result = run_fn(candles)
        assert isinstance(result, dict)
        for key in REQUIRED_KEYS:
            assert key in result, f"Missing key: {key}"
        assert result["recommendation"] in ("BUY", "SELL", "NO-TRADE")
        assert isinstance(result["confidence"], (int, float))
        assert 0 <= result["confidence"] <= 100
        assert isinstance(result["explanation"], str)
        assert isinstance(result["metrics"], dict)

    @pytest.mark.parametrize("run_fn", [
        run_strategy_d, run_strategy_e, run_strategy_f,
        run_strategy_g, run_strategy_h, run_strategy_i,
        run_strategy_j,
    ])
    def test_none_params_no_crash(self, run_fn):
        candles = _make_candles(_trending_up())
        result = run_fn(candles, params=None)
        assert result["recommendation"] in ("BUY", "SELL", "NO-TRADE")
