"""Unit tests for Strategy M — Claude AI (Pure Market Analysis).

All Claude API calls are mocked; no real API key is needed.
"""

import json
import numpy as np
import pytest
from unittest.mock import patch, MagicMock

from app.strategies.strategy_m_claude_ai import (
    run_strategy_m,
    _prepare_price_features,
    _prepare_depth_features,
    _prepare_24hr_features,
    _parse_claude_response,
    _format_prompt,
    DEFAULTS,
)


# ── Helpers ────────────────────────────────────────────────────────────

def _make_candles(closes: np.ndarray, volumes: np.ndarray | None = None) -> np.ndarray:
    n = len(closes)
    times = np.arange(n, dtype=float) * 60000
    opens = closes * 0.999
    highs = closes * 1.002
    lows = closes * 0.998
    vols = volumes if volumes is not None else np.ones(n) * 100
    return np.column_stack([times, opens, highs, lows, closes, vols])


def _trending_up(n=60, start=100.0, drift=0.002):
    np.random.seed(42)
    returns = np.random.normal(drift, 0.005, n)
    return start * np.cumprod(1 + returns)


def _sample_depth():
    return {
        "bids": [[str(100 - i * 0.1), str(1.0 + i * 0.5)] for i in range(10)],
        "asks": [[str(100.1 + i * 0.1), str(0.8 + i * 0.3)] for i in range(10)],
    }


def _sample_ticker():
    return {
        "priceChangePercent": "2.35",
        "highPrice": "105.00",
        "lowPrice": "97.50",
        "volume": "15000.0",
        "quoteVolume": "1500000.0",
        "count": 42000,
        "weightedAvgPrice": "101.25",
    }


def _claude_response(direction=45, intensity=70, confidence=80, reasoning="test"):
    return json.dumps({
        "direction": direction,
        "intensity": intensity,
        "confidence": confidence,
        "reasoning": reasoning,
    })


# ── Feature extraction tests ──────────────────────────────────────────

class TestPriceFeatures:
    def test_basic_features(self):
        closes = _trending_up(50)
        candles = _make_candles(closes)
        feats = _prepare_price_features(candles, 50)

        assert "recent_closes" in feats
        assert "log_returns" in feats
        assert "acceleration" in feats
        assert "volume_ratio" in feats
        assert "hour_utc" in feats
        assert "day_of_week" in feats
        assert "mean_ret" in feats
        assert "std_ret" in feats
        assert "skew" in feats
        assert "kurt" in feats

    def test_lookback_limits(self):
        closes = _trending_up(100)
        candles = _make_candles(closes)
        feats = _prepare_price_features(candles, 30)
        assert len(feats["recent_closes"]) == 30

    def test_volume_ratio(self):
        closes = np.ones(30) * 100
        volumes = np.ones(30) * 50
        volumes[-1] = 100  # double average
        candles = _make_candles(closes, volumes)
        feats = _prepare_price_features(candles, 30)
        # The SMA-20 window includes the last candle itself, so ratio
        # is 100 / avg(29 × 50 + 1 × 100 over 20) ≈ 1.905
        assert feats["volume_ratio"] > 1.8


class TestDepthFeatures:
    def test_valid_depth(self):
        depth = _sample_depth()
        feats = _prepare_depth_features(depth)
        assert feats is not None
        assert "spread_pct" in feats
        assert "imbalance" in feats
        assert "bid_total" in feats
        assert "ask_total" in feats

    def test_none_depth(self):
        assert _prepare_depth_features(None) is None

    def test_empty_depth(self):
        assert _prepare_depth_features({"bids": [], "asks": []}) is None


class TestTicker24hrFeatures:
    def test_valid_ticker(self):
        feats = _prepare_24hr_features(_sample_ticker())
        assert feats is not None
        assert feats["price_change_pct"] == pytest.approx(2.35)
        assert feats["trade_count"] == 42000
        assert feats["vwap"] == pytest.approx(101.25)

    def test_none_ticker(self):
        assert _prepare_24hr_features(None) is None


# ── Response parsing tests ────────────────────────────────────────────

class TestParsing:
    def test_valid_json(self):
        text = '{"direction": 50, "intensity": 60, "confidence": 70, "reasoning": "ok"}'
        result = _parse_claude_response(text)
        assert result is not None
        assert result["direction"] == 50
        assert result["intensity"] == 60
        assert result["confidence"] == 70

    def test_markdown_fenced_json(self):
        text = 'Here is my analysis:\n```json\n{"direction": -30, "intensity": 40, "confidence": 55, "reasoning": "bearish"}\n```'
        result = _parse_claude_response(text)
        assert result is not None
        assert result["direction"] == -30

    def test_clamping_high(self):
        text = '{"direction": 200, "intensity": 150, "confidence": 999, "reasoning": ""}'
        result = _parse_claude_response(text)
        assert result["direction"] == 100
        assert result["intensity"] == 100
        assert result["confidence"] == 100

    def test_clamping_low(self):
        text = '{"direction": -200, "intensity": -10, "confidence": -5, "reasoning": ""}'
        result = _parse_claude_response(text)
        assert result["direction"] == -100
        assert result["intensity"] == 0
        assert result["confidence"] == 0

    def test_invalid_json(self):
        assert _parse_claude_response("not json at all") is None

    def test_missing_fields(self):
        assert _parse_claude_response('{"direction": 10}') is None

    def test_reasoning_default(self):
        text = '{"direction": 0, "intensity": 0, "confidence": 0}'
        result = _parse_claude_response(text)
        assert result is not None
        assert result["reasoning"] == ""


# ── Prompt construction tests ─────────────────────────────────────────

class TestPrompt:
    def test_basic_prompt(self):
        closes = _trending_up(50)
        candles = _make_candles(closes)
        feats = _prepare_price_features(candles, 50)
        prompt = _format_prompt("BTCUSDT", feats, None, None, [])
        assert "BTCUSDT" in prompt
        assert "DATOS DE PRECIO" in prompt
        assert "CONTEXTO TEMPORAL" in prompt
        # No depth or 24hr sections
        assert "LIBRO DE ÓRDENES" not in prompt
        assert "ESTADÍSTICAS 24H" not in prompt

    def test_with_depth_and_ticker(self):
        closes = _trending_up(50)
        candles = _make_candles(closes)
        feats = _prepare_price_features(candles, 50)
        depth_f = _prepare_depth_features(_sample_depth())
        ticker_f = _prepare_24hr_features(_sample_ticker())
        prompt = _format_prompt("ETHUSDT", feats, depth_f, ticker_f, [])
        assert "LIBRO DE ÓRDENES" in prompt
        assert "ESTADÍSTICAS 24H" in prompt

    def test_with_past_outcomes(self):
        closes = _trending_up(50)
        candles = _make_candles(closes)
        feats = _prepare_price_features(candles, 50)
        outcomes = [
            {"timestamp": "2025-01-01T12:00", "direction": 30, "intensity": 50,
             "confidence": 60, "outcome": "CORRECT", "return_pct": 1.5, "horizon": "1h"},
        ]
        prompt = _format_prompt("BTCUSDT", feats, None, None, outcomes)
        assert "PREDICCIONES PASADAS" in prompt
        assert "CORRECT" in prompt


# ── Full pipeline tests (mocked Claude API) ───────────────────────────

class TestRunStrategyM:
    def _params(self, **extra):
        return {
            "_market_data": {
                "anthropic_api_key": "test-key-123",
                "symbol": "BTCUSDT",
                "depth": _sample_depth(),
                "ticker_24hr": _sample_ticker(),
                "past_outcomes": [],
            },
            **extra,
        }

    @patch("app.strategies.strategy_m_claude_ai._call_claude_sync")
    def test_buy_signal(self, mock_call):
        mock_call.return_value = _claude_response(direction=55, intensity=70, confidence=80)
        closes = _trending_up(60)
        candles = _make_candles(closes)
        result = run_strategy_m(candles, params=self._params())

        assert result["recommendation"] == "BUY"
        assert result["confidence"] == 55.0
        assert result["force"] == 0.7
        assert result["entry"] is not None
        assert result["stop_loss"] is not None
        assert result["take_profit"] is not None
        assert result["metrics"]["direction"] == 55
        assert result["metrics"]["intensity"] == 70
        assert result["metrics"]["ai_confidence"] == 80

    @patch("app.strategies.strategy_m_claude_ai._call_claude_sync")
    def test_sell_signal(self, mock_call):
        mock_call.return_value = _claude_response(direction=-60, intensity=80, confidence=75)
        closes = _trending_up(60)
        candles = _make_candles(closes)
        result = run_strategy_m(candles, params=self._params())

        assert result["recommendation"] == "SELL"
        assert result["confidence"] == 60.0
        assert result["force"] == 0.8

    @patch("app.strategies.strategy_m_claude_ai._call_claude_sync")
    def test_deadzone_produces_no_trade(self, mock_call):
        mock_call.return_value = _claude_response(direction=5, intensity=30, confidence=50)
        closes = _trending_up(60)
        candles = _make_candles(closes)
        result = run_strategy_m(candles, params=self._params())

        assert result["recommendation"] == "NO-TRADE"
        assert result["confidence"] == 0.0

    @patch("app.strategies.strategy_m_claude_ai._call_claude_sync")
    def test_custom_deadzone(self, mock_call):
        mock_call.return_value = _claude_response(direction=25, intensity=50, confidence=60)
        closes = _trending_up(60)
        candles = _make_candles(closes)
        result = run_strategy_m(candles, params=self._params(direction_deadzone=30))

        assert result["recommendation"] == "NO-TRADE"

    def test_no_api_key(self):
        closes = _trending_up(60)
        candles = _make_candles(closes)
        params = {"_market_data": {"symbol": "BTCUSDT"}}
        result = run_strategy_m(candles, params=params)

        assert result["recommendation"] == "NO-TRADE"
        assert "API key" in result["explanation"]

    def test_insufficient_data(self):
        candles = _make_candles(np.array([100.0] * 5))
        result = run_strategy_m(candles, params=self._params())

        assert result["recommendation"] == "NO-TRADE"
        assert "insuficientes" in result["explanation"]

    @patch("app.strategies.strategy_m_claude_ai._call_claude_sync")
    def test_api_error(self, mock_call):
        mock_call.side_effect = Exception("API timeout")
        closes = _trending_up(60)
        candles = _make_candles(closes)
        result = run_strategy_m(candles, params=self._params())

        assert result["recommendation"] == "NO-TRADE"
        assert "Error" in result["explanation"]

    @patch("app.strategies.strategy_m_claude_ai._call_claude_sync")
    def test_unparseable_response(self, mock_call):
        mock_call.return_value = "I cannot analyze this data properly."
        closes = _trending_up(60)
        candles = _make_candles(closes)
        result = run_strategy_m(candles, params=self._params())

        assert result["recommendation"] == "NO-TRADE"
        assert "interpretar" in result["explanation"]

    @patch("app.strategies.strategy_m_claude_ai._call_claude_sync")
    def test_metrics_populated(self, mock_call):
        mock_call.return_value = _claude_response(
            direction=40, intensity=65, confidence=85, reasoning="strong momentum"
        )
        closes = _trending_up(60)
        candles = _make_candles(closes)
        result = run_strategy_m(candles, params=self._params())

        m = result["metrics"]
        assert m["direction"] == 40
        assert m["intensity"] == 65
        assert m["ai_confidence"] == 85
        assert m["depth_available"] is True
        assert m["ticker_24hr_available"] is True
        assert "reasoning" in m

    @patch("app.strategies.strategy_m_claude_ai._call_claude_sync")
    def test_without_depth_and_ticker(self, mock_call):
        mock_call.return_value = _claude_response(direction=30, intensity=50, confidence=60)
        closes = _trending_up(60)
        candles = _make_candles(closes)
        params = {
            "_market_data": {
                "anthropic_api_key": "test-key",
                "symbol": "BTCUSDT",
                "depth": None,
                "ticker_24hr": None,
                "past_outcomes": [],
            },
            "include_depth": False,
            "include_24hr": False,
        }
        result = run_strategy_m(candles, params=params)

        assert result["recommendation"] == "BUY"
        assert result["metrics"]["depth_available"] is False
        assert result["metrics"]["ticker_24hr_available"] is False
