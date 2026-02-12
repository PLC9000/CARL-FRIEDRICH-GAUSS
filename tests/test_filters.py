"""Tests for binance_filters: rounding, validation, and automatic adjustment."""

import asyncio
from decimal import Decimal
from unittest.mock import AsyncMock, patch

import pytest

from app.services.binance_filters import (
    clear_cache,
    round_step,
    round_tick,
    step_decimals,
    validate_and_adjust_order,
)


# ── Helper ───────────────────────────────────────────────────────────────────

def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


# Realistic filters for SOLUSDT-style symbol
MOCK_FILTERS = {
    "step_size": "0.01000000",
    "min_qty": "0.01000000",
    "max_qty": "9000000.00000000",
    "tick_size": "0.01000000",
    "min_price": "0.01000000",
    "max_price": "10000.00000000",
    "min_notional": "5.00000000",
    "apply_to_market": True,
    "avg_price_mins": 5,
}

# Realistic filters for BTCUSDT-style symbol (small stepSize, large prices)
MOCK_FILTERS_BTC = {
    "step_size": "0.00001000",
    "min_qty": "0.00001000",
    "max_qty": "9000.00000000",
    "tick_size": "0.01000000",
    "min_price": "0.01000000",
    "max_price": "1000000.00000000",
    "min_notional": "5.00000000",
    "apply_to_market": True,
    "avg_price_mins": 5,
}


@pytest.fixture(autouse=True)
def _clear():
    clear_cache()
    yield
    clear_cache()


# ── round_step tests ─────────────────────────────────────────────────────────

class TestRoundStep:
    def test_rounds_down(self):
        assert round_step(0.12345, "0.001") == Decimal("0.123")

    def test_rounds_down_not_up(self):
        assert round_step(1.999, "0.01") == Decimal("1.99")

    def test_exact_value_unchanged(self):
        assert round_step(1.5, "0.5") == Decimal("1.5")

    def test_integer_step(self):
        assert round_step(7.8, "1") == Decimal("7")

    def test_small_step(self):
        assert round_step(0.000456, "0.00001") == Decimal("0.00045")

    def test_zero_step(self):
        result = round_step(1.234, "0")
        assert result == Decimal("1.234")

    def test_large_value(self):
        assert round_step(42000.789, "0.01") == Decimal("42000.78")

    def test_round_half_up_mode(self):
        result = round_step(0.125, "0.01", mode="up")
        assert result == Decimal("0.13")


# ── round_tick tests ─────────────────────────────────────────────────────────

class TestRoundTick:
    def test_rounds_to_tick(self):
        assert round_tick(42000.123, "0.01") == Decimal("42000.12")

    def test_half_up(self):
        assert round_tick(42000.125, "0.01") == Decimal("42000.13")

    def test_exact_value(self):
        assert round_tick(100.50, "0.01") == Decimal("100.50")

    def test_large_tick(self):
        assert round_tick(42005, "10") == Decimal("42010")


# ── step_decimals tests ─────────────────────────────────────────────────────

class TestStepDecimals:
    def test_normal(self):
        assert step_decimals("0.01000000") == 2

    def test_five_decimals(self):
        assert step_decimals("0.00001000") == 5

    def test_integer(self):
        assert step_decimals("1") == 0

    def test_one_decimal(self):
        assert step_decimals("0.10000000") == 1


# ── validate_and_adjust_order tests ──────────────────────────────────────────

class TestValidateMarketBuy:
    """MARKET BUY orders."""

    @patch("app.services.binance_filters._get_current_price", new_callable=AsyncMock)
    @patch("app.services.binance_filters.get_symbol_filters", new_callable=AsyncMock)
    def test_normal_order_no_adjustments(self, mock_filters, mock_price):
        mock_filters.return_value = MOCK_FILTERS
        mock_price.return_value = 150.0  # price=150, qty=1 → notional=150 > 5

        result = _run(validate_and_adjust_order(
            symbol="SOLUSDT", side="BUY", order_type="MARKET",
            quantity=1.0, price=None,
        ))
        assert result["quantity"] == "1.00"
        assert result["adjustments"] == []

    @patch("app.services.binance_filters._get_current_price", new_callable=AsyncMock)
    @patch("app.services.binance_filters.get_symbol_filters", new_callable=AsyncMock)
    def test_rounds_quantity_down(self, mock_filters, mock_price):
        mock_filters.return_value = MOCK_FILTERS
        mock_price.return_value = 150.0

        result = _run(validate_and_adjust_order(
            symbol="SOLUSDT", side="BUY", order_type="MARKET",
            quantity=1.12789, price=None,
        ))
        assert result["quantity"] == "1.12"
        assert any("LOT_SIZE" in a for a in result["adjustments"])

    @patch("app.services.binance_filters._get_current_price", new_callable=AsyncMock)
    @patch("app.services.binance_filters.get_symbol_filters", new_callable=AsyncMock)
    def test_below_notional_switches_to_quote_qty(self, mock_filters, mock_price):
        """When qty*price < minNotional, BUY MARKET uses quoteOrderQty."""
        mock_filters.return_value = MOCK_FILTERS
        mock_price.return_value = 150.0  # qty=0.01 → notional=1.5 < 5

        result = _run(validate_and_adjust_order(
            symbol="SOLUSDT", side="BUY", order_type="MARKET",
            quantity=0.01, price=None,
        ))
        assert "quoteOrderQty" in result
        assert "quantity" not in result
        quote_val = float(result["quoteOrderQty"])
        assert quote_val >= 5.0 * 1.02  # minNotional × 1.02
        assert any("NOTIONAL" in a for a in result["adjustments"])


class TestValidateMarketSell:
    """MARKET SELL orders."""

    @patch("app.services.binance_filters._get_current_price", new_callable=AsyncMock)
    @patch("app.services.binance_filters.get_symbol_filters", new_callable=AsyncMock)
    def test_sell_below_notional_raises(self, mock_filters, mock_price):
        """SELL below minNotional should raise NO-TRADE."""
        mock_filters.return_value = MOCK_FILTERS
        mock_price.return_value = 150.0  # qty=0.01 → notional=1.5 < 5

        with pytest.raises(ValueError, match="NO-TRADE"):
            _run(validate_and_adjust_order(
                symbol="SOLUSDT", side="SELL", order_type="MARKET",
                quantity=0.01, price=None,
            ))

    @patch("app.services.binance_filters._get_current_price", new_callable=AsyncMock)
    @patch("app.services.binance_filters.get_symbol_filters", new_callable=AsyncMock)
    def test_sell_normal(self, mock_filters, mock_price):
        mock_filters.return_value = MOCK_FILTERS
        mock_price.return_value = 150.0

        result = _run(validate_and_adjust_order(
            symbol="SOLUSDT", side="SELL", order_type="MARKET",
            quantity=1.0, price=None,
        ))
        assert result["quantity"] == "1.00"


class TestValidateLimitOrder:
    """LIMIT orders (used in OCO)."""

    @patch("app.services.binance_filters.get_symbol_filters", new_callable=AsyncMock)
    def test_price_rounded_to_tick(self, mock_filters):
        mock_filters.return_value = MOCK_FILTERS

        result = _run(validate_and_adjust_order(
            symbol="SOLUSDT", side="BUY", order_type="LIMIT",
            quantity=1.0, price=155.1234,
        ))
        assert result["price"] == "155.12"
        assert any("PRICE_FILTER" in a for a in result["adjustments"])

    @patch("app.services.binance_filters.get_symbol_filters", new_callable=AsyncMock)
    def test_limit_buy_below_notional_bumps_qty(self, mock_filters):
        """LIMIT BUY below minNotional increases qty to meet minimum."""
        mock_filters.return_value = MOCK_FILTERS

        result = _run(validate_and_adjust_order(
            symbol="SOLUSDT", side="BUY", order_type="LIMIT",
            quantity=0.01, price=150.0, balance_free=100.0,
        ))
        # Should have bumped qty: 5/150 = 0.033... → rounded + 1 step
        qty = Decimal(result["quantity"])
        assert qty * 150 >= 5  # notional met
        assert any("NOTIONAL" in a for a in result["adjustments"])

    @patch("app.services.binance_filters.get_symbol_filters", new_callable=AsyncMock)
    def test_limit_buy_insufficient_balance(self, mock_filters):
        mock_filters.return_value = MOCK_FILTERS

        with pytest.raises(ValueError, match="Balance insuficiente"):
            _run(validate_and_adjust_order(
                symbol="SOLUSDT", side="BUY", order_type="LIMIT",
                quantity=0.01, price=150.0, balance_free=0.50,
            ))

    @patch("app.services.binance_filters.get_symbol_filters", new_callable=AsyncMock)
    def test_price_above_max_raises(self, mock_filters):
        mock_filters.return_value = MOCK_FILTERS

        with pytest.raises(ValueError, match="máximo permitido"):
            _run(validate_and_adjust_order(
                symbol="SOLUSDT", side="BUY", order_type="LIMIT",
                quantity=1.0, price=99999.0,
            ))


class TestValidateBalanceCap:
    """Balance capping for BUY orders."""

    @patch("app.services.binance_filters._get_current_price", new_callable=AsyncMock)
    @patch("app.services.binance_filters.get_symbol_filters", new_callable=AsyncMock)
    def test_market_buy_caps_at_98pct(self, mock_filters, mock_price):
        """MARKET BUY should use at most 98% of balance."""
        mock_filters.return_value = MOCK_FILTERS
        mock_price.return_value = 100.0  # balance=20 → max=19.6/100=0.19

        result = _run(validate_and_adjust_order(
            symbol="SOLUSDT", side="BUY", order_type="MARKET",
            quantity=0.25, price=None, balance_free=20.0,
        ))
        qty = Decimal(result["quantity"])
        # 20 * 0.98 = 19.6 → 19.6/100 = 0.196 → floor(0.19)
        assert qty <= Decimal("0.20")
        assert any("BALANCE" in a for a in result["adjustments"])

    @patch("app.services.binance_filters.get_symbol_filters", new_callable=AsyncMock)
    def test_limit_buy_caps_at_99pct(self, mock_filters):
        mock_filters.return_value = MOCK_FILTERS

        result = _run(validate_and_adjust_order(
            symbol="SOLUSDT", side="BUY", order_type="LIMIT",
            quantity=10.0, price=100.0, balance_free=500.0,
        ))
        qty = Decimal(result["quantity"])
        # 500 * 0.99 = 495 → 495/100 = 4.95 → floor(4.95)
        assert qty <= Decimal("4.95")
        assert any("BALANCE" in a for a in result["adjustments"])


class TestValidateLotSizeLimits:
    """LOT_SIZE min/max enforcement."""

    @patch("app.services.binance_filters._get_current_price", new_callable=AsyncMock)
    @patch("app.services.binance_filters.get_symbol_filters", new_callable=AsyncMock)
    def test_below_min_qty_market_buy_bumps(self, mock_filters, mock_price):
        mock_filters.return_value = MOCK_FILTERS
        mock_price.return_value = 150.0

        result = _run(validate_and_adjust_order(
            symbol="SOLUSDT", side="BUY", order_type="MARKET",
            quantity=0.001, price=None,
        ))
        # 0.001 rounds down to 0.00 which is < minQty 0.01 → bumped to 0.01
        # But 0.01 * 150 = 1.5 < 5 minNotional → switches to quoteOrderQty
        assert "quoteOrderQty" in result

    @patch("app.services.binance_filters._get_current_price", new_callable=AsyncMock)
    @patch("app.services.binance_filters.get_symbol_filters", new_callable=AsyncMock)
    def test_above_max_qty_raises(self, mock_filters, mock_price):
        mock_filters.return_value = MOCK_FILTERS
        mock_price.return_value = 150.0

        with pytest.raises(ValueError, match="máximo permitido"):
            _run(validate_and_adjust_order(
                symbol="SOLUSDT", side="BUY", order_type="MARKET",
                quantity=99999999.0, price=None,
            ))


class TestBTCFilters:
    """Tests with BTC-style filters (5 decimal stepSize)."""

    @patch("app.services.binance_filters._get_current_price", new_callable=AsyncMock)
    @patch("app.services.binance_filters.get_symbol_filters", new_callable=AsyncMock)
    def test_btc_rounding(self, mock_filters, mock_price):
        mock_filters.return_value = MOCK_FILTERS_BTC
        mock_price.return_value = 70000.0

        result = _run(validate_and_adjust_order(
            symbol="BTCUSDT", side="BUY", order_type="MARKET",
            quantity=0.123456789, price=None,
        ))
        assert result["quantity"] == "0.12345"
        assert any("LOT_SIZE" in a for a in result["adjustments"])
