"""Tests for the approval gating and trade execution safety logic."""

import asyncio
import datetime
from unittest.mock import AsyncMock, patch

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.database import Base
from app.models import (
    User, Role, Recommendation, Approval, ApprovalStatus,
)
from app.services.approval_service import approve, reject
from app.services.trade_service import execute_trade
from app.auth.jwt_handler import hash_password
from app.auth.encryption import encrypt


@pytest.fixture()
def db():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(bind=engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    yield session
    session.close()


@pytest.fixture()
def admin_user(db):
    user = User(
        username="admin",
        hashed_password=hash_password("pass"),
        role=Role.admin,
        binance_api_key_enc=encrypt("test_api_key"),
        binance_api_secret_enc=encrypt("test_api_secret"),
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


@pytest.fixture()
def admin_no_keys(db):
    user = User(
        username="admin_nokeys",
        hashed_password=hash_password("pass"),
        role=Role.admin,
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


@pytest.fixture()
def regular_user(db):
    user = User(username="trader", hashed_password=hash_password("pass"), role=Role.user)
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


@pytest.fixture()
def pending_recommendation(db, regular_user):
    rec = Recommendation(
        user_id=regular_user.id,
        symbol="BTCUSDT",
        interval="1h",
        period_start="2025-01-01",
        period_end="2025-01-31",
        strategy="A",
        recommendation="BUY",
        confidence=75.0,
        entry_price=42000.0,
        stop_loss=41000.0,
        take_profit=44000.0,
        explanation="Test recommendation",
    )
    db.add(rec)
    db.flush()
    approval = Approval(recommendation_id=rec.id, status=ApprovalStatus.pending)
    db.add(approval)
    db.commit()
    db.refresh(approval)
    return rec, approval


def _run(coro):
    """Run an async coroutine synchronously."""
    return asyncio.get_event_loop().run_until_complete(coro)


MOCK_MARKET_RESULT = {
    "symbol": "BTCUSDT",
    "orderId": 12345,
    "status": "FILLED",
    "fills": [{"price": "42000.00", "qty": "0.5", "commission": "0.001"}],
}

MOCK_OCO_RESULT = {
    "orderListId": 99,
    "listStatusType": "EXEC_STARTED",
    "orders": [
        {"symbol": "BTCUSDT", "orderId": 100, "type": "LIMIT_MAKER"},
        {"symbol": "BTCUSDT", "orderId": 101, "type": "STOP_LOSS_LIMIT"},
    ],
}


class TestApprovalService:
    def test_approve_sets_status(self, db, admin_user, pending_recommendation):
        _, approval = pending_recommendation
        result = approve(approval.id, admin_user.id, "Looks good", db)
        assert result.status == ApprovalStatus.approved
        assert result.reviewed_by == admin_user.id

    def test_reject_sets_status(self, db, admin_user, pending_recommendation):
        _, approval = pending_recommendation
        result = reject(approval.id, admin_user.id, "Too risky", db)
        assert result.status == ApprovalStatus.rejected

    def test_double_approve_fails(self, db, admin_user, pending_recommendation):
        _, approval = pending_recommendation
        approve(approval.id, admin_user.id, "ok", db)
        with pytest.raises(ValueError, match="already"):
            approve(approval.id, admin_user.id, "again", db)

    def test_nonexistent_approval_fails(self, db, admin_user):
        with pytest.raises(ValueError, match="no encontrado"):
            approve(9999, admin_user.id, "nope", db)


class TestTradeExecution:
    def test_execute_requires_approved(self, db, admin_user, pending_recommendation):
        _, approval = pending_recommendation
        with pytest.raises(ValueError, match="approved"):
            _run(execute_trade(approval.id, admin_user.id, 0.5, "market", db))

    def test_execute_requires_api_keys(self, db, admin_no_keys, pending_recommendation):
        _, approval = pending_recommendation
        approve(approval.id, admin_no_keys.id, "go", db)
        with pytest.raises(ValueError, match="API keys"):
            _run(execute_trade(approval.id, admin_no_keys.id, 0.5, "market", db))

    @patch("app.services.trade_service.place_market_order", new_callable=AsyncMock)
    def test_execute_market_order(self, mock_market, db, admin_user, pending_recommendation):
        mock_market.return_value = MOCK_MARKET_RESULT
        _, approval = pending_recommendation
        approve(approval.id, admin_user.id, "go", db)
        trade = _run(execute_trade(approval.id, admin_user.id, 0.5, "market", db))
        assert trade.status == "filled"
        assert trade.is_live is True
        assert trade.symbol == "BTCUSDT"
        assert trade.side == "BUY"
        assert trade.quantity == 0.5
        mock_market.assert_called_once()

    @patch("app.services.trade_service.execute_market_then_oco", new_callable=AsyncMock)
    def test_execute_oco_order(self, mock_oco, db, admin_user, pending_recommendation):
        mock_oco.return_value = {
            "market_order": MOCK_MARKET_RESULT,
            "oco_order": MOCK_OCO_RESULT,
            "filled_price": 42000.0,
            "take_profit": 44000.0,
            "stop_loss": 41000.0,
        }
        _, approval = pending_recommendation
        approve(approval.id, admin_user.id, "go", db)
        trade = _run(execute_trade(approval.id, admin_user.id, 0.1, "oco", db))
        assert trade.status == "filled"
        assert trade.order_type == "oco"
        assert trade.stop_loss == 41000.0
        assert trade.take_profit == 44000.0
        assert trade.is_live is True
        mock_oco.assert_called_once()

    @patch("app.services.trade_service.place_market_order", new_callable=AsyncMock)
    def test_failed_order_records_failure(self, mock_market, db, admin_user, pending_recommendation):
        mock_market.side_effect = RuntimeError("Binance API timeout")
        _, approval = pending_recommendation
        approve(approval.id, admin_user.id, "go", db)
        trade = _run(execute_trade(approval.id, admin_user.id, 0.5, "market", db))
        assert trade.status == "failed"
        assert "error" in trade.result

    def test_no_trade_recommendation_blocked(self, db, admin_user):
        rec = Recommendation(
            user_id=admin_user.id,
            symbol="ETHUSDT",
            interval="1d",
            period_start="2025-01-01",
            period_end="2025-01-31",
            strategy="B",
            recommendation="NO-TRADE",
            confidence=10.0,
        )
        db.add(rec)
        db.flush()
        approval = Approval(recommendation_id=rec.id, status=ApprovalStatus.pending)
        db.add(approval)
        db.commit()
        approve(approval.id, admin_user.id, "testing", db)
        with pytest.raises(ValueError, match="NO-TRADE"):
            _run(execute_trade(approval.id, admin_user.id, 1.0, "market", db))
