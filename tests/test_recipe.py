"""Tests for recipe CRUD, score normalization, and evaluation engine logic."""

import datetime

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.database import Base
from app.models import (
    User, Role, Recipe, RecipeStatus, RecipeEvaluation,
    Recommendation, Approval, ApprovalStatus,
)
from app.auth.jwt_handler import hash_password
from app.models import StrategyConfig
from app.services.recipe_service import (
    create_recipe, update_recipe, toggle_recipe_status,
    delete_recipe, list_recipes, get_recipe_evaluations,
)
from app.services.evaluation_engine import (
    normalize_score, compute_weighted_score, determine_signal,
)


@pytest.fixture()
def db():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(bind=engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    # Seed all strategies as enabled for tests
    for key in ("A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K"):
        session.add(StrategyConfig(strategy_key=key, enabled=True))
    session.commit()
    yield session
    session.close()


@pytest.fixture()
def user(db):
    u = User(
        username="trader",
        hashed_password=hash_password("pass123"),
        role=Role.user,
    )
    db.add(u)
    db.commit()
    db.refresh(u)
    return u


# ── Score normalization tests ──────────────────────────────────────────


class TestNormalizeScore:
    def test_buy_positive(self):
        assert normalize_score("BUY", 72.0) == pytest.approx(0.72)

    def test_sell_negative(self):
        assert normalize_score("SELL", 85.0) == pytest.approx(-0.85)

    def test_no_trade_zero(self):
        assert normalize_score("NO-TRADE", 50.0) == 0.0

    def test_buy_100(self):
        assert normalize_score("BUY", 100.0) == pytest.approx(1.0)

    def test_sell_0(self):
        assert normalize_score("SELL", 0.0) == pytest.approx(0.0)


class TestComputeWeightedScore:
    def test_single_strategy(self):
        results = [{"score": 0.72, "weight": 1.0}]
        assert compute_weighted_score(results) == pytest.approx(0.72)

    def test_two_strategies_equal_weight(self):
        results = [
            {"score": 0.8, "weight": 0.5},
            {"score": -0.4, "weight": 0.5},
        ]
        assert compute_weighted_score(results) == pytest.approx(0.2)

    def test_weighted_average(self):
        results = [
            {"score": 0.9, "weight": 0.7},
            {"score": -0.3, "weight": 0.3},
        ]
        expected = (0.9 * 0.7 + (-0.3) * 0.3) / (0.7 + 0.3)
        assert compute_weighted_score(results) == pytest.approx(expected)

    def test_clamps_to_minus_one(self):
        results = [{"score": -1.0, "weight": 1.0}]
        assert compute_weighted_score(results) >= -1.0

    def test_clamps_to_plus_one(self):
        results = [{"score": 1.0, "weight": 1.0}]
        assert compute_weighted_score(results) <= 1.0

    def test_zero_weight_returns_zero(self):
        results = [{"score": 0.5, "weight": 0.0}]
        assert compute_weighted_score(results) == 0.0


class TestDetermineSignal:
    def test_buy_above_threshold(self):
        assert determine_signal(0.65, 0.5, 0.5) == "BUY"

    def test_sell_below_threshold(self):
        assert determine_signal(-0.7, 0.5, 0.5) == "SELL"

    def test_hold_in_neutral_zone(self):
        assert determine_signal(0.3, 0.5, 0.5) == "HOLD"

    def test_hold_slightly_negative(self):
        assert determine_signal(-0.3, 0.5, 0.5) == "HOLD"

    def test_exact_threshold_buy(self):
        assert determine_signal(0.5, 0.5, 0.5) == "BUY"

    def test_exact_threshold_sell(self):
        assert determine_signal(-0.5, 0.5, 0.5) == "SELL"

    def test_asymmetric_thresholds(self):
        assert determine_signal(0.3, 0.3, 0.8) == "BUY"
        assert determine_signal(-0.5, 0.3, 0.8) == "HOLD"
        assert determine_signal(-0.8, 0.3, 0.8) == "SELL"


# ── Recipe CRUD tests ─────────────────────────────────────────────────


class TestRecipeCRUD:
    def test_create_recipe(self, db, user):
        recipe = create_recipe(
            user_id=user.id,
            name="BTC Trend",
            symbol="BTCUSDT",
            strategies=[{"strategy": "A", "weight": 100}],
            interval="1h",
            lookback_days=7,
            buy_threshold=0.5,
            sell_threshold=0.5,
            auto_threshold=0.0,
            max_order_pct=5.0,
            stop_loss_pct=2.0,
            take_profit_pct=4.0,
            strategy_params=None,
            db=db,
        )
        assert recipe.id is not None
        assert recipe.name == "BTC Trend"
        assert recipe.symbol == "BTCUSDT"
        assert recipe.status == RecipeStatus.inactive
        assert len(recipe.strategies) == 1
        assert recipe.strategies[0]["weight"] == 1.0

    def test_create_converts_pct_to_fraction(self, db, user):
        recipe = create_recipe(
            user_id=user.id,
            name="Multi",
            symbol="ETHUSDT",
            strategies=[
                {"strategy": "A", "weight": 40},
                {"strategy": "B", "weight": 60},
            ],
            interval="4h",
            lookback_days=14,
            buy_threshold=0.6,
            sell_threshold=0.6,
            auto_threshold=0.0,
            max_order_pct=10.0,
            stop_loss_pct=3.0,
            take_profit_pct=6.0,
            strategy_params=None,
            db=db,
        )
        weights = [s["weight"] for s in recipe.strategies]
        assert sum(weights) == pytest.approx(1.0)
        assert weights[0] == pytest.approx(0.4)
        assert weights[1] == pytest.approx(0.6)

    def test_create_invalid_strategy_raises(self, db, user):
        with pytest.raises(ValueError, match="desconocida"):
            create_recipe(
                user_id=user.id,
                name="Bad",
                symbol="BTCUSDT",
                strategies=[{"strategy": "X", "weight": 100}],
                interval="1h",
                lookback_days=7,
                buy_threshold=0.5,
                sell_threshold=0.5,
                auto_threshold=0.0,
                max_order_pct=5.0,
                stop_loss_pct=2.0,
                take_profit_pct=4.0,
                strategy_params=None,
                db=db,
            )

    def test_list_recipes(self, db, user):
        for i in range(3):
            create_recipe(
                user_id=user.id,
                name=f"Recipe {i}",
                symbol="BTCUSDT",
                strategies=[{"strategy": "A", "weight": 100}],
                interval="1h",
                lookback_days=7,
                buy_threshold=0.5,
                sell_threshold=0.5,
                auto_threshold=0.0,
                max_order_pct=5.0,
                stop_loss_pct=2.0,
                take_profit_pct=4.0,
                strategy_params=None,
                db=db,
            )
        recipes = list_recipes(user.id, db)
        assert len(recipes) == 3

    def test_toggle_recipe_status(self, db, user):
        recipe = create_recipe(
            user_id=user.id,
            name="Toggle",
            symbol="BTCUSDT",
            strategies=[{"strategy": "B", "weight": 100}],
            interval="5m",
            lookback_days=1,
            buy_threshold=0.5,
            sell_threshold=0.5,
            auto_threshold=0.0,
            max_order_pct=5.0,
            stop_loss_pct=2.0,
            take_profit_pct=4.0,
            strategy_params=None,
            db=db,
        )
        assert recipe.status == RecipeStatus.inactive

        toggled = toggle_recipe_status(recipe.id, user.id, db)
        assert toggled.status == RecipeStatus.active

        toggled2 = toggle_recipe_status(recipe.id, user.id, db)
        assert toggled2.status == RecipeStatus.inactive

    def test_update_recipe(self, db, user):
        recipe = create_recipe(
            user_id=user.id,
            name="Original",
            symbol="BTCUSDT",
            strategies=[{"strategy": "A", "weight": 100}],
            interval="1h",
            lookback_days=7,
            buy_threshold=0.5,
            sell_threshold=0.5,
            auto_threshold=0.0,
            max_order_pct=5.0,
            stop_loss_pct=2.0,
            take_profit_pct=4.0,
            strategy_params=None,
            db=db,
        )
        updated = update_recipe(recipe.id, user.id, {"name": "Updated"}, db)
        assert updated.name == "Updated"

    def test_delete_recipe(self, db, user):
        recipe = create_recipe(
            user_id=user.id,
            name="ToDelete",
            symbol="BTCUSDT",
            strategies=[{"strategy": "C", "weight": 100}],
            interval="1d",
            lookback_days=30,
            buy_threshold=0.5,
            sell_threshold=0.5,
            auto_threshold=0.0,
            max_order_pct=5.0,
            stop_loss_pct=2.0,
            take_profit_pct=4.0,
            strategy_params=None,
            db=db,
        )
        delete_recipe(recipe.id, user.id, db)
        assert list_recipes(user.id, db) == []

    def test_delete_nonexistent_raises(self, db, user):
        with pytest.raises(ValueError, match="no encontrada"):
            delete_recipe(999, user.id, db)

    def test_toggle_nonexistent_raises(self, db, user):
        with pytest.raises(ValueError, match="no encontrada"):
            toggle_recipe_status(999, user.id, db)

    def test_update_nonexistent_raises(self, db, user):
        with pytest.raises(ValueError, match="no encontrada"):
            update_recipe(999, user.id, {"name": "X"}, db)


class TestAutoThreshold:
    def test_create_with_auto_threshold(self, db, user):
        recipe = create_recipe(
            user_id=user.id,
            name="Auto",
            symbol="BTCUSDT",
            strategies=[{"strategy": "A", "weight": 100}],
            interval="1h",
            lookback_days=7,
            buy_threshold=0.5,
            sell_threshold=0.5,
            auto_threshold=0.8,
            max_order_pct=5.0,
            stop_loss_pct=2.0,
            take_profit_pct=4.0,
            strategy_params=None,
            db=db,
        )
        assert recipe.auto_threshold == 0.8

    def test_auto_threshold_zero_disabled(self, db, user):
        recipe = create_recipe(
            user_id=user.id,
            name="NoAuto",
            symbol="BTCUSDT",
            strategies=[{"strategy": "A", "weight": 100}],
            interval="1h",
            lookback_days=7,
            buy_threshold=0.5,
            sell_threshold=0.5,
            auto_threshold=0.0,
            max_order_pct=5.0,
            stop_loss_pct=2.0,
            take_profit_pct=4.0,
            strategy_params=None,
            db=db,
        )
        assert recipe.auto_threshold == 0.0

    def test_auto_below_buy_raises(self, db, user):
        with pytest.raises(ValueError, match="auto-aprobación"):
            create_recipe(
                user_id=user.id,
                name="Bad",
                symbol="BTCUSDT",
                strategies=[{"strategy": "A", "weight": 100}],
                interval="1h",
                lookback_days=7,
                buy_threshold=0.5,
                sell_threshold=0.5,
                auto_threshold=0.3,
                max_order_pct=5.0,
                stop_loss_pct=2.0,
                take_profit_pct=4.0,
                strategy_params=None,
                db=db,
            )

    def test_update_auto_threshold(self, db, user):
        recipe = create_recipe(
            user_id=user.id,
            name="UpdAuto",
            symbol="BTCUSDT",
            strategies=[{"strategy": "A", "weight": 100}],
            interval="1h",
            lookback_days=7,
            buy_threshold=0.5,
            sell_threshold=0.5,
            auto_threshold=0.0,
            max_order_pct=5.0,
            stop_loss_pct=2.0,
            take_profit_pct=4.0,
            strategy_params=None,
            db=db,
        )
        updated = update_recipe(recipe.id, user.id, {"auto_threshold": 0.8}, db)
        assert updated.auto_threshold == 0.8

    def test_update_auto_below_existing_buy_raises(self, db, user):
        recipe = create_recipe(
            user_id=user.id,
            name="BadUpd",
            symbol="BTCUSDT",
            strategies=[{"strategy": "A", "weight": 100}],
            interval="1h",
            lookback_days=7,
            buy_threshold=0.5,
            sell_threshold=0.5,
            auto_threshold=0.0,
            max_order_pct=5.0,
            stop_loss_pct=2.0,
            take_profit_pct=4.0,
            strategy_params=None,
            db=db,
        )
        with pytest.raises(ValueError, match="auto-aprobación"):
            update_recipe(recipe.id, user.id, {"auto_threshold": 0.3}, db)
