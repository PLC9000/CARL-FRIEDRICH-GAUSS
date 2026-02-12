import datetime
from sqlalchemy import (
    Column, Integer, String, Float, Boolean, DateTime, Text, Enum as SAEnum,
    ForeignKey, JSON,
)
from sqlalchemy.orm import relationship
import enum

from app.database import Base


class Role(str, enum.Enum):
    user = "user"
    admin = "admin"


class ApprovalStatus(str, enum.Enum):
    pending = "pending"
    approved = "approved"
    rejected = "rejected"


class RecipeStatus(str, enum.Enum):
    active = "active"
    inactive = "inactive"


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(50), unique=True, nullable=False, index=True)
    hashed_password = Column(String(255), nullable=False)
    role = Column(SAEnum(Role), default=Role.user, nullable=False)
    auto_only = Column(Boolean, default=False, nullable=False)
    # Binance credentials (encrypted at rest)
    binance_api_key_enc = Column(Text, default="")
    binance_api_secret_enc = Column(Text, default="")
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    recommendations = relationship("Recommendation", back_populates="user")
    audit_logs = relationship("AuditLog", back_populates="user")
    recipes = relationship("Recipe", back_populates="user")


class Recommendation(Base):
    __tablename__ = "recommendations"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    symbol = Column(String(20), nullable=False)
    interval = Column(String(5), nullable=False)
    period_start = Column(String(30), nullable=False)
    period_end = Column(String(30), nullable=False)
    strategy = Column(String(10), nullable=False)
    metrics = Column(JSON, nullable=True)
    recommendation = Column(String(10), nullable=False)
    confidence = Column(Float, nullable=False)
    entry_price = Column(Float, nullable=True)
    stop_loss = Column(Float, nullable=True)
    take_profit = Column(Float, nullable=True)
    explanation = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    user = relationship("User", back_populates="recommendations")
    approval = relationship("Approval", back_populates="recommendation", uselist=False)


class Approval(Base):
    __tablename__ = "approvals"

    id = Column(Integer, primary_key=True, index=True)
    recommendation_id = Column(Integer, ForeignKey("recommendations.id"), unique=True, nullable=False)
    status = Column(SAEnum(ApprovalStatus), default=ApprovalStatus.pending, nullable=False, index=True)
    reviewed_by = Column(Integer, ForeignKey("users.id"), nullable=True)
    review_reason = Column(Text, nullable=True)
    reviewed_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow, index=True)

    recommendation = relationship("Recommendation", back_populates="approval")


class TradeExecution(Base):
    __tablename__ = "trade_executions"

    id = Column(Integer, primary_key=True, index=True)
    approval_id = Column(Integer, ForeignKey("approvals.id"), nullable=False, index=True)
    executed_by = Column(Integer, ForeignKey("users.id"), nullable=False)
    is_live = Column(Boolean, default=False)
    order_type = Column(String(10), default="market")  # market | oco
    symbol = Column(String(20), nullable=False)
    side = Column(String(4), nullable=False)
    quantity = Column(Float, nullable=True)
    price = Column(Float, nullable=True)
    stop_loss = Column(Float, nullable=True)
    take_profit = Column(Float, nullable=True)
    status = Column(String(20), default="simulated")
    result = Column(JSON, nullable=True)
    executed_at = Column(DateTime, default=datetime.datetime.utcnow)


class AuditLog(Base):
    __tablename__ = "audit_logs"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    action = Column(String(50), nullable=False)
    payload = Column(JSON, nullable=True)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow, index=True)

    user = relationship("User", back_populates="audit_logs")


class Recipe(Base):
    __tablename__ = "recipes"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    name = Column(String(100), nullable=False)
    symbol = Column(String(20), nullable=False)
    strategies = Column(JSON, nullable=False)  # [{strategy: "A", weight: 1.0}, ...]
    interval = Column(String(5), nullable=False)
    lookback_days = Column(Integer, nullable=False, default=7)
    buy_threshold = Column(Float, nullable=False, default=0.5)
    sell_threshold = Column(Float, nullable=False, default=0.5)
    auto_threshold = Column(Float, nullable=False, default=0.0)
    auto_quantity = Column(Float, nullable=True, default=None)  # legacy
    buy_quantity = Column(Float, nullable=True, default=None)
    sell_quantity = Column(Float, nullable=True, default=None)
    auto_order_type = Column(String(10), nullable=False, default="oco")  # legacy
    buy_order_type = Column(String(10), nullable=False, default="oco")
    sell_order_type = Column(String(10), nullable=False, default="oco")
    strength_threshold = Column(Float, nullable=False, default=0.0)
    auto_strength_threshold = Column(Float, nullable=False, default=0.0)
    turbo_threshold = Column(Float, nullable=False, default=0.0)
    confirmation_minutes = Column(Float, nullable=False, default=0.0)
    confirmation_seconds = Column(Integer, nullable=False, default=0)
    max_order_pct = Column(Float, nullable=False, default=5.0)
    stop_loss_pct = Column(Float, nullable=False, default=2.0)
    take_profit_pct = Column(Float, nullable=False, default=4.0)
    max_ops_count = Column(Integer, nullable=False, default=0)   # 0 = unlimited
    max_ops_hours = Column(Float, nullable=False, default=24.0)  # time window in hours
    strategy_params = Column(JSON, nullable=True)  # {A: {param: val}, ...}
    mode = Column(String(20), nullable=True, default=None)  # "weighted" | "roles" | None
    status = Column(SAEnum(RecipeStatus), default=RecipeStatus.inactive, nullable=False, index=True)
    last_evaluated_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.datetime.utcnow,
                        onupdate=datetime.datetime.utcnow)

    user = relationship("User", back_populates="recipes")
    evaluations = relationship("RecipeEvaluation", back_populates="recipe",
                               order_by="RecipeEvaluation.evaluated_at.desc()")


class RecipeEvaluation(Base):
    __tablename__ = "recipe_evaluations"

    id = Column(Integer, primary_key=True, index=True)
    recipe_id = Column(Integer, ForeignKey("recipes.id"), nullable=False, index=True)
    strategy_results = Column(JSON, nullable=False)
    final_score = Column(Float, nullable=False)
    signal = Column(String(10), nullable=False)
    direction_status = Column(String(10), nullable=True)
    strength_status = Column(String(10), nullable=True)
    direction_value = Column(Float, nullable=True)
    strength_value = Column(Float, nullable=True)
    triggered = Column(Boolean, default=False)
    recommendation_id = Column(Integer, ForeignKey("recommendations.id"), nullable=True)
    evaluated_at = Column(DateTime, default=datetime.datetime.utcnow, index=True)

    recipe = relationship("Recipe", back_populates="evaluations")
    recommendation = relationship("Recommendation")


class BacktestResult(Base):
    __tablename__ = "backtest_results"

    id = Column(Integer, primary_key=True, index=True)
    recipe_id = Column(Integer, ForeignKey("recipes.id"), nullable=False, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)

    # Backtest configuration
    symbol = Column(String(20), nullable=False)
    interval = Column(String(5), nullable=False)
    backtest_days = Column(Integer, nullable=False)
    lookback_candles = Column(Integer, nullable=False, default=0)
    start_date = Column(String(30), nullable=False, default="")
    end_date = Column(String(30), nullable=False, default="")
    total_candles = Column(Integer, nullable=False, default=0)

    # Results summary
    total_trades = Column(Integer, nullable=False, default=0)
    wins = Column(Integer, nullable=False, default=0)
    losses = Column(Integer, nullable=False, default=0)
    win_rate = Column(Float, nullable=False, default=0.0)
    net_pnl_pct = Column(Float, nullable=False, default=0.0)
    gross_pnl_pct = Column(Float, nullable=False, default=0.0)
    profit_factor = Column(Float, nullable=True)
    max_drawdown_pct = Column(Float, nullable=False, default=0.0)
    avg_win_pct = Column(Float, nullable=False, default=0.0)
    avg_loss_pct = Column(Float, nullable=False, default=0.0)
    max_consecutive_losses = Column(Integer, nullable=False, default=0)
    max_consecutive_wins = Column(Integer, nullable=False, default=0)
    annualized_return_pct = Column(Float, nullable=False, default=0.0)
    volatility_pct = Column(Float, nullable=False, default=0.0)
    sharpe_ratio = Column(Float, nullable=True)

    # Scores (0-100)
    score_performance = Column(Float, nullable=False, default=0.0)
    score_risk = Column(Float, nullable=False, default=0.0)
    score_consistency = Column(Float, nullable=False, default=0.0)
    score_reliability = Column(Float, nullable=False, default=0.0)
    score_global = Column(Float, nullable=False, default=0.0)
    classification = Column(String(20), nullable=False, default="Balanceada")

    # Detailed data (JSON)
    trades = Column(JSON, nullable=False, default=list)
    equity_curve = Column(JSON, nullable=False, default=list)
    monthly_pnl = Column(JSON, nullable=True)
    score_details = Column(JSON, nullable=True)

    # Status
    status = Column(String(20), nullable=False, default="running")
    error_message = Column(Text, nullable=True)
    duration_seconds = Column(Float, nullable=True)

    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    completed_at = Column(DateTime, nullable=True)

    recipe = relationship("Recipe")


class StrategyConfig(Base):
    __tablename__ = "strategy_configs"

    id = Column(Integer, primary_key=True, index=True)
    strategy_key = Column(String(5), unique=True, nullable=False, index=True)
    enabled = Column(Boolean, default=False, nullable=False)
    updated_at = Column(DateTime, default=datetime.datetime.utcnow,
                        onupdate=datetime.datetime.utcnow)
