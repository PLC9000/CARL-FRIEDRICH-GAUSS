"""Marketplace: backtest engine + scoring for recipe evaluation."""

import asyncio
import datetime
import logging
import math
import time
from collections import defaultdict

import numpy as np
from sqlalchemy.orm import Session

from app.binance_client import fetch_candles
from app.database import SessionLocal
from app.models import Recipe, BacktestResult
from app.strategies import STRATEGY_MAP
from app.services.evaluation_engine import (
    normalize_score,
    compute_weighted_score,
    compute_strength_factor,
    determine_signal,
    INTERVAL_SECONDS,
)

logger = logging.getLogger(__name__)

# Track running tasks to prevent duplicates
_backtest_tasks: dict[int, asyncio.Task] = {}

# Max backtest days per interval (performance guard)
_MAX_DAYS = {
    "1m": 90,
    "5m": 180,
    "15m": 365,
    "1h": 365,
    "4h": 365,
    "1d": 365,
}


# ── Launch / poll ──────────────────────────────────────────────────────

async def launch_backtest(
    recipe_id: int, user_id: int, backtest_days: int, db: Session,
) -> BacktestResult:
    """Create a running BacktestResult and spawn the background task."""
    recipe = db.query(Recipe).filter(
        Recipe.id == recipe_id, Recipe.user_id == user_id,
    ).first()
    if recipe is None:
        raise ValueError("Receta no encontrada")
    if not recipe.strategies:
        raise ValueError("La receta no tiene estrategias configuradas")

    max_d = _MAX_DAYS.get(recipe.interval, 365)
    if backtest_days > max_d:
        backtest_days = max_d

    # Check for running task
    existing = (
        db.query(BacktestResult)
        .filter(BacktestResult.recipe_id == recipe_id, BacktestResult.status == "running")
        .first()
    )
    if existing and existing.id in _backtest_tasks:
        return existing

    # Delete old completed results for this recipe (keep only latest)
    db.query(BacktestResult).filter(
        BacktestResult.recipe_id == recipe_id,
        BacktestResult.status != "running",
    ).delete()

    bt = BacktestResult(
        recipe_id=recipe_id,
        user_id=user_id,
        symbol=recipe.symbol,
        interval=recipe.interval,
        backtest_days=backtest_days,
        trades=[],
        equity_curve=[],
        status="running",
    )
    db.add(bt)
    db.commit()
    db.refresh(bt)

    task = asyncio.create_task(_run_backtest_task(bt.id, recipe_id, backtest_days))
    _backtest_tasks[bt.id] = task
    return bt


def get_backtest(recipe_id: int, user_id: int, db: Session) -> BacktestResult | None:
    """Return the latest BacktestResult for a recipe."""
    return (
        db.query(BacktestResult)
        .filter(BacktestResult.recipe_id == recipe_id, BacktestResult.user_id == user_id)
        .order_by(BacktestResult.created_at.desc())
        .first()
    )


# ── Background backtest task ──────────────────────────────────────────

async def _run_backtest_task(bt_id: int, recipe_id: int, backtest_days: int):
    """Run the full backtest in the background."""
    db = SessionLocal()
    try:
        bt = db.query(BacktestResult).get(bt_id)
        recipe = db.query(Recipe).get(recipe_id)
        if not bt or not recipe:
            return
        t0 = time.time()

        now = datetime.datetime.utcnow()
        start_dt = now - datetime.timedelta(days=backtest_days)
        end_dt = now

        # Fetch all candles
        candles = await fetch_candles(recipe.symbol, recipe.interval, start_dt, end_dt)
        if candles is None or len(candles) < 50:
            bt.status = "failed"
            bt.error_message = f"Datos insuficientes: {len(candles) if candles is not None else 0} velas"
            bt.completed_at = datetime.datetime.utcnow()
            db.commit()
            return

        total_candles = len(candles)
        interval_secs = INTERVAL_SECONDS.get(recipe.interval, 3600)
        candles_per_day = 86400 // interval_secs
        lookback_window = recipe.lookback_days * candles_per_day

        if lookback_window >= total_candles:
            bt.status = "failed"
            bt.error_message = f"Ventana de lookback ({lookback_window}) >= velas totales ({total_candles})"
            bt.completed_at = datetime.datetime.utcnow()
            db.commit()
            return

        bt.start_date = start_dt.isoformat()
        bt.end_date = end_dt.isoformat()
        bt.total_candles = total_candles
        bt.lookback_candles = lookback_window

        # Run backtest
        trades, equity_curve = _run_sliding_window(recipe, candles, lookback_window)

        # Compute summary stats
        stats = _compute_stats(trades, equity_curve, backtest_days)

        # Compute scores
        scores = _compute_scores(stats, trades, backtest_days)

        # Classification
        classification = _classify(scores)

        # Monthly P&L breakdown
        monthly = _compute_monthly_pnl(trades, candles, interval_secs, start_dt)

        # Downsample equity curve to max 500 points
        eq_out = equity_curve
        if len(eq_out) > 500:
            step = len(eq_out) / 500
            eq_out = [eq_out[int(i * step)] for i in range(500)]

        # Populate result
        bt.total_trades = stats["total_trades"]
        bt.wins = stats["wins"]
        bt.losses = stats["losses"]
        bt.win_rate = stats["win_rate"]
        bt.net_pnl_pct = stats["net_pnl_pct"]
        bt.gross_pnl_pct = stats["gross_pnl_pct"]
        bt.profit_factor = stats["profit_factor"]
        bt.max_drawdown_pct = stats["max_drawdown_pct"]
        bt.avg_win_pct = stats["avg_win_pct"]
        bt.avg_loss_pct = stats["avg_loss_pct"]
        bt.max_consecutive_losses = stats["max_consecutive_losses"]
        bt.max_consecutive_wins = stats["max_consecutive_wins"]
        bt.annualized_return_pct = stats["annualized_return_pct"]
        bt.volatility_pct = stats["volatility_pct"]
        bt.sharpe_ratio = stats["sharpe_ratio"]

        bt.score_performance = scores["performance"]
        bt.score_risk = scores["risk"]
        bt.score_consistency = scores["consistency"]
        bt.score_reliability = scores["reliability"]
        bt.score_global = scores["global"]
        bt.classification = classification

        bt.trades = trades
        bt.equity_curve = eq_out
        bt.monthly_pnl = monthly
        bt.score_details = scores

        bt.status = "completed"
        bt.duration_seconds = round(time.time() - t0, 2)
        bt.completed_at = datetime.datetime.utcnow()
        db.commit()
        logger.info(
            "Backtest %d completed: %d trades, score=%.1f, %.2fs",
            bt_id, len(trades), scores["global"], bt.duration_seconds,
        )
    except Exception as exc:
        logger.exception("Backtest %d failed", bt_id)
        try:
            bt = db.query(BacktestResult).get(bt_id)
            if bt:
                bt.status = "failed"
                bt.error_message = str(exc)[:500]
                bt.completed_at = datetime.datetime.utcnow()
                db.commit()
        except Exception:
            pass
    finally:
        db.close()
        _backtest_tasks.pop(bt_id, None)


# ── Sliding window backtest engine ────────────────────────────────────

def _run_sliding_window(
    recipe: Recipe, candles: np.ndarray, lookback_window: int,
) -> tuple[list[dict], list[dict]]:
    """Slide across candles, run recipe strategies, simulate trades.

    Returns (trades_list, equity_curve_list).
    """
    recipe_mode = recipe.mode or "roles"
    buy_th = recipe.buy_threshold
    sell_th = recipe.sell_threshold
    sl_pct = recipe.stop_loss_pct / 100
    tp_pct = recipe.take_profit_pct / 100
    strength_th = recipe.strength_threshold or 0.0

    total = len(candles)
    trades: list[dict] = []
    equity_curve: list[dict] = []
    equity = 100.0

    # Position state
    pos_side = None   # "BUY" or "SELL" or None
    pos_entry = 0.0
    pos_sl = 0.0
    pos_tp = 0.0
    pos_entry_idx = 0

    # Evaluate every N candles to keep runtime reasonable
    # For 1m interval with 90d → ~130k candles. Step every 5 candles = 26k evals.
    step = max(1, total // 5000)

    has_fng = any(s.get("strategy") == "K" for s in recipe.strategies)
    if has_fng:
        logger.info("Backtest: skipping strategy K (Fear & Greed) — requires external API")

    for i in range(lookback_window, total, step):
        close = float(candles[i, 4])
        high = float(candles[i, 2])
        low = float(candles[i, 3])

        # Check SL/TP if position is open
        if pos_side is not None:
            hit_sl = False
            hit_tp = False
            if pos_side == "BUY":
                if low <= pos_sl:
                    hit_sl = True
                if high >= pos_tp:
                    hit_tp = True
            else:  # SELL
                if high >= pos_sl:
                    hit_sl = True
                if low <= pos_tp:
                    hit_tp = True

            if hit_sl or hit_tp:
                # SL priority if both hit
                if hit_sl:
                    exit_price = pos_sl
                    exit_type = "SL"
                else:
                    exit_price = pos_tp
                    exit_type = "TP"

                if pos_side == "BUY":
                    pnl = (exit_price - pos_entry) / pos_entry * 100
                else:
                    pnl = (pos_entry - exit_price) / pos_entry * 100

                trades.append({
                    "entry_price": round(pos_entry, 8),
                    "exit_price": round(exit_price, 8),
                    "side": pos_side,
                    "entry_idx": pos_entry_idx,
                    "exit_idx": i,
                    "exit_type": exit_type,
                    "pnl_pct": round(pnl, 4),
                    "duration_candles": i - pos_entry_idx,
                })
                equity *= (1 + pnl / 100)
                pos_side = None

        # Record equity
        equity_curve.append({"idx": i, "equity": round(equity, 4)})

        # If no position, evaluate for entry
        if pos_side is None:
            window = candles[i - lookback_window: i + 1]
            signal = _evaluate_window(recipe, window, recipe_mode, buy_th, sell_th, strength_th, has_fng)

            if signal in ("BUY", "SELL"):
                pos_side = signal
                pos_entry = close
                pos_entry_idx = i
                if signal == "BUY":
                    pos_sl = round(close * (1 - sl_pct), 8)
                    pos_tp = round(close * (1 + tp_pct), 8)
                else:
                    pos_sl = round(close * (1 + sl_pct), 8)
                    pos_tp = round(close * (1 - tp_pct), 8)

    # Close open position at last close (EXPIRY)
    if pos_side is not None:
        last_close = float(candles[-1, 4])
        if pos_side == "BUY":
            pnl = (last_close - pos_entry) / pos_entry * 100
        else:
            pnl = (pos_entry - last_close) / pos_entry * 100
        trades.append({
            "entry_price": round(pos_entry, 8),
            "exit_price": round(last_close, 8),
            "side": pos_side,
            "entry_idx": pos_entry_idx,
            "exit_idx": len(candles) - 1,
            "exit_type": "EXPIRY",
            "pnl_pct": round(pnl, 4),
            "duration_candles": len(candles) - 1 - pos_entry_idx,
        })
        equity *= (1 + pnl / 100)
        equity_curve.append({"idx": len(candles) - 1, "equity": round(equity, 4)})

    return trades, equity_curve


def _evaluate_window(
    recipe: Recipe,
    window: np.ndarray,
    mode: str,
    buy_th: float,
    sell_th: float,
    strength_th: float,
    skip_fng: bool,
) -> str:
    """Run all strategies on a candle window and determine signal."""
    strategy_results = []

    for strat_config in recipe.strategies:
        strat_key = strat_config["strategy"]
        weight = strat_config.get("weight", 1.0)
        role = strat_config.get("role", "direction")

        if skip_fng and strat_key == "K":
            continue

        fn = STRATEGY_MAP.get(strat_key)
        if fn is None:
            continue

        params = None
        if recipe.strategy_params and strat_key in recipe.strategy_params:
            params = recipe.strategy_params[strat_key]

        try:
            result = fn(window, params=params)
        except Exception:
            continue

        score = normalize_score(result["recommendation"], result["confidence"])
        force = result.get("force", result["confidence"] / 100.0) if role == "strength" else None

        entry = {
            "strategy": strat_key,
            "weight": weight,
            "role": role,
            "recommendation": result["recommendation"],
            "confidence": result["confidence"],
            "score": score,
        }
        if force is not None:
            entry["force"] = force

        strategy_results.append(entry)

    if not strategy_results:
        return "HOLD"

    if mode == "weighted":
        total_w = sum(r["weight"] for r in strategy_results)
        final_score = (
            sum(r["score"] * r["weight"] for r in strategy_results) / total_w
        ) if total_w else 0.0
        final_score = max(-1.0, min(1.0, final_score))
        strength_factor = 1.0
        eff_strength_th = 0.0
    else:
        final_score = compute_weighted_score(strategy_results)
        strength_factor = compute_strength_factor(strategy_results)
        eff_strength_th = strength_th

    return determine_signal(
        final_score, buy_th, sell_th,
        strength_factor=strength_factor,
        strength_threshold=eff_strength_th,
    )


# ── Statistics ────────────────────────────────────────────────────────

def _compute_stats(
    trades: list[dict], equity_curve: list[dict], backtest_days: int,
) -> dict:
    """Compute summary statistics from trades and equity curve."""
    total = len(trades)
    if total == 0:
        return {
            "total_trades": 0, "wins": 0, "losses": 0, "win_rate": 0,
            "net_pnl_pct": 0, "gross_pnl_pct": 0, "profit_factor": None,
            "max_drawdown_pct": 0, "avg_win_pct": 0, "avg_loss_pct": 0,
            "max_consecutive_losses": 0, "max_consecutive_wins": 0,
            "annualized_return_pct": 0, "volatility_pct": 0, "sharpe_ratio": None,
        }

    pnls = [t["pnl_pct"] for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]

    gross_win = sum(wins) if wins else 0
    gross_loss = abs(sum(losses)) if losses else 0
    profit_factor = round(gross_win / gross_loss, 4) if gross_loss > 0 else None

    # Max consecutive wins/losses
    max_con_loss = 0
    max_con_win = 0
    cur_loss = 0
    cur_win = 0
    for p in pnls:
        if p > 0:
            cur_win += 1
            cur_loss = 0
            max_con_win = max(max_con_win, cur_win)
        else:
            cur_loss += 1
            cur_win = 0
            max_con_loss = max(max_con_loss, cur_loss)

    # Max drawdown from equity curve
    max_dd = 0.0
    peak = 100.0
    for pt in equity_curve:
        eq = pt["equity"]
        if eq > peak:
            peak = eq
        dd = (peak - eq) / peak * 100
        if dd > max_dd:
            max_dd = dd

    net_pnl = sum(pnls)
    vol = float(np.std(pnls)) if len(pnls) > 1 else 0.0

    # Annualized return
    final_eq = equity_curve[-1]["equity"] if equity_curve else 100
    total_return = (final_eq - 100) / 100
    years = backtest_days / 365
    annualized = ((1 + total_return) ** (1 / years) - 1) * 100 if years > 0 else 0

    # Sharpe ratio (annualized, assuming risk-free = 0)
    if vol > 0 and len(pnls) > 1:
        avg_trade_return = np.mean(pnls)
        trades_per_year = len(pnls) / years if years > 0 else len(pnls)
        sharpe = (avg_trade_return / vol) * math.sqrt(trades_per_year)
        sharpe = round(sharpe, 4)
    else:
        sharpe = None

    return {
        "total_trades": total,
        "wins": len(wins),
        "losses": len(losses),
        "win_rate": round(len(wins) / total * 100, 2),
        "net_pnl_pct": round(net_pnl, 4),
        "gross_pnl_pct": round(gross_win - gross_loss, 4),
        "profit_factor": profit_factor,
        "max_drawdown_pct": round(max_dd, 4),
        "avg_win_pct": round(np.mean(wins), 4) if wins else 0,
        "avg_loss_pct": round(np.mean(losses), 4) if losses else 0,
        "max_consecutive_losses": max_con_loss,
        "max_consecutive_wins": max_con_win,
        "annualized_return_pct": round(annualized, 4),
        "volatility_pct": round(vol, 4),
        "sharpe_ratio": sharpe,
    }


# ── Scoring system ────────────────────────────────────────────────────

def _sigmoid_map(val: float, center: float, scale: float) -> float:
    """Map a value to 0-100 using a sigmoid centered at `center`."""
    x = (val - center) / scale
    return 100 / (1 + math.exp(-x))


def _linear_map(val: float, lo: float, hi: float) -> float:
    """Linear map from [lo, hi] to [0, 100], clamped."""
    if hi <= lo:
        return 50.0
    return max(0.0, min(100.0, (val - lo) / (hi - lo) * 100))


def _compute_scores(stats: dict, trades: list[dict], backtest_days: int) -> dict:
    """Compute sub-scores and global score (all 0-100)."""

    # ── Performance (40%) ─────────────────────────────────────────
    s_pnl = _sigmoid_map(stats["net_pnl_pct"], 0, 25)
    s_pf = _linear_map(stats["profit_factor"] or 0, 0, 3) if stats["profit_factor"] else 0
    s_annual = _sigmoid_map(stats["annualized_return_pct"], 0, 50)
    perf = round((s_pnl + s_pf + s_annual) / 3, 2)

    # ── Risk (30%) — inverted: lower = better score ───────────────
    s_dd = _linear_map(50 - stats["max_drawdown_pct"], 0, 50)  # 0% DD → 100
    s_vol = _linear_map(10 - stats["volatility_pct"], 0, 10)   # 0% vol → 100
    s_con_loss = _linear_map(10 - stats["max_consecutive_losses"], 0, 10)
    risk = round((s_dd + s_vol + s_con_loss) / 3, 2)

    # ── Consistency (20%) ─────────────────────────────────────────
    s_wr = _linear_map(stats["win_rate"], 30, 80)
    # Monthly variance
    pnls = [t["pnl_pct"] for t in trades]
    if len(pnls) > 3:
        cv = float(np.std(pnls) / abs(np.mean(pnls))) if np.mean(pnls) != 0 else 5
        s_disp = _linear_map(5 - cv, 0, 5)  # lower CV = better
    else:
        s_disp = 0
    consist = round((s_wr + s_disp) / 2, 2)

    # ── Reliability (10%) ─────────────────────────────────────────
    s_trades = min(stats["total_trades"] / 50 * 100, 100)
    s_span = min(backtest_days / 90 * 100, 100)
    # Signal frequency: ratio of trades to potential slots
    total_candles_approx = backtest_days * 24  # rough estimate for 1h
    freq_pct = (stats["total_trades"] / max(total_candles_approx, 1)) * 100
    # Bell curve around 5-15%
    if freq_pct < 1:
        s_freq = freq_pct * 20
    elif freq_pct <= 10:
        s_freq = 20 + (freq_pct - 1) * (80 / 9)
    else:
        s_freq = max(0, 100 - (freq_pct - 10) * 5)
    reliab = round((s_trades + s_span + s_freq) / 3, 2)

    global_score = round(
        perf * 0.40 + risk * 0.30 + consist * 0.20 + reliab * 0.10, 2
    )

    return {
        "performance": perf,
        "risk": risk,
        "consistency": consist,
        "reliability": reliab,
        "global": global_score,
        # Sub-component details for transparency
        "details": {
            "perf_pnl": round(s_pnl, 1),
            "perf_pf": round(s_pf, 1),
            "perf_annual": round(s_annual, 1),
            "risk_dd": round(s_dd, 1),
            "risk_vol": round(s_vol, 1),
            "risk_con_loss": round(s_con_loss, 1),
            "consist_wr": round(s_wr, 1),
            "consist_disp": round(s_disp, 1),
            "reliab_trades": round(s_trades, 1),
            "reliab_span": round(s_span, 1),
            "reliab_freq": round(s_freq, 1),
        },
    }


def _classify(scores: dict) -> str:
    """Classify recipe based on score profile."""
    perf = scores["performance"]
    risk = scores["risk"]
    if risk > 70 and perf <= 60:
        return "Conservadora"
    if perf > 70 and risk < 40:
        return "Agresiva"
    return "Balanceada"


# ── Monthly P&L ───────────────────────────────────────────────────────

def _compute_monthly_pnl(
    trades: list[dict],
    candles: np.ndarray,
    interval_secs: int,
    start_dt: datetime.datetime,
) -> list[dict]:
    """Group trades by month for the monthly P&L breakdown."""
    if not trades:
        return []

    monthly: dict[str, dict] = defaultdict(
        lambda: {"month": "", "trades": 0, "wins": 0, "losses": 0, "pnl_pct": 0.0}
    )

    for t in trades:
        # Approximate the trade's date from candle index
        idx = t.get("exit_idx", 0)
        approx_secs = idx * interval_secs
        trade_dt = start_dt + datetime.timedelta(seconds=approx_secs)
        month_key = trade_dt.strftime("%Y-%m")

        m = monthly[month_key]
        m["month"] = month_key
        m["trades"] += 1
        m["pnl_pct"] = round(m["pnl_pct"] + t["pnl_pct"], 4)
        if t["pnl_pct"] > 0:
            m["wins"] += 1
        else:
            m["losses"] += 1

    result = sorted(monthly.values(), key=lambda x: x["month"])
    # Add cumulative
    cum = 0.0
    for m in result:
        cum += m["pnl_pct"]
        m["cumulative"] = round(cum, 4)
    return result
