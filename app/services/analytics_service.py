"""P&L analytics: summary, daily, and monthly breakdowns for closed trades."""

from collections import defaultdict

from sqlalchemy.orm import Session

from app.config import get_settings
from app.models import TradeExecution


def _compute_pnl(trade: TradeExecution, fee_pct: float) -> dict | None:
    """Compute P&L for a single closed trade. Returns None if not computable."""
    result = trade.result or {}
    if result.get("oco_status") != "ALL_DONE":
        return None

    exit_time = result.get("exit_time")
    if not exit_time:
        return None

    # Use pre-computed values if available, otherwise compute
    if "net_pnl_abs" in result:
        return {
            "pnl_abs": result["pnl_abs"],
            "net_pnl_abs": result["net_pnl_abs"],
            "exit_type": result.get("exit_type", "UNKNOWN"),
            "exit_time": exit_time,
            "symbol": trade.symbol,
        }

    # Backfill: compute from raw data
    exit_px = result.get("exit_price", 0)
    entry_px = trade.price or 0
    qty = trade.quantity or 0
    if entry_px <= 0 or exit_px <= 0:
        return None

    if trade.side == "BUY":
        pnl_pct = (exit_px - entry_px) / entry_px * 100
    else:
        pnl_pct = (entry_px - exit_px) / entry_px * 100

    pnl_abs = round(pnl_pct / 100 * entry_px * qty, 4)
    net_pnl_pct = pnl_pct - fee_pct * 2
    net_pnl_abs = round(net_pnl_pct / 100 * entry_px * qty, 4)

    return {
        "pnl_abs": pnl_abs,
        "net_pnl_abs": net_pnl_abs,
        "exit_type": result.get("exit_type", "UNKNOWN"),
        "exit_time": exit_time,
        "symbol": trade.symbol,
        "_backfill": {
            "pnl_pct": round(pnl_pct, 4),
            "pnl_abs": pnl_abs,
            "net_pnl_pct": round(net_pnl_pct, 4),
            "net_pnl_abs": net_pnl_abs,
            "commission_pct": fee_pct,
        },
    }


def compute_analytics(user_id: int, db: Session) -> dict:
    """Compute full P&L analytics for a user's closed trades."""
    settings = get_settings()
    fee_pct = settings.commission_pct

    trades = (
        db.query(TradeExecution)
        .filter(TradeExecution.executed_by == user_id, TradeExecution.status == "filled")
        .all()
    )

    closed: list[dict] = []
    for t in trades:
        pnl = _compute_pnl(t, fee_pct)
        if pnl is None:
            continue

        # Lazy backfill: persist computed P&L if missing
        if "_backfill" in pnl:
            updated = dict(t.result or {})
            updated.update(pnl["_backfill"])
            t.result = updated
            db.add(t)
            del pnl["_backfill"]

        closed.append(pnl)

    # Commit any backfills
    db.commit()

    if not closed:
        return {
            "summary": {
                "total_trades": 0, "wins": 0, "losses": 0, "win_rate": 0,
                "gross_pnl": 0, "total_commissions": 0, "net_pnl": 0,
                "avg_win": 0, "avg_loss": 0, "profit_factor": None,
                "best_trade": 0, "worst_trade": 0,
            },
            "daily": [],
            "monthly": [],
        }

    # Summary
    wins = [c for c in closed if c["exit_type"] == "TP"]
    losses = [c for c in closed if c["exit_type"] == "SL"]
    total = len(closed)
    win_count = len(wins)
    loss_count = len(losses)

    gross_pnl = round(sum(c["pnl_abs"] for c in closed), 2)
    net_pnl = round(sum(c["net_pnl_abs"] for c in closed), 2)
    total_commissions = round(gross_pnl - net_pnl, 2)

    win_amounts = [c["net_pnl_abs"] for c in wins]
    loss_amounts = [c["net_pnl_abs"] for c in losses]
    avg_win = round(sum(win_amounts) / len(win_amounts), 2) if win_amounts else 0
    avg_loss = round(sum(loss_amounts) / len(loss_amounts), 2) if loss_amounts else 0

    gross_wins = sum(c["pnl_abs"] for c in wins) if wins else 0
    gross_losses = abs(sum(c["pnl_abs"] for c in losses)) if losses else 0
    profit_factor = round(gross_wins / gross_losses, 2) if gross_losses > 0 else None

    all_net = [c["net_pnl_abs"] for c in closed]
    best_trade = round(max(all_net), 2) if all_net else 0
    worst_trade = round(min(all_net), 2) if all_net else 0

    summary = {
        "total_trades": total,
        "wins": win_count,
        "losses": loss_count,
        "win_rate": round(win_count / total * 100, 1) if total > 0 else 0,
        "gross_pnl": gross_pnl,
        "total_commissions": total_commissions,
        "net_pnl": net_pnl,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "profit_factor": profit_factor,
        "best_trade": best_trade,
        "worst_trade": worst_trade,
    }

    # Group by day and month
    daily_map: dict[str, dict] = defaultdict(lambda: {"trades": 0, "wins": 0, "losses": 0, "gross_pnl": 0.0, "net_pnl": 0.0})
    monthly_map: dict[str, dict] = defaultdict(lambda: {"trades": 0, "wins": 0, "losses": 0, "gross_pnl": 0.0, "net_pnl": 0.0})

    for c in closed:
        et = c["exit_time"]
        day = et[:10] if len(et) >= 10 else et
        month = et[:7] if len(et) >= 7 else et

        for m, key in [(daily_map, day), (monthly_map, month)]:
            m[key]["trades"] += 1
            if c["exit_type"] == "TP":
                m[key]["wins"] += 1
            elif c["exit_type"] == "SL":
                m[key]["losses"] += 1
            m[key]["gross_pnl"] += c["pnl_abs"]
            m[key]["net_pnl"] += c["net_pnl_abs"]

    def _build_periods(period_map: dict) -> list[dict]:
        periods = sorted(period_map.keys())
        result = []
        cumulative = 0.0
        for p in periods:
            d = period_map[p]
            cumulative += d["net_pnl"]
            result.append({
                "period": p,
                "trades": d["trades"],
                "wins": d["wins"],
                "losses": d["losses"],
                "gross_pnl": round(d["gross_pnl"], 2),
                "net_pnl": round(d["net_pnl"], 2),
                "cumulative": round(cumulative, 2),
            })
        return result

    return {
        "summary": summary,
        "daily": _build_periods(daily_map),
        "monthly": _build_periods(monthly_map),
    }
