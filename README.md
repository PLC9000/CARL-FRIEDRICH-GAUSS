# Trading Decision Assistant

A production-minded trading decision assistant that connects to the Binance Spot API, analyses a selected token over a user-chosen time period, evaluates one of three mathematically-driven prediction strategies, and outputs a recommended **BUY / SELL / NO-TRADE** decision with entry, stop-loss, and take-profit levels.

Trade execution is gated behind **admin approval** and defaults to **paper trading**.

---

## Setup

```bash
# 1. Clone / enter project
cd BotTrading

# 2. Create a virtual environment (recommended)
python3 -m venv .venv && source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment
cp .env.example .env
# Edit .env — at minimum set JWT_SECRET_KEY to a random string.
# Binance API keys are optional (public kline endpoints work without them).
```

### Environment Variables

| Variable | Description | Default |
|---|---|---|
| `BINANCE_API_KEY` | Binance API key (read-only recommended) | `""` |
| `BINANCE_API_SECRET` | Binance API secret | `""` |
| `BINANCE_BASE_URL` | Binance REST base URL | `https://api.binance.com` |
| `JWT_SECRET_KEY` | Secret for signing JWTs | `change-me` |
| `JWT_EXPIRE_MINUTES` | Token lifetime in minutes | `60` |
| `DATABASE_URL` | SQLAlchemy connection string | `sqlite:///./trading.db` |
| `LIVE_MODE` | Enable live order placement | `false` |

---

## Run the Dev Server

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

API docs available at `http://localhost:8000/docs` (Swagger UI).

---

## Run Tests

```bash
pytest tests/ -v
```

---

## Create an Admin User

```bash
curl -X POST http://localhost:8000/auth/register \
  -H "Content-Type: application/json" \
  -d '{"username":"admin","password":"strongpassword","role":"admin"}'
```

Regular users use `"role":"user"` (the default).

---

## Strategies

| ID | Name | Description |
|---|---|---|
| **A** | AR Time-Series | Fits AR(4) on differenced close prices, forecasts 5 steps, signals based on expected return vs threshold |
| **B** | Kalman Filter | State-space smoother tracking level + slope; signals on normalised trend strength with volatility gate |
| **C** | OU Mean-Reversion | Rolling z-score of log-returns; signals on extreme z-scores with volatility regime filter |

---

## Approval Flow

1. Any authenticated user calls `POST /recommendation` — this generates the analysis **and** creates a pending approval record.
2. An **admin** reviews and calls `POST /approval/{id}/approve` or `POST /approval/{id}/reject`.
3. Only after approval can an admin call `POST /trade/execute` to place a (paper) trade.
4. All actions are logged to the `audit_logs` table.

If `LIVE_MODE=false` (default), execution is always simulated.

---

## Example curl Commands

### 1. Login

```bash
# Login and capture token
TOKEN=$(curl -s -X POST http://localhost:8000/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username":"admin","password":"strongpassword"}' \
  | python3 -c "import sys,json; print(json.load(sys.stdin)['access_token'])")
```

### 2. Request a Recommendation

```bash
curl -s -X POST http://localhost:8000/recommendation \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $TOKEN" \
  -d '{
    "symbol": "BTCUSDT",
    "interval": "1h",
    "strategy": "A",
    "last_n_days": 7
  }' | python3 -m json.tool
```

You can also specify explicit dates instead of `last_n_days`:

```bash
  -d '{
    "symbol": "ETHUSDT",
    "interval": "4h",
    "strategy": "B",
    "start": "2025-12-01",
    "end": "2025-12-31"
  }'
```

### 3. Approve

```bash
curl -s -X POST http://localhost:8000/approval/1/approve \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $TOKEN" \
  -d '{"reason": "Metrics look solid, proceed with paper test"}'
```

### 4. Execute Trade (Paper)

```bash
curl -s -X POST http://localhost:8000/trade/execute \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $TOKEN" \
  -d '{"approval_id": 1, "quantity": 0.01}' | python3 -m json.tool
```

---

## Project Structure

```
app/
├── main.py                  # FastAPI entry point, router mounting
├── config.py                # .env loader (pydantic-settings)
├── database.py              # SQLAlchemy engine / session
├── models.py                # ORM models (User, Recommendation, Approval, TradeExecution, AuditLog)
├── schemas.py               # Pydantic request/response schemas
├── auth/
│   ├── jwt_handler.py       # Password hashing (PBKDF2), JWT creation/decode
│   └── dependencies.py      # FastAPI dependencies (get_current_user, require_admin)
├── binance_client.py        # Async Binance kline fetcher with pagination + retry
├── strategies/
│   ├── strategy_a_arima.py  # AR(4) on differenced prices
│   ├── strategy_b_kalman.py # Kalman filter trend detector
│   └── strategy_c_ou_mean_reversion.py  # OU z-score mean-reversion
├── services/
│   ├── analysis_service.py  # Orchestrator: fetch → strategy → persist
│   ├── approval_service.py  # Approve / reject with audit logging
│   └── trade_service.py     # Paper / live execution gating
├── routes/
│   ├── auth_routes.py       # /auth/register, /auth/login
│   ├── recommendation_routes.py  # POST /recommendation
│   ├── approval_routes.py   # POST /approval/{id}/approve|reject
│   └── trade_routes.py      # POST /trade/execute
└── utils/
    ├── validation.py        # Time-period parsing
    └── retry.py             # HTTP retry with exponential backoff
tests/
├── test_strategies.py       # Unit tests for all three strategies
└── test_approval.py         # Approval gating + trade execution tests
```

---

## Safety Rules

- Live orders are **never** placed unless `LIVE_MODE=true` **and** the trade has an approved approval record **and** the caller is an admin.
- Even with `LIVE_MODE=true`, the current implementation returns simulated fills — actual Binance order signing is a placeholder. This is intentional for safety.
- API keys are never logged.
- All approval and trade actions are written to `audit_logs`.
