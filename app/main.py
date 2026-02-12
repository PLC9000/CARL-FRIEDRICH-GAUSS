"""Asistente de Decisiones de Trading — Punto de entrada FastAPI."""

import logging
import pathlib
import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from app.database import engine, Base
from app.routes.auth_routes import router as auth_router
from app.routes.recommendation_routes import router as rec_router
from app.routes.approval_routes import router as approval_router
from app.routes.trade_routes import router as trade_router
from app.routes.settings_routes import router as settings_router
from app.routes.recipe_routes import router as recipe_router
from app.routes.strategy_routes import router as strategy_router
from app.routes.setup_routes import router as setup_router
from app.routes.marketplace_routes import router as marketplace_router
from app.services.evaluation_engine import start_engine, stop_engine
from app.services.http_client import start_client, stop_client

# ── Logging ─────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

STATIC_DIR = pathlib.Path(__file__).resolve().parent / "static"

# ── Lifespan ───────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Start evaluation engine on startup, stop on shutdown."""
    Base.metadata.create_all(bind=engine)
    # Ensure new columns exist in SQLite (create_all won't add to existing tables)
    from sqlalchemy import text, inspect as sa_inspect
    with engine.connect() as conn:
        insp = sa_inspect(engine)
        cols = {c["name"] for c in insp.get_columns("recipes")}
        if "buy_order_type" not in cols:
            conn.execute(text("ALTER TABLE recipes ADD COLUMN buy_order_type VARCHAR(10) NOT NULL DEFAULT 'oco'"))
        if "sell_order_type" not in cols:
            conn.execute(text("ALTER TABLE recipes ADD COLUMN sell_order_type VARCHAR(10) NOT NULL DEFAULT 'oco'"))
        if "strength_threshold" not in cols:
            conn.execute(text("ALTER TABLE recipes ADD COLUMN strength_threshold REAL NOT NULL DEFAULT 0.0"))
        if "confirmation_minutes" not in cols:
            conn.execute(text("ALTER TABLE recipes ADD COLUMN confirmation_minutes REAL NOT NULL DEFAULT 0.0"))
        need_qty_migrate = "buy_quantity" not in cols
        if "buy_quantity" not in cols:
            conn.execute(text("ALTER TABLE recipes ADD COLUMN buy_quantity REAL"))
        if "sell_quantity" not in cols:
            conn.execute(text("ALTER TABLE recipes ADD COLUMN sell_quantity REAL"))
        if need_qty_migrate:
            conn.execute(text("UPDATE recipes SET buy_quantity = auto_quantity, sell_quantity = auto_quantity WHERE auto_quantity IS NOT NULL AND buy_quantity IS NULL"))
        if "confirmation_seconds" not in cols:
            conn.execute(text("ALTER TABLE recipes ADD COLUMN confirmation_seconds INTEGER NOT NULL DEFAULT 0"))
        if "mode" not in cols:
            conn.execute(text("ALTER TABLE recipes ADD COLUMN mode VARCHAR(20)"))
        if "auto_strength_threshold" not in cols:
            conn.execute(text("ALTER TABLE recipes ADD COLUMN auto_strength_threshold REAL NOT NULL DEFAULT 0.0"))
        # Per-strategy tracking columns on recipe_evaluations
        eval_cols = {c["name"] for c in insp.get_columns("recipe_evaluations")}
        if "direction_status" not in eval_cols:
            conn.execute(text("ALTER TABLE recipe_evaluations ADD COLUMN direction_status VARCHAR(10)"))
        if "strength_status" not in eval_cols:
            conn.execute(text("ALTER TABLE recipe_evaluations ADD COLUMN strength_status VARCHAR(10)"))
        if "direction_value" not in eval_cols:
            conn.execute(text("ALTER TABLE recipe_evaluations ADD COLUMN direction_value REAL"))
        if "strength_value" not in eval_cols:
            conn.execute(text("ALTER TABLE recipe_evaluations ADD COLUMN strength_value REAL"))
        # User preferences
        user_cols = {c["name"] for c in insp.get_columns("users")}
        if "auto_only" not in user_cols:
            conn.execute(text("ALTER TABLE users ADD COLUMN auto_only BOOLEAN NOT NULL DEFAULT 0"))
        conn.commit()
    from app.database import SessionLocal
    from app.services.setup_service import seed_strategy_configs
    db = SessionLocal()
    try:
        seed_strategy_configs(db)
    finally:
        db.close()
    await start_client()
    await start_engine()
    yield
    await stop_engine()
    await stop_client()


# ── App ─────────────────────────────────────────────────────────────────────
app = FastAPI(
    lifespan=lifespan,
    title="Carl Friedrich Gauss",
    version="2.0.0",
    description=(
        "Sistema de análisis y recomendaciones de trading conectado a Binance.\n\n"
        "Seleccioná un par de monedas, un período de tiempo y una de las tres "
        "estrategias matemáticas disponibles. El sistema te devuelve una "
        "recomendación **COMPRAR / VENDER / NO OPERAR** con niveles de entrada, "
        "stop-loss y take-profit.\n\n"
        "**Flujo de aprobación:** las recomendaciones las puede pedir cualquier "
        "usuario autenticado, pero para ejecutar un trade se necesita aprobación "
        "de un administrador. Por defecto se opera en modo papel (simulado).\n\n"
        "---\n\n"
        "### Estrategias disponibles\n\n"
        "| ID | Nombre | Qué hace |\n"
        "|---|---|---|\n"
        "| **A** | Autoregresivo (AR) | Ajusta un modelo AR(4) sobre precios diferenciados y proyecta los próximos 5 períodos |\n"
        "| **B** | Filtro de Kalman | Suaviza la serie de precios y detecta tendencia con filtro de volatilidad |\n"
        "| **C** | Reversión a la media (OU) | Usa z-score de retornos para detectar desviaciones extremas del precio promedio |\n"
    ),
    openapi_tags=[
        {
            "name": "Autenticación",
            "description": "Registro de usuarios e inicio de sesión. "
            "Acá obtenés el token JWT que necesitás para el resto de los endpoints.",
        },
        {
            "name": "Recomendaciones",
            "description": "Pedí un análisis para cualquier par listado en Binance. "
            "El sistema baja las velas, corre la estrategia elegida y te devuelve "
            "la recomendación con métricas y niveles de precio.",
        },
        {
            "name": "Aprobaciones",
            "description": "Flujo de aprobación para ejecutar trades. Solo administradores "
            "pueden aprobar o rechazar. Cada acción queda registrada en el log de auditoría.",
        },
        {
            "name": "Operaciones",
            "description": "Ejecución de trades (simulados por defecto). Requiere rol de admin "
            "y una aprobación previa. Si LIVE_MODE está desactivado, siempre se simula.",
        },
        {
            "name": "Recetas",
            "description": "Planificación de recetas de trading. Configurá estrategias, "
            "umbrales y gestión de riesgo. El motor de evaluación ejecuta las "
            "estrategias automáticamente según la frecuencia definida.",
        },
    ],
)

# ── Traducción de errores de validación ────────────────────────────────────

_FIELD_NAMES = {
    "symbol": "Par de trading", "strategies": "Estrategias",
    "strategy": "Estrategia", "weight": "Peso", "name": "Nombre",
    "interval": "Intervalo", "lookback_days": "Dias historicos",
    "buy_threshold": "Umbral de compra", "sell_threshold": "Umbral de venta",
    "max_order_pct": "Max orden %", "stop_loss_pct": "Stop-Loss %",
    "take_profit_pct": "Take-Profit %", "strategy_params": "Parametros",
    "username": "Usuario", "password": "Contrasena",
    "api_key": "API Key", "api_secret": "API Secret",
    "auto_threshold": "Umbral de auto-aprobacion",
    "quantity": "Cantidad", "approval_id": "ID aprobacion", "reason": "Motivo",
}

_MSG_TRANSLATIONS = [
    ("Field required", "campo obligatorio"),
    ("Value error, ", ""),
    ("Input should be", "debe ser"),
    ("String should have at least", "debe tener al menos"),
    ("String should have at most", "debe tener como maximo"),
    ("ensure this value is greater than", "debe ser mayor que"),
    ("ensure this value is less than", "debe ser menor que"),
    ("value is not a valid", "formato invalido para"),
    ("Input should be a valid number", "debe ser un numero valido"),
    ("Input should be a valid string", "debe ser texto valido"),
    ("Input should be a valid integer", "debe ser un numero entero"),
    ("List should have at least", "debe tener al menos"),
    ("List should have at most", "debe tener como maximo"),
    (" items", " elementos"),
    (" characters", " caracteres"),
    ("greater than or equal to", "mayor o igual a"),
    ("less than or equal to", "menor o igual a"),
    ("greater than", "mayor que"),
    ("less than", "menor que"),
    ("after validation, not", "despues de validar, no"),
]


def _translate_msg(msg: str) -> str:
    for en, es in _MSG_TRANSLATIONS:
        msg = msg.replace(en, es)
    return msg


def _translate_field(loc: list) -> str:
    parts = [p for p in loc if p != "body"]
    named = [_FIELD_NAMES.get(str(p), str(p)) for p in parts]
    return " > ".join(named) or "Campo"


@app.exception_handler(RequestValidationError)
async def validation_error_handler(_request: Request, exc: RequestValidationError):
    errors = []
    for e in exc.errors():
        field = _translate_field(e.get("loc", []))
        msg = _translate_msg(e.get("msg", "valor invalido"))
        errors.append({"campo": field, "mensaje": msg})

    readable = ". ".join(f"{e['campo']}: {e['mensaje']}" for e in errors)
    return JSONResponse(
        status_code=422,
        content={"detail": readable},
    )


# Montar routers de API
app.include_router(auth_router)
app.include_router(rec_router)
app.include_router(approval_router)
app.include_router(trade_router)
app.include_router(settings_router)
app.include_router(recipe_router)
app.include_router(strategy_router)
app.include_router(setup_router)
app.include_router(marketplace_router)


@app.get("/health", summary="Verificar estado del servicio", tags=["Sistema"])
def health():
    """Devuelve `ok` si el servidor está levantado y funcionando."""
    return {"estado": "ok"}


# ── Frontend ────────────────────────────────────────────────────────────────

@app.get("/", include_in_schema=False)
def serve_frontend():
    html = (STATIC_DIR / "index.html").read_text(encoding="utf-8")
    return HTMLResponse(
        content=html,
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache",
            "Expires": "0",
        },
    )


app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
