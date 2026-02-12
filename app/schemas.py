from __future__ import annotations

import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


# ── Autenticación ──────────────────────────────────────────────────────────

class UserCreate(BaseModel):
    """Datos para registrar un usuario nuevo."""
    username: str = Field(min_length=3, max_length=50, description="Nombre de usuario único. Entre 3 y 50 caracteres, sin espacios.", examples=["juan_trader"])
    password: str = Field(min_length=6, max_length=128, description="Contraseña. Mínimo 6 caracteres. Se guarda hasheada, nunca en texto plano.", examples=["miClave123"])
    role: Literal["user", "admin"] = Field(default="user", description="Rol del usuario. 'user' puede pedir recomendaciones; 'admin' además puede aprobar y ejecutar trades.")


class UserOut(BaseModel):
    """Datos públicos de un usuario registrado."""
    id: int = Field(description="Identificador único del usuario.")
    username: str = Field(description="Nombre de usuario.")
    role: str = Field(description="Rol asignado: 'user' o 'admin'.")
    has_binance_keys: bool = Field(description="Si el usuario tiene configuradas sus API keys de Binance.")
    created_at: datetime.datetime = Field(description="Fecha y hora de registro (UTC).")
    model_config = {"from_attributes": True}


class Token(BaseModel):
    """Token de acceso JWT devuelto tras un login exitoso."""
    access_token: str = Field(description="Token JWT. Usalo en el header 'Authorization: Bearer <token>'.")
    token_type: str = Field(default="bearer", description="Tipo de token. Siempre 'bearer'.")


class LoginRequest(BaseModel):
    """Credenciales para iniciar sesión."""
    username: str = Field(description="Tu nombre de usuario registrado.")
    password: str = Field(description="Tu contraseña.")


# ── Configuración Binance ──────────────────────────────────────────────────

class BinanceConfigRequest(BaseModel):
    """API keys de Binance. Se guardan encriptadas en la base de datos."""
    api_key: str = Field(description="Tu API Key de Binance. Recomendamos permisos solo de lectura + spot trading.", examples=["vmPUZ..."])
    api_secret: str = Field(description="Tu API Secret de Binance. Nunca se muestra en texto plano después de guardarlo.", examples=["NhqPt..."])


class BinanceConfigOut(BaseModel):
    """Estado de la configuración de Binance."""
    configured: bool = Field(description="Si tenés API keys configuradas.")
    api_key_hint: str = Field(description="Primeros 6 caracteres de tu API key (para verificación).")


# ── Recomendación ──────────────────────────────────────────────────────────

VALID_INTERVALS = {"1m", "5m", "15m", "1h", "4h", "1d"}


class RecommendationRequest(BaseModel):
    """Parámetros para solicitar una recomendación de trading.

    Podés definir el período con `last_n_days` (más sencillo) o con `start`/`end`.
    """
    symbol: str = Field(examples=["BTCUSDT"], description="Par de trading en Binance. Ej: BTCUSDT, ETHFDUSD.")
    interval: str = Field(examples=["1h"], description="Intervalo de cada vela: 1m, 5m, 15m, 1h, 4h, 1d.")
    strategy: Literal["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K"] = Field(description="Estrategia: A-I.")
    last_n_days: int | None = Field(default=None, ge=1, le=365, description="Días hacia atrás a analizar.", examples=[7])
    start: str | None = Field(default=None, description="Fecha inicio (AAAA-MM-DD).", examples=["2025-01-01"])
    end: str | None = Field(default=None, description="Fecha fin (AAAA-MM-DD).", examples=["2025-01-31"])
    params: dict[str, float] | None = Field(
        default=None,
        description=(
            "Parámetros avanzados de la estrategia (opcional). "
            "A: ar_order, forecast_horizon, return_threshold, sl_multiplier, tp_multiplier. "
            "B: process_noise, measurement_noise, trend_threshold, volatility_cap, sl_multiplier, tp_multiplier. "
            "C: lookback, z_threshold, vol_regime_multiplier, sl_multiplier, tp_multiplier."
        ),
        examples=[{"sl_multiplier": 2.0, "tp_multiplier": 3.0}],
    )

    @field_validator("interval")
    @classmethod
    def validate_interval(cls, v: str) -> str:
        if v not in VALID_INTERVALS:
            raise ValueError(f"El intervalo debe ser uno de: {VALID_INTERVALS}")
        return v

    @field_validator("symbol")
    @classmethod
    def validate_symbol(cls, v: str) -> str:
        v = v.upper().strip()
        if not v.isalnum() or len(v) < 2 or len(v) > 20:
            raise ValueError("Formato de símbolo inválido.")
        return v


class LevelsOut(BaseModel):
    """Niveles de precio sugeridos."""
    entry: float | None = Field(description="Precio de entrada sugerido.")
    stop_loss: float | None = Field(description="Stop-loss: precio para limitar pérdidas.")
    take_profit: float | None = Field(description="Take-profit: precio objetivo para tomar ganancias.")


class ReturnsOut(BaseModel):
    """Retornos potenciales considerando comisiones de Binance."""
    commission_pct: float = Field(description="Comisión por operación (%). Ej: 0.1 = Binance estándar.")
    total_fees_pct: float = Field(description="Comisiones totales ida+vuelta (%). Ej: 0.2 = entrada + salida.")
    gross_profit_pct: float | None = Field(description="Ganancia bruta si se alcanza el take-profit (%).")
    net_profit_pct: float | None = Field(description="Ganancia neta descontando comisiones (%).")
    gross_loss_pct: float | None = Field(description="Pérdida bruta si se toca el stop-loss (%).")
    net_loss_pct: float | None = Field(description="Pérdida neta incluyendo comisiones (%).")
    risk_reward: float | None = Field(description="Ratio riesgo/recompensa. >1 = más ganancia que pérdida potencial.")


class RecommendationOut(BaseModel):
    """Resultado completo del análisis."""
    id: int = Field(description="ID de la recomendación.")
    symbol: str = Field(description="Par analizado.")
    interval: str = Field(description="Intervalo de vela.")
    period: dict[str, str] = Field(description="Período analizado (start/end).")
    strategy: str = Field(description="Estrategia usada: A, B o C.")
    metrics: dict[str, Any] | None = Field(description="Métricas internas de la estrategia.")
    recommendation: str = Field(description="Recomendación: BUY, SELL o NO-TRADE.")
    confidence: float = Field(description="Confianza 0-100.")
    levels: LevelsOut = Field(description="Niveles de precio sugeridos.")
    returns: ReturnsOut = Field(description="Retornos potenciales con comisiones descontadas.")
    explanation: str | None = Field(description="Explicación de la recomendación.")
    model_config = {"from_attributes": True}


# ── Aprobación ─────────────────────────────────────────────────────────────

class ApprovalAction(BaseModel):
    """Datos para aprobar o rechazar."""
    reason: str = Field(default="", max_length=500, description="Motivo. Queda en el log de auditoría.", examples=["Métricas sólidas, aprobado para papel"])


class ApprovalOut(BaseModel):
    """Estado de la aprobación."""
    id: int = Field(description="ID de la aprobación.")
    recommendation_id: int = Field(description="ID de la recomendación asociada.")
    status: str = Field(description="Estado: pending, approved, rejected.")
    reviewed_by: int | None = Field(description="ID del admin que revisó.")
    review_reason: str | None = Field(description="Motivo del admin.")
    reviewed_at: datetime.datetime | None = Field(description="Fecha de revisión.")
    created_at: datetime.datetime = Field(description="Fecha de creación.")
    model_config = {"from_attributes": True}


# ── Operaciones ────────────────────────────────────────────────────────────

class TradeExecuteRequest(BaseModel):
    """Datos para ejecutar un trade aprobado."""
    approval_id: int = Field(description="ID de la aprobación (debe estar en 'approved').")
    quantity: float = Field(gt=0, description="Cantidad en la moneda base (ej: 0.01 BTC).", examples=[0.01])
    order_type: Literal["market", "oco", "limit"] = Field(
        default="market",
        description=(
            "Tipo de orden:\n"
            "- **market**: compra/venta a precio de mercado.\n"
            "- **oco**: compra a mercado + orden OCO automática "
            "(take-profit y stop-loss en una sola orden que se cancelan entre sí)."
        ),
    )


class TradeExecutionOut(BaseModel):
    """Resultado de la ejecución."""
    id: int = Field(description="ID de la ejecución.")
    approval_id: int = Field(description="ID de la aprobación.")
    order_type: str = Field(description="Tipo: market u oco.")
    symbol: str = Field(description="Par operado.")
    side: str = Field(description="BUY o SELL.")
    quantity: float | None = Field(description="Cantidad operada.")
    price: float | None = Field(description="Precio de entrada.")
    stop_loss: float | None = Field(description="Stop-loss (solo OCO).")
    take_profit: float | None = Field(description="Take-profit (solo OCO).")
    status: str = Field(description="Estado: filled o failed.")
    result: dict[str, Any] | None = Field(description="Detalle del resultado de Binance.")
    executed_at: datetime.datetime = Field(description="Fecha de ejecución.")
    model_config = {"from_attributes": True}


class TradeOperationEnrichedOut(BaseModel):
    """Operación enriquecida con datos pre-computados para la tabla."""
    id: int
    approval_id: int
    order_type: str
    symbol: str
    side: str
    quantity: float | None = None
    price: float | None = None
    stop_loss: float | None = None
    take_profit: float | None = None
    status: str
    result: dict[str, Any] | None = None
    executed_at: datetime.datetime
    signal_price: float | None = None
    filled_price: float | None = None
    total_commission: float | None = None
    commission_asset: str | None = None
    recipe_name: str | None = None
    recipe_id: int | None = None
    confidence: float | None = None
    oco_status: str | None = None
    exit_type: str | None = None
    exit_price: float | None = None
    exit_time: str | None = None
    pnl_pct: float | None = None
    net_pnl_pct: float | None = None
    net_pnl_abs: float | None = None
    is_closed: bool = False
    is_active: bool = False
    closed_manually: bool = False
    binance_order_id: int | None = None
    binance_order_list_id: int | None = None
    validation_status: str | None = None
    validation_reason: str | None = None
    error_message: str | None = None
    filter_adjustments: list | None = None
    model_config = {"from_attributes": True}


class TradeTrackingOut(BaseModel):
    """Seguimiento en tiempo real de una operación ejecutada."""
    trade_id: int = Field(description="ID de la operación.")
    symbol: str = Field(description="Par operado.")
    side: str = Field(description="BUY o SELL.")
    entry_price: float = Field(description="Precio de entrada.")
    current_price: float = Field(description="Precio actual de Binance.")
    pnl_pct: float = Field(description="P&L en porcentaje.")
    pnl_abs: float = Field(description="P&L en USD.")
    stop_loss: float | None = Field(description="Stop-loss de la operación.")
    take_profit: float | None = Field(description="Take-profit de la operación.")
    sl_distance_pct: float | None = Field(description="Distancia al SL en % desde entry.")
    tp_distance_pct: float | None = Field(description="Distancia al TP en % desde precio actual.")
    tp_progress_pct: float | None = Field(description="Progreso hacia TP (0=entry, 100=TP).")
    evaluation: str = Field(description="EN GANANCIA, EN PERDIDA, SL ALCANZADO, TP ALCANZADO, NEUTRAL.")
    elapsed: str = Field(description="Tiempo transcurrido desde ejecución.")
    timestamp: str = Field(description="Momento de la consulta (ISO).")
    oco_status: str | None = Field(default=None, description="Estado OCO si aplica: EXEC_STARTED o ALL_DONE.")
    exit_type: str | None = Field(default=None, description="TP o SL si OCO completada.")
    exit_price: float | None = Field(default=None, description="Precio real de salida si OCO completada.")


class OCOStatusOut(BaseModel):
    """Estado de una orden OCO consultada contra Binance."""
    trade_id: int = Field(description="ID de la operación.")
    oco_status: str = Field(description="Estado OCO: EXEC_STARTED (activa) o ALL_DONE (completada).")
    exit_type: str | None = Field(default=None, description="TP o SL (solo cuando ALL_DONE).")
    exit_price: float | None = Field(default=None, description="Precio de salida (solo cuando ALL_DONE).")
    exit_time: str | None = Field(default=None, description="Timestamp de salida ISO (solo cuando ALL_DONE).")


class TradePricePathOut(BaseModel):
    """Historial de precios para graficar una operación cerrada."""
    trade_id: int
    closes: list[float]
    times: list[int]
    entry_price: float | None
    exit_price: float | None
    stop_loss: float | None
    take_profit: float | None
    entry_time: str
    exit_time: str
    exit_type: str | None
    side: str


# ── Analíticas ────────────────────────────────────────────────────────────

class TradeSummaryOut(BaseModel):
    """Resumen general de P&L."""
    total_trades: int = Field(description="Total de operaciones cerradas.")
    wins: int = Field(description="Operaciones ganadoras (TP).")
    losses: int = Field(description="Operaciones perdedoras (SL).")
    win_rate: float = Field(description="Tasa de acierto 0-100.")
    gross_pnl: float = Field(description="P&L bruto total (USD).")
    total_commissions: float = Field(description="Comisiones totales pagadas (USD).")
    net_pnl: float = Field(description="P&L neto total (USD).")
    avg_win: float = Field(description="Ganancia promedio por trade ganador (USD).")
    avg_loss: float = Field(description="Pérdida promedio por trade perdedor (USD).")
    profit_factor: float | None = Field(description="Ganancias brutas / pérdidas brutas.")
    best_trade: float = Field(description="Mejor operación (USD).")
    worst_trade: float = Field(description="Peor operación (USD).")


class PeriodPnlOut(BaseModel):
    """P&L para un período (día o mes)."""
    period: str = Field(description="Período: AAAA-MM-DD (diario) o AAAA-MM (mensual).")
    trades: int = Field(description="Cantidad de operaciones.")
    wins: int = Field(description="Ganadoras.")
    losses: int = Field(description="Perdedoras.")
    gross_pnl: float = Field(description="P&L bruto (USD).")
    net_pnl: float = Field(description="P&L neto (USD).")
    cumulative: float = Field(description="P&L neto acumulado (USD).")


class AnalyticsOut(BaseModel):
    """Analíticas completas de P&L."""
    summary: TradeSummaryOut = Field(description="Resumen general.")
    daily: list[PeriodPnlOut] = Field(description="Desglose diario.")
    monthly: list[PeriodPnlOut] = Field(description="Desglose mensual.")


# ── Recetas ──────────────────────────────────────────────────────────────

class StrategyConfig(BaseModel):
    """Configuración de una estrategia dentro de una receta."""
    strategy: Literal["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K"] = Field(
        description="ID de estrategia: A-K.")
    weight: float = Field(default=0, ge=0, le=100,
        description="Peso en porcentaje (0-100). La suma de pesos debe ser 100%.")
    role: Literal["direction", "strength"] = Field(default="direction",
        description="Rol: 'direction' (determina compra/venta) o 'strength' (amplifica/atenúa).")


class RecipeCreate(BaseModel):
    """Datos para crear una receta de trading."""
    name: str = Field(min_length=1, max_length=100,
        description="Nombre de la receta.", examples=["BTC Tendencia 1h"])
    mode: Literal["weighted", "roles"] | None = Field(default=None,
        description="Modo: 'weighted' (ponderado) o 'roles' (roles independientes).")
    symbol: str = Field(examples=["BTCUSDT"],
        description="Par de trading en Binance.")
    strategies: list[StrategyConfig] = Field(min_length=1, max_length=5,
        description="De 1 a 5 estrategias. Los pesos deben sumar 100%.")
    interval: str = Field(examples=["1h"],
        description="Intervalo de velas: 1m, 5m, 15m, 1h, 4h, 1d.")
    lookback_days: int = Field(default=7, ge=1, le=365,
        description="Días de datos históricos a analizar.")
    buy_threshold: float = Field(default=0.5, ge=0.1, le=1.0,
        description="Umbral mínimo de score para señal de COMPRA.")
    sell_threshold: float = Field(default=0.5, ge=0.1, le=1.0,
        description="Umbral mínimo de score (absoluto) para señal de VENTA.")
    auto_threshold: float = Field(default=0.0, ge=0.0, le=1.0,
        description="Umbral de auto-aprobación. Si |score| >= este valor, la aprobación es automática. 0 = desactivado.")
    auto_quantity: float | None = Field(default=None, ge=0,
        description="(Legacy) Cantidad única. Usar buy_quantity/sell_quantity.")
    buy_quantity: float | None = Field(default=None, ge=0,
        description="Cantidad a comprar automáticamente cuando se auto-aprueba. None/0 = solo aprueba, no ejecuta.")
    sell_quantity: float | None = Field(default=None, ge=0,
        description="Cantidad a vender automáticamente cuando se auto-aprueba. None/0 = solo aprueba, no ejecuta.")
    auto_order_type: str = Field(default="oco",
        description="(Legacy) Tipo de orden por defecto.")
    buy_order_type: str = Field(default="oco",
        description="Tipo de orden para señal de COMPRA: 'market', 'oco' o 'limit'.")
    sell_order_type: str = Field(default="oco",
        description="Tipo de orden para señal de VENTA: 'market', 'oco' o 'limit'.")
    strength_threshold: float = Field(default=0.0, ge=0.0, le=1.0,
        description="Umbral mínimo de fuerza para disparar señal (aprobación). 0 = sin gate de fuerza.")
    auto_strength_threshold: float = Field(default=0.0, ge=0.0, le=1.0,
        description="Umbral de fuerza para auto-ejecución. Si fuerza >= este valor + dirección >= auto_threshold → auto. 0 = usa strength_threshold.")
    turbo_threshold: float = Field(default=0.0, ge=0.0, le=1.0,
        description="Umbral de turbo. Si |score| >= este valor, el engine evalúa cada 15s en vez de esperar el intervalo. 0 = desactivado.")
    confirmation_minutes: float = Field(default=0.0, ge=0.0, le=1440,
        description="Minutos de confirmación (legacy). Usar confirmation_seconds para mayor precision. 0 = disparo inmediato.")
    confirmation_seconds: int = Field(default=0, ge=0, le=86400,
        description="Segundos de confirmación. La señal debe mantenerse durante este período. Tiene prioridad sobre confirmation_minutes. 0 = disparo inmediato.")
    max_order_pct: float = Field(default=5.0, ge=0.1, le=100,
        description="Tamaño máximo de orden (% del balance).")
    stop_loss_pct: float = Field(default=2.0, ge=0.1, le=50,
        description="Stop-loss en % desde el precio de entrada.")
    take_profit_pct: float = Field(default=4.0, ge=0.1, le=100,
        description="Take-profit en % desde el precio de entrada.")
    max_ops_count: int = Field(default=0, ge=0, le=1000,
        description="Máximo de operaciones permitidas en la ventana de tiempo. 0 = sin límite.")
    max_ops_hours: float = Field(default=24.0, ge=0.1, le=720,
        description="Ventana de tiempo en horas para contar operaciones.")
    strategy_params: dict[str, dict[str, Any]] | None = Field(default=None,
        description="Parámetros avanzados por estrategia: {'A': {'ar_order': 6}, 'D': {'ma_type': 'SMA'}, ...}")

    @field_validator("interval")
    @classmethod
    def validate_interval(cls, v: str) -> str:
        if v not in VALID_INTERVALS:
            raise ValueError(f"El intervalo debe ser uno de: {VALID_INTERVALS}")
        return v

    @field_validator("symbol")
    @classmethod
    def validate_symbol(cls, v: str) -> str:
        v = v.upper().strip()
        if not v.isalnum() or len(v) < 2 or len(v) > 20:
            raise ValueError("Formato de símbolo inválido.")
        return v


class RecipeUpdate(BaseModel):
    """Actualización parcial de una receta."""
    name: str | None = Field(default=None, min_length=1, max_length=100)
    mode: Literal["weighted", "roles"] | None = None
    symbol: str | None = Field(default=None)
    strategies: list[StrategyConfig] | None = Field(
        default=None, max_length=5,
        description="De 1 a 5 estrategias. Los pesos deben sumar 100%.")
    interval: str | None = None
    lookback_days: int | None = Field(default=None, ge=1, le=365)
    buy_threshold: float | None = Field(default=None, ge=0.1, le=1.0)
    sell_threshold: float | None = Field(default=None, ge=0.1, le=1.0)
    auto_threshold: float | None = Field(default=None, ge=0.0, le=1.0)
    auto_quantity: float | None = Field(default=None, ge=0)
    buy_quantity: float | None = Field(default=None, ge=0)
    sell_quantity: float | None = Field(default=None, ge=0)
    auto_order_type: str | None = Field(default=None)
    buy_order_type: str | None = Field(default=None)
    sell_order_type: str | None = Field(default=None)
    strength_threshold: float | None = Field(default=None, ge=0.0, le=1.0)
    auto_strength_threshold: float | None = Field(default=None, ge=0.0, le=1.0)
    turbo_threshold: float | None = Field(default=None, ge=0.0, le=1.0)
    confirmation_minutes: float | None = Field(default=None, ge=0.0, le=1440)
    confirmation_seconds: int | None = Field(default=None, ge=0, le=86400)
    max_order_pct: float | None = Field(default=None, ge=0.1, le=100)
    stop_loss_pct: float | None = Field(default=None, ge=0.1, le=50)
    take_profit_pct: float | None = Field(default=None, ge=0.1, le=100)
    max_ops_count: int | None = Field(default=None, ge=0, le=1000)
    max_ops_hours: float | None = Field(default=None, ge=0.1, le=720)
    strategy_params: dict[str, dict[str, Any]] | None = None

    @field_validator("symbol")
    @classmethod
    def validate_symbol(cls, v: str | None) -> str | None:
        if v is not None:
            v = v.upper().strip()
            if not v.isalnum() or len(v) < 2 or len(v) > 20:
                raise ValueError("Formato de símbolo inválido.")
        return v

    @field_validator("interval")
    @classmethod
    def validate_interval(cls, v: str | None) -> str | None:
        if v is not None and v not in VALID_INTERVALS:
            raise ValueError(f"El intervalo debe ser uno de: {VALID_INTERVALS}")
        return v


class RecipeOut(BaseModel):
    """Receta completa."""
    id: int
    name: str
    mode: str | None = None
    symbol: str
    strategies: list[dict[str, Any]]
    interval: str
    lookback_days: int
    buy_threshold: float
    sell_threshold: float
    auto_threshold: float
    auto_quantity: float | None
    buy_quantity: float | None
    sell_quantity: float | None
    auto_order_type: str
    buy_order_type: str
    sell_order_type: str
    strength_threshold: float
    auto_strength_threshold: float
    turbo_threshold: float
    confirmation_minutes: float
    confirmation_seconds: int
    max_order_pct: float
    stop_loss_pct: float
    take_profit_pct: float
    max_ops_count: int = 0
    max_ops_hours: float = 24.0
    ops_used: int = 0
    strategy_params: dict[str, Any] | None
    status: str
    last_evaluated_at: datetime.datetime | None
    last_score: float | None = None
    last_signal: str | None = None
    last_strategy_results: list[dict[str, Any]] | None = None
    last_strength_factor: float | None = None
    last_direction_status: str | None = None
    last_strength_status: str | None = None
    last_triggered: bool | None = None
    confirmation_elapsed_secs: float | None = None
    confirmation_required_secs: int | None = None
    created_at: datetime.datetime
    updated_at: datetime.datetime
    model_config = {"from_attributes": True}


class RecipeEvaluationOut(BaseModel):
    """Resultado de una evaluación de receta."""
    id: int
    recipe_id: int
    strategy_results: list[dict[str, Any]]
    final_score: float
    signal: str
    direction_status: str | None = None
    strength_status: str | None = None
    direction_value: float | None = None
    strength_value: float | None = None
    triggered: bool
    recommendation_id: int | None
    evaluated_at: datetime.datetime
    model_config = {"from_attributes": True}


class PendingApprovalOut(BaseModel):
    """Aprobación pendiente o auto-aprobada sin ejecutar."""
    approval_id: int
    recommendation_id: int
    symbol: str
    strategies_used: str
    final_score: float | None
    recommendation: str
    confidence: float
    entry_price: float | None
    stop_loss: float | None
    take_profit: float | None
    explanation: str | None
    recipe_name: str | None
    order_type: str | None = None
    status: str = "pending"
    created_at: datetime.datetime
    model_config = {"from_attributes": True}


class ApprovalHistoryOut(PendingApprovalOut):
    """Aprobación histórica (aprobada, rechazada o auto-aprobada)."""
    status: str
    review_reason: str | None
    reviewed_at: datetime.datetime | None


# ── Marketplace ───────────────────────────────────────────────────────

class RecipeTradeOut(BaseModel):
    """Trade individual cerrado de una receta."""
    trade_id: int
    symbol: str
    side: str
    entry_price: float
    exit_price: float
    quantity: float | None = None
    exit_type: str
    pnl_pct: float
    net_pnl_abs: float
    executed_at: str
    exit_time: str

class RecipeRealStatsOut(BaseModel):
    """Estadísticas reales de operaciones cerradas para una receta."""
    recipe_id: int
    recipe_name: str
    symbol: str
    interval: str
    strategies: list[dict[str, Any]]
    mode: str | None
    total_trades: int
    wins: int
    losses: int
    win_rate: float
    net_pnl: float
    gross_pnl: float
    avg_win: float
    avg_loss: float
    profit_factor: float | None = None
    best_trade: float
    worst_trade: float
    trades: list[RecipeTradeOut]

class CompareRequest(BaseModel):
    """Request para comparar recetas."""
    recipe_ids: list[int] = Field(min_length=2)
