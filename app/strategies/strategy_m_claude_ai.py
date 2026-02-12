"""
Strategy M — Claude AI (Pure Market Analysis).

Sends raw market data to Anthropic's Claude API and receives a structured
trading signal.  No traditional technical indicators are computed; the model
analyses price series, log returns, volume, order-book depth, and temporal
features directly.

Three outputs:
  direction  (-100 to +100) → recommendation + confidence
  intensity  (  0  to  100) → force (strength role)
  confidence (  0  to  100) → stored in metrics

In-context learning: past predictions and their real-world outcomes are
included in the prompt so the model can adjust its analysis over time.
"""

from __future__ import annotations

import json
import logging
import math
import re
from datetime import datetime, timezone
from typing import Any

import numpy as np

from app.strategies._common import no_trade, simple_atr, compute_sl_tp

logger = logging.getLogger(__name__)

# ── Default parameters ────────────────────────────────────────────────────────

DEFAULTS: dict[str, Any] = {
    "claude_model": "claude-haiku-4-5-20251001",
    "temperature": 0.2,
    "max_tokens": 512,
    "lookback_candles": 50,
    "include_depth": True,
    "include_24hr": True,
    "history_window": 10,
    "direction_deadzone": 10,
}


# ── Feature extraction (raw data only, NO indicators) ────────────────────────

def _prepare_price_features(candles: np.ndarray, lookback: int) -> dict:
    """Extract raw numerical features from OHLCV candles."""
    n = min(lookback, len(candles))
    recent = candles[-n:]

    closes = recent[:, 4].astype(float)
    volumes = recent[:, 5].astype(float)
    highs = recent[:, 2].astype(float)
    lows = recent[:, 3].astype(float)
    times_ms = recent[:, 0].astype(float)

    # Log returns
    log_rets = np.diff(np.log(np.maximum(closes, 1e-10)))

    # Price acceleration (second derivative of close)
    first_diff = np.diff(closes)
    accel = np.diff(first_diff) if len(first_diff) > 1 else np.array([0.0])

    # Volume ratio vs simple moving average (20-period or available)
    vol_window = min(20, len(volumes))
    vol_sma = np.mean(volumes[-vol_window:]) if vol_window > 0 else 1.0
    vol_ratio = float(volumes[-1] / vol_sma) if vol_sma > 0 else 1.0

    # High-low range normalized by close
    hl_range = (highs - lows) / np.maximum(closes, 1e-10)

    # Time features from last candle
    last_ts = times_ms[-1] / 1000.0
    dt = datetime.fromtimestamp(last_ts, tz=timezone.utc)
    hour_utc = dt.hour
    day_of_week = dt.strftime("%A")

    # Return stats
    mean_ret = float(np.mean(log_rets)) if len(log_rets) > 0 else 0.0
    std_ret = float(np.std(log_rets)) if len(log_rets) > 0 else 0.0
    skew = _skewness(log_rets)
    kurt = _kurtosis(log_rets)

    return {
        "recent_closes": closes,
        "log_returns": log_rets,
        "acceleration": accel,
        "volumes": volumes,
        "volume_ratio": vol_ratio,
        "hl_range": hl_range,
        "hour_utc": hour_utc,
        "day_of_week": day_of_week,
        "mean_ret": mean_ret,
        "std_ret": std_ret,
        "skew": skew,
        "kurt": kurt,
    }


def _skewness(arr: np.ndarray) -> float:
    if len(arr) < 3:
        return 0.0
    m = np.mean(arr)
    s = np.std(arr)
    if s == 0:
        return 0.0
    return float(np.mean(((arr - m) / s) ** 3))


def _kurtosis(arr: np.ndarray) -> float:
    if len(arr) < 4:
        return 0.0
    m = np.mean(arr)
    s = np.std(arr)
    if s == 0:
        return 0.0
    return float(np.mean(((arr - m) / s) ** 4) - 3.0)


def _prepare_depth_features(depth_data: dict | None) -> dict | None:
    """Parse Binance order book into compact features."""
    if not depth_data:
        return None
    try:
        bids = depth_data.get("bids", [])
        asks = depth_data.get("asks", [])
        if not bids or not asks:
            return None

        bid_prices = [float(b[0]) for b in bids[:10]]
        bid_qtys = [float(b[1]) for b in bids[:10]]
        ask_prices = [float(a[0]) for a in asks[:10]]
        ask_qtys = [float(a[1]) for a in asks[:10]]

        best_bid = bid_prices[0] if bid_prices else 0
        best_ask = ask_prices[0] if ask_prices else 0
        mid = (best_bid + best_ask) / 2 if (best_bid + best_ask) > 0 else 1

        spread_pct = (best_ask - best_bid) / mid * 100 if mid > 0 else 0
        bid_total = sum(bid_qtys)
        ask_total = sum(ask_qtys)
        denom = bid_total + ask_total
        imbalance = (bid_total - ask_total) / denom if denom > 0 else 0

        top_bids = ", ".join(f"{p:.2f}×{q:.4f}" for p, q in zip(bid_prices[:5], bid_qtys[:5]))
        top_asks = ", ".join(f"{p:.2f}×{q:.4f}" for p, q in zip(ask_prices[:5], ask_qtys[:5]))

        return {
            "spread_pct": spread_pct,
            "bid_total": bid_total,
            "ask_total": ask_total,
            "imbalance": imbalance,
            "top_bids": top_bids,
            "top_asks": top_asks,
        }
    except Exception:
        return None


def _prepare_24hr_features(ticker_data: dict | None) -> dict | None:
    """Parse Binance 24hr ticker into compact features."""
    if not ticker_data:
        return None
    try:
        return {
            "price_change_pct": float(ticker_data.get("priceChangePercent", 0)),
            "high": float(ticker_data.get("highPrice", 0)),
            "low": float(ticker_data.get("lowPrice", 0)),
            "volume": float(ticker_data.get("volume", 0)),
            "quote_volume": float(ticker_data.get("quoteVolume", 0)),
            "trade_count": int(ticker_data.get("count", 0)),
            "vwap": float(ticker_data.get("weightedAvgPrice", 0)),
        }
    except Exception:
        return None


# ── Prompt construction ───────────────────────────────────────────────────────

def _compact_array(arr: np.ndarray, decimals: int = 4) -> str:
    """Format array as compact comma-separated string."""
    vals = arr[-30:] if len(arr) > 30 else arr  # max 30 values
    return ", ".join(f"{v:.{decimals}f}" for v in vals)


def _format_prompt(
    symbol: str,
    price_features: dict,
    depth_features: dict | None,
    ticker_features: dict | None,
    past_outcomes: list[dict],
) -> str:
    sections: list[str] = []

    # 1. System instruction
    sections.append(
        "Eres un analista cuantitativo de mercados financieros. "
        "Analiza los siguientes datos CRUDOS de mercado y produce una señal de trading. "
        "NO debes usar ni referenciar indicadores técnicos tradicionales "
        "(nada de EMA, RSI, MACD, Bollinger, ADX, etc.). "
        "Basa tu análisis SOLAMENTE en los datos numéricos crudos proporcionados: "
        "patrones de precio, distribución de retornos, comportamiento del volumen, "
        "estructura del libro de órdenes y patrones temporales.\n"
        "\n"
        "Responde SOLAMENTE con un objeto JSON (sin markdown, sin texto fuera del JSON):\n"
        '{"direction": <int -100 a +100>, "intensity": <int 0 a 100>, '
        '"confidence": <int 0 a 100>, "reasoning": "<explicación breve>"}\n'
        "\n"
        "direction: negativo=bajista, positivo=alcista, magnitud=fuerza de la señal\n"
        "intensity: qué tan agresivamente operar (0=no operar, 100=máxima posición)\n"
        "confidence: confianza estadística en tu análisis (0-100)\n"
    )

    # 2. Price data
    closes = price_features["recent_closes"]
    log_rets = price_features["log_returns"]
    sections.append(
        f"=== {symbol} DATOS DE PRECIO ===\n"
        f"Últimos {len(closes)} cierres: {_compact_array(closes, 2)}\n"
        f"Precio actual: {closes[-1]:.2f}\n"
        f"Retornos log (últimos 20): {_compact_array(log_rets[-20:], 6)}\n"
        f"Estadísticas retornos: media={price_features['mean_ret']:.6f} "
        f"std={price_features['std_ret']:.6f} "
        f"skew={price_features['skew']:.4f} "
        f"kurt={price_features['kurt']:.4f}\n"
        f"Aceleración precio (últimos 10): {_compact_array(price_features['acceleration'][-10:], 4)}\n"
        f"Volumen (últimos 20): {_compact_array(price_features['volumes'][-20:], 2)}\n"
        f"Ratio volumen (actual/SMA20): {price_features['volume_ratio']:.2f}\n"
        f"Rango (H-L)/C (últimos 10): {_compact_array(price_features['hl_range'][-10:], 6)}\n"
    )

    # 3. Time context
    sections.append(
        f"=== CONTEXTO TEMPORAL ===\n"
        f"Hora (UTC): {price_features['hour_utc']}\n"
        f"Día de semana: {price_features['day_of_week']}\n"
    )

    # 4. Order book + 24hr
    if depth_features:
        sections.append(
            f"=== LIBRO DE ÓRDENES ===\n"
            f"Spread: {depth_features['spread_pct']:.4f}%\n"
            f"Profundidad bids (total): {depth_features['bid_total']:.4f}\n"
            f"Profundidad asks (total): {depth_features['ask_total']:.4f}\n"
            f"Desbalance: {depth_features['imbalance']:.4f} "
            f"(+1=todas bids, -1=todas asks)\n"
            f"Top 5 bids: {depth_features['top_bids']}\n"
            f"Top 5 asks: {depth_features['top_asks']}\n"
        )

    if ticker_features:
        sections.append(
            f"=== ESTADÍSTICAS 24H ===\n"
            f"Cambio: {ticker_features['price_change_pct']:.2f}%\n"
            f"Máximo: {ticker_features['high']:.2f}  Mínimo: {ticker_features['low']:.2f}\n"
            f"Volumen: {ticker_features['volume']:.2f}\n"
            f"Trades: {ticker_features['trade_count']}\n"
            f"VWAP: {ticker_features['vwap']:.2f}\n"
        )

    # 5. Past outcomes (in-context learning)
    if past_outcomes:
        sections.append("=== PREDICCIONES PASADAS Y RESULTADOS ===\n")
        for po in past_outcomes:
            sections.append(
                f"  [{po['timestamp']}] Predicción: dir={po['direction']:+d} "
                f"int={po['intensity']} conf={po['confidence']} | "
                f"Real: {po['outcome']} ({po['return_pct']:+.2f}% en {po['horizon']})\n"
            )
        sections.append(
            "Aprende de estos resultados. Si tus predicciones pasadas fueron "
            "incorrectas, ajusta tu análisis.\n"
        )

    return "\n".join(sections)


# ── Claude API call (sync, via httpx) ─────────────────────────────────────────

def _call_claude_sync(
    api_key: str,
    model: str,
    prompt: str,
    temperature: float = 0.2,
    max_tokens: int = 512,
) -> str:
    """Synchronous POST to the Anthropic Messages API.

    Uses a short-lived sync httpx client (not the shared async one)
    because this runs inside a synchronous strategy function.
    """
    import httpx

    resp = httpx.post(
        "https://api.anthropic.com/v1/messages",
        headers={
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
        json={
            "model": model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": [{"role": "user", "content": prompt}],
        },
        timeout=30.0,
    )
    resp.raise_for_status()
    data = resp.json()
    return data["content"][0]["text"]


# ── Response parsing ──────────────────────────────────────────────────────────

def _parse_claude_response(text: str) -> dict | None:
    """Extract and validate the JSON response from Claude.

    Handles both raw JSON and JSON wrapped in markdown code fences.
    Returns None if parsing fails.
    """
    # Try to extract JSON from markdown fences
    md_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if md_match:
        text = md_match.group(1)

    # Try to find a JSON object
    json_match = re.search(r"\{[^{}]*\}", text, re.DOTALL)
    if not json_match:
        return None

    try:
        data = json.loads(json_match.group())
    except json.JSONDecodeError:
        return None

    # Validate required fields
    if not all(k in data for k in ("direction", "intensity", "confidence")):
        return None

    # Clamp values
    data["direction"] = max(-100, min(100, int(data["direction"])))
    data["intensity"] = max(0, min(100, int(data["intensity"])))
    data["confidence"] = max(0, min(100, int(data["confidence"])))
    data.setdefault("reasoning", "")

    return data


# ── Main strategy function ────────────────────────────────────────────────────

def run_strategy_m(candles: np.ndarray, params: dict | None = None) -> dict:
    """Claude AI strategy — pure market data analysis, no indicators.

    Parameters
    ----------
    candles : (N, 6) array — [time_ms, open, high, low, close, volume]
    params  : overrides + ``_market_data`` dict injected by eval engine

    Returns standard strategy dict.
    """
    p = {**DEFAULTS, **(params or {})}

    # Minimum data check
    if len(candles) < 10:
        return no_trade("Datos insuficientes (se necesitan al menos 10 velas)")

    # Extract injected market data
    market_data: dict = p.get("_market_data", {})
    api_key: str = market_data.get("anthropic_api_key", "") or ""

    if not api_key:
        return no_trade("No se configuró la API key de Anthropic (ANTHROPIC_API_KEY)")

    # Prepare features
    lookback = int(p["lookback_candles"])
    price_features = _prepare_price_features(candles, lookback)

    depth_features = None
    if p.get("include_depth"):
        depth_features = _prepare_depth_features(market_data.get("depth"))

    ticker_features = None
    if p.get("include_24hr"):
        ticker_features = _prepare_24hr_features(market_data.get("ticker_24hr"))

    symbol = market_data.get("symbol", "UNKNOWN")
    past_outcomes = market_data.get("past_outcomes", [])
    history_window = int(p.get("history_window", 10))
    past_outcomes = past_outcomes[-history_window:] if past_outcomes else []

    # Build prompt
    prompt = _format_prompt(symbol, price_features, depth_features,
                            ticker_features, past_outcomes)

    # Call Claude
    try:
        response_text = _call_claude_sync(
            api_key=api_key,
            model=str(p["claude_model"]),
            prompt=prompt,
            temperature=float(p["temperature"]),
            max_tokens=int(p["max_tokens"]),
        )
    except Exception as exc:
        logger.warning("Claude API error: %s", exc)
        return no_trade(f"Error en la API de Claude: {exc}")

    # Parse response
    parsed = _parse_claude_response(response_text)
    if parsed is None:
        logger.warning("Unparseable Claude response: %s", response_text[:200])
        return no_trade("No se pudo interpretar la respuesta de Claude")

    direction = parsed["direction"]
    intensity = parsed["intensity"]
    ai_confidence = parsed["confidence"]
    reasoning = parsed.get("reasoning", "")

    # Map to strategy interface
    deadzone = float(p["direction_deadzone"])
    if abs(direction) < deadzone:
        signal = "NO-TRADE"
        confidence = 0.0
    elif direction > 0:
        signal = "BUY"
        confidence = min(float(abs(direction)), 100.0)
    else:
        signal = "SELL"
        confidence = min(float(abs(direction)), 100.0)

    force = round(intensity / 100.0, 4)

    # Price levels
    last_close = float(candles[-1, 4])
    entry = last_close
    atr_val = simple_atr(candles)
    sl, tp = compute_sl_tp(signal, entry, atr_val, 1.5, 2.5)

    return {
        "recommendation": signal,
        "confidence": round(confidence, 1),
        "force": force,
        "entry": round(entry, 8) if signal != "NO-TRADE" else None,
        "stop_loss": sl,
        "take_profit": tp,
        "explanation": (
            f"Claude AI ({p['claude_model']}): dirección={direction:+d}, "
            f"intensidad={intensity}, confianza_IA={ai_confidence}%. "
            f"{reasoning[:300]}"
        ),
        "metrics": {
            "direction": direction,
            "intensity": intensity,
            "ai_confidence": ai_confidence,
            "model": str(p["claude_model"]),
            "reasoning": reasoning,
            "depth_available": depth_features is not None,
            "ticker_24hr_available": ticker_features is not None,
            "past_outcomes_count": len(past_outcomes),
        },
    }
