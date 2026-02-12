from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    # Binance
    binance_api_key: str = ""
    binance_api_secret: str = ""
    binance_base_url: str = "https://api.binance.com"
    binance_max_candles: int = 1000
    binance_retry_attempts: int = 3
    binance_retry_backoff: float = 1.0

    # JWT
    jwt_secret_key: str = "change-me"
    jwt_algorithm: str = "HS256"
    jwt_expire_minutes: int = 60

    # Database
    database_url: str = "sqlite:///./trading.db"

    # Trading
    commission_pct: float = 0.1  # Binance spot fee per trade (0.1% = standard)

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


@lru_cache()
def get_settings() -> Settings:
    return Settings()
