from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.database import get_db
from app.models import User, AuditLog
from app.schemas import BinanceConfigRequest, BinanceConfigOut
from app.auth.dependencies import get_current_user
from app.auth.encryption import encrypt, decrypt
from app.services.binance_account_service import get_api_permissions
from app.services.http_client import get_client

router = APIRouter(prefix="/settings", tags=["Configuración"])


@router.post(
    "/binance",
    response_model=BinanceConfigOut,
    summary="Guardar API keys de Binance",
)
def save_binance_config(
    body: BinanceConfigRequest,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Guarda tus API keys de Binance encriptadas en la base de datos.

    Las keys se encriptan con AES (Fernet) usando la clave secreta del servidor.
    Nunca se almacenan ni se muestran en texto plano.

    **Recomendación:** creá API keys con permisos de solo lectura + spot trading.
    No habilites retiros.
    """
    user.binance_api_key_enc = encrypt(body.api_key.strip())
    user.binance_api_secret_enc = encrypt(body.api_secret.strip())
    db.add(AuditLog(
        user_id=user.id,
        action="settings.binance_keys_saved",
        payload={"api_key_hint": body.api_key[:6] + "..."},
    ))
    db.commit()
    return BinanceConfigOut(
        configured=True,
        api_key_hint=body.api_key[:6] + "..." if len(body.api_key) >= 6 else "***",
    )


@router.get(
    "/binance",
    response_model=BinanceConfigOut,
    summary="Ver estado de la configuración de Binance",
)
def get_binance_config(
    user: User = Depends(get_current_user),
):
    """Devuelve si tenés API keys configuradas y los primeros caracteres para verificar."""
    if not user.binance_api_key_enc:
        return BinanceConfigOut(configured=False, api_key_hint="")
    try:
        raw_key = decrypt(user.binance_api_key_enc)
        hint = raw_key[:6] + "..." if len(raw_key) >= 6 else "***"
    except Exception:
        hint = "error"
    return BinanceConfigOut(configured=True, api_key_hint=hint)


@router.delete(
    "/binance",
    summary="Eliminar API keys de Binance",
)
def delete_binance_config(
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Borra las API keys guardadas. Después de esto no se podrán hacer órdenes en vivo."""
    user.binance_api_key_enc = ""
    user.binance_api_secret_enc = ""
    db.add(AuditLog(
        user_id=user.id,
        action="settings.binance_keys_deleted",
        payload={},
    ))
    db.commit()
    return {"detail": "API keys eliminadas"}


@router.get(
    "/binance/permissions",
    summary="Ver permisos de la API key de Binance",
)
async def get_binance_permissions(
    user: User = Depends(get_current_user),
):
    """Consulta a Binance qué permisos tiene tu API key configurada.

    Devuelve flags como lectura, spot trading, retiros, margin, futuros, etc.
    """
    if not user.binance_api_key_enc:
        raise HTTPException(status_code=400, detail="No tenés API keys configuradas")
    try:
        perms = await get_api_permissions(
            user.binance_api_key_enc,
            user.binance_api_secret_enc,
        )
        return perms
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=str(exc))


# ── Server WAN IP ────────────────────────────────────────────────────

@router.get(
    "/server-ip",
    summary="Ver IP pública del servidor",
)
async def get_server_ip(
    user: User = Depends(get_current_user),
):
    """Devuelve la IP pública (WAN) del servidor donde corre la aplicación.

    Usá esta IP para configurar la restricción de IP en tu API key de Binance.
    """
    try:
        client = get_client()
        resp = await client.get("https://api.ipify.org?format=json")
        data = resp.json()
        return {"server_ip": data["ip"]}
    except Exception:
        raise HTTPException(
            status_code=502,
            detail="No se pudo obtener la IP pública del servidor",
        )


# ── Auto-only preference ──────────────────────────────────────────────

@router.get("/auto-only", summary="Estado de modo Solo Automáticas")
def get_auto_only(user: User = Depends(get_current_user)):
    """Devuelve si el usuario tiene activado el modo Solo Automáticas."""
    return {"auto_only": bool(getattr(user, "auto_only", False))}


@router.post("/auto-only", summary="Cambiar modo Solo Automáticas")
def set_auto_only(
    body: dict,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Activa/desactiva el modo Solo Automáticas.

    Cuando está activo, solo se ejecutan órdenes auto-aprobadas.
    Las señales que requerirían aprobación manual se ignoran.
    """
    new_val = bool(body.get("auto_only", False))
    user.auto_only = new_val
    db.add(AuditLog(
        user_id=user.id,
        action="settings.auto_only",
        payload={"auto_only": new_val},
    ))
    db.commit()
    return {"auto_only": new_val}
