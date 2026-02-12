from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.database import get_db
from app.models import User, AuditLog
from app.schemas import UserCreate, UserOut, LoginRequest, Token
from app.auth.jwt_handler import hash_password, verify_password, create_access_token
from app.auth.dependencies import get_current_user

router = APIRouter(prefix="/auth", tags=["Autenticación"])


def _user_out(user: User) -> UserOut:
    return UserOut(
        id=user.id,
        username=user.username,
        role=user.role.value,
        has_binance_keys=bool(user.binance_api_key_enc),
        created_at=user.created_at,
    )


@router.post(
    "/register",
    response_model=UserOut,
    status_code=status.HTTP_201_CREATED,
    summary="Registrar usuario nuevo",
)
def register(body: UserCreate, db: Session = Depends(get_db)):
    """Crea una cuenta nueva en el sistema.

    - Con rol **user** podés pedir recomendaciones de trading.
    - Con rol **admin** además podés aprobar/rechazar y ejecutar trades.
    """
    if db.query(User).filter(User.username == body.username).first():
        raise HTTPException(status_code=400, detail="Ese nombre de usuario ya está en uso")

    user = User(
        username=body.username,
        hashed_password=hash_password(body.password),
        role=body.role,
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return _user_out(user)


@router.post(
    "/login",
    response_model=Token,
    summary="Iniciar sesión",
)
def login(body: LoginRequest, db: Session = Depends(get_db)):
    """Devuelve un token JWT que dura 60 minutos (configurable)."""
    user = db.query(User).filter(User.username == body.username).first()
    if user is None or not verify_password(body.password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Credenciales inválidas")

    token = create_access_token({"user_id": user.id, "role": user.role.value})

    db.add(AuditLog(user_id=user.id, action="auth.login", payload={"username": user.username}))
    db.commit()

    return Token(access_token=token)


@router.get(
    "/me",
    response_model=UserOut,
    summary="Ver mi perfil",
)
def me(user: User = Depends(get_current_user)):
    """Devuelve los datos del usuario logueado, incluyendo si tiene API keys configuradas."""
    return _user_out(user)
