"""
═══════════════════════════════════════════
 Auth — JWT-based authentication
═══════════════════════════════════════════
"""

import os
from datetime import datetime, timedelta
from typing import Optional

from jose import JWTError, jwt
from passlib.context import CryptContext
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.orm import Session

from backend.database import get_db, User

# ── Config ──
SECRET_KEY = os.getenv("JWT_SECRET", "vibeml-change-this-in-production-to-a-random-64-char-string")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_HOURS = int(os.getenv("TOKEN_EXPIRE_HOURS", "72"))

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
security = HTTPBearer(auto_error=False)


# ── Password Hashing ──

def hash_password(password: str) -> str:
    return pwd_context.hash(password)


def verify_password(plain: str, hashed: str) -> bool:
    return pwd_context.verify(plain, hashed)


# ── JWT Tokens ──

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(hours=ACCESS_TOKEN_EXPIRE_HOURS))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


def decode_token(token: str) -> dict:
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
        )


# ── Dependencies ──

async def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
    db: Session = Depends(get_db),
) -> Optional[User]:
    """
    Returns current user if token provided, None otherwise.
    This allows endpoints to work for both anonymous and authenticated users.
    """
    if credentials is None:
        return None

    payload = decode_token(credentials.credentials)
    user_id = payload.get("sub")
    if user_id is None:
        return None

    user = db.query(User).filter(User.id == int(user_id)).first()
    return user


async def require_auth(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: Session = Depends(get_db),
) -> User:
    """Strict auth — returns 401 if not authenticated."""
    if credentials is None:
        raise HTTPException(status_code=401, detail="Authentication required")

    payload = decode_token(credentials.credentials)
    user_id = payload.get("sub")
    if user_id is None:
        raise HTTPException(status_code=401, detail="Invalid token")

    user = db.query(User).filter(User.id == int(user_id)).first()
    if user is None:
        raise HTTPException(status_code=401, detail="User not found")

    return user


# ── Plan Limits ──

PLAN_LIMITS = {
    "free":  {"pipelines_per_month": 10,     "max_rows": 20_000,  "deep_learning": False, "download_model": True},
    "pro":   {"pipelines_per_month": 999999, "max_rows": 500_000, "deep_learning": True,  "download_model": True},
    "team":  {"pipelines_per_month": 999999, "max_rows": 500_000, "deep_learning": True,  "download_model": True},
}


def check_plan_limit(user: Optional[User], row_count: int):
    """Check if user is within their plan limits."""
    plan = user.plan if user else "free"
    limits = PLAN_LIMITS[plan]

    if user and user.pipelines_this_month >= limits["pipelines_per_month"]:
        raise HTTPException(
            status_code=403,
            detail=f"Monthly pipeline limit reached ({limits['pipelines_per_month']}). Upgrade your plan.",
        )

    if row_count > limits["max_rows"]:
        raise HTTPException(
            status_code=403,
            detail=f"Your plan supports up to {limits['max_rows']:,} rows. This file has {row_count:,}. Upgrade to Pro.",
        )
