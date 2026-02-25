"""
Auth Routes — Signup, Login, Profile
"""

from datetime import datetime
from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.orm import Session
from pydantic import BaseModel, EmailStr

from backend.database import get_db, User
from backend.auth import (
    hash_password, verify_password, create_access_token, require_auth
)

router = APIRouter()


class SignupRequest(BaseModel):
    name: str
    email: str
    password: str


class LoginRequest(BaseModel):
    email: str
    password: str


@router.post("/signup")
async def signup(req: SignupRequest, db: Session = Depends(get_db)):
    """Create a new account."""
    if len(req.password) < 6:
        raise HTTPException(400, "Password must be at least 6 characters.")

    existing = db.query(User).filter(User.email == req.email.lower()).first()
    if existing:
        raise HTTPException(409, "Email already registered. Please log in.")

    user = User(
        name=req.name.strip(),
        email=req.email.lower().strip(),
        password_hash=hash_password(req.password),
        plan="free",
    )
    db.add(user)
    db.commit()
    db.refresh(user)

    token = create_access_token({"sub": str(user.id)})

    return {
        "token": token,
        "user": {
            "id": user.id,
            "name": user.name,
            "email": user.email,
            "plan": user.plan,
        },
    }


@router.post("/login")
async def login(req: LoginRequest, db: Session = Depends(get_db)):
    """Log in and get JWT token."""
    user = db.query(User).filter(User.email == req.email.lower().strip()).first()
    if not user or not verify_password(req.password, user.password_hash):
        raise HTTPException(401, "Invalid email or password.")

    user.last_login = datetime.utcnow()
    db.commit()

    token = create_access_token({"sub": str(user.id)})

    return {
        "token": token,
        "user": {
            "id": user.id,
            "name": user.name,
            "email": user.email,
            "plan": user.plan,
            "pipelines_this_month": user.pipelines_this_month,
        },
    }


@router.get("/me")
async def get_profile(user: User = Depends(require_auth)):
    """Get current user profile."""
    return {
        "id": user.id,
        "name": user.name,
        "email": user.email,
        "plan": user.plan,
        "pipelines_this_month": user.pipelines_this_month,
        "created_at": user.created_at.isoformat(),
    }


@router.post("/upgrade")
async def upgrade_plan(user: User = Depends(require_auth), db: Session = Depends(get_db)):
    """
    Placeholder for payment integration.
    In production: Razorpay webhook calls this after successful payment.
    """
    # TODO: Integrate Razorpay
    # 1. Create Razorpay order: razorpay_client.order.create({amount: 99900, currency: 'INR'})
    # 2. Frontend opens Razorpay checkout
    # 3. Razorpay webhook hits /api/auth/payment-webhook
    # 4. Verify signature and upgrade plan

    return {
        "message": "Payment integration coming soon. Contact us for Pro access.",
        "razorpay_note": "Will integrate Razorpay. Key steps: create order → frontend checkout → webhook verification → upgrade plan.",
    }
