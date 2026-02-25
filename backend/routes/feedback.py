"""
Feedback Routes — Submit and manage feedback/complaints
"""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from pydantic import BaseModel

from backend.database import get_db, Feedback

router = APIRouter()


class FeedbackRequest(BaseModel):
    name: str
    email: str
    type: str  # feedback, bug, complaint, feature
    message: str


@router.post("/submit")
async def submit_feedback(req: FeedbackRequest, db: Session = Depends(get_db)):
    """Submit feedback, bug report, complaint, or feature request."""
    if req.type not in ("feedback", "bug", "complaint", "feature"):
        raise HTTPException(400, "Type must be: feedback, bug, complaint, or feature")

    if len(req.message.strip()) < 10:
        raise HTTPException(400, "Message must be at least 10 characters.")

    fb = Feedback(
        name=req.name.strip(),
        email=req.email.strip().lower(),
        type=req.type,
        message=req.message.strip(),
    )
    db.add(fb)
    db.commit()

    # TODO: Send email notification to founder
    # In production: integrate with SendGrid/Resend/AWS SES
    # send_email(
    #     to="founder@vibeml.in",
    #     subject=f"[{req.type.upper()}] New feedback from {req.name}",
    #     body=req.message,
    # )

    return {"status": "received", "message": "Thank you! We'll respond within 24 hours."}


@router.get("/list")
async def list_feedback(db: Session = Depends(get_db)):
    """
    Public endpoint — shows feedback count by type.
    Full feedback list is admin-only (TODO: add admin auth).
    """
    from sqlalchemy import func
    counts = db.query(Feedback.type, func.count(Feedback.id)).group_by(Feedback.type).all()
    total = db.query(func.count(Feedback.id)).scalar()

    return {
        "total": total,
        "by_type": {t: c for t, c in counts},
    }
