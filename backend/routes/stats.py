"""
Stats Routes — Live stats for the landing page
"""

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from backend.database import get_db, GlobalStats

router = APIRouter()


@router.get("/live")
async def get_live_stats(db: Session = Depends(get_db)):
    """
    Get real-time stats for the landing page.
    Every number is REAL. Zero means zero.
    """
    stats = db.query(GlobalStats).first()

    if not stats:
        return {
            "datasets_processed": 0,
            "code_lines_generated": 0,
            "avg_model_accuracy": None,
            "models_trained": 0,
            "data_retained_bytes": 0,
        }

    avg_accuracy = None
    if stats.total_models_trained > 0:
        avg_accuracy = round(stats.sum_accuracy / stats.total_models_trained * 100, 1)

    return {
        "datasets_processed": stats.total_datasets,
        "code_lines_generated": stats.total_code_lines,
        "avg_model_accuracy": avg_accuracy,
        "models_trained": stats.total_models_trained,
        "data_retained_bytes": 0,  # Always zero. That's the promise.
    }
