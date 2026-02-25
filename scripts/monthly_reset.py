"""
Monthly Reset — Run on 1st of each month via cron
Resets pipelines_this_month counter for all users.
Cron: 0 0 1 * * python3 /app/scripts/monthly_reset.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.database import SessionLocal, User

def reset_monthly_counters():
    db = SessionLocal()
    try:
        updated = db.query(User).update({User.pipelines_this_month: 0})
        db.commit()
        print(f"[Monthly Reset] Reset pipeline counters for {updated} users.")
    finally:
        db.close()

if __name__ == "__main__":
    reset_monthly_counters()
