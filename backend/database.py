"""
═══════════════════════════════════════════
 Database — SQLAlchemy + SQLite
 Tables: users, pipelines, feedback, stats
═══════════════════════════════════════════
"""

import os
from datetime import datetime
from sqlalchemy import (
    create_engine, Column, Integer, String, Float, Text,
    DateTime, Boolean, ForeignKey, JSON
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///storage/vibeml.db")

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


# ── Models ──

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String(255), unique=True, index=True, nullable=False)
    name = Column(String(255), nullable=False)
    password_hash = Column(String(255), nullable=False)
    plan = Column(String(20), default="free")  # free, pro, team
    pipelines_this_month = Column(Integer, default=0)
    created_at = Column(DateTime, default=datetime.utcnow)
    last_login = Column(DateTime, default=datetime.utcnow)
    is_active = Column(Boolean, default=True)

    # Relationships
    pipelines = relationship("Pipeline", back_populates="user")


class Pipeline(Base):
    __tablename__ = "pipelines"

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String(36), unique=True, index=True, nullable=False)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True)  # nullable for free anonymous use
    filename = Column(String(255))
    rows_original = Column(Integer)
    rows_cleaned = Column(Integer)
    columns = Column(Integer)
    column_types = Column(JSON)  # {"col_name": "numeric", ...}
    quality_score_before = Column(Integer)
    quality_score_after = Column(Integer)
    issues_found = Column(JSON)  # {"missing": {...}, "outliers": {...}, "duplicates": N}
    features_original = Column(Integer)
    features_engineered = Column(Integer)
    best_model = Column(String(100))
    best_accuracy = Column(Float)
    all_model_scores = Column(JSON)  # [{"name": "RF", "score": 0.94}, ...]
    problem_type = Column(String(20))  # classification, regression
    target_column = Column(String(255))
    code_lines = Column(Integer)
    status = Column(String(20), default="uploaded")  # uploaded, profiled, cleaned, engineered, trained, completed
    created_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime, nullable=True)

    # Relationships
    user = relationship("User", back_populates="pipelines")


class Feedback(Base):
    __tablename__ = "feedback"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), nullable=False)
    email = Column(String(255), nullable=False)
    type = Column(String(20), nullable=False)  # feedback, bug, complaint, feature
    message = Column(Text, nullable=False)
    status = Column(String(20), default="new")  # new, read, replied, resolved
    admin_notes = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)


class GlobalStats(Base):
    """Single-row table for live stats shown on landing page."""
    __tablename__ = "global_stats"

    id = Column(Integer, primary_key=True, default=1)
    total_datasets = Column(Integer, default=0)
    total_code_lines = Column(Integer, default=0)
    total_models_trained = Column(Integer, default=0)
    sum_accuracy = Column(Float, default=0.0)  # sum of all accuracies (divide by total_models for avg)
    updated_at = Column(DateTime, default=datetime.utcnow)


# ── Init ──

def init_db():
    """Create all tables."""
    Base.metadata.create_all(bind=engine)
    # Ensure global stats row exists
    db = SessionLocal()
    try:
        stats = db.query(GlobalStats).first()
        if not stats:
            db.add(GlobalStats(id=1))
            db.commit()
    finally:
        db.close()


def get_db():
    """Dependency: yields DB session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
