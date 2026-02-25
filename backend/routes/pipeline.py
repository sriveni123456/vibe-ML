"""
═══════════════════════════════════════════
 Pipeline Routes — The Core API
 Upload → Profile → Clean → Engineer → Train → Download
═══════════════════════════════════════════
"""

import os
import uuid
import shutil
from datetime import datetime
from typing import Optional

import pandas as pd
from fastapi import APIRouter, UploadFile, File, HTTPException, Depends, Query
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session
from pydantic import BaseModel

from backend.database import get_db, Pipeline, GlobalStats, User
from backend.auth import get_current_user, check_plan_limit
from backend.ml_engine import (
    detect_column_types,
    profile_data,
    clean_data,
    engineer_features,
    train_models,
    detect_problem_type,
    generate_full_pipeline_code,
    save_pipeline_artifacts,
)

router = APIRouter()

# In-memory session store (use Redis in production)
sessions = {}


class TargetColumnRequest(BaseModel):
    session_id: str
    target_column: str


class SessionRequest(BaseModel):
    session_id: str


# ══════════════════════════════════
#  1. UPLOAD
# ══════════════════════════════════

@router.post("/upload")
async def upload_file(
    file: UploadFile = File(...),
    user: Optional[User] = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Upload a CSV file. Returns session_id and initial profile.
    """
    # Validate file type
    if not file.filename.endswith((".csv", ".tsv", ".txt")):
        raise HTTPException(400, "Only CSV, TSV, and TXT files are supported.")

    # Read file
    try:
        content = await file.read()
        # Try different encodings
        for encoding in ["utf-8", "latin-1", "cp1252"]:
            try:
                text = content.decode(encoding)
                break
            except UnicodeDecodeError:
                continue
        else:
            raise HTTPException(400, "Could not decode file. Please use UTF-8 encoding.")

        # Detect separator
        first_line = text.split("\n")[0]
        sep = "\t" if "\t" in first_line else ","

        from io import StringIO
        df = pd.read_csv(StringIO(text), sep=sep)

    except Exception as e:
        raise HTTPException(400, f"Error parsing file: {str(e)}")

    if len(df) == 0 or len(df.columns) == 0:
        raise HTTPException(400, "File is empty or has no columns.")

    if len(df.columns) < 2:
        raise HTTPException(400, "Need at least 2 columns (features + target).")

    # Check plan limits
    check_plan_limit(user, len(df))

    # Create session
    session_id = str(uuid.uuid4())
    col_types = detect_column_types(df)
    profile = profile_data(df, col_types)

    # Save to disk
    upload_dir = os.path.join("storage", "uploads", session_id)
    os.makedirs(upload_dir, exist_ok=True)
    df.to_csv(os.path.join(upload_dir, "original.csv"), index=False)

    # Store in session
    sessions[session_id] = {
        "df": df,
        "col_types": col_types,
        "profile": profile,
        "filename": file.filename,
        "user_id": user.id if user else None,
        "status": "uploaded",
        "created_at": datetime.utcnow(),
    }

    # Create DB record
    pipeline_record = Pipeline(
        session_id=session_id,
        user_id=user.id if user else None,
        filename=file.filename,
        rows_original=len(df),
        columns=len(df.columns),
        column_types=col_types,
        quality_score_before=profile["quality_score"],
        issues_found={
            "missing": profile["missing"],
            "outliers": profile["outliers"],
            "duplicates": profile["duplicates"],
        },
        status="uploaded",
    )
    db.add(pipeline_record)
    db.commit()

    return {
        "session_id": session_id,
        "filename": file.filename,
        "profile": profile,
        "column_types": col_types,
        "preview": df.head(10).to_dict(orient="records"),
        "suggested_target": df.columns[-1],
    }


# ══════════════════════════════════
#  2. PROFILE (detailed)
# ══════════════════════════════════

@router.post("/profile")
async def get_profile(req: SessionRequest):
    """Get detailed data profile for a session."""
    session = sessions.get(req.session_id)
    if not session:
        raise HTTPException(404, "Session not found. Please upload again.")

    return {
        "session_id": req.session_id,
        "profile": session["profile"],
        "column_types": session["col_types"],
        "preview": session["df"].head(10).to_dict(orient="records"),
    }


# ══════════════════════════════════
#  3. CLEAN
# ══════════════════════════════════

@router.post("/clean")
async def clean_dataset(
    req: SessionRequest,
    db: Session = Depends(get_db),
):
    """Run auto-cleaning on the uploaded data."""
    session = sessions.get(req.session_id)
    if not session:
        raise HTTPException(404, "Session not found.")

    df = session["df"]
    col_types = session["col_types"]

    cleaned_df, clean_report = clean_data(df, col_types)
    clean_profile = profile_data(cleaned_df, col_types)

    # Update session
    session["cleaned_df"] = cleaned_df
    session["clean_report"] = clean_report
    session["col_types"] = col_types  # may have been updated (currency → numeric)
    session["status"] = "cleaned"

    # Update DB
    pipeline = db.query(Pipeline).filter(Pipeline.session_id == req.session_id).first()
    if pipeline:
        pipeline.rows_cleaned = len(cleaned_df)
        pipeline.quality_score_after = clean_profile["quality_score"]
        pipeline.status = "cleaned"
        db.commit()

    return {
        "session_id": req.session_id,
        "clean_report": clean_report,
        "profile_after": clean_profile,
        "preview": cleaned_df.head(10).to_dict(orient="records"),
    }


# ══════════════════════════════════
#  4. ENGINEER FEATURES
# ══════════════════════════════════

@router.post("/engineer")
async def engineer(
    req: TargetColumnRequest,
    db: Session = Depends(get_db),
):
    """Run feature engineering. Requires target column selection."""
    session = sessions.get(req.session_id)
    if not session:
        raise HTTPException(404, "Session not found.")

    df = session.get("cleaned_df", session["df"])
    col_types = session["col_types"]

    if req.target_column not in df.columns:
        raise HTTPException(400, f"Target column '{req.target_column}' not found.")

    engineered_df, encoders, eng_report = engineer_features(df, col_types)

    # Update session
    session["engineered_df"] = engineered_df
    session["encoders"] = encoders
    session["eng_report"] = eng_report
    session["target_column"] = req.target_column
    session["status"] = "engineered"

    # Update DB
    pipeline = db.query(Pipeline).filter(Pipeline.session_id == req.session_id).first()
    if pipeline:
        pipeline.features_original = eng_report["features_before"]
        pipeline.features_engineered = eng_report["features_after"]
        pipeline.target_column = req.target_column
        pipeline.status = "engineered"
        db.commit()

    return {
        "session_id": req.session_id,
        "eng_report": eng_report,
        "features_before": eng_report["features_before"],
        "features_after": eng_report["features_after"],
        "columns": list(engineered_df.columns),
    }


# ══════════════════════════════════
#  5. TRAIN MODEL
# ══════════════════════════════════

@router.post("/train")
async def train(
    req: SessionRequest,
    db: Session = Depends(get_db),
):
    """Train models, cross-validate, return best."""
    session = sessions.get(req.session_id)
    if not session:
        raise HTTPException(404, "Session not found.")

    df = session.get("engineered_df", session.get("cleaned_df", session["df"]))
    target = session.get("target_column", df.columns[-1])

    # Detect problem type
    problem_type = detect_problem_type(df, target)

    try:
        train_result = train_models(df, target, problem_type)
    except Exception as e:
        raise HTTPException(500, f"Training failed: {str(e)}")

    # Generate full Python code
    col_types = session["col_types"]
    profile = session["profile"]
    clean_report = session.get("clean_report", {"rows_before": len(session["df"]), "rows_after": len(df), "steps": []})
    eng_report = session.get("eng_report", {"steps": [], "features_before": len(df.columns), "features_after": len(df.columns)})

    full_code = generate_full_pipeline_code(
        col_types, profile, clean_report, eng_report, train_result, session["filename"]
    )

    # Save artifacts
    clean_df = session.get("cleaned_df", session["df"])
    encoders = session.get("encoders", {})
    artifacts = save_pipeline_artifacts(
        req.session_id, train_result["model"], clean_df, full_code, encoders
    )

    # Update session
    session["train_result"] = {k: v for k, v in train_result.items() if k != "model"}
    session["full_code"] = full_code
    session["artifacts"] = artifacts
    session["status"] = "trained"

    code_lines = len(full_code.split("\n"))

    # Update DB
    pipeline = db.query(Pipeline).filter(Pipeline.session_id == req.session_id).first()
    if pipeline:
        pipeline.best_model = train_result["best_model_name"]
        pipeline.best_accuracy = train_result["best_cv_score"]
        pipeline.all_model_scores = train_result["all_scores"]
        pipeline.problem_type = problem_type
        pipeline.code_lines = code_lines
        pipeline.status = "trained"
        pipeline.completed_at = datetime.utcnow()
        db.commit()

    # Update global stats
    gstats = db.query(GlobalStats).first()
    if gstats:
        gstats.total_datasets += 1
        gstats.total_code_lines += code_lines
        gstats.total_models_trained += 1
        gstats.sum_accuracy += train_result["best_cv_score"]
        gstats.updated_at = datetime.utcnow()
        db.commit()

    # Increment user's monthly pipeline count
    if session.get("user_id"):
        user = db.query(User).filter(User.id == session["user_id"]).first()
        if user:
            user.pipelines_this_month += 1
            db.commit()

    return {
        "session_id": req.session_id,
        "problem_type": problem_type,
        "best_model": train_result["best_model_name"],
        "best_score": train_result["best_cv_score"],
        "all_scores": train_result["all_scores"],
        "eval_metrics": train_result["eval_metrics"],
        "feature_importance": train_result["feature_importance"],
        "explanation": train_result["explanation"],
        "code_lines": code_lines,
    }


# ══════════════════════════════════
#  6. DOWNLOAD
# ══════════════════════════════════

@router.get("/download/{session_id}/{file_type}")
async def download_file(
    session_id: str,
    file_type: str,  # code, data, model
    user: Optional[User] = Depends(get_current_user),
):
    """Download generated artifacts."""
    base_dir = os.path.join("storage", "outputs", session_id)

    file_map = {
        "code": ("vibeml_pipeline.py", "text/x-python"),
        "data": ("clean_data.csv", "text/csv"),
        "model": ("model.pkl", "application/octet-stream"),
    }

    if file_type not in file_map:
        raise HTTPException(400, "Invalid file type. Use: code, data, model")

    # Model download restricted to paid plans
    if file_type == "model":
        if not user or user.plan == "free":
            raise HTTPException(403, "Model download requires Pro plan. Upgrade to download .pkl files.")

    filename, media_type = file_map[file_type]
    filepath = os.path.join(base_dir, filename)

    if not os.path.exists(filepath):
        raise HTTPException(404, "File not found. Pipeline may have expired.")

    return FileResponse(filepath, filename=filename, media_type=media_type)


@router.get("/code/{session_id}")
async def get_code(session_id: str):
    """Get generated Python code as text (for display in frontend)."""
    session = sessions.get(session_id)
    if session and "full_code" in session:
        return {"code": session["full_code"]}

    # Try from disk
    code_path = os.path.join("storage", "outputs", session_id, "vibeml_pipeline.py")
    if os.path.exists(code_path):
        with open(code_path) as f:
            return {"code": f.read()}

    raise HTTPException(404, "Code not found.")


# ══════════════════════════════════
#  CLEANUP (delete session data)
# ══════════════════════════════════

@router.delete("/session/{session_id}")
async def delete_session(session_id: str):
    """Delete all data for a session. Zero retention."""
    # Remove from memory
    sessions.pop(session_id, None)

    # Remove from disk
    for subdir in ["uploads", "outputs"]:
        path = os.path.join("storage", subdir, session_id)
        if os.path.exists(path):
            shutil.rmtree(path)

    return {"status": "deleted", "session_id": session_id}
