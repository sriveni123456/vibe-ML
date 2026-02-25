"""
═══════════════════════════════════════════
 Vibe ML — Backend API Server
 FastAPI + SQLite + Scikit-learn + XGBoost
═══════════════════════════════════════════
"""

import os
import uuid
import shutil
from datetime import datetime
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, HTMLResponse

from backend.database import init_db, get_db, SessionLocal
from backend.routes import pipeline, users, feedback, stats


# ── Lifecycle ──
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup/shutdown events."""
    init_db()
    # Create upload/output dirs
    os.makedirs("storage/uploads", exist_ok=True)
    os.makedirs("storage/outputs", exist_ok=True)
    os.makedirs("storage/models", exist_ok=True)
    yield
    # Cleanup on shutdown (optional)


# ── App ──
app = FastAPI(
    title="Vibe ML API",
    description="From messy CSV to trained AI model with full Python code",
    version="1.0.0",
    lifespan=lifespan,
)

# ── CORS ──
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Rate Limiting Middleware ──
from collections import defaultdict
import time

request_counts = defaultdict(list)
RATE_LIMIT = int(os.getenv("RATE_LIMIT_PER_MINUTE", "30"))


@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    """Simple in-memory rate limiter. Replace with Redis in production."""
    client_ip = request.client.host
    now = time.time()
    # Clean old entries
    request_counts[client_ip] = [t for t in request_counts[client_ip] if now - t < 60]
    if len(request_counts[client_ip]) >= RATE_LIMIT:
        from fastapi.responses import JSONResponse
        return JSONResponse(
            status_code=429,
            content={"detail": "Too many requests. Please wait a minute."},
        )
    request_counts[client_ip].append(now)
    response = await call_next(request)
    return response


# ── Routers ──
app.include_router(pipeline.router, prefix="/api/pipeline", tags=["Pipeline"])
app.include_router(users.router, prefix="/api/auth", tags=["Authentication"])
app.include_router(feedback.router, prefix="/api/feedback", tags=["Feedback"])
app.include_router(stats.router, prefix="/api/stats", tags=["Stats"])


# ── Serve Frontend ──
# Mount static frontend files
if os.path.exists("frontend"):
    app.mount("/static", StaticFiles(directory="frontend"), name="static")


@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    """Serve the single-page frontend."""
    frontend_path = os.path.join("frontend", "index.html")
    if os.path.exists(frontend_path):
        with open(frontend_path, encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse(content="<h1>Vibe ML API</h1><p>Frontend not found. Place index.html in /frontend/</p>")


@app.get("/health")
async def health_check():
    return {"status": "healthy", "version": "1.0.0", "timestamp": datetime.utcnow().isoformat()}
