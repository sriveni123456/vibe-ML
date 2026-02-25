# ═══════════════════════════════════════════
#  Vibe ML — Dockerfile
# ═══════════════════════════════════════════

FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system deps (for scipy, sklearn)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ && \
    rm -rf /var/lib/apt/lists/*

# Install Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Create storage dirs
RUN mkdir -p storage/uploads storage/outputs storage/models

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
