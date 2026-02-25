# ⚡ Vibe ML

**From messy CSV to trained AI model — with full Python code.**

Upload your data. We auto-clean it, engineer features, select the best algorithm, train the model — and show you every line of Python code. Download the code, clean data, and trained model. No coding needed.

---

## Architecture

```
┌──────────────────────────────────────────────────┐
│                    Frontend                       │
│          (Single-page HTML/JS app)                │
│   Landing Page ←→ App Mode (Upload → Pipeline)   │
└──────────────────────┬───────────────────────────┘
                       │ REST API
┌──────────────────────▼───────────────────────────┐
│              FastAPI Backend                       │
│                                                    │
│  ┌─────────┐  ┌──────────┐  ┌─────────────────┐  │
│  │  Auth    │  │ Pipeline │  │   Feedback       │  │
│  │ (JWT)   │  │  Routes  │  │   + Stats        │  │
│  └────┬────┘  └────┬─────┘  └────────┬────────┘  │
│       │            │                  │            │
│  ┌────▼────────────▼──────────────────▼────────┐  │
│  │            ML Engine                         │  │
│  │  Profile → Clean → Engineer → Train → Code   │  │
│  │  (pandas, sklearn, xgboost, scipy)           │  │
│  └──────────────────┬──────────────────────────┘  │
│                     │                              │
│  ┌──────────────────▼──────────────────────────┐  │
│  │           SQLite Database                    │  │
│  │  users | pipelines | feedback | global_stats │  │
│  └─────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────┘
```

## Tech Stack

| Layer        | Technology                            |
|-------------|---------------------------------------|
| Frontend    | Vanilla HTML/JS + PapaParse           |
| Backend     | Python 3.11 + FastAPI                 |
| ML Engine   | pandas, scikit-learn, XGBoost, scipy  |
| Database    | SQLite (swap to PostgreSQL for scale)  |
| Auth        | JWT (python-jose + bcrypt)            |
| Deployment  | Docker + Nginx + Let's Encrypt        |
| Payments    | Razorpay (TODO)                       |

## Quick Start (Local Development)

```bash
# 1. Clone
git clone https://github.com/yourusername/vibeml.git
cd vibeml

# 2. Run
chmod +x run.sh
./run.sh

# 3. Open
# App:  http://localhost:8000
# Docs: http://localhost:8000/docs
```

Or manually:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
mkdir -p storage frontend
uvicorn backend.main:app --reload --port 8000
```

## API Endpoints

### Pipeline (core product)

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/pipeline/upload` | Upload CSV, get profile |
| POST | `/api/pipeline/profile` | Get detailed data profile |
| POST | `/api/pipeline/clean` | Auto-clean data |
| POST | `/api/pipeline/engineer` | Feature engineering |
| POST | `/api/pipeline/train` | Train models, get best |
| GET | `/api/pipeline/download/{session_id}/{type}` | Download code/data/model |
| GET | `/api/pipeline/code/{session_id}` | Get generated Python code |
| DELETE | `/api/pipeline/session/{session_id}` | Delete all session data |

### Auth

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/auth/signup` | Create account |
| POST | `/api/auth/login` | Login, get JWT |
| GET | `/api/auth/me` | Get profile |

### Feedback & Stats

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/feedback/submit` | Submit feedback |
| GET | `/api/feedback/list` | Feedback counts |
| GET | `/api/stats/live` | Live stats for landing page |

### Example: Full Pipeline Flow

```bash
# 1. Upload
curl -X POST http://localhost:8000/api/pipeline/upload \
  -F "file=@sales_data.csv"
# Returns: { "session_id": "abc-123", "profile": {...} }

# 2. Clean
curl -X POST http://localhost:8000/api/pipeline/clean \
  -H "Content-Type: application/json" \
  -d '{"session_id": "abc-123"}'

# 3. Engineer features (specify target column)
curl -X POST http://localhost:8000/api/pipeline/engineer \
  -H "Content-Type: application/json" \
  -d '{"session_id": "abc-123", "target_column": "revenue"}'

# 4. Train
curl -X POST http://localhost:8000/api/pipeline/train \
  -H "Content-Type: application/json" \
  -d '{"session_id": "abc-123"}'
# Returns: { "best_model": "XGBoost", "best_score": 0.94, ... }

# 5. Download code
curl http://localhost:8000/api/pipeline/download/abc-123/code -o pipeline.py

# 6. Download clean data
curl http://localhost:8000/api/pipeline/download/abc-123/data -o clean_data.csv

# 7. Delete session data
curl -X DELETE http://localhost:8000/api/pipeline/session/abc-123
```

## Deploy to Production

### Option A: VPS (DigitalOcean, AWS EC2, Hetzner)

```bash
# 1. SSH into your server
ssh root@your-server-ip

# 2. Clone project
git clone https://github.com/yourusername/vibeml.git
cd vibeml

# 3. Run deploy script
chmod +x deploy.sh
./deploy.sh

# 4. Setup domain + SSL
sudo apt install certbot python3-certbot-nginx
sudo certbot --nginx -d vibeml.in -d www.vibeml.in
```

**Recommended VPS:**
- DigitalOcean: $12/mo (2GB RAM, 1 vCPU) — enough to start
- Hetzner: €4.50/mo (2GB RAM) — cheapest option
- AWS Lightsail: $10/mo

### Option B: Docker

```bash
cp .env.example .env
# Edit .env with your settings
docker-compose up -d
```

### Option C: Railway / Render (one-click)

1. Push to GitHub
2. Connect to Railway.app or Render.com
3. Set environment variables from .env.example
4. Deploy

## Project Structure

```
vibeml/
├── backend/
│   ├── __init__.py
│   ├── main.py              # FastAPI app + middleware
│   ├── database.py          # SQLAlchemy models (User, Pipeline, Feedback, Stats)
│   ├── auth.py              # JWT auth + plan limits
│   ├── ml_engine.py         # THE ML ENGINE (profile, clean, engineer, train, codegen)
│   └── routes/
│       ├── __init__.py
│       ├── pipeline.py      # Upload → Profile → Clean → Engineer → Train → Download
│       ├── users.py         # Signup, Login, Profile
│       ├── feedback.py      # Submit + list feedback
│       └── stats.py         # Live stats
├── frontend/
│   └── index.html           # Complete frontend (landing + app)
├── storage/                  # Created at runtime (gitignored)
│   ├── uploads/             # Uploaded CSVs (deleted after processing)
│   ├── outputs/             # Generated code, clean data, models
│   └── vibeml.db            # SQLite database
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── nginx.conf               # Production reverse proxy
├── deploy.sh                # VPS deployment script
├── run.sh                   # Local dev script
├── .env.example
├── .gitignore
└── README.md
```

## What the ML Engine Actually Does

1. **Upload**: Parse CSV (handles UTF-8, Latin-1, CP1252 encodings, comma/tab separators)
2. **Detect Types**: Auto-classify each column as numeric, categorical, date, or text
3. **Profile**: Missing values, duplicates, outliers (IQR), correlations, quality score
4. **Clean**: Remove duplicates → strip currency symbols → fill missing (median/mode) → remove outliers
5. **Engineer**: Extract date features → label encode categoricals → StandardScale numerics
6. **Train**: Test Random Forest, Gradient Boosting, XGBoost (+ Logistic/Ridge) with 5-fold CV
7. **Generate Code**: Full runnable Python script (~150-200 lines) using standard libraries
8. **Save**: model.pkl + clean_data.csv + pipeline.py → ready for download

## TODO / Roadmap

- [ ] **Razorpay integration** for Pro/Team payments
- [ ] **Email notifications** for feedback (SendGrid/Resend)
- [ ] **PostgreSQL** migration for scale
- [ ] **Redis** for session storage + rate limiting
- [ ] **Deep learning** option (TensorFlow/PyTorch for Pro users)
- [ ] **Regression visualizations** (actual vs predicted plots)
- [ ] **Admin dashboard** for managing feedback + users
- [ ] **Monthly cron** to reset pipelines_this_month counter
- [ ] **File cleanup cron** to delete old session data
- [ ] **Celery/background tasks** for large dataset training

## License

MIT — Use it however you want. Code is yours.
