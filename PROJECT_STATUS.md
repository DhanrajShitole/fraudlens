# FraudLens — Project Status & Implementation Tracking

**Last Updated:** November 1, 2026  
**Status:** All 12 Phases Implemented & Verified. Deployment-ready.

---

## 1. Executive Summary

FraudLens is an end-to-end, explainable financial fraud detection system paired with an agentic investigation copilot. All 12 phases from initial problem definition to held-out test evaluation, MLOps, CI/CD automation, and deployment packaging are complete.

The test suite contains **51 automated tests**, all passing across agent logic, API contracts, drift detection, model loading, and SHAP explainability.

---

## 2. Phase-by-Phase Completion Matrix

| Phase | Description | Status | Verification & Deliverables |
|---|---|---|---|
| **Phase 1** | Research & Problem Definition | **Complete** | IEEE-CIS & PaySim problem framing in `README.md` |
| **Phase 2** | Data Collection & Exploration | **Complete** | `notebooks/01_eda.ipynb`, class imbalance analysis |
| **Phase 3** | Feature Engineering | **Complete** | `src/features.py` (576 features, time-based leakage-safe splits) |
| **Phase 4** | Baseline Modeling | **Complete** | `src/train_baseline.py`, `notebooks/phase4_summary.md` (LogReg, LightGBM, IsolationForest) |
| **Phase 5** | Advanced Modeling & SHAP | **Complete** | `src/train_advanced.py` (Tuned LightGBM+SMOTE, PR-AUC 0.598, SHAP top-5 features) |
| **Phase 6** | Agentic Investigation Copilot | **Complete** | `src/agent.py` (Point-in-time tools, policy TF-IDF, hardened numerical validator) |
| **Phase 7** | Backend Serving API | **Complete** | `api/main.py` (FastAPI, 5 endpoints, CORS, SQLite case store) |
| **Phase 8** | Analyst Dashboard | **Complete** | `frontend/dashboard.html` (Dark console UI, score, reason codes, case files) |
| **Phase 9** | MLOps: Registry & Drift | **Complete** | `src/register_model.py` (alias `@production`), `src/drift_check.py` (PSI report) |
| **Phase 10** | Containerization & Deployment | **Complete** | `Dockerfile`, `docker-compose.yml`, hybrid deployment architecture |
| **Phase 11** | Testing & Test-Set Evaluation | **Complete** | 51 pytest tests passing; `src/evaluate_test.py` held-out test evaluation |
| **Phase 12** | Documentation & Research Draft | **Complete** | `README.md`, `PROJECT_AUDIT.md`, `reports/research_paper_draft.md` |

---

## 3. Test Suite Verification

Run locally using Python 3.13:
```bash
C:\Users\DHANRAJ\AppData\Local\Programs\Python\Python313\python.exe -m pytest tests/ -v
```

### Breakdown of 51 Passing Tests:
- **`tests/test_agent_logic.py` (19 tests):**
  - Point-in-time correctness: transactions do not observe future fraud or own labels.
  - Validator numerical grounding: rejects fabricated numbers and numbers sourced solely from policy text.
  - Rejection of unsupported comparative framing (e.g. self-referential mean comparison).
  - Policy escalation logic and zero-model-call fast fallback for missing entities.
  - LLM failure resilience: produces full deterministic case file when Ollama is unavailable.
- **`tests/test_api.py` (12 tests):**
  - `GET /health`: service health, model load status, model source, and model type reporting.
  - `POST /score/{id}`: probability in [0, 1], 2% FP threshold flag check, SHAP reason codes.
  - `POST /score/{id}`: 404 handling on unknown transaction IDs.
  - `POST /investigate/{id}`: 404 for unknown transaction, 503 error handling on Ollama unavailability.
  - `GET /cases` and `GET /cases/{id}`: retrieval, pagination limit, and SQLite persistence.
- **`tests/test_drift.py` (6 tests):**
  - Population Stability Index (PSI) calculation under identical distributions, large shifts, near-constant features, and NaNs.
- **`tests/test_model.py` (14 tests):**
  - Verifies presence and structure of `models/ieee_lightgbm_tuned.joblib` and metadata.
  - 545 input feature compatibility and -999 NaN imputation.
  - `predict_proba` probability distribution properties.
  - SHAP `TreeExplainer` integration and top-5 reason code extraction.

---

## 4. MLOps Status

- **Model Registry:** MLflow SQLite backend (`mlflow.db`) with model registered as `fraudlens-ieee-lightgbm`.
- **Production Alias:** Pointed to the active tuned version using MLflow's modern alias API (`@production`).
- **Serving Fallback:** `api/main.py` dynamically resolves `models:/fraudlens-ieee-lightgbm@production`. If registry lookup is unavailable (e.g. within Docker where host paths differ), it transparently falls back to `models/ieee_lightgbm_tuned.joblib` without downtime.
- **Drift Monitoring:** PSI computation between train (reference) and validation (comparison) sets confirms 535/545 features stable (<0.10 PSI), 2 moderate, and 7 genuine drift candidates (1.3%) in device/infrastructure features.

---

## 5. DevOps & CI/CD Pipeline

- **GitHub Actions Workflow (`.github/workflows/ci.yml`):**
  - Triggered automatically on all pushes and pull requests to `main`.
  - Step 1: Environment setup on Python 3.13 with pip caching.
  - Step 2: Dependency installation matching `requirements.txt`.
  - Step 3: Bytecode compilation and AST syntax validation across `src/`, `api/`, `tests/`.
  - Step 4: HTML syntax parsing of `frontend/dashboard.html`.
  - Step 5: Complete automated pytest execution (`pytest tests/`).
  - Step 6: Docker image build validation (`docker build -t fraudlens-api:latest .`).

---

## 6. Deployment Architecture

### Hybrid Cloud/Local Model:
1. **Lightweight Scoring Backend (Cloud or Container Ready):**
   - FastAPI + LightGBM + SHAP runs on `python:3.13-slim` (~500MB RAM requirement).
   - Easily deployable to any free or low-cost cloud container service (Render, Fly.io, AWS ECS / Fargate, Railway).
2. **Investigation Copilot (Local / Dedicated Instance):**
   - The Llama 3.1 8B parameter model runs locally via Ollama to ensure complete data privacy (no PII or banking transaction data sent to third-party cloud APIs) and zero recurring inference costs.
   - For a production deployment without a local machine, Ollama can be hosted on an instance with 8GB-16GB RAM or configured to call a private enterprise LLM endpoint via `OLLAMA_HOST`.

---

## 7. How to Run FraudLens Locally

### Step 1: Start the Backend API
```bash
C:\Users\DHANRAJ\AppData\Local\Programs\Python\Python313\python.exe -m uvicorn api.main:app --host 127.0.0.1 --port 8000 --reload
```
Check health:
```bash
curl http://127.0.0.1:8000/health
```

### Step 2: Open the Analyst Console
Open `frontend/dashboard.html` directly in any web browser.
- Enter a TransactionID (e.g., `3141064` or `3400379`).
- Click **Get Score** (or press Enter) for real-time risk score + SHAP reason codes.
- Click **Investigate** to generate a complete multi-section evidence case file.

### Step 3: Run with Docker Compose (Optional)
```bash
docker compose up --build
```

