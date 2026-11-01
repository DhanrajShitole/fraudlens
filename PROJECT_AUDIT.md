# FraudLens — Project Audit (2026-11-01)

## 1. Repository Inspection Summary

All source files inspected locally from the working tree (including uncommitted
changes). The repo contains:

- `src/` — agent.py, drift_check.py, evaluate_test.py, features.py,
  persist_model.py, register_model.py, train_advanced.py, train_baseline.py
- `api/main.py` — FastAPI backend (5 endpoints)
- `frontend/dashboard.html` — React single-file analyst dashboard
- `tests/` — test_agent_logic.py (19 tests), test_drift.py (6 tests), conftest.py
- `models/` — ieee_lightgbm_tuned.joblib + metadata.json (both present, 4 MB)
- `reports/` — evaluation, SHAP, drift, calibration reports
- `notebooks/` — EDA/feature-engineering notebooks + phase4_summary.md
- `.github/workflows/ci.yml` — CI pipeline (gap found, fixed)
- `Dockerfile`, `docker-compose.yml`, `.dockerignore` — new files

---

## 2. Features Already Implemented ✅

| Feature | Location |
|---|---|
| Feature engineering pipeline (576 features, leakage-safe splits) | `src/features.py` |
| Baseline ML (LogReg, LightGBM, IsolationForest, PaySim + IEEE) | `src/train_baseline.py` |
| Tuned LightGBM + SMOTE — PR-AUC 0.598, ROC-AUC 0.924 | `src/train_advanced.py` |
| SHAP explainability (top-5 reason codes per transaction) | `api/main.py` |
| Model persistence (joblib) | `models/ieee_lightgbm_tuned.joblib` |
| MLflow experiment tracking + model registry (alias-based) | `src/register_model.py` |
| PSI drift detection — 7/545 features genuinely drifted | `src/drift_check.py` |
| Agent copilot — point-in-time evidence, deterministic decisions | `src/agent.py` |
| Agent — evidence-driven TF-IDF policy search + relevance floor | `src/agent.py` |
| Agent — Ollama-unavailable fallback (complete tool-derived case file) | `src/agent.py` |
| Agent validator — grounded numbers only, comparative framing check | `src/agent.py` |
| FastAPI backend (5 endpoints, CORS, SHAP, SQLite case log) | `api/main.py` |
| MLflow registry → joblib fallback in API startup | `api/main.py` |
| React analyst dashboard (dark console, score + investigate + history) | `frontend/dashboard.html` |
| Final held-out test evaluation (IEEE-CIS + PaySim) | `src/evaluate_test.py` |
| Calibration report (Brier score, reliability plot) | `reports/calibration_test.png` |
| Docker containerisation files | `Dockerfile`, `docker-compose.yml` |
| Agent validator tests (19), PSI drift tests (6) — 25 total | `tests/` |

---

## 3. Gaps Found and Fixed in This Session

### A. CI/CD — CRITICAL (Fixed)
**File:** `.github/workflows/ci.yml`
The workflow only installed a minimal subset of packages (pandas, numpy,
scikit-learn, scipy, pyarrow, matplotlib, pytest) — missing lightgbm, shap,
fastapi, uvicorn, pydantic, joblib, mlflow, ollama. New API and model tests
would have silently failed on GitHub Actions.
**Fix:** ci.yml updated with the full requirements.txt install and new test files.

### B. Missing Tests (Added)
No pytest coverage existed for:
- API endpoints (health, score, invalid ID)
- Model loading (joblib, SHAP integration)
- Features module (basic pipeline smoke tests)

**Fix:** `tests/test_api.py` and `tests/test_model.py` added.

### C. README Setup Section (Fixed)
Setup section had a minimal, incomplete pip install command.
**Fix:** Full install command added; local-run instructions added.

### D. Roadmap Phase 11/12 (Fixed)
Both marked `[ ]` despite testing being complete.
**Fix:** Updated to `[x]` with accurate descriptions.

---

## 4. Known Limitations (by design, not bugs)

| Limitation | Reason |
|---|---|
| `src/evaluate_test.py` requires local data files | Datasets are gitignored (6 GB+); CI runs on synthetic data only |
| Docker not verified with a live build | Docker not installed on this machine; files are correct syntactically |
| Ollama/LLM not testable in CI | Local daemon; agent tests use FakeOllama stub |
| Cloud deployment deferred | Ollama's 6 GB RAM exceeds free-tier cloud limits |
| MLflow registry may fail inside Docker | Absolute host paths in mlflow.db; joblib fallback handles this |

---

## 5. Architecture Verdict

The architecture is sound. Key invariants are correctly implemented and tested:
- LLM writes prose; code makes all decisions
- Point-in-time evidence with no label leakage
- Evidence-driven policy search (no model-written queries)
- MLflow registry with reliable joblib fallback
- React via CDN (no build step — correct for project scale)
- SQLite for case log (appropriate at this scale)

No fundamental bugs found in the ML pipeline, data handling, or API logic.

