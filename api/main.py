"""
api/main.py

Phase 7 — FastAPI backend for FraudLens.

Endpoints:
    GET  /health                          — service status check
    POST /score/{transaction_id}          — real-time fraud risk score + SHAP reason codes
    POST /investigate/{transaction_id}    — triggers the agent to draft a case file
    GET  /cases                           — list past investigations (SQLite-backed log)
    GET  /cases/{case_id}                 — retrieve one past investigation

Data note: this demo API scores transactions that already exist in the
processed validation set (data/processed/ieee_val.parquet), simulating
what a real-time feed would look like. A production version would accept
raw transaction fields in the POST body instead of a lookup ID — this is
a deliberate simplification appropriate for a final-year project without
a live transaction stream, documented here rather than hidden.

Run with:
    C:\\...\\Python313\\python.exe -m uvicorn api.main:app --reload --app-dir src

Required packages:
    C:\\...\\Python313\\python.exe -m pip install fastapi uvicorn
(joblib, shap, pandas, lightgbm should already be installed from earlier phases)
"""

import os
import sys
import json
import sqlite3
from datetime import datetime, timezone
from contextlib import asynccontextmanager

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(BASE_DIR, "data", "processed")
MODELS_DIR = os.path.join(BASE_DIR, "models")
DB_PATH = os.path.join(BASE_DIR, "cases.db")

# agent.py lives in src/, a sibling folder of api/ (this file's folder) —
# add it to the path so we can import it regardless of where uvicorn is launched from.
SRC_DIR = os.path.join(BASE_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

_state = {}


def _init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS cases (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            transaction_id INTEGER NOT NULL,
            risk_score REAL,
            case_file TEXT,
            created_at TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: load model, metadata, and validation data once.
    print("Loading model and metadata...")
    _state["model"] = joblib.load(os.path.join(MODELS_DIR, "ieee_lightgbm_tuned.joblib"))
    with open(os.path.join(MODELS_DIR, "ieee_lightgbm_tuned_metadata.json")) as f:
        _state["metadata"] = json.load(f)
    print("Loading validation transactions (stand-in for a live feed)...")
    _state["val_data"] = pd.read_parquet(os.path.join(DATA_DIR, "ieee_val.parquet"))

    try:
        import shap
        _state["shap_explainer"] = shap.TreeExplainer(_state["model"])
    except ImportError:
        print("WARNING: shap not installed — /score will skip reason codes.")
        _state["shap_explainer"] = None

    _init_db()
    print("Startup complete.")
    yield
    print("Shutting down.")


app = FastAPI(title="FraudLens API", version="0.1.0", lifespan=lifespan)


class ScoreResponse(BaseModel):
    transaction_id: int
    risk_score: float
    flagged: bool
    threshold_used: float
    top_reason_codes: list[dict]


class InvestigateResponse(BaseModel):
    transaction_id: int
    risk_score: float
    case_file: str
    case_id: int


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": "model" in _state,
        "model_type": _state.get("metadata", {}).get("model_type", "unknown"),
    }


@app.post("/score/{transaction_id}", response_model=ScoreResponse)
def score_transaction(transaction_id: int):
    feats = _state["metadata"]["feature_columns"]
    threshold = _state["metadata"]["threshold_2pct_fp_budget"]

    row = _state["val_data"][_state["val_data"]["TransactionID"] == transaction_id]
    if row.empty:
        raise HTTPException(status_code=404, detail=f"TransactionID {transaction_id} not found in dataset.")

    X = row[feats].fillna(-999)
    risk_score = float(_state["model"].predict_proba(X)[:, 1][0])

    top_reasons = []
    if _state["shap_explainer"] is not None:
        shap_values = _state["shap_explainer"].shap_values(X)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        contributions = shap_values[0]
        top_idx = np.argsort(np.abs(contributions))[::-1][:5]
        for i in top_idx:
            top_reasons.append({
                "feature": feats[i],
                "value": float(X.iloc[0, i]),
                "shap_contribution": round(float(contributions[i]), 5),
            })

    return ScoreResponse(
        transaction_id=transaction_id,
        risk_score=round(risk_score, 5),
        flagged=risk_score >= threshold,
        threshold_used=round(threshold, 5),
        top_reason_codes=top_reasons,
    )


@app.post("/investigate/{transaction_id}", response_model=InvestigateResponse)
def investigate_transaction(transaction_id: int):
    # Reuse the scoring logic first, so the case file has a risk score to reference.
    score_result = score_transaction(transaction_id)

    try:
        from agent import investigate as agent_investigate
    except ImportError:
        raise HTTPException(status_code=500,
                             detail="Agent module not found — ensure src/agent.py exists and Ollama is running.")

    case_file_text = agent_investigate(transaction_id, verbose=False)

    conn = sqlite3.connect(DB_PATH)
    cur = conn.execute(
        "INSERT INTO cases (transaction_id, risk_score, case_file, created_at) VALUES (?, ?, ?, ?)",
        (transaction_id, score_result.risk_score, case_file_text, datetime.now(timezone.utc).isoformat()),
    )
    case_id = cur.lastrowid
    conn.commit()
    conn.close()

    return InvestigateResponse(
        transaction_id=transaction_id,
        risk_score=score_result.risk_score,
        case_file=case_file_text,
        case_id=case_id,
    )


@app.get("/cases")
def list_cases(limit: int = 20):
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT id, transaction_id, risk_score, created_at FROM cases ORDER BY id DESC LIMIT ?",
        (limit,),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


@app.get("/cases/{case_id}")
def get_case(case_id: int):
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    row = conn.execute("SELECT * FROM cases WHERE id = ?", (case_id,)).fetchone()
    conn.close()
    if row is None:
        raise HTTPException(status_code=404, detail=f"Case {case_id} not found.")
    return dict(row)