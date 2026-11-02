"""
tests/test_api.py
=================
API tests for api/main.py.

Tests endpoint logic, error codes (404, 503), response models, and database
persistence directly through the endpoint handlers with isolated, stubbed state.
This avoids loopback socket hangs on Windows while rigorously testing the full
API contract and behaviour.
"""

from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from fastapi import HTTPException

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import api.main as app_module

# ---------------------------------------------------------------------------
# Test Stubs & Constants
# ---------------------------------------------------------------------------

_FEATURE_COLS = ["TransactionAmt", "card1", "addr1"]
_THRESHOLD = 0.129365
_KNOWN_TXN = 3141064
_MISSING_TXN = 9999999999


class _FakeModel:
    def __init__(self, proba: float = 0.75):
        self._proba = proba

    def predict_proba(self, X):
        n = len(X)
        return np.column_stack([np.full(n, 1.0 - self._proba), np.full(n, self._proba)])


class _FakeExplainer:
    def shap_values(self, X):
        # 3 features -> return mock contributions
        return np.array([[0.05, -0.02, 0.15]])


@pytest.fixture(autouse=True)
def setup_api_state(tmp_path):
    """
    Isolate _state and DB_PATH for each test to ensure no side effects.
    """
    test_db = str(tmp_path / "test_cases.db")
    orig_db = app_module.DB_PATH
    orig_state = dict(app_module._state)

    app_module.DB_PATH = test_db
    app_module._init_db()

    val_df = pd.DataFrame({
        "TransactionID": [_KNOWN_TXN],
        "TransactionAmt": [150.0],
        "card1": [100.0],
        "addr1": [200.0],
    })

    app_module._state.clear()
    app_module._state.update({
        "model": _FakeModel(0.75),
        "model_source": "test-model-source",
        "metadata": {
            "model_type": "LightGBM (test-tuned)",
            "feature_columns": _FEATURE_COLS,
            "threshold_2pct_fp_budget": _THRESHOLD,
        },
        "val_data": val_df,
        "shap_explainer": _FakeExplainer(),
    })

    yield

    app_module.DB_PATH = orig_db
    app_module._state.clear()
    app_module._state.update(orig_state)


# ---------------------------------------------------------------------------
# /health
# ---------------------------------------------------------------------------

class TestHealthEndpoint:
    def test_health_returns_ok_and_metadata(self):
        res = app_module.health()
        assert res["status"] == "ok"
        assert res["model_loaded"] is True
        assert res["model_type"] == "LightGBM (test-tuned)"
        assert res["model_source"] == "test-model-source"

    def test_health_when_model_unloaded(self):
        app_module._state.clear()
        res = app_module.health()
        assert res["status"] == "ok"
        assert res["model_loaded"] is False
        assert res["model_type"] == "unknown"


# ---------------------------------------------------------------------------
# /score/{transaction_id}
# ---------------------------------------------------------------------------

class TestScoreEndpoint:
    def test_score_known_transaction_success(self):
        res = app_module.score_transaction(_KNOWN_TXN)
        assert res.transaction_id == _KNOWN_TXN
        assert res.risk_score == pytest.approx(0.75, abs=1e-4)
        assert res.flagged is True
        assert res.threshold_used == pytest.approx(_THRESHOLD, abs=1e-4)
        assert len(res.top_reason_codes) == 3
        # Ensure reason codes sorted by magnitude
        assert res.top_reason_codes[0]["feature"] == "addr1"
        assert res.top_reason_codes[0]["shap_contribution"] == 0.15

    def test_score_unflagged_when_below_threshold(self):
        app_module._state["model"] = _FakeModel(0.05)
        res = app_module.score_transaction(_KNOWN_TXN)
        assert res.risk_score == pytest.approx(0.05, abs=1e-4)
        assert res.flagged is False

    def test_score_unknown_transaction_raises_404(self):
        with pytest.raises(HTTPException) as exc_info:
            app_module.score_transaction(_MISSING_TXN)
        assert exc_info.value.status_code == 404
        assert "not found in dataset" in exc_info.value.detail

    def test_score_without_shap_returns_empty_reasons(self):
        app_module._state["shap_explainer"] = None
        res = app_module.score_transaction(_KNOWN_TXN)
        assert res.top_reason_codes == []


# ---------------------------------------------------------------------------
# /investigate/{transaction_id}
# ---------------------------------------------------------------------------

class TestInvestigateEndpoint:
    def test_investigate_unknown_transaction_raises_404(self):
        with patch.object(app_module, "_load_agent_investigator") as load_agent:
            with pytest.raises(HTTPException) as exc_info:
                app_module.investigate_transaction(_MISSING_TXN)
        assert exc_info.value.status_code == 404
        load_agent.assert_not_called()

    def test_investigate_ollama_failure_raises_503(self):
        mock_investigate = MagicMock(side_effect=RuntimeError("Connection refused by Ollama"))
        with patch("agent.investigate", mock_investigate):
            with pytest.raises(HTTPException) as exc_info:
                app_module.investigate_transaction(_KNOWN_TXN)
            assert exc_info.value.status_code == 503
            assert "Ollama service may have crashed" in exc_info.value.detail

    def test_investigate_agent_initialization_failure_raises_503(self):
        with patch.object(
            app_module,
            "_load_agent_investigator",
            side_effect=FileNotFoundError("ieee_train.parquet missing"),
        ):
            with pytest.raises(HTTPException) as exc_info:
                app_module.investigate_transaction(_KNOWN_TXN)
        assert exc_info.value.status_code == 503
        assert "could not initialize" in exc_info.value.detail
        assert "ieee_train.parquet missing" in exc_info.value.detail

    def test_investigate_success_persists_case(self):
        mock_case_file = (
            "ENTITY SUMMARY: Test entity summary\n"
            "TOP RISK FACTORS:\n• Prior transactions: 1\n"
            "RELATED ENTITY SIGNAL: 1 domain\n"
            "RELEVANT POLICY: None\n"
            "RECOMMENDED ACTION: Monitor"
        )
        with patch("agent.investigate", return_value=mock_case_file):
            res = app_module.investigate_transaction(_KNOWN_TXN)

        assert res.transaction_id == _KNOWN_TXN
        assert res.case_file == mock_case_file
        assert res.case_id > 0

        # Verify persisted into DB
        case = app_module.get_case(res.case_id)
        assert case["transaction_id"] == _KNOWN_TXN
        assert case["case_file"] == mock_case_file
        assert case["flagged"] is True


# ---------------------------------------------------------------------------
# /cases and /cases/{case_id}
# ---------------------------------------------------------------------------

class TestCasesEndpoints:
    def test_list_cases_initially_empty(self):
        res = app_module.list_cases()
        assert res == []

    def test_list_and_get_case(self):
        # Insert test case directly into DB
        conn = sqlite3.connect(app_module.DB_PATH)
        conn.execute(
            "INSERT INTO cases (transaction_id, risk_score, threshold_used, flagged, top_reason_codes, case_file, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (555, 0.88, 0.12, 1, json.dumps([{"feature": "f1", "shap_contribution": 0.5}]), "Sample Case File", "2026-11-01T10:00:00Z")
        )
        conn.commit()
        conn.close()

        cases = app_module.list_cases(limit=10)
        assert len(cases) == 1
        assert cases[0]["transaction_id"] == 555

        case = app_module.get_case(cases[0]["id"])
        assert case["transaction_id"] == 555
        assert case["risk_score"] == pytest.approx(0.88)
        assert case["flagged"] is True
        assert len(case["top_reason_codes"]) == 1

    def test_get_nonexistent_case_raises_404(self):
        with pytest.raises(HTTPException) as exc_info:
            app_module.get_case(99999)
        assert exc_info.value.status_code == 404
        assert "Case 99999 not found" in exc_info.value.detail
