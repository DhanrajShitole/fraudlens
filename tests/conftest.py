"""
tests/conftest.py
=================
Shared pytest fixtures for FraudLens tests.

Key design decisions:
- Builds a tiny synthetic ieee_train.parquet + ieee_val.parquet in a tmp dir.
- monkeypatch.chdir(tmp) BEFORE importing src/agent.py (it loads parquet at
  import time using cwd-relative paths).
- Stubs the `ollama` module via sys.modules with a scriptable fake chat().
- Imports agent fresh via importlib so state doesn't leak between tests.
- No real data or Ollama needed — safe to run in CI.
"""

import importlib
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Synthetic data builder
# ---------------------------------------------------------------------------

def _make_ieee_parquet(tmp_path: Path) -> None:
    """
    Build tiny ieee_train.parquet + ieee_val.parquet in tmp_path / data / processed.
    Columns match what agent.py reads: TransactionID, TransactionDT, card1, addr1,
    P_emaildomain, TransactionAmt, isFraud.

    Entity structure used in tests:
      card1=100, addr1=200 -> TxIDs 1001 (earlier, legit) and 1002 (later, fraud)
      This lets test_agent_logic verify point-in-time correctness.
      TxID 1003: missing addr1 -> triggers the no-entity-data path
    """
    rng = np.random.default_rng(0)
    n_train = 100
    n_val = 50

    # --- train rows ---
    train_records = []
    # Special entity rows — deterministic
    train_records.append({
        "TransactionID": 1001,
        "TransactionDT": 1000,
        "card1": 100,
        "addr1": 200,
        "P_emaildomain": "gmail.com",
        "TransactionAmt": 50.0,
        "isFraud": 0,
    })
    train_records.append({
        "TransactionID": 1002,
        "TransactionDT": 2000,   # LATER than 1001
        "card1": 100,
        "addr1": 200,
        "P_emaildomain": "yahoo.com",
        "TransactionAmt": 500.0,
        "isFraud": 1,            # FRAUD — must NOT appear in get_entity_history(1001)
    })
    # Missing addr1 row
    train_records.append({
        "TransactionID": 1003,
        "TransactionDT": 3000,
        "card1": float("nan"),
        "addr1": float("nan"),
        "P_emaildomain": "outlook.com",
        "TransactionAmt": 75.0,
        "isFraud": 0,
    })

    # Pad to n_train rows with random legit data
    for i in range(n_train - len(train_records)):
        train_records.append({
            "TransactionID": 2000 + i,
            "TransactionDT": 4000 + i * 10,
            "card1": int(rng.integers(300, 310)),
            "addr1": int(rng.integers(400, 410)),
            "P_emaildomain": "example.com",
            "TransactionAmt": float(rng.uniform(10, 200)),
            "isFraud": 0,
        })

    # --- val rows ---
    val_records = []
    for i in range(n_val):
        val_records.append({
            "TransactionID": 9000 + i,
            "TransactionDT": 50000 + i * 10,
            "card1": int(rng.integers(300, 310)),
            "addr1": int(rng.integers(400, 410)),
            "P_emaildomain": "example.com",
            "TransactionAmt": float(rng.uniform(10, 200)),
            "isFraud": int(i % 20 == 0),   # 1-in-20 fraud so there's at least 1
        })

    data_dir = tmp_path / "data" / "processed"
    data_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(train_records).to_parquet(data_dir / "ieee_train.parquet", index=False)
    pd.DataFrame(val_records).to_parquet(data_dir / "ieee_val.parquet", index=False)


# ---------------------------------------------------------------------------
# Fake ollama module
# ---------------------------------------------------------------------------

class FakeOllama:
    """
    Drop-in replacement for the `ollama` module.
    Tests configure `responses` (a list of strings) and then call chat();
    each call pops the next response. If `raise_on_call` is set, chat()
    raises that exception instead.
    """

    def __init__(self):
        self.responses: list[str] = []
        self.raise_on_call: Exception | None = None
        self.call_count: int = 0

    def chat(self, model=None, messages=None, **kwargs):
        self.call_count += 1
        if self.raise_on_call is not None:
            raise self.raise_on_call
        if self.responses:
            content = self.responses.pop(0)
        else:
            # Default: valid two-section response so tests don't break unless
            # they explicitly set responses or raise_on_call.
            content = (
                "ENTITY SUMMARY: This entity has 2 transactions totaling $550.\n"
                "TOP RISK FACTORS:\n• Prior transactions: 2\n• Amount: $550"
            )
        return {"message": {"content": content}}


# ---------------------------------------------------------------------------
# Session-scoped fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="function")
def tmp_agent_env(tmp_path, monkeypatch):
    """
    Sets up the synthetic data, stubs ollama, changes cwd, then imports
    agent fresh. Yields (agent_module, fake_ollama).
    """
    # 1. Build the synthetic parquet files.
    _make_ieee_parquet(tmp_path)

    # 2. Stub the ollama module BEFORE agent.py is imported.
    fake_ollama = FakeOllama()
    monkeypatch.setitem(sys.modules, "ollama", fake_ollama)

    # 3. Change cwd so agent.py's path resolution finds the synthetic data.
    monkeypatch.chdir(tmp_path)

    # 4. Remove any previously cached agent module so state doesn't leak.
    for key in list(sys.modules.keys()):
        if key == "agent" or key.startswith("agent."):
            del sys.modules[key]

    # 5. Ensure src/ is on sys.path so `import agent` works.
    src_dir = str(Path(__file__).resolve().parents[1] / "src")
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)

    # 6. Import agent fresh.
    agent = importlib.import_module("agent")

    yield agent, fake_ollama

    # 7. Cleanup: remove agent from sys.modules for next test.
    for key in list(sys.modules.keys()):
        if key == "agent" or key.startswith("agent."):
            del sys.modules[key]
