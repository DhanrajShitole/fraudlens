"""
src/agent.py

Phase 6 — Agentic Investigation Copilot for FraudLens.

A narrow, tool-scoped agent: given a flagged TransactionID, it calls three
READ-ONLY tools to gather context, then drafts a structured case file for
a human analyst to review. It never blocks a transaction or takes any
write action — evidence assembly only, by design.

Runs on a local Ollama model (default: llama3.1:8b) — no API key, no
internet required after the model is pulled, no data leaves your machine.

Usage:
    C:\\...\\Python313\\python.exe src\\agent.py

Required packages (install with the full Python 3.13 path):
    C:\\...\\Python313\\python.exe -m pip install ollama scikit-learn
(pandas should already be installed from earlier phases)

Prerequisite: Ollama running locally with the model pulled:
    ollama pull llama3.1:8b
"""

import os
import json
import textwrap
import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

try:
    import ollama
except ImportError:
    raise SystemExit("ollama package not found — install it first (see module docstring).")

BASE_DIR = os.path.abspath(os.path.join(os.getcwd(), ".."))
if not os.path.isdir(os.path.join(BASE_DIR, "data")):
    BASE_DIR = os.getcwd()
DATA_DIR = os.path.join(BASE_DIR, "data", "processed")

MODEL_NAME = "llama3.1:8b"

# ---------------------------------------------------------------------
# In-memory "database": the processed IEEE-CIS data stands in for what
# would be a real Postgres table in the deployed system (Phase 7+).
# Loaded once at import time.
# ---------------------------------------------------------------------

print("Loading transaction data for the agent's tools (this happens once)...")
_train = pd.read_parquet(os.path.join(DATA_DIR, "ieee_train.parquet"))
_val = pd.read_parquet(os.path.join(DATA_DIR, "ieee_val.parquet"))
_all_txns = pd.concat([_train, _val], ignore_index=True)
print(f"  Loaded {len(_all_txns):,} transactions for lookup.")

# A small, hand-written synthetic policy corpus, standing in for a real
# bank's fraud policy manual. Retrieved via TF-IDF (a lightweight, honest
# stand-in for a full RAG/vector-DB setup — no extra infra needed for a
# corpus this small; swap for pgvector if the corpus grows significantly).
_POLICY_CLAUSES = [
    {"id": "POL-001", "title": "Shared Card-Address Velocity",
     "text": "If more than 5 distinct transactions occur on the same card and billing "
             "address combination within a 24-hour window, the case must be escalated "
             "for manual review regardless of individual transaction risk score."},
    {"id": "POL-002", "title": "New Account High-Value Transaction",
     "text": "Transactions exceeding $500 originating from an account or card first "
             "observed within the past 7 days require secondary verification before "
             "processing."},
    {"id": "POL-003", "title": "Email Domain Fraud Ring Indicator",
     "text": "A single billing address associated with more than 3 distinct email "
             "domains within a 30-day period is a strong indicator of a coordinated "
             "fraud ring and should be flagged for network-level investigation, not "
             "just the individual transaction."},
    {"id": "POL-004", "title": "Product Category Risk Weighting",
     "text": "Digital goods and gift-card-equivalent product categories carry elevated "
             "fraud risk due to instant, irreversible fulfillment. Transactions in "
             "these categories flagged by the model should be reviewed with a lower "
             "tolerance threshold than physical-goods transactions."},
    {"id": "POL-005", "title": "Repeat False-Positive Entities",
     "text": "If a card or account has been reviewed and cleared as a false positive "
             "more than twice in the past 90 days, analysts should consider "
             "suppressing future low-confidence alerts for that entity to reduce "
             "alert fatigue, while still escalating high-confidence flags."},
]

_policy_texts = [c["text"] for c in _POLICY_CLAUSES]
_vectorizer = TfidfVectorizer(stop_words="english")
_policy_matrix = _vectorizer.fit_transform(_policy_texts)


# ---------------------------------------------------------------------
# TOOL 1: get_entity_history — read-only lookup by TransactionID
# ---------------------------------------------------------------------

def get_entity_history(transaction_id: int) -> dict:
    """Given a TransactionID, return the entity's (card1+addr1) recent
    transaction history: count, total amount, and how many were fraud."""
    row = _all_txns[_all_txns["TransactionID"] == transaction_id]
    if row.empty:
        return {"error": f"TransactionID {transaction_id} not found."}
    row = row.iloc[0]

    card1, addr1 = row.get("card1"), row.get("addr1")
    if pd.isna(card1) or pd.isna(addr1):
        return {
            "transaction_id": int(transaction_id),
            "note": "card1/addr1 missing for this transaction — limited entity history available.",
        }

    entity_txns = _all_txns[(_all_txns["card1"] == card1) & (_all_txns["addr1"] == addr1)]
    return {
        "transaction_id": int(transaction_id),
        "entity_key": f"card1={card1}, addr1={addr1}",
        "total_transactions_by_entity": int(len(entity_txns)),
        "total_amount_by_entity": round(float(entity_txns["TransactionAmt"].sum()), 2),
        "mean_amount_by_entity": round(float(entity_txns["TransactionAmt"].mean()), 2),
        "known_fraud_count_for_entity": int(entity_txns["isFraud"].sum()),
        "this_transaction_amount": round(float(row["TransactionAmt"]), 2),
    }


# ---------------------------------------------------------------------
# TOOL 2: get_related_entities — read-only lookup of shared-identifier signal
# ---------------------------------------------------------------------

def get_related_entities(transaction_id: int) -> dict:
    """Given a TransactionID, return how many distinct emails share this
    exact card+address combination — a proxy for coordinated fraud-ring
    activity. Deliberately grouped by (card1, addr1) TOGETHER, not addr1
    alone — addr1 alone is a coarse regional code in this dataset and
    produces meaningless, noisy counts if used by itself."""
    row = _all_txns[_all_txns["TransactionID"] == transaction_id]
    if row.empty:
        return {"error": f"TransactionID {transaction_id} not found."}
    row = row.iloc[0]

    card1, addr1 = row.get("card1"), row.get("addr1")
    if pd.isna(card1) or pd.isna(addr1):
        return {"note": "card1/addr1 missing for this transaction — cannot compute related entities."}

    same_entity = _all_txns[(_all_txns["card1"] == card1) & (_all_txns["addr1"] == addr1)]
    distinct_emails = (same_entity["P_emaildomain"].nunique()
                        if "P_emaildomain" in same_entity.columns else None)

    return {
        "transaction_id": int(transaction_id),
        "entity_key": f"card1={card1}, addr1={addr1}",
        "distinct_email_domains_for_this_card_and_address": int(distinct_emails) if distinct_emails is not None else "unknown",
        "total_transactions_for_this_exact_entity": int(len(same_entity)),
        "note": "addr1 alone is a coarse regional code in this dataset, not a literal address — "
                "grouping by card1+addr1 together gives a meaningfully specific entity.",
    }


# ---------------------------------------------------------------------
# TOOL 3: retrieve_policy_clause — TF-IDF retrieval over the policy corpus
# ---------------------------------------------------------------------

def retrieve_policy_clause(query: str) -> dict:
    """Given a natural-language query, return the most relevant fraud policy clause."""
    query_vec = _vectorizer.transform([query])
    sims = cosine_similarity(query_vec, _policy_matrix)[0]
    best_idx = int(np.argmax(sims))
    best = _POLICY_CLAUSES[best_idx]
    return {
        "policy_id": best["id"],
        "title": best["title"],
        "text": best["text"],
        "relevance_score": round(float(sims[best_idx]), 3),
    }


# ---------------------------------------------------------------------
# Tool schemas (Ollama/OpenAI-style function-calling format)
# ---------------------------------------------------------------------

_TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "get_entity_history",
            "description": "Get the transaction history for the entity (card+address) behind a given TransactionID.",
            "parameters": {
                "type": "object",
                "properties": {
                    "transaction_id": {"type": "integer", "description": "The TransactionID to investigate."}
                },
                "required": ["transaction_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_related_entities",
            "description": "Get shared-identifier signal (distinct cards/emails at the same billing address) for a given TransactionID — a proxy for coordinated fraud rings.",
            "parameters": {
                "type": "object",
                "properties": {
                    "transaction_id": {"type": "integer", "description": "The TransactionID to investigate."}
                },
                "required": ["transaction_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "retrieve_policy_clause",
            "description": "Retrieve the most relevant fraud policy clause for a natural-language description of the situation.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Natural-language description of the fraud pattern observed, e.g. 'many cards sharing one address'."}
                },
                "required": ["query"],
            },
        },
    },
]

_TOOL_FUNCTIONS = {
    "get_entity_history": get_entity_history,
    "get_related_entities": get_related_entities,
    "retrieve_policy_clause": retrieve_policy_clause,
}

_SYSTEM_PROMPT = textwrap.dedent("""
    You are a fraud investigation assistant. You are given a TransactionID
    that has already been flagged as high-risk by a machine learning model.
    Your job is to gather evidence using the tools available, then produce
    a structured case file for a human analyst to review.

    You do NOT decide whether the transaction is fraud. You do NOT block or
    approve anything. You only assemble evidence and summarize it clearly.

    Use the tools to gather: (1) the entity's transaction history, (2) any
    related-entity / fraud-ring signal, and (3) the most relevant policy
    clause given what you find. Then write a final case file with this
    exact structure:

    ENTITY SUMMARY: <one or two sentences>
    TOP RISK FACTORS: <bulleted list, based only on tool output, no invented facts>
    RELATED ENTITY SIGNAL: <what get_related_entities found, and whether it's concerning>
    RELEVANT POLICY: <policy id and one-sentence explanation of why it applies>
    RECOMMENDED ACTION: <one of: "Escalate for manual review", "Monitor, no immediate action",
    "Insufficient data for a recommendation" — with a one-sentence justification>

    Only state facts you actually retrieved from the tools. If a tool
    returns missing/unknown data, say so plainly rather than guessing.

    IMPORTANT: addr1 is a coarse internal billing region code in this
    dataset — it is NOT a country, a street address, or any kind of
    geographic identifier you can name. Never describe it as a "country,"
    never claim "cross-border" activity, and never say a billing address
    "differs" from anything — none of that information exists in the tool
    outputs. If you are tempted to invent a geographic explanation, stop
    and state only the literal numeric facts instead.

    Your tools report LIFETIME totals for an entity, not time-windowed
    counts. If a policy clause mentions a specific time window (e.g.
    "within 24 hours"), do NOT claim that window was verified — say the
    policy's underlying concern applies in spirit, but note explicitly
    that your tools report lifetime totals, not a specific time period.
""").strip()


# ---------------------------------------------------------------------
# Agent loop
# ---------------------------------------------------------------------

def investigate(transaction_id: int, max_tool_rounds: int = 5, verbose: bool = True) -> str:
    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": f"Investigate TransactionID {transaction_id} and produce the case file."},
    ]

    for round_num in range(max_tool_rounds):
        response = ollama.chat(model=MODEL_NAME, messages=messages, tools=_TOOLS_SCHEMA)
        msg = response["message"]
        messages.append(msg)

        tool_calls = msg.get("tool_calls")
        if not tool_calls:
            # No more tool calls — the model produced its final answer.
            return msg.get("content", "").strip()

        for call in tool_calls:
            fn_name = call["function"]["name"]
            fn_args = call["function"]["arguments"]
            if verbose:
                print(f"  [round {round_num+1}] Agent calls {fn_name}({fn_args})")

            if fn_name not in _TOOL_FUNCTIONS:
                result = {"error": f"Unknown tool '{fn_name}' — not permitted."}
            else:
                try:
                    result = _TOOL_FUNCTIONS[fn_name](**fn_args)
                except Exception as e:
                    result = {"error": f"Tool execution failed: {e}"}

            messages.append({
                "role": "tool",
                "content": json.dumps(result),
            })

    return "Agent did not produce a final case file within the tool-call limit — check for a loop or unclear instructions."


# ---------------------------------------------------------------------
# Test harness: pick a real flagged (fraud=1) transaction and investigate it
# ---------------------------------------------------------------------

if __name__ == "__main__":
    fraud_rows = _val[_val["isFraud"] == 1]
    if len(fraud_rows) == 0:
        raise SystemExit("No fraud examples found in validation set to test with.")

    sample_txn_id = int(fraud_rows.iloc[0]["TransactionID"])
    print(f"\nTesting agent on a known fraud case: TransactionID {sample_txn_id}\n")
    print("=" * 70)

    case_file = investigate(sample_txn_id)

    print("=" * 70)
    print("\nFINAL CASE FILE:\n")
    print(case_file)