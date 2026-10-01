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

Architecture note (latency): earlier versions let the model decide, across
2-3 tool-calling rounds, which of the three tools to call and when. The
validation suite showed the model needed all three tools in every tested
case, and only 2 sentences of its output (ENTITY SUMMARY, TOP RISK FACTORS)
are actually used — everything else is deterministic. So the evidence tools
now always run up front (no model decision needed), and the model is asked
for those two sentences in a SINGLE completion call given the evidence
directly, with at most one retry if that response fails validation. This
cuts the common case from 2-3 model calls to 1, without changing any of the
validated deterministic logic below.

Known limitations, found during evaluation and mitigated below:

1. An 8B local model does not always use the structured tool-calling
   mechanism reliably across multiple rounds — it sometimes writes a tool
   call out as plain JSON text instead of a real structured tool_call.
   The single-call redesign sidesteps this by not asking the model to
   call tools at all; validation + one retry still guards the two lines
   it does write.

2. Policy retrieval used to take a free-text query written by the model in
   the SAME round as the evidence tools, i.e. before it had seen any
   evidence. It searched generically, TF-IDF returned a poor match
   (e.g. a "cross-border" clause with no cross-border evidence), and the
   model then rationalised the mismatch and downgraded its recommendation.
   Fix: the policy query is now built by the SYSTEM from the evidence the
   other two tools actually returned, and a relevance floor returns "no
   applicable policy" rather than a weak match.
"""

import os
import re
import json
import time
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

REQUIRED_SECTIONS = [
    "ENTITY SUMMARY:",
    "TOP RISK FACTORS:",
    "RELATED ENTITY SIGNAL:",
    "RELEVANT POLICY:",
    "RECOMMENDED ACTION:",
]

# Below this TF-IDF cosine similarity, a retrieved clause is treated as "no
# match" instead of being handed to the model to rationalise.
POLICY_MIN_RELEVANCE = 0.15

# Thresholds taken from the policy corpus itself (POL-001: >5, POL-003: >3).
# They are compared against LIFETIME totals, since that is all the tools report.
RING_DOMAIN_THRESHOLD = 3
VELOCITY_TXN_THRESHOLD = 5

# TransactionDT is in seconds, so policy windows can be verified exactly.
SECONDS_24H = 24 * 60 * 60
SECONDS_30D = 30 * 24 * 60 * 60

print("Loading transaction data for the agent's tools (this happens once)...")
_train = pd.read_parquet(os.path.join(DATA_DIR, "ieee_train.parquet"))
_val = pd.read_parquet(os.path.join(DATA_DIR, "ieee_val.parquet"))
_all_txns = pd.concat([_train, _val], ignore_index=True)
print(f"  Loaded {len(_all_txns):,} transactions for lookup.")

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
    {"id": "POL-006", "title": "Cross-Border Transaction Handling",
     "text": "Transactions where the billing address country differs from the "
             "typical account history require additional identity confirmation "
             "before the case can be closed as legitimate."},
]

_policy_texts = [c["text"] for c in _POLICY_CLAUSES]
_vectorizer = TfidfVectorizer(stop_words="english")
_policy_matrix = _vectorizer.fit_transform(_policy_texts)


def _entity_as_of(transaction_id: int):
    """Point-in-time view of the entity (card1+addr1) behind a transaction:
    only rows with TransactionDT <= this transaction's TransactionDT, i.e.
    what a live system could have known when the transaction arrived.
    Returns (row, as_of_df, error_dict_or_None)."""
    row = _all_txns[_all_txns["TransactionID"] == transaction_id]
    if row.empty:
        return None, None, {"error": f"TransactionID {transaction_id} not found."}
    row = row.iloc[0]

    card1, addr1 = row.get("card1"), row.get("addr1")
    if pd.isna(card1) or pd.isna(addr1):
        return row, None, {
            "transaction_id": int(transaction_id),
            "note": "card1/addr1 missing for this transaction — limited entity history available.",
        }

    entity = _all_txns[(_all_txns["card1"] == card1) & (_all_txns["addr1"] == addr1)]
    if "TransactionDT" in entity.columns and pd.notna(row.get("TransactionDT")):
        entity = entity[entity["TransactionDT"] <= row["TransactionDT"]]
    return row, entity, None


def get_entity_history(transaction_id: int) -> dict:
    """Given a TransactionID, return the entity's (card1+addr1) history AS OF
    that transaction: count, total amount, and how many EARLIER transactions
    were known fraud. Never uses transactions that happened later, and never
    counts the transaction's own label (that would leak the answer)."""
    row, entity, err = _entity_as_of(transaction_id)
    if err is not None:
        return err

    if "TransactionDT" in entity.columns and pd.notna(row.get("TransactionDT")):
        earlier = entity[(entity["TransactionDT"] < row["TransactionDT"])
                         & (entity["TransactionID"] != transaction_id)]
    else:
        earlier = entity[entity["TransactionID"] != transaction_id]

    windowed = {}
    if "TransactionDT" in entity.columns and pd.notna(row.get("TransactionDT")):
        last_24h = entity[entity["TransactionDT"] > row["TransactionDT"] - SECONDS_24H]
        windowed["transactions_last_24_hours"] = int(len(last_24h))

    return {
        **windowed,
        "transaction_id": int(transaction_id),
        "entity_key": f"card1={row['card1']}, addr1={row['addr1']}",
        "total_transactions_by_entity": int(len(entity)),
        "total_amount_by_entity": round(float(entity["TransactionAmt"].sum()), 2),
        "mean_amount_by_entity": round(float(entity["TransactionAmt"].mean()), 2),
        "known_fraud_count_for_entity": int(earlier["isFraud"].sum()),
        "this_transaction_amount": round(float(row["TransactionAmt"]), 2),
        "note": "Totals cover this entity up to and including this transaction only "
                "(transactions_last_24_hours is the 24 hours up to this transaction). "
                "known_fraud_count_for_entity counts strictly EARLIER transactions "
                "confirmed as fraud, never this one and never later ones.",
    }


def get_related_entities(transaction_id: int) -> dict:
    """Given a TransactionID, return how many distinct emails share this
    exact card+address combination up to this transaction — a proxy for
    coordinated fraud-ring activity. Deliberately grouped by (card1, addr1)
    TOGETHER, not addr1 alone — addr1 alone is a coarse regional code in this
    dataset and produces meaningless, noisy counts if used by itself."""
    row, same_entity, err = _entity_as_of(transaction_id)
    if err is not None:
        return err if "error" in err else {
            "note": "card1/addr1 missing for this transaction — cannot compute related entities."}

    distinct_emails = (same_entity["P_emaildomain"].nunique()
                        if "P_emaildomain" in same_entity.columns else None)

    windowed = {}
    if ("P_emaildomain" in same_entity.columns and "TransactionDT" in same_entity.columns
            and pd.notna(row.get("TransactionDT"))):
        last_30d = same_entity[same_entity["TransactionDT"] > row["TransactionDT"] - SECONDS_30D]
        windowed["distinct_email_domains_last_30_days"] = int(last_30d["P_emaildomain"].nunique())

    return {
        **windowed,
        "transaction_id": int(transaction_id),
        "entity_key": f"card1={row['card1']}, addr1={row['addr1']}",
        "distinct_email_domains_for_this_card_and_address": int(distinct_emails) if distinct_emails is not None else "unknown",
        "total_transactions_for_this_exact_entity": int(len(same_entity)),
        "note": "Counts cover transactions up to and including this one (no later data). "
                "addr1 alone is a coarse regional code in this dataset, not a literal address — "
                "grouping by card1+addr1 together gives a meaningfully specific entity.",
    }


def retrieve_policy_clause(query: str) -> dict:
    """Low-level TF-IDF lookup: return the policy clause most similar to `query`.
    investigate() does NOT pass model-written queries to this any more — it
    goes through _policy_from_evidence(), which builds the query from tool
    output and applies a relevance floor."""
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


def _evidence_policy_queries(tool_results: dict) -> list:
    """Turn the evidence the tools actually returned into policy search
    queries, in priority order: [(signal_description, query, window_verified)].
    Uses the policy's own time window (30 days / 24 hours, computed from
    TransactionDT as of the transaction) whenever the tools provide it, and
    only falls back to lifetime totals — flagged as unverified — if not."""
    history = tool_results.get("get_entity_history", {})
    related = tool_results.get("get_related_entities", {})
    queries = []

    n30 = related.get("distinct_email_domains_last_30_days")
    if isinstance(n30, int):
        if n30 > RING_DOMAIN_THRESHOLD:
            queries.append((
                f"{n30} distinct email domains for this card and address in the 30 days "
                f"up to this transaction (policy threshold: more than {RING_DOMAIN_THRESHOLD})",
                "multiple distinct email domains one billing address coordinated fraud ring",
                True,
            ))
    else:
        n_domains = related.get("distinct_email_domains_for_this_card_and_address")
        if isinstance(n_domains, int) and n_domains > RING_DOMAIN_THRESHOLD:
            queries.append((
                f"{n_domains} distinct email domains for this card and address, lifetime "
                f"(policy threshold: more than {RING_DOMAIN_THRESHOLD} within 30 days; window not verified)",
                "multiple distinct email domains one billing address coordinated fraud ring",
                False,
            ))

    n24 = history.get("transactions_last_24_hours")
    if isinstance(n24, int):
        if n24 > VELOCITY_TXN_THRESHOLD:
            queries.append((
                f"{n24} transactions on this card and address in the 24 hours up to this "
                f"transaction (policy threshold: more than {VELOCITY_TXN_THRESHOLD})",
                "several transactions same card and billing address velocity",
                True,
            ))
    else:
        n_txns = related.get("total_transactions_for_this_exact_entity",
                             history.get("total_transactions_by_entity"))
        if isinstance(n_txns, int) and n_txns > VELOCITY_TXN_THRESHOLD:
            queries.append((
                f"{n_txns} lifetime transactions on this card and address "
                f"(policy threshold: more than {VELOCITY_TXN_THRESHOLD} in 24 hours; window not verified)",
                "several transactions same card and billing address velocity",
                False,
            ))
    return queries


def _policy_from_evidence(tool_results: dict) -> dict:
    """Evidence-driven replacement for a model-written policy query. The
    first signal (priority order) that clears the relevance floor becomes
    the primary match; other matching clauses are listed alongside it."""
    matches = []
    for signal, query, verified in _evidence_policy_queries(tool_results):
        hit = retrieve_policy_clause(query)
        if hit["relevance_score"] >= POLICY_MIN_RELEVANCE:
            hit["matched_on"] = signal
            hit["window_verified"] = verified
            matches.append(hit)

    if not matches:
        return {
            "policy_id": None,
            "note": "No policy clause is triggered by the evidence retrieved. "
                    "Do not name or infer a policy.",
        }

    primary = dict(matches[0])
    primary["also_matching"] = [m["policy_id"] for m in matches[1:] if m["policy_id"] != primary["policy_id"]]
    return primary


_TOOL_FUNCTIONS = {
    "get_entity_history": get_entity_history,
    "get_related_entities": get_related_entities,
    "retrieve_policy_clause": retrieve_policy_clause,
}

_ADJ = r"(?:high|large|low|small|significant|unusual|frequent|excessive)"
_NOUN = r"(?:number|count|volume|amount|level|frequency|quantity)"


def _scrub_magnitude_language(text: str) -> str:
    """Deterministically remove unsupported magnitude wording instead of
    paying for another model generation: "has a high number of X" -> "has X",
    "the high number" -> "the number". Literal numbers stay untouched."""
    text = re.sub(rf"\b(?:a|an)\s+{_ADJ}\s+{_NOUN}\s+of\s+", "", text, flags=re.IGNORECASE)
    text = re.sub(rf"\b{_ADJ}\s+({_NOUN})\b", r"\1", text, flags=re.IGNORECASE)
    return text


_WINDOW_NOTE = ("Note: the tools could not verify the policy's time window, so this count "
                "is a lifetime total up to this transaction.")


def _related_entity_signal(tool_results: dict) -> str:
    """Written by code from tool output, never by the model: it kept mislabelling
    lifetime counts as 30-day counts and inventing comparisons."""
    r = tool_results.get("get_related_entities", {})
    lifetime = r.get("distinct_email_domains_for_this_card_and_address")
    if not isinstance(lifetime, int):
        return "Related-entity data is unavailable for this transaction (missing card/address identifiers or lookup error)."
    text = (f"{lifetime} distinct email domains across this card and address's history "
            f"up to this transaction")
    d30 = r.get("distinct_email_domains_last_30_days")
    if isinstance(d30, int):
        text += f"; {d30} in the 30 days up to this transaction"
    return text + "."


def _policy_section(tool_results: dict) -> str:
    p = tool_results.get("retrieve_policy_clause", {})
    if not p.get("policy_id"):
        return "No policy clause is triggered by the retrieved evidence."
    text = f"{p['policy_id']}: {p.get('title', '')} — {p.get('text', '')} Triggered by: {p.get('matched_on', 'threshold exceeded')}."
    if p.get("window_verified") is False:
        text += f" {_WINDOW_NOTE}"
    return text


def _recommended_action(tool_results: dict) -> str:
    """Rules-based recommendation derived only from tool evidence. The
    language model writes narrative sections; it never picks the action,
    because an 8B model repeatedly contradicted its own evidence here."""
    h = tool_results.get("get_entity_history", {})
    p = tool_results.get("retrieve_policy_clause", {})
    if "known_fraud_count_for_entity" not in h or "error" in h:
        return ("Insufficient data for a recommendation — entity history could not be "
                "assembled for this transaction (missing card/address identifiers or lookup error).")

    reasons = []
    if p.get("policy_id"):
        reasons.append(f"{p['policy_id']} is triggered: {p.get('matched_on', 'threshold exceeded')}")
    prior = h.get("known_fraud_count_for_entity", 0)
    if isinstance(prior, int) and prior > 0:
        reasons.append(f"{prior} prior confirmed fraud case(s) for this entity")

    if reasons:
        return "Escalate for manual review — " + "; ".join(reasons) + "."
    return ("Monitor, no immediate action — no policy clause is triggered by the evidence "
            "retrieved and this entity has 0 prior confirmed fraud cases.")


def _entity_summary(tool_results: dict) -> str:
    h = tool_results.get("get_entity_history", {})
    if "total_transactions_by_entity" not in h:
        return "Entity history is unavailable for this transaction."
    return (f"This card+address entity has {h['total_transactions_by_entity']} transactions "
            f"totaling ${h.get('total_amount_by_entity', 'unknown')}, up to and including this one.")


def _model_section(text: str, label: str) -> str:
    """Pull one section body out of the model's case file."""
    i = text.find(label)
    if i == -1:
        return ""
    start = i + len(label)
    ends = [j for j in (text.find(l, start) for l in REQUIRED_SECTIONS) if j != -1]
    return text[start:(min(ends) if ends else len(text))].strip()


# ---- validation of the two model-written sections ---------------------------
_NUM_RE = re.compile(r"\d[\d,]*(?:\.\d+)?")
_LIST_MARKER_RE = re.compile(r"(?m)^\s*(?:[•*\-]|\d+[.)])\s+")
_SNAKE_RE = re.compile(r"\b[a-z]+(?:_[a-z0-9]+)+\b")
_BANNED_TEXT_RE = re.compile(
    r"\b(concern(?:ing|s|ed)?|warrants?|suspicious|suspected?|flagged|high-risk|low-risk|"
    r"cross-border|countr(?:y|ies)|fraud ring|unusual|abnormal|anomal\w*)\b", re.IGNORECASE)
_WINDOW_30D_RE = re.compile(r"\b30[- ]?days?\b|\bmonth\b", re.IGNORECASE)
_WINDOW_24H_RE = re.compile(r"\b24[- ]?hours?\b|\blast day\b", re.IGNORECASE)


def _numbers_in(text: str) -> set:
    text = _LIST_MARKER_RE.sub("", text)
    return {round(float(m.replace(",", "").rstrip(".")), 2) for m in _NUM_RE.findall(text)}


def _grounded_numbers(value) -> set:
    """Every number that literally appears as a value in the tool results."""
    if isinstance(value, bool):
        return set()
    if isinstance(value, (int, float)):
        return {round(float(value), 2)}
    if isinstance(value, str):
        return _numbers_in(value)
    if isinstance(value, dict):
        return set().union(*[_grounded_numbers(v) for v in value.values()]) if value else set()
    if isinstance(value, (list, tuple)):
        return set().union(*[_grounded_numbers(v) for v in value]) if value else set()
    return set()


def _model_text_clean(text: str, tool_results: dict) -> bool:
    """True only if a model-written section makes no claim the tool results
    don't support: no unsupported numbers, no time window attached to a count
    that isn't that window's count, no evaluative/geographic words, no field names."""
    if not text:
        return False
    if _BANNED_TEXT_RE.search(text) or _SNAKE_RE.search(text):
        return False
    if not _numbers_in(text) <= _grounded_numbers(tool_results):
        return False
    n24 = tool_results.get("get_entity_history", {}).get("transactions_last_24_hours")
    d30 = tool_results.get("get_related_entities", {}).get("distinct_email_domains_last_30_days")
    for chunk in re.split(r"(?<=[.!?])\s+|\n", text):
        nums = _numbers_in(chunk)
        if _WINDOW_30D_RE.search(chunk) and not (isinstance(d30, int) and float(d30) in nums):
            return False
        if _WINDOW_24H_RE.search(chunk) and not (isinstance(n24, int) and float(n24) in nums):
            return False
    return True


def _fallback_risk_bullets(tool_results: dict) -> str:
    h = tool_results.get("get_entity_history", {})
    return "\n".join([
        f"• Lifetime transactions: {h.get('total_transactions_by_entity', 'unknown')}",
        f"• Mean transaction amount: ${h.get('mean_amount_by_entity', 'unknown')}",
        f"• Prior confirmed fraud cases for this entity: {h.get('known_fraud_count_for_entity', 'unknown')}",
        f"• This transaction amount: ${h.get('this_transaction_amount', 'unknown')}",
    ])


def _assemble_case_file(summary: str, risks: str, tool_results: dict) -> str:
    """Build the final case file from: the model's two narrative sections
    (already validated by the caller, with fallbacks already applied) plus
    the three always-deterministic sections."""
    return "\n".join([
        f"ENTITY SUMMARY: {summary}",
        f"TOP RISK FACTORS:\n{risks}",
        f"RELATED ENTITY SIGNAL: {_related_entity_signal(tool_results)}",
        f"RELEVANT POLICY: {_policy_section(tool_results)}",
        f"RECOMMENDED ACTION: {_recommended_action(tool_results)}",
    ])


def _fallback_case_file(tool_results: dict,
                        reason: str = "automated LLM synthesis failed validation") -> str:
    """Last resort: build a well-formed case file directly from the raw tool
    outputs, with no LLM involvement, so the analyst still receives accurate
    evidence in the standard format."""
    return "\n".join([
        f"[NOTE: {reason}; this case file was assembled directly from the tool outputs "
        "with no model-written text.]",
        "",
        f"ENTITY SUMMARY: {_entity_summary(tool_results)}",
        f"TOP RISK FACTORS:\n{_fallback_risk_bullets(tool_results)}",
        f"RELATED ENTITY SIGNAL: {_related_entity_signal(tool_results)}",
        f"RELEVANT POLICY: {_policy_section(tool_results)}",
        f"RECOMMENDED ACTION: {_recommended_action(tool_results)}",
    ])


class _LLMError(Exception):
    """Raised when the Ollama call itself fails (e.g. the CUDA crash)."""


_SYSTEM_PROMPT = textwrap.dedent("""
    You are a fraud investigation assistant. You will be given the complete
    evidence already gathered by the system for one TransactionID — you do
    NOT call any tools, and you are NOT told what any model scored this
    transaction, so never say it was "flagged", "high-risk" or "low-risk".

    Using ONLY the evidence provided, respond with EXACTLY two lines, in
    this exact format and nothing else — no preamble, no other sections,
    no markdown:

    ENTITY SUMMARY: <one or two sentences describing the card+address entity's
    transaction count and total amount, from the evidence>
    TOP RISK FACTORS:
    <bulleted list of literal facts from the evidence, one per line, starting with "• ">

    Only state facts that literally appear in the evidence below. Never
    invent a comparison, a benchmark, or an "overall"/"typical"/"average"
    reference point — if you want to say a number is notable, you have
    no baseline to judge that against, so simply state the number and
    stop there.

    addr1 is a coarse internal billing region code — NOT a country, a
    street address, or any geographic identifier. Never call it a
    "country" and never claim "cross-border" activity.

    Counts are lifetime totals for the entity up to this transaction,
    except transactions_last_24_hours (the 24 hours up to it) and
    distinct_email_domains_last_30_days (the 30 days up to it). Never
    attach a time window to a number unless the evidence field name
    states that window.

    Do NOT describe any number as "high," "low," "large," "small,"
    "frequent," "unusual," "concerning," or "suspicious" — you have no
    baseline for what is normal for this kind of entity. State literal
    numbers only and let the analyst judge them.

    Never write a raw field name (such as known_fraud_count_for_entity)
    — say "prior confirmed fraud cases" instead.
""").strip()

_RETRY_NOTE = ("\n\nYour previous response did not follow the required format or made an "
               "unsupported claim. Follow the instructions exactly this time: two lines only, "
               "literal numbers only, no invented comparisons.")


def _build_evidence_prompt(tool_results: dict) -> str:
    return ("Evidence gathered for this investigation:\n\n"
            f"get_entity_history: {json.dumps(tool_results.get('get_entity_history', {}))}\n\n"
            f"get_related_entities: {json.dumps(tool_results.get('get_related_entities', {}))}\n\n"
            "Write the two lines now.")


def _generate_narrative(tool_results: dict, verbose: bool, stats: dict, messages_log: list) -> tuple:
    """One completion call (plus at most one retry) asking ONLY for the two
    model-owned lines, given evidence that was already gathered
    deterministically. Returns (summary, risks, model_output_used: bool)."""
    user_prompt = _build_evidence_prompt(tool_results)
    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    for attempt in range(2):  # first try + one retry, never more
        t = time.perf_counter()
        try:
            response = ollama.chat(model=MODEL_NAME, messages=messages)
        except Exception as e:
            raise _LLMError(f"{type(e).__name__}: {str(e)[:200]}") from e
        finally:
            stats["secs"] += time.perf_counter() - t
            stats["calls"] += 1

        text = (response["message"].get("content") or "").strip()
        if verbose:
            print(f"  [model call {attempt+1}] {len(text)} chars")

        summary = _model_section(text, "ENTITY SUMMARY:") or _model_section("ENTITY SUMMARY: " + text if "ENTITY SUMMARY:" not in text else text, "ENTITY SUMMARY:")
        # Simpler and more robust for a 2-section-only response than the old
        # multi-section parser: split on the TOP RISK FACTORS: label directly.
        if "ENTITY SUMMARY:" in text and "TOP RISK FACTORS:" in text:
            summary = text.split("ENTITY SUMMARY:", 1)[1].split("TOP RISK FACTORS:", 1)[0].strip()
            risks = text.split("TOP RISK FACTORS:", 1)[1].strip()
        else:
            summary, risks = "", ""

        if _model_text_clean(summary, tool_results) and _model_text_clean(risks, tool_results):
            return summary, risks, True

        if verbose:
            print(f"  [!] Attempt {attempt+1} failed validation — "
                  f"{'retrying' if attempt == 0 else 'giving up, using tool-derived text'}.")
        messages.append({"role": "assistant", "content": text})
        messages.append({"role": "user", "content": _RETRY_NOTE})

    return _entity_summary(tool_results), _fallback_risk_bullets(tool_results), False


def investigate(transaction_id: int, verbose: bool = True) -> str:
    stats = {"calls": 0, "secs": 0.0}
    t_start = time.perf_counter()
    messages_log = []

    # Evidence is always needed and is never a model decision, so gather it
    # deterministically up front — this is what cut the common case from
    # 2-3 model calls down to 1 (see module docstring).
    tool_results = {
        "get_entity_history": get_entity_history(transaction_id),
        "get_related_entities": get_related_entities(transaction_id),
    }
    if verbose:
        print(f"  [evidence] get_entity_history, get_related_entities gathered deterministically")
    tool_results["retrieve_policy_clause"] = _policy_from_evidence(tool_results)
    if verbose:
        print(f"  [evidence] policy search -> {tool_results['retrieve_policy_clause'].get('policy_id')}")

    def _report(path: str) -> None:
        total = time.perf_counter() - t_start
        print(f"[agent] TransactionID {transaction_id}: {stats['calls']} model call(s), "
              f"{stats['secs']:.1f}s in model of {total:.1f}s total, path={path}")

    # If the entity itself couldn't be resolved (missing card1/addr1, or the
    # transaction wasn't found), there's nothing for the model to write about —
    # skip straight to the deterministic case file, no model call at all.
    if "total_transactions_by_entity" not in tool_results["get_entity_history"]:
        _report("no_entity_data")
        return _fallback_case_file(tool_results, reason="entity history unavailable for this transaction")

    try:
        summary, risks, used_model = _generate_narrative(tool_results, verbose, stats, messages_log)
        case_file = _assemble_case_file(_scrub_magnitude_language(summary) if used_model else summary,
                                        _scrub_magnitude_language(risks) if used_model else risks,
                                        tool_results)
        _report("ok_model_narrative" if used_model else "ok_fallback_narrative")
        return case_file

    except _LLMError as e:
        # The evidence tools are deterministic, so the analyst still gets a
        # complete, accurate case file even when the language model is down.
        print(f"[agent] language model call failed ({e}) — building the case file from tool outputs only.")
        _report("llm_unavailable")
        return _fallback_case_file(
            tool_results, reason="the language model was unavailable, so no model-written text is included")


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