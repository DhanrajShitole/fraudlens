# FraudLens — Agentic AI & Investigation Copilot Deep Dive

**Author:** Dhanraj Shitole  
**Target Audience:** AI Engineers, LLM System Architects, Compliance Officers  
**Source Code:** `src/agent.py`  
**Evaluation Logs:** `reports/`  

---

## 1. Introduction: Generative AI in High-Stakes Compliance

In banking and fraud prevention, deploying unconstrained Large Language Models (LLMs) as autonomous decision-makers is fundamentally unacceptable due to legal, regulatory, and financial compliance requirements:
1. **Hallucination Risk:** Generative language models predict token sequences based on distributional probabilities, not logical truth. In compliance reports, fabricating an entity transaction count or a policy violation can expose an institution to regulatory fines.
2. **Temporal Label Leakage:** In naive tool calling, an LLM querying a customer's history might inadvertently observe future confirmed fraud flags that occurred days *after* the transaction being evaluated, silently leaking the target label.
3. **Sycophancy & Drift:** An LLM prompted with an elevated fraud score tends to rationalize that score, creating fabricated justifications to agree with the model rather than assessing raw facts objectively.

**FraudLens resolves this through an evidence-grounded agent architecture governed by one non-negotiable rule:**
> **Code makes decisions, models score, and language models write prose.**
> The LLM is never permitted to decide whether a transaction is fraudulent, never selects the policy action, and is strictly restricted to drafting two narrative sentences whose numerical contents are validated against tool output.

---

## 2. Agent Redesign: From Multi-Turn Tool Calling to Single-Turn Synthesis

### 2.1 The Latency & Reliability Problem in Multi-Turn Tool Calling
In early prototypes (Evaluation Round 1), FraudLens followed standard multi-turn tool calling:
- Round 1: Model receives Transaction ID and decides whether to call `get_entity_history` or `get_related_entities`.
- Round 2: Model receives tool outputs and writes a search query to call `retrieve_policy_clause`.
- Round 3: Model receives policy text and outputs a 5-section case file.

**Evaluation revealed fatal flaws in this multi-turn pattern:**
1. **Tool Invocation Failure:** Small local models (Llama 3.1 8B) frequently emitted raw JSON strings in plain text instead of executing native tool calling, causing parsing crashes.
2. **Blind Policy Retrieval:** In Round 2, the model attempted to write a policy search query *before* fully evaluating entity evidence. It searched generically (e.g., `"foreign transaction fraud"`), TF-IDF matched an irrelevant clause (e.g., cross-border policy), and the model rationalized the mismatch.
3. **Severe Latency:** Executing 3 sequential model completions on consumer hardware (especially CPU-mode Ollama) required 45 to 90 seconds per investigation.

### 2.2 The Single-Turn Deterministic Pipeline
Empirical validation showed that human analysts required all three evidence categories in 100% of cases, and the model's actual contribution was confined to two narrative sections:
- `ENTITY SUMMARY`
- `TOP RISK FACTORS`

All remaining sections (`RELATED ENTITY SIGNAL`, `RELEVANT POLICY`, `RECOMMENDED ACTION`) are purely deterministic logic.

**The Redesigned Architecture:**
1. Deterministic Python executes `get_entity_history` and `get_related_entities` up front.
2. Deterministic Python builds the policy search query directly from tool counts and queries TF-IDF with a relevance floor.
3. If the entity lacks identifiers (`card1` or `addr1` is NaN), the system immediately outputs an "insufficient data" template in **0 model calls**.
4. If entity data exists, a single structured prompt is sent to Llama 3.1 8B asking for exactly two lines.
5. An automated validator inspects the response; if clean, the case file is assembled. If invalid, the model receives one retry; if still invalid, the system falls back to tool-derived bullets.

*Latency Result:* Reduced model calls from 2–3 down to **exactly 1**, slashing runtime by over 65% with zero degradation in analytical quality.

---

## 3. The Four Hallucination Classes & How They Were Solved

During iterative validation against held-out fraud cases, four distinct hallucination classes were discovered and systematically eliminated:

### Hallucination Class 1: Geographic Hallucination
- **Failure Mode:** The model read the `addr1` column (e.g. `addr1 = 299`) and wrote: *"The transaction originated from country 299, indicating an unauthorized cross-border payment."*
- **Root Cause:** `addr1` in IEEE-CIS is an anonymized internal billing zip/region code, not an ISO country code.
- **Fix:** Explicit system prompt constraint prohibiting geographic inferences: *"addr1 is a coarse internal billing region code — NOT a country, street address, or geographic identifier. Never call it a country and never claim cross-border activity."*

### Hallucination Class 2: Data-Granularity Aggregation Bug
- **Failure Mode:** The `get_related_entities` tool originally grouped transactions on `addr1` alone, reporting: *"723 other cards share this address."*
- **Root Cause:** In IEEE-CIS, thousands of distinct cardholders share coarse billing region `299`. Grouping by `addr1` alone produced meaningless false fraud-ring signals.
- **Fix:** Redesigned tool to group strictly by the composite tuple `(card1, addr1)` together, matching the granularity of engineered modeling features.

### Hallucination Class 3: Temporal Label Leakage & Window Overreach
- **Failure Mode:** The model asserted: *"Entity had 5 transactions within a 24-hour window, violating Policy POL-001."* However, the tool only reported lifetime totals.
- **Root Cause:** The entity history tool lacked point-in-time window filtering. Furthermore, calculating prior fraud originally included the transaction being investigated, silently leaking the label.
- **Fix:**
  1. Gated all tool queries strictly to `TransactionDT <= this_transaction.TransactionDT`.
  2. Prior fraud count strictly requires `TransactionDT < this_transaction.TransactionDT` and `TransactionID != this_transaction.TransactionID`.
  3. Added explicit 24-hour and 30-day window calculators using `SECONDS_24H` ($86,400$) and `SECONDS_30D` ($2,592,000$).

### Hallucination Class 4: Vacuous Comparative Framing
- **Failure Mode:** The model wrote: *"The mean transaction amount of $144.14 is greater than the overall mean of $144.14."*
- **Root Cause:** The number was technically present in the evidence tool output, and the phrasing used no banned words, so early validators let it through. However, the comparison was logically vacuous and self-referential.
- **Fix:** Hardened validator with regex `_COMPARATIVE_RE`. Any sentence containing comparative phrasing (*greater than, less than, higher than, lower than, more than, above, below, compared to, versus, relative to, than the overall/average/typical/mean*) is rejected unless the sentence contains at least **two distinct grounded numbers**.

---

## 4. The Three Read-Only Deterministic Tools

### 4.1 `get_entity_history(transaction_id: int) -> dict`
Extracts point-in-time card+address transaction metrics as of the investigated transaction:

```python
{
    "transaction_id": 3141064,
    "entity_key": "card1=100, addr1=200",
    "total_transactions_by_entity": 12,
    "total_amount_by_entity": 1729.68,
    "mean_amount_by_entity": 144.14,
    "known_fraud_count_for_entity": 1,
    "this_transaction_amount": 350.00,
    "transactions_last_24_hours": 6,
    "note": "Totals cover this entity up to and including this transaction only..."
}
```

### 4.2 `get_related_entities(transaction_id: int) -> dict`
Evaluates multi-identity clustering across the exact same card and billing address tuple:

```python
{
    "transaction_id": 3141064,
    "entity_key": "card1=100, addr1=200",
    "distinct_email_domains_for_this_card_and_address": 4,
    "distinct_email_domains_last_30_days": 4,
    "total_transactions_for_this_exact_entity": 12,
    "note": "Counts cover transactions up to and including this one..."
}
```

### 4.3 `retrieve_policy_clause(query: str) -> dict`
Applies TF-IDF vectorization and cosine similarity over the six enterprise fraud policy clauses:
- `POL-001`: Shared Card-Address Velocity (>5 txns in 24h).
- `POL-002`: New Account High-Value Transaction (>$500 within 7 days).
- `POL-003`: Email Domain Fraud Ring Indicator (>3 distinct email domains in 30 days).
- `POL-004`: Product Category Risk Weighting (Digital goods lower tolerance).
- `POL-005`: Repeat False-Positive Entities (>2 cleared false-positives in 90 days).
- `POL-006`: Cross-Border Transaction Handling (Billing country mismatch).

*Relevance Floor:* If maximum cosine similarity $< 0.15$, the system returns `policy_id: null`, preventing spurious policy citations.

---

## 5. Hardened Automated Validator Implementation

Before any text authored by Llama 3.1 8B is incorporated into a case file, it must pass `_model_text_clean(text, tool_results)`:

```python
def _model_text_clean(text: str, tool_results: dict) -> bool:
    if not text:
        return False
    # 1. Banned evaluative / speculative words
    if _BANNED_TEXT_RE.search(text) or _SNAKE_RE.search(text):
        return False
    # 2. Number grounding in evidence tools
    grounded = _grounded_numbers(tool_results)
    if not _numbers_in(text) <= grounded:
        return False
    # 3. Temporal window checks
    n24 = tool_results.get("get_entity_history", {}).get("transactions_last_24_hours")
    d30 = tool_results.get("get_related_entities", {}).get("distinct_email_domains_last_30_days")
    for chunk in re.split(r"(?<=[.!?])\s+|\n", text):
        nums = _numbers_in(chunk)
        if _WINDOW_30D_RE.search(chunk) and not (isinstance(d30, int) and float(d30) in nums):
            return False
        if _WINDOW_24H_RE.search(chunk) and not (isinstance(n24, int) and float(n24) in nums):
            return False
        # 4. Comparative framing guard: requires >= 2 distinct grounded numbers
        if _COMPARATIVE_RE.search(chunk):
            sentence_grounded = nums & grounded
            if len(sentence_grounded) < 2:
                return False
    return True
```

### Numerical Grounding Isolation (`_grounded_numbers`)
To prevent the model from citing numbers found in policy text or system notes:
- The entire `retrieve_policy_clause` subdictionary is skipped.
- Any dictionary key named `"note"` at any nesting level is skipped.
- Numbers embedded inside the `entity_key` string (e.g. `card1=100, addr1=200`) are accepted as grounded identifiers.

---

## 6. Deterministic Case File Synthesis & Resilience

### 6.1 The 5 Required Sections
The final case file returned to the analyst strictly adheres to standard compliance reporting:
1. `ENTITY SUMMARY:` Narrative synthesis of card+address transaction volume and spend.
2. `TOP RISK FACTORS:` Bulleted literal facts from evidence.
3. `RELATED ENTITY SIGNAL:` Written deterministically by code from `get_related_entities`.
4. `RELEVANT POLICY:` Written deterministically by code, displaying clause ID, title, matched threshold, and verification status.
5. `RECOMMENDED ACTION:` Written deterministically by code:
   - If prior confirmed fraud $> 0$ OR any policy is triggered $\to$ `"Escalate for manual review — [reasons]."`
   - Otherwise $\to$ `"Monitor, no immediate action — no policy clause is triggered and this entity has 0 prior confirmed fraud cases."`

### 6.2 Zero-Data-Loss Failure Fallback
If Ollama is stopped, or if the model fails validation on both attempt 1 and retry attempt 2:
The agent catches `_LLMError` and immediately synthesizes a fallback case file using `_fallback_case_file(tool_results)`:
- Prepends: `[NOTE: automated LLM synthesis failed validation / language model was unavailable; this case file was assembled directly from tool outputs with no model-written text.]`
- Automatically fills `ENTITY SUMMARY` and `TOP RISK FACTORS` using raw Python templates populated with grounded tool data.
- The analyst receives a complete, formatted case file with 100% accurate evidence.

