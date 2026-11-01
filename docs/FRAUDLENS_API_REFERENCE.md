# FraudLens — REST API Reference & Specification

**Service Name:** FraudLens Scoring & Investigation API  
**Default Base URL:** `http://127.0.0.1:8000`  
**Framework:** FastAPI 0.136.1 (ASGI)  
**OpenAPI Specification:** Available interactively at `/docs` (Swagger UI) and `/redoc` (ReDoc)  

---

## 1. Overview & General Standards

The FraudLens backend exposes RESTful endpoints supporting real-time fraud probability scoring, game-theoretic SHAP reason-code decomposition, and agentic investigation synthesis.

### Core Protocol Principles:
- **Transport:** HTTP / 1.1 JSON over TCP.
- **Content-Type:** `application/json` for all requests and responses.
- **Cross-Origin Resource Sharing (CORS):** Enabled for all origins (`allow_origins=["*"]`) to allow the local browser-based analyst console (`frontend/dashboard.html`) to access endpoints without CORS restrictions.
- **Data Lookup Scope:** In the current demonstration version, transactions are queried by `TransactionID` from the combined IEEE-CIS processed dataset (train + validation, totaling 501,959 transactions).

---

## 2. Global Error Handling & HTTP Status Codes

The API returns standard RFC 7807 problem details in error conditions:

| Status Code | Meaning | Typical Trigger |
|---|---|---|
| **200 OK** | Successful execution | Transaction scored, case retrieved, or health check passed. |
| **404 Not Found** | Resource missing | `TransactionID` does not exist in dataset, or `case_id` does not exist in SQLite. |
| **500 Internal Error** | Module failure | Missing agent module or database connection failure. |
| **503 Service Unavailable** | Dependency unavailable | Local Ollama service is stopped, unresponsive, or crashed during LLM generation. |

Error Response Format:
```json
{
  "detail": "TransactionID 9999999999 not found in dataset."
}
```

---

## 3. Endpoint Specifications

### 3.1 Service Health Check

#### `GET /health`
Verifies that the API service is operational and reports which model backend is active.

- **Request Headers:** None required.
- **Request Parameters:** None.
- **Response Model:** `dict`

#### Success Response (`200 OK`):
```json
{
  "status": "ok",
  "model_loaded": true,
  "model_type": "LightGBM (tuned, Phase 5 hyperparameters, SMOTE-trained)",
  "model_source": "registry:fraudlens-ieee-lightgbm@production"
}
```
*Note on `model_source`:* If MLflow registry resolution succeeds, `model_source` will indicate the registry model name and alias (`registry:fraudlens-ieee-lightgbm@production`). If the registry path cannot be resolved (such as within a Docker container pointing to a local host path), the service falls back gracefully to `joblib-fallback`.

#### Example cURL:
```bash
curl -X GET http://127.0.0.1:8000/health
```

---

### 3.2 Real-Time Transaction Scoring

#### `POST /score/{transaction_id}`
Computes the fraud probability for the specified `TransactionID` using the tuned LightGBM model, applies the 2% false-positive budget threshold, and calculates the top-5 feature contributions using TreeSHAP.

- **Path Parameter:**
  - `transaction_id` (`integer`, required): The unique integer ID of the transaction (e.g. `3141064`).
- **Request Body:** None.
- **Response Model:** `ScoreResponse`

#### Response Schema (`ScoreResponse`):
```json
{
  "transaction_id": 3141064,
  "risk_score": 0.81245,
  "flagged": true,
  "threshold_used": 0.12937,
  "top_reason_codes": [
    {
      "feature": "card1_addr1_amt_mean",
      "value": 144.14,
      "shap_contribution": 0.32145
    },
    {
      "feature": "card1_addr1_count",
      "value": 12.0,
      "shap_contribution": 0.19842
    },
    {
      "feature": "C13",
      "value": 5.0,
      "shap_contribution": 0.14231
    },
    {
      "feature": "TransactionAmt",
      "value": 350.0,
      "shap_contribution": 0.08912
    },
    {
      "feature": "dist1",
      "value": 18.0,
      "shap_contribution": -0.04123
    }
  ]
}
```

#### Field Explanations:
- `risk_score`: Calibrated probability $P(\text{isFraud} = 1 \mid X)$ rounded to 5 decimal places.
- `flagged`: Boolean indicator; `true` if `risk_score >= threshold_used`, otherwise `false`.
- `threshold_used`: The operational decision threshold established on the validation split at a 2.0% false-positive budget ($\tau = 0.129365$).
- `top_reason_codes`: Array of the 5 features with the highest absolute Shapley values ($|\phi_i|$).
  - `feature`: Exact name of the modeling feature.
  - `value`: Raw or imputed numerical value for this transaction.
  - `shap_contribution`: Additive margin contribution $\phi_i$. Positive values push risk higher; negative values push risk lower.

#### Error Responses:
- **`404 Not Found`**:
  ```json
  {
    "detail": "TransactionID 9999999999 not found in dataset."
  }
  ```

#### Example cURL:
```bash
curl -X POST http://127.0.0.1:8000/score/3141064
```

---

### 3.3 Trigger Agent Investigation

#### `POST /investigate/{transaction_id}`
Executes transaction scoring and triggers the AI investigation copilot (`src/agent.py`). Gathers point-in-time entity history, retrieves policy citations via TF-IDF, generates and validates narrative facts via local Llama 3.1 8B, logs the complete case file to SQLite (`cases.db`), and returns the investigation record.

- **Path Parameter:**
  - `transaction_id` (`integer`, required): The unique integer ID of the transaction.
- **Request Body:** None.
- **Response Model:** `InvestigateResponse`

#### Response Schema (`InvestigateResponse`):
```json
{
  "transaction_id": 3141064,
  "risk_score": 0.81245,
  "flagged": true,
  "threshold_used": 0.12937,
  "top_reason_codes": [
    {
      "feature": "card1_addr1_amt_mean",
      "value": 144.14,
      "shap_contribution": 0.32145
    }
  ],
  "case_file": "ENTITY SUMMARY: This card+address entity has 12 transactions totaling $1729.68, up to and including this one.\nTOP RISK FACTORS:\n• Lifetime transactions: 12\n• Mean transaction amount: $144.14\n• Prior confirmed fraud cases for this entity: 1\n• This transaction amount: $350.00\nRELATED ENTITY SIGNAL: 4 distinct email domains across this card and address's history up to this transaction; 4 in the 30 days up to this transaction.\nRELEVANT POLICY: POL-003: Email Domain Fraud Ring Indicator — A single billing address associated with more than 3 distinct email domains within a 30-day period is a strong indicator of a coordinated fraud ring. Triggered by: 4 distinct email domains in 30 days.\nRECOMMENDED ACTION: Escalate for manual review — POL-003 is triggered; 1 prior confirmed fraud case(s) for this entity.",
  "case_id": 1
}
```

#### Error Responses:
- **`404 Not Found`**: If the transaction is not in the dataset.
- **`503 Service Unavailable`**: If Ollama cannot be reached:
  ```json
  {
    "detail": "Agent investigation failed — the local Ollama service may have crashed or isn't running. Try 'ollama run llama3.1:8b' in a terminal to check it responds, then retry. Underlying error: Connection refused"
  }
  ```

#### Example cURL:
```bash
curl -X POST http://127.0.0.1:8000/investigate/3141064
```

---

### 3.4 List Investigation Cases

#### `GET /cases`
Retrieves a paginated list of previously logged investigation cases from SQLite (`cases.db`), ordered by descending case ID (most recent first).

- **Query Parameters:**
  - `limit` (`integer`, optional, default = 20): Maximum number of case summaries to return.
- **Response Model:** `list[CaseSummary]`

#### Success Response (`200 OK`):
```json
[
  {
    "id": 1,
    "transaction_id": 3141064,
    "risk_score": 0.81245,
    "created_at": "2026-11-01T10:45:00.123456+00:00"
  }
]
```

#### Example cURL:
```bash
curl -X GET "http://127.0.0.1:8000/cases?limit=10"
```

---

### 3.5 Retrieve Investigation Case Detail

#### `GET /cases/{case_id}`
Retrieves the complete historical record of a previously completed investigation, including stored risk score, threshold, reason codes, and full case file text.

- **Path Parameter:**
  - `case_id` (`integer`, required): The primary key ID of the stored case.
- **Response Model:** `CaseDetail`

#### Success Response (`200 OK`):
```json
{
  "id": 1,
  "transaction_id": 3141064,
  "risk_score": 0.81245,
  "threshold_used": 0.12937,
  "flagged": true,
  "top_reason_codes": [
    {
      "feature": "card1_addr1_amt_mean",
      "value": 144.14,
      "shap_contribution": 0.32145
    }
  ],
  "case_file": "ENTITY SUMMARY: This card+address entity has 12 transactions totaling $1729.68...\nRECOMMENDED ACTION: Escalate for manual review...",
  "created_at": "2026-11-01T10:45:00.123456+00:00"
}
```

#### Error Responses:
- **`404 Not Found`**:
  ```json
  {
    "detail": "Case 999999 not found."
  }
  ```

#### Example cURL:
```bash
curl -X GET http://127.0.0.1:8000/cases/1
```

---

## 4. Client Integration Examples

### Python Client (requests)
```python
import requests

BASE_URL = "http://127.0.0.1:8000"

# 1. Health check
health = requests.get(f"{BASE_URL}/health").json()
print("API Status:", health["status"], "| Model:", health["model_source"])

# 2. Score a transaction
txn_id = 3141064
score_resp = requests.post(f"{BASE_URL}/score/{txn_id}")
if score_resp.status_code == 200:
    data = score_resp.json()
    print(f"Transaction {txn_id}: Risk Score = {data['risk_score'] * 100:.1f}%")
    print(f"Flagged for Review: {data['flagged']}")
    print("Top Reason:", data["top_reason_codes"][0]["feature"])

# 3. Trigger Investigation
inv_resp = requests.post(f"{BASE_URL}/investigate/{txn_id}")
if inv_resp.status_code == 200:
    case_data = inv_resp.json()
    print("Investigation Case ID:", case_data["case_id"])
    print("\n--- CASE FILE ---\n")
    print(case_data["case_file"])
```

### JavaScript Client (Browser Fetch)
```javascript
const API_BASE = "http://127.0.0.1:8000";

async function scoreAndInvestigate(txnId) {
  // Score transaction
  const scoreRes = await fetch(`${API_BASE}/score/${txnId}`, { method: "POST" });
  if (!scoreRes.ok) throw new Error("Scoring failed");
  const scoreData = await scoreRes.json();
  console.log(`Risk: ${(scoreData.risk_score * 100).toFixed(1)}%`);

  // Investigate if flagged
  if (scoreData.flagged) {
    const invRes = await fetch(`${API_BASE}/investigate/${txnId}`, { method: "POST" });
    const invData = await invRes.json();
    console.log("Generated Case File:", invData.case_file);
  }
}
```

