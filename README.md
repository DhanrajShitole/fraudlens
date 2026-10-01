# FraudLens

**A real-time, explainable fraud detection platform with an agentic investigation copilot.**

> Status: 🚧 In development — BTech final-year project. EDA, feature engineering, baseline modeling, hyperparameter tuning, SHAP explainability, the agentic investigation copilot, and a working FastAPI backend are complete; frontend and cloud deployment in progress.

---

## 1. Problem Statement

Financial institutions process millions of transactions daily; fraud-detection models exist, but the bottleneck has shifted from *detection* to *investigation throughput* — analysts spend disproportionate time manually assembling context per flagged case. FraudLens targets both: accurate, calibrated fraud scoring, and (upcoming) an automated evidence-assembly layer that turns a bare fraud score into an analyst-ready case file.

## 2. Architecture

*(To be finalized once the backend/API layer is built — see roadmap below.)*

## 3. Tech Stack

| Layer | Technology | Why |
|---|---|---|
| Data | Pandas, NumPy, PostgreSQL (planned) | Feature engineering and eventual storage layer |
| ML | scikit-learn, LightGBM, imbalanced-learn (SMOTE) | Cost-sensitive classification under class imbalance |
| Experiment Tracking | MLflow (SQLite backend) | Every model run logged with params/metrics for honest comparison |
| GenAI | Llama 3.1 8B (Ollama, local), scikit-learn TF-IDF | Zero-cost, offline agent layer — see Section 8 |
| Backend | FastAPI, SQLite | Real-time scoring, agent-triggered investigation, case log — 5 endpoints, working |
| Frontend | React (CDN, no build step) | Analyst dashboard — single HTML file, see Section 10 |
| Infra | Docker, GitHub Actions (planned) | Containerized deployment, CI |
| Cloud | AWS (S3, Fargate, RDS, CloudWatch) (planned) | See full blueprint for service-by-service justification |

## 4. Dataset

- **IEEE-CIS Fraud Detection** (Kaggle competition, 590,540 labeled transactions, 27.4:1 class imbalance)
- **PaySim** (synthetic mobile-money, 6.3M+ transactions, 1,221.6:1 class imbalance)

See `notebooks/01_eda.ipynb` for full exploratory analysis, and `notebooks/02_feature_engineering.ipynb` for the feature pipeline (576 final modeling features for IEEE-CIS, built with leakage-safe time-based train/val/test splits).

## 5. Model Approach

Baseline models trained and honestly compared across both datasets (see `src/train_baseline.py`), tracked in MLflow:

- **Logistic Regression** (class-weighted) — sanity-check floor
- **LightGBM** with `scale_pos_weight` — cost-sensitive gradient boosting
- **LightGBM + SMOTE** — oversampling comparison (IEEE-CIS only)
- **Isolation Forest** — unsupervised anomaly detection (PaySim only, as a comparison point against the extreme 1,221:1 imbalance)

## 6. MLOps

MLflow experiment tracking with a SQLite backend (`mlflow.db`); every training run logged with hyperparameters and metrics. Model registry, drift monitoring (Evidently AI), and CI/CD deployment are planned for later phases.

## 7. Results

### IEEE-CIS (27.4:1 imbalance)

| Model | PR-AUC | ROC-AUC | F1 @ 2% FP budget |
|---|---|---|---|
| Logistic Regression | 0.305 | 0.818 | 0.354 |
| LightGBM + scale_pos_weight | 0.552 | 0.916 | 0.506 |
| LightGBM + SMOTE (Phase 4 baseline) | 0.557 | 0.904 | 0.515 |
| **LightGBM + SMOTE, tuned (Phase 5, best)** | **0.598** | 0.924 | **0.541** |

Hyperparameter tuning (RandomizedSearchCV, PR-AUC-scored, 3-fold CV) improved PR-AUC by **+7.3% relative** over the untuned baseline. Full SHAP explainability was run on the tuned model across all 545 candidate features — see `reports/ieee_shap_summary.png` and `reports/ieee_shap_feature_importance.csv`.

**Entity-graph feature validation:** the engineered entity-relationship features (built from shared card/address/email identifiers, adapted from the original device/IP plan due to high missingness — see Phase 3 notes) rank genuinely high by SHAP importance:

| Feature | SHAP Rank | Percentile |
|---|---|---|
| `card1_addr1_amt_mean` | 13 / 545 | top 2.4% |
| `card1_addr1_count` | 17 / 545 | top 3.1% |
| `card1_addr1_amt_std` | 24 / 545 | top 4.4% |
| `card1_addr1_email_count` | 39 / 545 | top 7.2% |

All four rank in the top 7% of features — validating the Phase 3 design decision to build entity signal from these fields instead of the sparser device/IP columns.

### PaySim (1,221.6:1 imbalance)

| Model | PR-AUC | ROC-AUC | Recall @ 2% FP budget | F1 @ budget |
|---|---|---|---|---|
| **LightGBM + scale_pos_weight (best)** | **0.011** | **0.949** | **94.5%** | **0.023** |
| Isolation Forest (unsupervised) | 0.002 | 0.840 | 2.5% | 0.001 |

**Key finding:** we hypothesized PaySim's extreme imbalance would favor unsupervised anomaly detection over supervised learning. The opposite was true — supervised LightGBM won decisively. We attribute this to PaySim's highly structured fraud pattern (confined to 2 transaction types with a clean account-draining signature), which a supervised model learns efficiently even from few positive examples, while Isolation Forest instead flags generic statistical outliers that are mostly legitimate. This negative result for the anomaly-detection hypothesis is documented rather than hidden — see `phase4_summary.md` for the full writeup, including the caveat that PaySim's synthetic fraud pattern is more learnable than real-world fraud would likely be.

## 8. Agentic Investigation Copilot (Phase 6)

Built a narrow, tool-scoped LLM agent (`src/agent.py`) that, given a TransactionID, gathers evidence via three read-only tools and drafts a structured case file — it never decides fraud/not-fraud and never takes an autonomous action.

**Stack choice:** locally-hosted **Llama 3.1 8B via Ollama** — zero cost, fully offline, no data leaves the machine. A deliberate engineering trade-off given project constraints, documented honestly rather than hidden.

**Tools:**
- `get_entity_history` — this card+address entity's transaction history **as of** the transaction under investigation (point-in-time only — see leakage fix below)
- `get_related_entities` — distinct email domains sharing the exact same card+address combination, both lifetime and in the policy-relevant 30-day window
- `retrieve_policy_clause` — TF-IDF retrieval over a small synthetic fraud-policy corpus, queried by the system from actual evidence rather than by the model (see redesign below)

### Evaluation round 1 — three hallucination classes found in early prompt-only testing

1. **Geographic hallucination:** the model misread `addr1` (a coarse internal billing-region code) as a "country" and fabricated a cross-border justification. Fixed with an explicit constraint against unsupported geographic claims.
2. **Data-granularity bug:** `get_related_entities` originally grouped by `addr1` alone — a coarse regional code shared by thousands of unrelated transactions — producing meaningless counts (e.g. "723 other cards"). Fixed by grouping on `card1`+`addr1` together, matching the granularity of the Phase 5-validated `card1_addr1_count` feature.
3. **Time-window overreach:** the model asserted a specific "24-hour window" a policy clause required, which no tool had actually verified. Fixed with an instruction against unverified temporal claims.

### Evaluation round 2 — a deeper problem, and an architectural rewrite

Continued testing surfaced two issues that prompt instructions alone couldn't fix:

- **Label leakage:** the entity-history tool could include the transaction being investigated (and any *later* transactions for that entity) when computing its "known prior fraud" count — silently leaking the answer into its own evidence.
- **Blind policy retrieval:** the model wrote its policy search query in the same turn as the evidence tools, *before* seeing any evidence. It searched generically, TF-IDF returned a plausible-sounding but wrong match (a "cross-border" policy with zero supporting evidence), and the model then rationalized the mismatch rather than reporting it as a non-match.

Both are evidence-pipeline design flaws, not prompt-wording problems, so `agent.py` was substantially rewritten:

- **Point-in-time evidence:** both tools now filter strictly to `TransactionDT <= this transaction`, and prior-fraud counts use strictly *earlier* transactions only — no leakage of the current or future labels.
- **Evidence-driven policy search:** the search query is now built by the system from the actual tool results (not written by the model pre-evidence), with a relevance floor below which no policy is cited at all rather than forcing a weak match.
- **Deterministic decision layer:** `RECOMMENDED ACTION`, `RELATED ENTITY SIGNAL`, and `RELEVANT POLICY` are now computed in code from tool output, not written by the LLM — the model's role is narrowed to orchestrating tool calls and writing two fact-checked narrative lines (`ENTITY SUMMARY`, `TOP RISK FACTORS`). Every number the model writes is validated against the literal tool output before being trusted; unsupported or evaluative language triggers an automatic fallback to code-generated text.
- **Ollama-crash resilience:** given the recurring local CUDA crash encountered during this project, a full LLM-unavailable path now returns a complete, accurate, tool-derived case file instead of an error — an analyst gets a usable result even when the model itself is down.

### Formal validation (5-case test suite against the real model)

A held-out test — 2 known-fraud cases (one exercising each policy path), 1 known-legitimate case, 1 case with missing identifiers, and 1 repeat run to check non-determinism — found:

- **The recovery path, not the clean path, is the norm:** only 1 of 5 runs completed cleanly on the first attempt; the other 4 required the recovery mechanism after the model skipped one or more tool calls. This validates that the robustness layer wasn't precautionary — for an 8B local model, it's load-bearing.
- **A fourth hallucination class**, caught by manual review of the "clean" output rather than the validator itself: the model wrote *"mean transaction amount of $144.14 is greater than the overall mean of $144.14"* — a self-referential, vacuous comparison. The number was technically grounded (it appeared in tool output) and used no banned language, so the automated validator correctly let it through; only a human reading the sentence for logical coherence caught it. Documented as a known limitation rather than silently patched, since it didn't meet the bar for a code change given the low severity — the concrete fix (reject comparative phrasing unless two distinct grounded numbers are present) is identified but not yet implemented.
- On the missing-identifier case, the agent degraded gracefully to an "insufficient data" case file with no fabricated numbers, rather than crashing.
- The Case 1 → Case 5 repeat run produced different narrative wording but identical policy match and recommended action — confirming non-determinism affects phrasing, not the decision-relevant output.

See `reports/` for example case files and the full validation transcript.

### Performance rearchitecture — from 2-3 model calls to 1

Once correctness was validated, real usage surfaced a usability problem: investigations were slow, especially on this project's local hardware (an intermittent GPU driver issue on the dev machine forced CPU-mode Ollama for stretches of development). Profiling against the validation report's own numbers showed why — the original design let the model *decide*, across up to 3 tool-calling rounds, whether and when to call each tool, and the validation suite had already shown the model needed all three tools in every real test case anyway. The model's actual contribution had also been narrowed, by the earlier fixes, to just two fact-checked sentences (`ENTITY SUMMARY`, `TOP RISK FACTORS`) — multi-round tool orchestration was overhead for a decision the model was never really making.

**Fix:** the two evidence tools and the policy match now always run deterministically up front (no model decision involved), and the model is asked for its two sentences in a **single** completion call given that evidence directly, with at most one retry if the response fails validation. This cut the common case from 2-3 model calls to 1 — validated with 5 synthetic test scenarios (happy path, validation-failure-then-retry, double-failure-to-fallback, missing-entity-data, and a simulated Ollama crash) covering every code path before being tested against the real model, where it was confirmed both faster and no regression in output quality.

## 9. Backend API (Phase 7)

A working FastAPI service (`api/main.py`) ties the model, SHAP explainability, and the agent together into 5 real endpoints:

| Endpoint | Purpose |
|---|---|
| `GET /health` | Service status, confirms model is loaded |
| `POST /score/{transaction_id}` | Real-time risk score + top-5 SHAP reason codes for a transaction |
| `POST /investigate/{transaction_id}` | Triggers the agent to draft a full case file |
| `GET /cases` | Lists past investigations (SQLite-backed log) |
| `GET /cases/{case_id}` | Retrieves one past investigation in full |

**Model persistence:** the Phase 5 tuned model was retrained with its known-best hyperparameters and saved to `models/ieee_lightgbm_tuned.joblib` (`src/persist_model.py`) — it previously only existed in memory during that script's run, which would have made a real API impossible without this step.

**Documented simplification:** since this project has no live transaction stream, `/score` and `/investigate` look up transactions that already exist in the processed dataset by ID (train+val combined, matching the scope `agent.py`'s tools already searched — an early version only searched val, which caused real lookup failures for train-set IDs used during agent testing), rather than accepting arbitrary raw transaction data in the request body. One honest caveat: scoring a **train-set** transaction will look unrealistically confident, since the model saw that row's label during training — the Phase 4/5 evaluation metrics (PR-AUC, recall@budget) remain valid regardless, as those were computed on held-out val/test data only, never train. For a demo that fairly represents real model performance, deliberately pick a val-set ID.

**Case persistence:** investigations are logged to a local SQLite database (`cases.db`, gitignored — regenerate by running the API) rather than the originally-planned PostgreSQL, since a single-file database is a reasonable, honest choice at this project's current scale; migrating to Postgres remains a natural next step if this were pushed toward production.

## 10. Frontend Dashboard (Phase 8)

A single self-contained HTML file (`frontend/dashboard.html`) — React via CDN, no npm install or build step, deliberately avoiding another layer of environment setup after the repeated Python-environment friction earlier in this project. Designed as an analyst investigation console (dark, data-dense, monospace for IDs/scores) rather than a generic SaaS dashboard.

**Functionality:** enter a TransactionID → **Get Score** (risk %, flagged/clear badge, SHAP reason-code bars) or **Investigate** (triggers the agent, renders the full case file with section labels highlighted). A sidebar lists recent investigations pulled from `/cases`, clickable to reopen.

**Two real bugs found and fixed during integration testing:**
1. **CORS:** FastAPI doesn't send CORS headers by default, so opening the dashboard as a local `file://` page silently blocked the browser from reading the API's (successful) responses — the server logs showed `200 OK` while the dashboard showed "can't reach the API." Fixed with explicit `CORSMiddleware`.
2. **Stale/missing threshold on direct Investigate:** clicking Investigate without first clicking Get Score left the dashboard's "flagged" comparison defaulting to a threshold of 0, so every transaction showed as flagged regardless of actual risk. Fixed by having `/investigate` return the same threshold, flagged status, and reason codes as `/score` directly, so the dashboard never depends on a separate, possibly-stale earlier call — and by storing these fields with each logged case, so reopening history shows accurate data too.

**Verified end-to-end:** all 5 cases from the agent's formal validation suite (Section 8) were re-run through the actual dashboard UI, not just the backend script, confirming the full stack — model, SHAP, agent, API, and UI — agree with each other.

## 11. Roadmap

- [x] Phase 1 — Research & problem definition
- [x] Phase 2 — Data collection
- [x] Phase 3 — Data engineering (576 features, leakage-safe time-based splits)
- [x] Phase 4 — Baseline ML (LightGBM best on both datasets; honest anomaly-detection comparison documented)
- [x] Phase 5 — Advanced ML/DL (hyperparameter tuning: +7.3% PR-AUC; SHAP explainability; entity-graph features validated in top 7% of 545 features)
- [x] Phase 6 — AI/GenAI integration (tool-scoped LLM agent, Llama 3.1/Ollama; rewritten for point-in-time evidence and deterministic decision logic after 4 hallucination classes found across two evaluation rounds; formally validated with a 5-case test suite)
- [x] Phase 7 — Backend (FastAPI: scoring, agent-triggered investigation, SQLite case log — 5 endpoints, all tested working)
- [x] Phase 8 — Frontend (single-file React dashboard, no build step; full-stack verified against all 5 validation cases: score, SHAP reason codes, agent case files, investigation history)
- [ ] Phase 9 — MLOps (model registry, drift detection)
- [ ] Phase 10 — Cloud deployment
- [ ] Phase 11 — Testing & evaluation
- [ ] Phase 12 — Documentation & research paper

## Setup

```bash
git clone https://github.com/DhanrajShitole/FraudLens
cd FraudLens
pip install pandas numpy scikit-learn lightgbm imbalanced-learn mlflow matplotlib seaborn pyarrow
```

## Data Access

Raw data is not committed to this repo. To reproduce:
1. Download the IEEE-CIS Fraud Detection dataset from Kaggle (competition, not the community re-uploads) and place the 4 CSVs in `data/ieee-cis/`.
2. Download the PaySim dataset from Kaggle and place it in `data/paysim/`.

## License

MIT
