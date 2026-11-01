# FraudLens — Executive Summary & Project Brief

**Project Name:** FraudLens  
**Subtitle:** A Real-Time, Explainable Financial Fraud Detection Platform with an Evidence-Grounded Agentic Investigation Copilot  
**Author:** Dhanraj Shitole  
**Status:** Completed & Validated (Phases 1–12 Complete)  
**Repository:** [https://github.com/DhanrajShitole/fraudlens](https://github.com/DhanrajShitole/fraudlens)  

---

## 1. Problem Statement & Motivation

Financial institutions process billions of transactions daily, but fraudulent events constitute a tiny fraction of total volume (typically 0.1% to 3.5%). This extreme **class imbalance** makes accurate automated detection difficult. More critically, the operational bottleneck in modern anti-fraud operations has shifted from *detection* to *investigation throughput*:

- When a machine learning model flags an alert, a human compliance analyst must manually cross-reference customer profiles, recent card velocity, IP/device associations, and regulatory compliance rules.
- Assembling this contextual evidence takes 15 to 45 minutes per suspicious transaction.
- Traditional rules-based engines produce high false-positive rates, inundating analysts with alert fatigue.
- Generic Generative AI / Large Language Models (LLMs) cannot be deployed naively in compliance workflows because unconstrained models hallucinate facts, invent geographic claims, leak future labels, and contradict their own data.

**FraudLens** was conceived and engineered to solve both sides of this equation:
1. **Accurate, Cost-Sensitive Fraud Scoring:** Gradient-boosted decision trees (LightGBM) optimized under strict false-positive budget constraints (2% false-positive budget) paired with TreeSHAP explainability to provide instant mathematical reason codes.
2. **Grounded Agentic Copilot:** A locally hosted, tool-scoped AI agent (Llama 3.1 8B via Ollama) governed by the core architectural rule: *“Code makes decisions, language models write prose.”* The copilot deterministically extracts point-in-time entity evidence and organizational policy citations, while the LLM drafts structured narrative summaries that are strictly validated against tool evidence before reaching the human investigator.

---

## 2. High-Level System Overview

```
                                  FRAUDLENS PLATFORM
┌───────────────────────────────────────────────────────────────────────────────────┐
│                                                                                   │
│   OFFLINE TRAINING & MLOps PIPELINE                                               │
│   ┌───────────────────┐      ┌────────────────────┐      ┌────────────────────┐   │
│   │ IEEE-CIS (590K)   │ ───► │ Feature Pipeline   │ ───► │ Tuned LightGBM     │   │
│   │ PaySim (6.3M)     │      │ 545 Features       │      │ + SMOTE Resampling │   │
│   └───────────────────┘      └────────────────────┘      └─────────┬──────────┘   │
│                                                                    │              │
│                                                                    ▼              │
│                                                          ┌────────────────────┐   │
│                                                          │ MLflow Registry    │   │
│                                                          │ Alias: @production │   │
│                                                          └─────────┬──────────┘   │
│                                                                    │              │
│   ONLINE SERVING & INVESTIGATION STACK                             │              │
│   ┌───────────────────┐      ┌────────────────────┐                │              │
│   │ Analyst Dashboard │ ───► │ FastAPI Backend    │ ◄──────────────┘              │
│   │ (React Dark Theme)│      │ (/score, /invest.) │ (Dual loader: Registry/Joblib)│
│   └───────────────────┘      └─────────┬──────────┘                               │
│                                        │                                          │
│                    ┌───────────────────┴───────────────────┐                      │
│                    ▼                                       ▼                      │
│        ┌────────────────────────┐              ┌────────────────────────┐         │
│        │ Real-Time Scoring      │              │ Agentic Investigation  │         │
│        │ • LightGBM Probability │              │ • Point-in-time tools  │         │
│        │ • Top-5 SHAP Reasons   │              │ • TF-IDF Policy Search │         │
│        │ • 2% FP Budget Flag    │              │ • Local Llama 3.1 8B   │         │
│        └────────────────────────┘              │ • Hardened Validator   │         │
│                                                └────────────────────────┘         │
└───────────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Key Achievements & Verified Performance Metrics

All metrics reported below are extracted directly from the frozen, held-out test evaluation reports (`reports/test_evaluation.md`) and validation experiments (`reports/`):

### A. Machine Learning Scoring Performance (IEEE-CIS Benchmark)
- **Validation PR-AUC:** **0.5975** (Tuned LightGBM + SMOTE) — a **+7.3% relative improvement** over the untuned baseline (0.5566).
- **Validation ROC-AUC:** **0.9242** (vs Logistic Regression 0.8180).
- **Validation F1 @ 2% False-Positive Budget:** **0.5405** (Precision: 50.71%, Recall: 57.86% at fixed threshold $\tau = 0.129365$).
- **Held-Out Test Set Generalization (One-Shot):**
  - **Test PR-AUC:** **0.5324** — delivering a **15.30x lift** over the test set base fraud rate (3.48%).
  - **Test ROC-AUC:** **0.9066**.
  - **Test F1 @ Fixed Threshold:** **0.4821** (Recall: 53.39%, Precision: 43.95%, Actual False-Positive Rate: 2.46%).
- **Probability Calibration:** Test Brier score of **0.02213** vs. non-informative baseline of **0.03359**, yielding a **Brier Skill Score of 34.12%**.

### B. The Anomaly Detection Hypothesis Test (PaySim Benchmark)
- Benchmarked supervised gradient boosting against unsupervised anomaly detection (Isolation Forest) across 6.36 million transactions (1,221.6:1 class imbalance).
- **Finding:** Supervised LightGBM dramatically outperformed Isolation Forest (ROC-AUC **0.949** vs. **0.840**; Recall at 2% FP budget **94.5%** vs. **2.5%**).
- **Conclusion:** Synthetic mobile money fraud follows crisp, discrete account-draining logic that supervised trees learn efficiently, while Isolation Forest flags high-value benign transfers.

### C. Game-Theoretic Feature Attribution (TreeSHAP)
- Engineered point-in-time entity features grouping physical card and billing region `(card1, addr1)` proved overwhelmingly significant:
  - `card1_addr1_amt_mean`: Ranked **13 / 545** (Top 2.4% most influential features).
  - `card1_addr1_count`: Ranked **17 / 545** (Top 3.1%).
  - `card1_addr1_amt_std`: Ranked **24 / 545** (Top 4.4%).
  - `card1_addr1_email_count`: Ranked **39 / 545** (Top 7.2%).

### D. Grounded Agent Reliability
- Solved 4 major LLM failure modes (geographic hallucination, data granularity errors, temporal label leakage, and vacuous comparisons).
- Tightened automated validator (`_model_text_clean`): Rejects ungrounded numbers, numbers sourced from policy text or notes, and comparative framing lacking two distinct grounded figures.
- Seamless fallback: If the local LLM crashes or fails validation after one retry, the agent generates a complete, deterministic, evidence-backed case file with 0% data loss.

---

## 4. Technology Stack & Operational Architecture

| Layer | Selected Technology | Technical Justification |
|---|---|---|
| **Language & Runtime** | Python 3.13 | Pinned, modern runtime with native typing and performance improvements. |
| **Data Processing** | Pandas 3.0, NumPy 2.2, PyArrow 25.0 | High-performance Parquet I/O, vectorized split filtering, zero memory copies. |
| **Predictive ML** | LightGBM 4.7, Scikit-Learn 1.8, Imbalanced-Learn 0.14 | Gradient-boosted decision trees with histogram binning and SMOTE resampling. |
| **Explainability** | SHAP 0.52 (TreeExplainer) | Exact Shapley value calculation in polynomial time for tree ensembles. |
| **Local LLM Engine** | Ollama 0.6.2, Llama 3.1 8B | Local, offline inference ensuring 100% data privacy (zero PII sent to cloud APIs). |
| **Serving Backend** | FastAPI 0.136, Uvicorn 0.46, Pydantic 2.13 | Async REST endpoints, sub-15ms scoring latency, strict schema validation. |
| **Case Store** | SQLite 3 (`cases.db`) | Zero-configuration, serverless relational persistence for investigation audit trails. |
| **Analyst UI** | React 18 (via CDN), CSS3 Variables | Single-file analyst dashboard, dark theme, zero npm build dependencies. |
| **MLOps & Monitoring**| MLflow 3.16, direct PSI computation | Experiment tracking, model registry with `@production` alias, Population Stability Index drift tracking. |
| **DevOps & CI/CD** | Docker, Docker Compose, GitHub Actions | Pinned multi-stage container build and automated CI running 51 unit/integration tests. |

---

## 5. Quality Assurance & Test Verification

FraudLens maintains a comprehensive test suite of **51 automated tests**, achieving 100% pass rates in local and CI environments:

```bash
python -m pytest tests/ -v
# Result: 51 passed, 2 warnings in ~10.7 seconds
```

- **`tests/test_agent_logic.py` (19 tests):** Validates point-in-time isolation, label leakage protection, numerical grounding, comparative framing rejection, policy matching, and Ollama failure resilience.
- **`tests/test_api.py` (12 tests):** Validates `/health`, `/score`, `/investigate`, `/cases`, HTTP status codes (200, 404, 503), and SQLite persistence.
- **`tests/test_drift.py` (6 tests):** Validates Population Stability Index calculation across stable distributions, large shifts, near-constant features, and missing values.
- **`tests/test_model.py` (14 tests):** Validates model file integrity, 545-feature schema alignment, probability distribution axioms, and TreeSHAP top-5 indexing.

---

## 6. How FraudLens Is Used

1. **Scoring a Transaction:**
   - The compliance analyst inputs a Transaction ID into the dashboard.
   - The backend runs LightGBM inference and computes TreeSHAP reason codes.
   - The dashboard displays the calibrated risk score, an alert badge (`FLAGGED FOR REVIEW` vs `BELOW THRESHOLD` based on the 2% false-positive budget threshold $\tau = 0.129365$), and horizontal bars indicating which features increased or decreased risk.
2. **Triggering an AI Investigation:**
   - The analyst clicks **Investigate**.
   - The backend gathers historical entity transactions (point-in-time, up to this transaction only), calculates distinct email clusters on the same card/address, and matches organizational fraud policies via TF-IDF with a relevance floor.
   - Local Llama 3.1 8B synthesizes the entity overview and bulleted risk factors.
   - The output is validated: if clean, it is merged with code-generated policies and recommended actions; if ungrounded, the system transparently falls back to deterministic evidence bullets.
   - The full case file is rendered in the dashboard and permanently logged to SQLite.

