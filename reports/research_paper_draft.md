# FraudLens: An Explainable Financial Fraud Detection Platform with an Evidence-Grounded Agentic Investigation Copilot

**Author:** Dhanraj Shitole  
**Affiliation:** Department of Computer Science & Engineering (BTech Project)  
**Date:** November 2026  

---

## Abstract

Financial institutions process millions of transactions per day, with fraud occurrence exhibiting severe class imbalance (from 27:1 in real-world benchmarks to over 1,200:1 in synthetic payment streams). While state-of-the-art gradient-boosted decision trees achieve high discriminative ability, operational fraud prevention is bottlenecked not by detection, but by *investigation throughput*—human analysts spend excessive manual effort cross-referencing multi-table histories, evaluating organization policies, and synthesizing narrative case files. In this paper, we present **FraudLens**, an end-to-end open-source platform that bridges high-throughput fraud scoring, tree-based explainability, and an evidence-grounded agentic investigation copilot. 

Our modeling pipeline combines class-weighted gradient boosting (LightGBM) with Synthetic Minority Over-sampling (SMOTE) across 545 engineered features, incorporating point-in-time entity aggregations on shared card-address clusters that rank in the top 7% of SHAP feature importance. We test on the real-world IEEE-CIS dataset (590K transactions) and synthetic PaySim (6.3M transactions), honestly evaluating one-shot generalization on held-out test splits. On IEEE-CIS, the tuned model achieves a PR-AUC of 0.598 on validation (a +7.3% relative lift over untuned baseline) and 0.532 on unseen test data (15.30x lift over base rate), with a Brier skill score of 34.1%. To assist analysts, we design a tool-scoped Local LLM copilot (Llama 3.1 8B via Ollama). Grounded in the architectural principle *“code makes decisions, language models write prose,”* the agent enforces deterministic policy citations, point-in-time isolation, and an automated validator that rejects ungrounded numerical claims and vacuous comparative framing. The complete system includes a FastAPI serving backend, reactive analyst dashboard, MLflow registry with alias-based staging, and Population Stability Index (PSI) drift monitoring.

---

## 1. Introduction

Automated fraud detection systems operate under acute adversarial pressure and asymmetric costs. Blocking legitimate transactions creates customer friction and interchange loss, whereas failing to intercept fraudulent transactions incurs chargebacks and direct financial liability. Consequently, production systems operate within strict false-positive (FP) budget constraints—typically capping FP rates at 1% to 2%.

However, flagging a transaction is only the initial step. Each flagged alert routes to human fraud analysts who must manually reconstruct the temporal context:
1. What is the historical velocity and spend pattern of this entity?
2. Are multiple cards or distinct emails clustering around this billing address?
3. Which organizational fraud policies (e.g., velocity rules, geographic ring indicators) are triggered?
4. What is the recommended disposition (escalate for manual review vs. monitor)?

In practice, this manual contextualization constitutes an operational bottleneck. While large language models (LLMs) offer strong narrative synthesis capabilities, deploying unconstrained generative models in financial compliance introduces severe reliability hazards: hallucinated entity statistics, temporal label leakage, fabricated policies, and sycophantic reasoning.

**FraudLens** addresses both challenges simultaneously through three integrated contributions:
1. **Explainable Discriminative Modeling:** A tuned LightGBM architecture with SHAP tree explainability, validated across both real-world e-commerce (IEEE-CIS) and mobile money (PaySim) datasets under rigorous time-based evaluation splits.
2. **Grounded Agentic Copilot:** A privacy-preserving, local LLM agent that synthesizes evidence while delegating all decisions, policy citations, and numerical facts to deterministic, point-in-time code execution.
3. **End-to-End MLOps & Production Architecture:** Transparent model registry aliasing, Population Stability Index drift tracking, a FastAPI serving layer with graceful fallback, and automated containerization.

---

## 2. Dataset Engineering & Class Imbalance

We evaluate FraudLens across two benchmark datasets representing divergent fraud typologies and operational scales:

| Dataset | Total Transactions | Target Fraud Rate | Imbalance Ratio | Nature |
|---|---|---|---|---|
| **IEEE-CIS Fraud Detection** | 590,540 | 3.50% | 27.4 : 1 | Real-world card transactions |
| **PaySim Mobile Money** | 6,362,620 | 0.13% | 1,221.6 : 1 | Synthetic multi-agent simulation |

### 2.1 Leakage-Safe Time-Based Partitioning
Random $k$-fold cross-validation in temporal transactional data introduces substantial lookahead leakage. In FraudLens, both datasets are partitioned strictly chronologically:
- **Training Set (70% earliest):** Used exclusively for fitting categorical frequency encoders, missingness imputers, and model training.
- **Validation Set (15% middle):** Used for hyperparameter tuning and establishing operational decision thresholds at a fixed 2.0% false-positive budget.
- **Held-Out Test Set (15% latest):** Evaluated strictly **one-shot** with frozen parameters to measure true temporal generalization.

### 2.2 Entity Aggregation Engineering
Because device and IP fields in IEEE-CIS suffer from up to 80% missingness in production logs, we engineered composite entity identifiers using the tuple $(card1, addr1)$, representing the physical card issuer and billing region. We computed running point-in-time aggregations including:
- Historical transaction counts (`card1_addr1_count`)
- Mean and standard deviation of transaction amounts (`card1_addr1_amt_mean`, `card1_addr1_amt_std`)
- Distinct email domains associated with the entity (`card1_addr1_email_count`)

---

## 3. Machine Learning Methodology & Evaluation

### 3.1 Model Comparison & The Anomaly Detection Fallacy
We systematically benchmarked four candidate modeling strategies:
1. **Class-Weighted Logistic Regression:** Baseline linear benchmark.
2. **Cost-Sensitive LightGBM (`scale_pos_weight`):** Gradient boosting weighted by class imbalance.
3. **LightGBM + SMOTE:** Resampling the minority class with Synthetic Minority Over-sampling to a balanced sampling ratio of 0.1.
4. **Isolation Forest:** Unsupervised anomaly detection.

A prevalent hypothesis in fraud literature suggests that extreme class imbalance (such as PaySim’s 1,221:1 ratio) favors unsupervised anomaly detection over supervised learning. **Our experiments decisively refute this hypothesis:**

| Dataset | Model | PR-AUC | ROC-AUC | Recall @ 2% FP Budget | F1 @ Budget |
|---|---|---|---|---|---|
| **IEEE-CIS** | Logistic Regression | 0.305 | 0.818 | 33.6% | 0.354 |
| | LightGBM + scale_pos_weight | 0.552 | 0.916 | 52.9% | 0.506 |
| | LightGBM + SMOTE (Baseline) | 0.557 | 0.904 | 54.1% | 0.515 |
| | **LightGBM + SMOTE (Tuned)** | **0.598** | **0.924** | **57.9%** | **0.541** |
| **PaySim** | Isolation Forest (Unsupervised) | 0.002 | 0.840 | 2.5% | 0.001 |
| | **LightGBM + scale_pos_weight** | **0.011** | **0.949** | **94.5%** | **0.023** |

*Analysis:* PaySim fraud consists of specific account-draining patterns (confined exclusively to `TRANSFER` and `CASH_OUT` transaction types). Supervised LightGBM efficiently isolates these discrete logical signatures even from scarce positive labels. Conversely, Isolation Forest flags generic statistical outliers (e.g., unusually large legitimate commercial transfers), yielding an operational recall of only 2.5% at the 2% FP budget.

### 3.2 SHAP Explainability & Entity Feature Validation
To validate our Phase 3 entity features, we performed game-theoretic feature attribution using TreeSHAP across all 545 candidate features:

| Feature | Description | SHAP Rank | Percentile |
|---|---|---|---|
| `card1_addr1_amt_mean` | Historical mean transaction amount for card+address | 13 / 545 | Top 2.4% |
| `card1_addr1_count` | Historical transaction velocity for card+address | 17 / 545 | Top 3.1% |
| `card1_addr1_amt_std` | Spending variability for card+address | 24 / 545 | Top 4.4% |
| `card1_addr1_email_count` | Distinct emails tied to card+address | 39 / 545 | Top 7.2% |

All four engineered entity features rank within the **top 7.2% of all 545 modeling features**, confirming that shared card-address behavioral signatures provide strong signal for detecting account takeover and coordinated fraud rings.

### 3.3 Held-Out Test Set Generalization & Calibration
On the final held-out IEEE-CIS test set ($N = 88,581$), the model achieved a PR-AUC of 0.5324 (a 15.30x lift over the 3.48% base rate) and an ROC-AUC of 0.9066. At the operational threshold fixed on validation ($\tau = 0.129365$), the model intercepted 53.39% of all fraud (recall) with 43.95% precision, while the actual test FP rate was 2.46% (slightly above the 2.0% target due to natural temporal drift).

Probability calibration evaluated across 10 quantile bins yielded a Brier score of 0.02213 compared to 0.03359 for a non-informative constant base-rate predictor—representing a **Brier Skill Score of 34.12%**.

---

## 4. Agentic Investigation Copilot Architecture

### 4.1 The Hallucination Classes in Unconstrained LLM Agents
During iterative development of an investigation copilot using an 8-billion parameter local model (Llama 3.1 8B), prompt-only implementations exhibited four distinct failure modes:
1. **Geographic Hallucination:** Misinterpreting coarse regional code `addr1` as an international country identifier and fabricating cross-border fraud narratives.
2. **Data-Granularity Bugs:** Aggregating data on `addr1` alone rather than the specific $(card1, addr1)$ tuple, generating false signals of coordinated fraud rings.
3. **Temporal Label Leakage & Overreach:** Counting the queried transaction’s own label or subsequent transactions within "historical" entity fraud counts.
4. **Vacuous Comparative Framing:** Emitting tautological assertions that appear mathematically grounded but lack logical substance (e.g., *"the mean transaction amount of $144.14 is greater than the overall mean of $144.14"*).

### 4.2 System Redesign: "Code Decides, Model Narrates"
To eliminate these vulnerabilities, FraudLens implements a decoupled architecture:

```
[ Flagged TransactionID ]
            │
            ├──► 1. get_entity_history(id)      (Deterministic Python, DT <= current)
            ├──► 2. get_related_entities(id)    (Point-in-time domain counts)
            └──► 3. retrieve_policy_clause()    (TF-IDF over policy corpus, relevance floor)
                        │
                        ▼
            [ Assembled Evidence Context ]
                        │
                        ├──► Deterministic Code Layer:
                        │      • RELATED ENTITY SIGNAL
                        │      • RELEVANT POLICY
                        │      • RECOMMENDED ACTION (Escalate vs Monitor)
                        │
                        └──► Single Local LLM Call (Llama 3.1 8B):
                               • ENTITY SUMMARY (1-2 sentences)
                               • TOP RISK FACTORS (bulleted facts)
                                       │
                                       ▼
                        [ Hardened Automated Validator ]
                               ├── Grounded number check (Evidence only; no policy/notes)
                               ├── Rejection of unsupported comparisons
                               └── Failure? ──► Fallback to Tool-Derived Template
```

### 4.3 Automated Validator Hardening
Before any LLM-authored text reaches an analyst, it must pass a strict validator:
1. **Numerical Grounding:** Every extracted number must be a member of the grounded evidence set from `get_entity_history` or `get_related_entities`. Numbers appearing in policy text (e.g., "within 24 hours" or "more than 3 domains") and internal note strings are strictly excluded from the grounded set to prevent policy bleeding.
2. **Comparative Framing Guard:** Any sentence containing comparative terms (*greater than, less than, higher than, lower than, more than, above, below, compared to, versus, relative to*) is rejected **unless** the sentence contains at least two distinct grounded numbers.
3. **Resilience to Service Failure:** If Ollama is unavailable or if the model fails validation after one retry, the system automatically falls back to deterministic code-generated bullets, guaranteeing zero analyst downtime.

---

## 5. System Implementation & Production Readiness

### 5.1 Serving API & Database Persistence
The scoring backend (`api/main.py`) is implemented with FastAPI:
- `GET /health`: Service health check, model version, and model source (`registry` vs `joblib-fallback`).
- `POST /score/{id}`: Real-time fraud scoring ($<15$ ms inference latency) with top-5 SHAP reason codes.
- `POST /investigate/{id}`: Orchestrates evidence retrieval, LLM narrative synthesis, and validation.
- `GET /cases` & `GET /cases/{id}`: SQLite-backed immutable case audit log.

### 5.2 MLOps: Registry Staging & PSI Drift Monitoring
- **Model Registry:** Models are registered in MLflow (`mlflow.db`) and promoted using the `@production` alias.
- **Drift Detection:** We implement Population Stability Index (PSI) tracking without third-party dependencies. Comparing the training distribution to the validation split across all 545 features:
  - 535 features (98.2%) were stable ($\text{PSI} < 0.10$).
  - 2 features exhibited moderate shift ($0.10 \le \text{PSI} < 0.25$).
  - 7 features (1.3%) exhibited genuine drift ($\text{PSI} \ge 0.25$), concentrated primarily in device and infrastructure fingerprint features (`id_13`, `id_20`, `id_02`).

### 5.3 Automated CI/CD & Containerization
FraudLens includes a full GitHub Actions pipeline (`.github/workflows/ci.yml`) performing automated Python bytecode compilation, HTML validation, Docker build checks, and execution of the 51-test suite across every pull request and push to `main`.

---

## 6. Conclusion & Future Work

FraudLens demonstrates that high-precision fraud scoring and generative AI can be safely integrated into financial operations when strict architectural guardrails are enforced. By constraining the LLM to narrative synthesis while delegating decision logic, evidence gathering, and policy enforcement to deterministic code, the system eliminates hallucination risks while reducing analyst context-assembly time from minutes to milliseconds.

Future extensions will explore streaming event ingestion (e.g., Apache Kafka), conformal prediction for calibrated confidence sets under non-stationary fraud drift, and federated learning across multi-institution consortium data.

---

## References

1. IEEE Computational Intelligence Society. *IEEE-CIS Fraud Detection Dataset*. Kaggle, 2019.
2. Lopez-Rojas, E. A., Elmir, A., & Axelsson, S. *PaySim: A financial mobile money simulator for fraud detection*. In Proceedings of the 28th European Modeling and Simulation Symposium (EMSS), 2016.
3. Lundberg, S. M., & Lee, S. I. *A unified approach to interpreting model predictions*. In Advances in Neural Information Processing Systems (NeurIPS), 2017.
4. Ke, G., Meng, Q., Finley, T., et al. *LightGBM: A highly efficient gradient boosting decision tree*. In Advances in Neural Information Processing Systems (NeurIPS), 2017.
5. Chawla, N. V., Bowyer, K. W., Hall, L. O., & Kegelmeyer, W. P. *SMOTE: Synthetic Minority Over-sampling Technique*. Journal of Artificial Intelligence Research (JAIR), 2002.

