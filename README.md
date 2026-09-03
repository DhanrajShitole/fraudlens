# FraudLens

**A real-time, explainable fraud detection platform with an agentic investigation copilot.**

> Status: 🚧 In development — BTech final-year project.

---

## 1. Problem Statement

*(Fill in once you've done the EDA — a tight 3-5 sentence version of: what fraud-detection gap this addresses, and why the investigation-assistance layer matters, not just the detection.)*

## 2. Architecture

*(Paste/adapt the ASCII architecture diagram once the pipeline is finalized. Update as the system evolves — this section should always reflect what's actually built, not the original plan.)*

## 3. Tech Stack

| Layer | Technology | Why |
|---|---|---|
| Data | Pandas, NumPy, PostgreSQL | ... |
| ML | scikit-learn, XGBoost/LightGBM, PyTorch | ... |
| GenAI | LLM API, pgvector | ... |
| Backend | FastAPI | ... |
| Frontend | React, TypeScript, Tailwind | ... |
| MLOps | MLflow, Evidently AI | ... |
| Infra | Docker, GitHub Actions | ... |
| Cloud | AWS (S3, Fargate, RDS, CloudWatch) | ... |

*(Fill in the "Why" column honestly as you build — this is the part recruiters actually read.)*

## 4. Dataset

- **IEEE-CIS Fraud Detection** (Kaggle competition, ~590K labeled transactions)
- **PaySim** (synthetic mobile-money transactions)

*(Add: class imbalance ratio, feature summary, and known data quality issues once EDA is done.)*

## 5. Model Approach

*(Baseline → advanced model progression, calibration approach, and how the anomaly/graph layers fit in. Update after each modeling phase.)*

## 6. MLOps

*(MLflow experiment tracking setup, model registry stages, drift monitoring approach — link to dashboards/screenshots once running.)*

## 7. Results

*(Fill in with real numbers once evaluation is complete: PR-AUC, F1 at fixed FP budget, calibration, agent factuality/latency metrics. Be honest about failure cases — this is a strength, not a weakness, in a portfolio project.)*

## 8. Roadmap

- [ ] Phase 1 — Research & problem definition
- [ ] Phase 2 — Data collection
- [ ] Phase 3 — Data engineering
- [ ] Phase 4 — Baseline ML
- [ ] Phase 5 — Advanced ML/DL
- [ ] Phase 6 — AI/GenAI integration
- [ ] Phase 7 — Backend
- [ ] Phase 8 — Frontend
- [ ] Phase 9 — MLOps
- [ ] Phase 10 — Cloud deployment
- [ ] Phase 11 — Testing & evaluation
- [ ] Phase 12 — Documentation & research paper

## Setup

```bash
git clone <your-repo-url>
cd fraudlens
# instructions to follow as the project takes shape
```

## Data Access

Raw data is not committed to this repo. To reproduce:
1. Download the IEEE-CIS Fraud Detection dataset from Kaggle (competition, not the community re-uploads) and place the 4 CSVs in `data/ieee-cis/`.
2. Download the PaySim dataset from Kaggle and place it in `data/paysim/`.

## License

*(Choose MIT or Apache-2.0 when you initialize the repo on GitHub.)*
