# FraudLens

**A real-time, explainable fraud detection platform with an agentic investigation copilot.**

> Status: 🚧 In development — BTech final-year project. EDA, feature engineering, baseline modeling, hyperparameter tuning, and SHAP explainability complete; agent layer, backend, and deployment in progress.

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
| GenAI | LLM API (Claude), pgvector (planned) | Agentic investigation copilot — in progress |
| Backend | FastAPI (planned) | Real-time scoring + case endpoints |
| Frontend | React, TypeScript, Tailwind (planned) | Analyst dashboard |
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

## 8. Roadmap

- [x] Phase 1 — Research & problem definition
- [x] Phase 2 — Data collection
- [x] Phase 3 — Data engineering (576 features, leakage-safe time-based splits)
- [x] Phase 4 — Baseline ML (LightGBM best on both datasets; honest anomaly-detection comparison documented)
- [x] Phase 5 — Advanced ML/DL (hyperparameter tuning: +7.3% PR-AUC; SHAP explainability; entity-graph features validated in top 7% of 545 features)
- [ ] Phase 6 — AI/GenAI integration (agentic investigation copilot)
- [ ] Phase 7 — Backend
- [ ] Phase 8 — Frontend
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
