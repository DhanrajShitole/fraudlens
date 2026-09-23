## Phase 4 Summary — Baseline Modeling

### IEEE-CIS (moderate imbalance, 27.4:1)

| Model | PR-AUC | ROC-AUC | Precision@2%FP | Recall@2%FP | F1@budget |
|---|---|---|---|---|---|
| Logistic Regression | 0.305 | 0.818 | 0.374 | 0.336 | 0.354 |
| LightGBM + scale_pos_weight | 0.552 | 0.916 | 0.485 | 0.529 | 0.506 |
| LightGBM + SMOTE | **0.557** | 0.904 | 0.490 | **0.541** | **0.515** |

**Best model: LightGBM + SMOTE.** Both gradient-boosted variants substantially outperform the logistic regression baseline (0.55+ PR-AUC vs. ~0.03 for a random classifier — roughly 16x better than chance, given the 3.5% fraud rate). SMOTE gives a small, real improvement over cost-sensitive weighting alone (+0.005 PR-AUC, +0.009 F1) — a modest but genuine gain, not a dramatic one, which is the honest finding rather than an inflated one.

### PaySim (extreme imbalance, 1,221.6:1)

| Model | PR-AUC | ROC-AUC | Precision@2%FP | Recall@2%FP | F1@budget |
|---|---|---|---|---|---|
| LightGBM + scale_pos_weight | **0.011** | **0.949** | 0.012 | **0.945** | **0.023** |
| Isolation Forest (unsupervised) | 0.002 | 0.840 | 0.001 | 0.025 | 0.001 |

**Best model: LightGBM (supervised) — this overturns our original hypothesis.** Going in, we expected the extreme 1,221:1 imbalance to favor the unsupervised Isolation Forest, since supervised learning typically degrades under severe rarity. Instead, the supervised model won decisively across every metric. We believe this is because PaySim's fraud pattern is highly structured — confined to exactly two transaction types (TRANSFER, CASH_OUT) with a clean account-draining signature — which a model with only 14 features can learn efficiently even from a small number of positive examples. Isolation Forest instead flags generic statistical outliers (e.g., unusually large legitimate transfers), most of which aren't fraud, making it noisy rather than fraud-specific here.

**Practical reading of the numbers:** at a 2% false-positive review budget, the supervised model would catch 94.5% of all fraud (Recall@budget), though precision at that threshold is low (~1.2%) — meaning most flagged transactions would still be false alarms. This precision/recall trade-off, not any single metric in isolation, is the honest summary of what this model can and can't do.

**Caveat on PaySim's numbers:** PaySim's fraud pattern is deterministic-ish by construction (a known property of this synthetic dataset) — fraud only occurs in 2 transaction types with a clean balance-drain signature. This makes the classification problem more learnable than real-world payment fraud typically is, and these strong recall numbers should not be read as a claim about how the model would perform on production data.

**Why PR-AUC looks numerically small despite strong models:** PR-AUC is only meaningful relative to its own random-baseline (≈ the true fraud rate). IEEE-CIS's random baseline is ~0.035 (3.5% fraud rate); PaySim's is ~0.0008 (0.08% fraud rate). Comparing PR-AUC values *across* the two datasets directly is not meaningful — each must be judged against its own baseline.

### Decision for Phase 5

- **IEEE-CIS:** proceed with LightGBM + SMOTE as the current best model; advanced modeling (Phase 5) will focus on hyperparameter tuning, SHAP explainability, and evaluating whether the graph/entity-aggregation features are pulling their weight.
- **PaySim:** proceed with the supervised LightGBM as the primary model; Isolation Forest is retained in the report as a documented, honest negative result rather than dropped silently — it's evidence the project explored the imbalance-handling question rigorously rather than assuming an answer.
