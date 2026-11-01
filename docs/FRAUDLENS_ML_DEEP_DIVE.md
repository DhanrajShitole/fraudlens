# FraudLens — Machine Learning & Explainability Deep Dive

**Author:** Dhanraj Shitole  
**Target Audience:** Machine Learning Engineers, Data Scientists, Academic Evaluators  
**Scope:** Algorithms, Data Engineering, Hyperparameter Optimization, TreeSHAP, and Empirical Evaluation  

---

## 1. The Machine Learning Problem Formulation

Financial fraud detection is formulated as a **cost-sensitive binary classification problem** under extreme class imbalance. Let:
- $\mathcal{X} \subset \mathbb{R}^d$ denote the $d$-dimensional feature space ($d = 545$).
- $\mathcal{Y} \in \{0, 1\}$ denote the binary ground-truth label, where $y = 1$ indicates confirmed fraudulent activity and $y = 0$ indicates legitimate commercial activity.
- The machine learning model parameterized by $\theta$ produces an estimated fraud probability:
  $$\hat{p}(x) = P(Y = 1 \mid X = x; \theta) \in [0, 1]$$
- An operational decision rule maps the probability to a discrete action via threshold $\tau$:
  $$\hat{y}(x) = \begin{cases} 1 & \text{if } \hat{p}(x) \ge \tau \quad (\text{Flag for manual compliance review}) \\ 0 & \text{if } \hat{p}(x) < \tau \quad (\text{Auto-approve / Process transaction}) \end{cases}$$

### The Asymmetric Cost Matrix
In traditional symmetric machine learning (e.g. image classification), False Positives (FP) and False Negatives (FN) incur identical unit penalties. In financial transactions:
- **False Negative Cost ($C_{\text{FN}}$):** Direct financial chargeback, interchange penalties, loss of goods, and compliance sanctions. $C_{\text{FN}} \propto \text{TransactionAmt}$.
- **False Positive Cost ($C_{\text{FP}}$):** Human analyst review cost ($\approx \$10 - \$30$ per manual case), customer friction, and merchant drop-off.

Because $C_{\text{FN}} \gg C_{\text{FP}}$, optimizing standard accuracy is mathematically invalid. A trivial dummy model that predicts $\hat{y} = 0$ for all transactions achieves **96.5% accuracy** on IEEE-CIS and **99.87% accuracy** on PaySim, while intercepting **0% of fraud**.

---

## 2. Benchmark Datasets & Temporal Partitioning

| Parameter | IEEE-CIS Fraud Detection | PaySim Mobile Money |
|---|---|---|
| **Domain** | E-commerce credit card transactions | Synthetic mobile money transfers |
| **Total Transactions** | 590,540 | 6,362,620 |
| **Fraud Cases** | 20,663 | 8,213 |
| **Class Imbalance** | 27.4 : 1 (3.50% fraud) | 1,221.6 : 1 (0.13% fraud) |
| **Split Strategy** | 70% Train / 15% Val / 15% Held-Out Test | 70% Train / 15% Val / 15% Held-Out Test |
| **Train Observations** | 413,378 rows | 4,453,834 rows |
| **Validation Observations** | 88,581 rows | 954,393 rows |
| **Held-Out Test Observations**| 88,581 rows | 954,393 rows |
| **Partitioning Variable** | Chronological on `TransactionDT` | Chronological on temporal `step` |

### Why Chronological Splitting Is Mandatory
If random $k$-fold cross-validation is applied to temporal financial transactions:
1. An entity that commits multiple fraudulent transactions over a 48-hour window would have transactions scattered across both training and validation sets.
2. The tree model would memorize the entity identifier (`card1`, `addr1`) and predict validation fraud based on future observations of the same card.
3. This creates massive **lookahead leakage**, resulting in an artificially inflated validation score that collapses in production.

In FraudLens, all encoders, imputers, and SMOTE resamplers are **fit strictly on the 70% training split**.

---

## 3. Feature Engineering Architecture (`src/features.py`)

Starting from raw tabular schemas, FraudLens engineers 576 modeling features (pruned to 545 non-redundant candidates):

### 3.1 Temporal Feature Transformations
`TransactionDT` represents elapsed seconds from a base reference date:
- Day-of-week: $D = \lfloor \frac{\text{TransactionDT}}{86400} \rfloor \pmod 7$
- Hour-of-day: $H = \lfloor \frac{\text{TransactionDT}}{3600} \rfloor \pmod{24}$

### 3.2 High-Cardinality Frequency Encoding
Categorical features (`card1`, `card2`, `addr1`, `P_emaildomain`, `R_emaildomain`) contain tens of thousands of unique categories. One-hot encoding would explode dimensionality and sparsify tree splits. We compute frequency encodings strictly on the training partition:
$$\text{freq}(c) = \frac{\sum_{i \in \text{Train}} \mathbb{I}(X_i = c)}{|\text{Train}|}$$
Unseen categories appearing in validation or test are assigned the fallback minimum frequency.

### 3.3 Composite Entity Aggregations
Device IDs and IP addresses suffer from up to 80% missingness in production payment gateways. FraudLens constructs composite behavioral entities using the tuple:
$$\text{Entity} = (\text{card1}, \text{addr1})$$
Where `card1` identifies the payment card issuer bank and cardholder series, and `addr1` represents the geographic billing district code. We compute cumulative point-in-time statistics:
1. `card1_addr1_count`: Number of historical transactions on this exact card and address.
2. `card1_addr1_amt_mean`: Running mean transaction amount for this entity.
3. `card1_addr1_amt_std`: Spending standard deviation (captures anomalous deviation from typical spend).
4. `card1_addr1_email_count`: Number of distinct email domains ever associated with this card+address.

---

## 4. The Anomaly Detection Fallacy (PaySim Benchmark)

A common assumption in academic literature asserts that when fraud imbalance exceeds 1,000:1, unsupervised anomaly detection (e.g. Isolation Forest) should outperform supervised classifiers because positive labels are too scarce to learn decision boundaries.

**FraudLens tested this hypothesis directly on PaySim’s 6.36 million transactions (1,221.6:1 imbalance):**

```
PaySim Evaluation: Supervised LightGBM vs. Unsupervised Isolation Forest
┌──────────────────────────────────────┬──────────┬──────────┬────────────────────────┐
│ Model Architecture                   │ PR-AUC   │ ROC-AUC  │ Recall @ 2% FP Budget  │
├──────────────────────────────────────┼──────────┼──────────┼────────────────────────┤
│ Isolation Forest (Unsupervised)      │ 0.0020   │ 0.8400   │  2.5%                  │
│ LightGBM + scale_pos_weight (Superv.)│ 0.0110   │ 0.9490   │ 94.5%                  │
└──────────────────────────────────────┴──────────┴──────────┴────────────────────────┘
```

### Empirical Insight:
- Isolation Forest isolates statistical outliers across all feature dimensions simultaneously. In financial data, benign high-net-worth commercial transfers are extreme statistical outliers (huge dollar amounts, unusual hours), so Isolation Forest flags them as anomalies.
- Supervised LightGBM learns the specific **structural fraud signature**: PaySim fraud occurs exclusively in `TRANSFER` followed by `CASH_OUT`, draining the originating account balance to zero. Because supervised trees learn split criteria directly conditioned on the label $y$, they isolate the account-draining rule with 94.5% recall, whereas unsupervised anomaly detection fails completely.

---

## 5. Machine Learning Algorithms & Optimization

### 5.1 LightGBM Mathematical Foundation
LightGBM builds an ensemble of $K$ regression trees:
$$\hat{y}_i = \sum_{k=1}^{K} f_k(x_i), \quad f_k \in \mathcal{F}$$
Minimizing the regularized binary cross-entropy objective:
$$\mathcal{L} = \sum_{i=1}^{N} l(y_i, \hat{y}_i) + \sum_{k=1}^{K} \Omega(f_k)$$
Where $l(y_i, \hat{y}_i) = - [y_i \ln \sigma(\hat{y}_i) + (1 - y_i) \ln(1 - \sigma(\hat{y}_i))]$.

#### Key LightGBM Architectural Advantages:
1. **Histogram-Based Binning:** Continuous numerical values are grouped into 256 discrete bins, reducing split-finding memory consumption from $O(N \cdot d)$ to $O(256 \cdot d)$.
2. **Leaf-Wise Tree Growth:** Instead of level-wise growth (XGBoost), LightGBM splits the single leaf with maximum loss reduction, achieving lower error with fewer tree levels.
3. **Handling of Missing Values (`-999`):** Tree splits naturally allocate missing values to whichever child branch maximizes reduction in variance, avoiding artificial imputation distortion.

### 5.2 Synthetic Minority Over-sampling Technique (SMOTE)
In our Phase 5 advanced model, we apply SMOTE to the training split with `sampling_strategy = 0.1` (bringing the minority class ratio from 1:27.4 up to 1:10).

```
Feature 2
   ▲
   │        o (Synthetic Fraud)
   │     x ─────── o ─────── x (Minority Fraud Instances)
   │        \
   │         x
   │
   │   •   •   •   •   • (Majority Non-Fraud Instances)
   └────────────────────────────────► Feature 1
```

For each minority example $x_i$:
1. Compute Euclidean distance to all minority class neighbors; select $k$-nearest neighbors ($k=5$).
2. Choose a random neighbor $x_{zi}$.
3. Generate synthetic instance:
   $$x_{\text{new}} = x_i + \lambda (x_{zi} - x_i), \quad \lambda \sim U(0, 1)$$

### 5.3 Hyperparameter Optimization
Hyperparameters were tuned via `RandomizedSearchCV` scored against **PR-AUC** across 3 time-based cross-validation folds:

```python
{
    "n_estimators": 500,
    "learning_rate": 0.03,
    "num_leaves": 63,
    "max_depth": -1,
    "min_child_samples": 50,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "scale_pos_weight": 1.0  # SMOTE handles resampling balance directly
}
```

---

## 6. TreeSHAP Explainability (`api/main.py`)

A primary reason traditional black-box ensembles are rejected by banking regulators is the lack of individual decision explanations. FraudLens integrates **TreeSHAP** (SHapley Additive exPlanations):

### 6.1 Shapley Value Axioms
For any individual transaction prediction $f(x)$, TreeSHAP allocates an additive attribution $\phi_i$ to each feature $i \in \{1, \dots, d\}$ such that:
$$f(x) = \phi_0 + \sum_{i=1}^{d} \phi_i(x)$$
Where $\phi_0 = \mathbb{E}[f(x)]$ is the base expected model prediction.

Shapley values are the **unique** solution satisfying four foundational economic axioms:
1. **Efficiency:** Attributions sum exactly to the difference between prediction and expected base value.
2. **Symmetry:** If two features contribute identically across all possible feature subsets, their attributions are equal.
3. **Dummy / Null Player:** A feature that has no impact on predictions receives $\phi_i = 0$.
4. **Additivity:** When combining model components, attributions sum linearly.

### 6.2 Top-5 Reason Code Extraction
In `api/main.py`, upon scoring transaction $X$:
1. `explainer.shap_values(X)` computes the vector of attributions $[\phi_1, \dots, \phi_{545}]$.
2. The indices are sorted by absolute magnitude:
   $$\text{rank} = \text{argsort}(|\phi|)[\::-1]$$
3. The top 5 features with their values and attributions are formatted into JSON reason codes:
   $$\text{top\_reasons} = \left\{ (\text{feature}_j, \text{value}_j, \phi_j) \mid j \in \text{rank}[:5] \right\}$$

---

## 7. Comprehensive Model Benchmark & Held-Out Test Evaluation

### 7.1 Validation Set Benchmark (IEEE-CIS, $N=88,581$)

| Model Architecture | PR-AUC | ROC-AUC | Precision @ 2% FP | Recall @ 2% FP | F1 @ Budget |
|---|---|---|---|---|---|
| **Logistic Regression** (Class-Weighted) | 0.3050 | 0.8180 | 0.3740 | 0.3360 | 0.3540 |
| **LightGBM + scale_pos_weight** | 0.5520 | 0.9160 | 0.4850 | 0.5290 | 0.5060 |
| **LightGBM + SMOTE** (Phase 4 Baseline) | 0.5566 | 0.9042 | 0.4900 | 0.5410 | 0.5150 |
| **LightGBM + SMOTE Tuned** (Phase 5, Serving) | **0.5975** | **0.9242** | **0.5071** | **0.5786** | **0.5405** |

*Key Result:* Hyperparameter tuning yielded a **+7.3% relative lift** in PR-AUC (from 0.5566 to 0.5975) and increased operational recall at the 2% FP budget from 54.1% to 57.86%.

### 7.2 Final Held-Out Test Evaluation (IEEE-CIS, $N=88,581$)
Evaluated strictly one-shot on unseen future data with the decision threshold frozen at $\tau = 0.129365$:

| Evaluation Metric | Validation Set | Held-Out Test Set | Delta ($\Delta$) | Analysis |
|---|---|---|---|---|
| **Rows ($N$)** | 88,581 | 88,581 | 0 | Equal sample sizes |
| **Fraud Rate (Base Rate)** | 3.4341% | 3.4804% | +0.0463 pp | Stable base prevalence |
| **PR-AUC** | **0.5975** | **0.5324** | -0.0651 | Expected mild degradation on unseen future data |
| **PR-AUC Lift over Base Rate** | 17.40x | 15.30x | -2.10x | Substantial discriminative power retained |
| **ROC-AUC** | 0.9242 | 0.9066 | -0.0176 | Excellent global ranking preserved |
| **Precision @ Threshold** | 50.71% | 43.95% | -6.76 pp | 44% of flagged alerts are true fraud |
| **Recall @ Threshold** | 57.86% | 53.39% | -4.47 pp | Over 53% of all fraud intercepted |
| **F1 Score @ Threshold** | 0.5405 | 0.4821 | -0.0584 | Strong operational performance |
| **Actual False-Positive Rate**| **2.0000%** | **2.4550%** | +0.455 pp | Slightly exceeded 2.0% budget due to mild drift |
| **% of All Transactions Flagged**| 3.92% | 4.23% | +0.31 pp | Manageable queue volume for human analysts |

### 7.3 Calibration & Brier Skill Score (`reports/calibration_test.png`)
A model predicting $P(\text{fraud}) = 0.80$ should ideally see 80 out of 100 such transactions be actual fraud.
- **Mean Predicted Probability:** 3.19%
- **Observed Fraud Prevalence:** 3.48%
- **Brier Score:** $BS = \frac{1}{N} \sum_{i=1}^{N} (\hat{p}_i - y_i)^2 = \mathbf{0.02213}$
- **Baseline Brier Score (Constant Base Rate):** $BS_{\text{base}} = \bar{y}(1 - \bar{y}) = \mathbf{0.03359}$
- **Brier Skill Score:**
  $$\text{BSS} = 1 - \frac{BS}{BS_{\text{base}}} = 1 - \frac{0.02213}{0.03359} = \mathbf{0.34115} \quad (\mathbf{34.12\%})$$
A Brier Skill Score of 34.12% verifies strong probabilistic accuracy and calibration over uninformative baselines.

