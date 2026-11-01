# FraudLens — Presentation, Viva Examination & Technical Interview Guide

**Author:** Dhanraj Shitole  
**Purpose:** Comprehensive preparation material for project reviews, viva voce examinations, technical interviews, and industry presentations.  

---

## Part 1: Spoken Presentation Scripts

### 1. 30-Second Elevator Pitch
> *"Financial institutions lose billions to fraud, but the real operational bottleneck today is investigation throughput—compliance analysts spend up to 45 minutes manually assembling evidence for every flagged alert. I built **FraudLens**, an end-to-end platform combining cost-sensitive gradient boosted trees with a privacy-preserving local AI investigation copilot. It scores transactions in under 15 milliseconds with mathematical SHAP reason codes, and automatically generates audit-ready, evidence-backed case files under the rule: code makes decisions, models score, and AI writes prose. Tested on over 6.9 million transactions, FraudLens cuts analyst investigation time from minutes to seconds."*

---

### 2. 1-Minute Project Overview
> *"Hello everyone. In modern banking, fraud detection faces two major hurdles: extreme class imbalance—where fraud accounts for under 1% of transactions—and analyst alert fatigue. Traditional rules generate too many false alarms, while naive LLMs hallucinate facts and leak future data.*
>
> *To solve this, I developed **FraudLens**. On the predictive side, FraudLens uses a tuned LightGBM model trained on leakage-safe temporal splits of the IEEE-CIS dataset, achieving a PR-AUC of 0.598—a 7.3% improvement over untuned baselines—and an ROC-AUC of 0.924. Crucially, we calibrate our decision threshold against a strict 2% false-positive budget to control operational costs, and calculate top-5 TreeSHAP reason codes for instant transparency.*
>
> *On the investigation side, FraudLens introduces a tool-scoped local AI agent powered by Llama 3.1 8B. Instead of letting the LLM guess, deterministic Python tools extract point-in-time entity velocity and check enterprise fraud policies with a relevance floor. An automated validator enforces that every number is grounded in evidence before an analyst sees it. The system is fully containerized, backed by an MLflow registry and Population Stability Index drift tracking, and validated with a 51-test automated suite."*

---

### 3. 5-Minute Academic / Viva Presentation Script
> *"Good morning respected evaluators. Today I am presenting my final-year project, **FraudLens: A Real-Time, Explainable Financial Fraud Detection Platform with an Evidence-Grounded Agentic Investigation Copilot.**
>
> **1. The Problem:**
> When machine learning models flag suspicious transactions, compliance officers must cross-reference account histories, device velocity, and organizational rules to decide whether to freeze the account. This manual context-assembly takes 15 to 45 minutes per alert.
>
> **2. The Data & Leakage Prevention:**
> We evaluated FraudLens on two major benchmarks: the real-world IEEE-CIS dataset (590,000 transactions) and synthetic PaySim (6.3 million transactions). We strictly rejected random train-test splitting because it causes severe lookahead leakage in temporal transaction streams. Instead, we implemented chronological 70/15/15 partitioning: training on the earliest 70%, validating on the middle 15%, and evaluating one-shot generalization on the final held-out 15%.
>
> **3. Feature Engineering & The Anomaly Detection Fallacy:**
> We engineered 545 candidate features. Crucially, because device IDs were 80% missing, we built composite entity aggregations on the tuple `(card1, addr1)`—the physical card issuer and billing district. Using TreeSHAP, we proved these entity features rank in the top 7% of all 545 features. On PaySim, we tested the common hypothesis that unsupervised anomaly detection outperforms supervised models under extreme 1,200:1 imbalance. Our results disproved this: supervised LightGBM achieved 94.5% recall at a 2% FP budget versus only 2.5% for Isolation Forest, because supervised trees efficiently learn discrete account-draining logic.
>
> **4. Model Evaluation & Calibration:**
> On IEEE-CIS, our tuned LightGBM model with SMOTE achieved a validation PR-AUC of 0.5975, an ROC-AUC of 0.9242, and an F1 score of 0.5405 at a 2% false-positive budget threshold of $\tau = 0.129365$. On the held-out test set, it retained a 15.3x lift over base fraud prevalence and achieved a Brier Skill Score of 34.12%, proving solid probability calibration.
>
> **5. The Grounded Investigation Copilot:**
> To eliminate analyst investigation bottlenecks, we built an agentic copilot. In early testing, we discovered 4 distinct hallucination classes in standard LLMs: geographic errors, data-granularity bugs, label leakage, and vacuous self-referential comparisons. We resolved this through the architectural principle: 'Code makes decisions, models score, and LLMs narrate.' The agent executes deterministic, point-in-time Python tools first, builds a policy search query from the evidence, and runs TF-IDF with a relevance floor. Llama 3.1 8B running locally via Ollama drafts two narrative lines, which are passed through an automated regex and numerical validator before being merged into the case file. If Ollama crashes or validation fails, a fallback engine outputs a 100% accurate tool-derived case file with zero data loss.
>
> **6. Production Architecture & MLOps:**
> FraudLens is served via a FastAPI backend delivering sub-15ms scoring, persistent SQLite case logging, a responsive React analyst dashboard, MLflow model registry aliasing (`@production`), Population Stability Index (PSI) drift monitoring across all 545 features, and a 51-test automated CI/CD pipeline in GitHub Actions.
>
> Thank you, and I am now ready for your questions."*

---

### 4. Step-by-Step Live Demonstration Guide
1. **System Health Check:**
   - Open browser or terminal to `http://127.0.0.1:8000/health`. Show that `model_loaded: true` and `model_source` points to `registry:fraudlens-ieee-lightgbm@production` (or `joblib-fallback`).
2. **Scoring a Normal Transaction:**
   - Open `frontend/dashboard.html`.
   - Enter a known legitimate Transaction ID (e.g. `2987000`).
   - Click **Get Score** (or press Enter).
   - Show the low probability (< 5%), green `BELOW THRESHOLD` badge, and explain that the threshold is calibrated to a 2% false-positive budget ($\tau = 0.129365$).
   - Point out the SHAP reason bars showing which features reduced risk.
3. **Scoring a Flagged Fraud Transaction:**
   - Enter a known flagged Transaction ID (e.g. `3141064` or `3400379`).
   - Click **Get Score**.
   - Show the high probability (> 80%), red `FLAGGED FOR REVIEW` badge, and highlight the top SHAP contributor: `card1_addr1_amt_mean` pushing risk up.
4. **Triggering the Investigation Copilot:**
   - With the flagged transaction loaded, click the gold **Investigate** button.
   - Watch the investigation assemble. Show the 5 structured sections:
     - `ENTITY SUMMARY`: Total transactions and dollar spend.
     - `TOP RISK FACTORS`: Bulleted grounded facts.
     - `RELATED ENTITY SIGNAL`: Multi-domain velocity across card+address.
     - `RELEVANT POLICY`: Matched enterprise policy clause (e.g. `POL-003: Email Domain Fraud Ring`).
     - `RECOMMENDED ACTION`: Code-generated escalation recommendation.
5. **Demonstrating Audit History:**
   - Point to the sidebar under **Recent Investigations**. Show the newly created case.
   - Click the case to show instant retrieval from SQLite (`cases.db`).

---

## Part 2: Comprehensive Viva & Technical Interview Question Bank (100+ Questions)

### Category A: General Project & System Architecture
**Q1: What is the core problem FraudLens solves?**  
*Answer:* FraudLens addresses the dual challenge of high-throughput automated fraud scoring under severe class imbalance and eliminating the 15-to-45-minute manual evidence-gathering bottleneck faced by compliance officers reviewing flagged transactions.

**Q2: What is the high-level architecture of FraudLens?**  
*Answer:* A decoupled hybrid pattern consisting of an offline training/MLOps pipeline (LightGBM, SMOTE, MLflow, PSI drift check) and an online serving stack (FastAPI backend, TreeSHAP reason codes, read-only deterministic tools, local Llama 3.1 8B copilot, SQLite case persistence, and a single-file React analyst dashboard).

**Q3: What makes FraudLens different from standard AI fraud detection tools?**  
*Answer:* Standard tools either rely solely on black-box probabilities or use unconstrained generative chatbots that hallucinate facts. FraudLens enforces the principle that *code makes decisions, models score, and LLMs write prose*. The model cannot alter the fraud verdict, and all numerical claims are validated against point-in-time evidence.

**Q4: Who are the primary end users of FraudLens?**  
*Answer:* Anti-Money Laundering (AML) analysts, Level-1 and Level-2 fraud investigation officers, and compliance risk teams in financial institutions and payment gateways.

**Q5: Why did you not build a live transaction ingestion stream with Kafka?**  
*Answer:* In our BTech academic scope, historical parquet files simulate the lookup storage layer. In a production enterprise setting, Kafka would stream transaction payloads to the `/score` endpoint, while historical feature aggregates would reside in an operational feature store (e.g. Feast or Redis).

---

### Category B: AI & Machine Learning Fundamentals
**Q6: What type of machine learning task is fraud detection?**  
*Answer:* Supervised binary classification under extreme class imbalance.

**Q7: What is the target variable in IEEE-CIS and what do values mean?**  
*Answer:* `isFraud`, where `1` denotes confirmed fraudulent transactions (chargebacks, compromised cards) and `0` denotes legitimate transactions.

**Q8: Why is standard classification accuracy a useless metric in fraud detection?**  
*Answer:* Because of class imbalance. In a dataset with 1% fraud, a naive classifier predicting 0 for every transaction achieves 99% accuracy while detecting zero fraud.

**Q9: What is the bias-variance tradeoff in gradient boosting?**  
*Answer:* Bias represents error from overly simplistic assumptions (underfitting); variance represents sensitivity to small fluctuations in training data (overfitting). LightGBM controls variance through regularization parameters (`num_leaves`, `min_child_samples`, `colsample_bytree`, and `subsample`).

**Q10: What is data leakage, and how did FraudLens prevent it?**  
*Answer:* Data leakage occurs when information from outside the training partition influences the model. We prevented it by enforcing chronological 70/15/15 time splits and fitting all encoders, imputers, and SMOTE resamplers strictly on training data.

---

### Category C: Datasets, Preprocessing & Feature Engineering
**Q11: Why is random k-fold cross-validation flawed for fraud detection?**  
*Answer:* Random splits allow transactions from the same card/entity across a single fraud attack to be split between train and test. The model memorizes entity identifiers and predicts past fraud using future records, inflating validation scores artificially.

**Q12: How did you handle missing values in IEEE-CIS?**  
*Answer:* Categorical missing values were assigned a `'missing'` string token. Numerical missing values were imputed with `-999`, accompanied by binary missingness indicator columns. Tree algorithms naturally isolate `-999` to specific branch routes without distorting distributions.

**Q13: How did you handle high-cardinality categorical variables?**  
*Answer:* Frequency encoding fit strictly on the training partition. Each category is replaced by its normalized frequency in training data, avoiding the curse of dimensionality caused by one-hot encoding.

**Q14: What composite entity was engineered, and why?**  
*Answer:* The tuple `(card1, addr1)` representing card issuer bank and geographic billing district. Device and IP columns had >80% missingness, whereas `(card1, addr1)` was available on almost all transactions and effectively captured entity-level behavioral clusters.

**Q15: What point-in-time statistics were computed for this entity?**  
*Answer:* Historical transaction count (`card1_addr1_count`), mean spend (`card1_addr1_amt_mean`), spend standard deviation (`card1_addr1_amt_std`), and distinct email count (`card1_addr1_email_count`).

---

### Category D: LightGBM, Gradient Boosting & Tree Algorithms
**Q16: How does gradient boosting work?**  
*Answer:* It builds an additive ensemble of weak decision trees sequentially. Each new tree fits to the pseudo-residuals (negative gradient of the loss function) of the existing ensemble, progressively minimizing overall loss.

**Q17: Why did you choose LightGBM over XGBoost and Random Forest?**  
*Answer:* Random Forest does not optimize sequential residuals and handles imbalance poorly. XGBoost uses exact greedy split-finding or pre-sorted algorithms that are slow on large datasets. LightGBM uses histogram-based binning (256 discrete bins) and leaf-wise tree growth, reducing training memory by over 70% and accelerating training by 5x–10x.

**Q18: What is leaf-wise tree growth vs level-wise growth?**  
*Answer:* Level-wise (XGBoost) grows all nodes in a tree level symmetrically. Leaf-wise (LightGBM) splits only the single leaf with the maximum delta loss reduction, achieving lower error for a fixed number of leaves.

**Q19: How does LightGBM handle numerical missing values natively?**  
*Answer:* During split evaluation, missing values are placed entirely into the left child, then the right child; the direction yielding higher gain is saved as the default route for unseen missing instances.

**Q20: What is `scale_pos_weight` in LightGBM?**  
*Answer:* A multiplier applied to the gradient of the positive class. Setting `scale_pos_weight = N_neg / N_pos` forces the tree optimizer to treat false negatives as severely as false positives.

---

### Category E: Imbalanced Learning & SMOTE
**Q21: What is SMOTE and how does it generate synthetic samples?**  
*Answer:* Synthetic Minority Over-sampling Technique. For each minority instance, it finds its $k$-nearest minority neighbors in feature space and interpolates a synthetic point along the line segment connecting them: $x_{\text{new}} = x_i + \lambda(x_{zi} - x_i)$ where $\lambda \sim U(0, 1)$.

**Q22: Why did you apply SMOTE with `sampling_strategy = 0.1` instead of `1.0`?**  
*Answer:* Resampling the minority class to 50/50 balance (1.0) severely distorts feature space geometry and leads to high false positives. Balancing to 1:10 (0.1) provides sufficient minority density for gradient boosting without flooding the dataset with synthetic artifacts.

**Q23: How did you prevent SMOTE from leaking data into validation?**  
*Answer:* SMOTE was applied exclusively to the training split *after* temporal partitioning. The validation and held-out test sets remained completely untouched with their natural 27.4:1 class distribution.

**Q24: What are the risks of using SMOTE in high-dimensional tabular data?**  
*Answer:* When features are sparse or collinear, linear interpolation can create synthetic samples in empty regions of feature space that do not correspond to realistic transactions. Combining SMOTE with leaf-wise gradient boosting and tree pruning mitigates this.

---

### Category F: Model Evaluation Metrics & The 2% False-Positive Budget
**Q25: Why is PR-AUC preferred over ROC-AUC in fraud detection?**  
*Answer:* ROC-AUC plots True Positive Rate vs False Positive Rate ($\text{FP} / \text{All Legit}$). Because legitimate transactions dominate by 27:1 or 1,200:1, thousands of false positives barely move the false-positive rate, making ROC-AUC look deceptively optimistic (>0.92). PR-AUC plots Precision ($\text{TP} / [\text{TP} + \text{FP}]$) vs Recall; any influx of false positives directly crushes Precision.

**Q26: What was your model's validation PR-AUC, and what was the lift over baseline?**  
*Answer:* The tuned model achieved a validation PR-AUC of **0.5975**, a **+7.3% relative lift** over untuned LightGBM (0.5566) and nearly double Logistic Regression (0.3050).

**Q27: What is the 2% false-positive budget, and why did you choose it?**  
*Answer:* In real banking compliance, human review teams have fixed staffing capacity. Capping the false-positive rate at 2.0% guarantees that no more than 2% of legitimate transactions trigger manual alerts. The operational threshold $\tau = 0.129365$ was chosen at the 98th percentile of legitimate validation scores.

**Q28: What were the test set results at this threshold on IEEE-CIS?**  
*Answer:* On the held-out test set ($N=88,581$), the model achieved 53.39% Recall, 43.95% Precision, an F1 score of 0.4821, and an actual false-positive rate of 2.46%.

**Q29: What is a Brier Score and Brier Skill Score?**  
*Answer:* The Brier score measures mean squared probability error: $BS = \frac{1}{N} \sum (\hat{p}_i - y_i)^2$. On test, our model achieved $BS = 0.02213$ vs $0.03359$ for a non-informative constant base rate, yielding a Brier Skill Score of **34.12%**, confirming strong probabilistic calibration.

---

### Category G: TreeSHAP & Explainable AI
**Q30: What is the difference between global and local explainability?**  
*Answer:* Global explainability measures overall feature importance across the entire dataset (e.g. mean $|SHAP|$). Local explainability explains the specific prediction for a single transaction by showing exactly how much each feature shifted the score above or below the base rate.

**Q31: What are Shapley values from game theory?**  
*Answer:* Shapley values allocate the payout of a cooperative game fairly among players based on their marginal contributions across all possible player coalitions. In machine learning, the players are features, and the payout is the model prediction.

**Q32: Why use TreeSHAP instead of KernelSHAP or LIME?**  
*Answer:* KernelSHAP and LIME rely on heuristic sampling and take exponential time $O(2^d)$ or minutes per sample. TreeSHAP exploits tree graph structures to compute exact Shapley values in polynomial time $O(T L D^2)$, completing local explanations in milliseconds.

**Q33: How does the dashboard visualize SHAP values?**  
*Answer:* The top 5 features by absolute magnitude $|\phi_i|$ are displayed as horizontal bars: red bars indicate positive contributions pushing risk up; green bars indicate risk-reducing contributions.

---

### Category H: Agentic AI & Investigation Copilot
**Q34: What is the difference between a chatbot and an AI agent?**  
*Answer:* A chatbot generates text based solely on user prompts. An AI agent is tool-scoped: it observes an environment, executes read/write tools, inspects deterministic state, and synthesizes structured actions.

**Q35: Why should an LLM never decide the fraud verdict?**  
*Answer:* Because LLMs are non-deterministic, probabilistic text generators. Giving an LLM decision authority over banking blocks introduces non-reproducible compliance decisions, hallucinated justifications, and legal liabilities.

**Q36: What model powers the agent, and why was it selected?**  
*Answer:* Llama 3.1 8B running locally through Ollama. It was selected because it runs completely offline without sending customer PII to cloud APIs, incurs zero per-token API costs, and supports concise structured prompting.

**Q37: What was the geographic hallucination bug in early testing?**  
*Answer:* The model misconstrued `addr1 = 299` as an international country code and fabricated a cross-border fraud justification. It was fixed with explicit prompt constraints explaining `addr1` is an internal regional billing code.

**Q38: How does the agent prevent temporal label leakage?**  
*Answer:* In `src/agent.py`, all evidence tool queries strictly filter to `TransactionDT <= this transaction`. Prior confirmed fraud counts strictly require `TransactionDT < this transaction` and `TransactionID != this transaction`.

**Q39: What is the vacuous comparative framing bug, and how is it blocked?**  
*Answer:* The model wrote statements like *"mean transaction amount of $144.14 is greater than the overall mean of $144.14"*. The hardened validator uses regex `_COMPARATIVE_RE` to reject comparative phrasing unless the sentence contains at least two distinct grounded numbers.

**Q40: What happens if Ollama crashes during an investigation?**  
*Answer:* The backend catches `_LLMError` and immediately triggers `_fallback_case_file()`, assembling all tool evidence and policy citations into the standard 5-section case file with 0% data loss.

---

### Category I: Backend, API & Database
**Q41: Why was FastAPI chosen over Flask or Django?**  
*Answer:* FastAPI provides native async ASGI concurrency, automatic Pydantic data validation, sub-15ms response times, and automated interactive OpenAPI/Swagger documentation at `/docs`.

**Q42: What endpoints does the backend expose?**  
*Answer:* `GET /health`, `POST /score/{transaction_id}`, `POST /investigate/{transaction_id}`, `GET /cases`, and `GET /cases/{case_id}`.

**Q43: How does the backend store investigation cases?**  
*Answer:* Cases are logged to a local SQLite database (`cases.db`) in table `cases` containing `id`, `transaction_id`, `risk_score`, `threshold_used`, `flagged`, `top_reason_codes` (JSON), `case_file`, and UTC `created_at`.

**Q44: Why use SQLite instead of PostgreSQL for this phase?**  
*Answer:* SQLite is serverless, zero-configuration, self-contained in a single file, and easily reproducible across academic evaluation environments. Migrating to PostgreSQL is a seamless next step for multi-analyst cloud deployment.

---

### Category J: MLOps, Drift Detection & DevOps
**Q45: What is the Population Stability Index (PSI)?**  
*Answer:* A statistical metric measuring the divergence between a reference distribution $P$ and comparison distribution $Q$: $\text{PSI} = \sum (Q_b - P_b) \ln(Q_b / P_b)$. $\text{PSI} < 0.1$ indicates stability.

**Q46: Why did `TransactionDT` exhibit a PSI of 8.28, and why is that not real drift?**  
*Answer:* `TransactionDT` is the chronological time variable used to split train (earliest 70%) from validation (next 15%). The two distributions have completely disjoint supports; it is a split artifact, not behavioral drift.

**Q47: Which features exhibited genuine drift in your analysis?**  
*Answer:* Exactly 7 features (1.3% of feature space), primarily device and network fingerprint headers (`id_13`, `id_20`, `id_02`) reflecting evolving operating system versions.

**Q48: How does MLflow handle model registry promotion?**  
*Answer:* We use MLflow's modern Model Aliases API (`client.set_registered_model_alias`), attaching the `@production` alias to the active version of `fraudlens-ieee-lightgbm`.

**Q49: What checks are performed in GitHub Actions CI?**  
*Answer:* Python 3.13 dependency installation, bytecode compilation syntax check (`compileall`), HTML syntax parsing of the dashboard, automated pytest execution of 51 tests, and Docker container build verification.

**Q50: Why can the LLM agent not easily run on free cloud instances?**  
*Answer:* Llama 3.1 8B requires $\ge 6$ GB of RAM for model weights and KV cache. Free cloud tiers typically provide 512MB to 1GB RAM. FraudLens resolves this with a hybrid architecture: scoring runs in lightweight containers, while the LLM runs locally or on a private high-RAM server.

