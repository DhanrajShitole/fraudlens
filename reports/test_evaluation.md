# FraudLens — Final Held-Out Test Evaluation

## IEEE-CIS

### Core metrics (threshold-independent)

| Metric | VAL | TEST |
| --- | --- | --- |
| Rows | 88,581 | 88,581 |
| Fraud rate (base rate) | 3.4341% | 3.4804% |
| PR-AUC | 0.59753 | 0.53241 |
| PR-AUC / base rate (lift) | 17.4 | 15.297 |
| ROC-AUC | 0.92419 | 0.90657 |

### Threshold metrics (threshold = 0.129365, chosen on VAL)

| Metric | VAL | TEST |
| --- | --- | --- |
| Precision | 0.50706 | 0.43952 |
| Recall | 0.57857 | 0.5339 |
| F1 | 0.54046 | 0.48213 |
| Actual FP rate (FP / all legit) | 2.0000% | 2.4550% |
| % of all transactions flagged | 3.92% | 4.23% |
| Legit tied at threshold | 0 | 0 |

### Calibration (TEST set only)

- **Mean predicted probability**: 3.19%
- **Observed fraud rate**: 3.48%
- **Brier score**: 0.02213
- **Baseline Brier score (constant base rate p*(1-p))**: 0.03359
- **Brier skill score (1 - Brier / Baseline)**: 0.34115 (34.11%)

> Context: The IEEE-CIS model was trained on SMOTE-resampled data (sampling_strategy=0.1).

**10-bin quantile reliability table (TEST):**

| Mean pred prob | Observed fraud rate | Ratio (pred / obs) |
| --- | --- | --- |
| 0.00095 | 0.00113 | 0.84 |
| 0.0019 | 0.0026 | 0.73 |
| 0.00289 | 0.00305 | 0.95 |
| 0.00408 | 0.00384 | 1.06 |
| 0.0056 | 0.00485 | 1.15 |
| 0.00762 | 0.00824 | 0.92 |
| 0.01062 | 0.01287 | 0.82 |
| 0.01592 | 0.02517 | 0.63 |
| 0.02958 | 0.04504 | 0.66 |
| 0.23943 | 0.24125 | 0.99 |

**Calibration verdict**: Across the 10 quantile bins, 2 have pred/obs > 1 and 8 have pred/obs < 1; overall mean predicted probability is 3.19% vs observed fraud rate 3.48%, confirming that the model **understates** true risk. The Brier score is 0.02213 compared to 0.03359 for a constant base-rate predictor (Brier skill score = 0.34115), confirming substantial predictive skill over baseline despite calibration differences (SMOTE was used during training as context).

## PaySim

### Core metrics (threshold-independent)

| Metric | VAL | TEST |
| --- | --- | --- |
| Rows | 954,393 | 954,393 |
| Fraud rate (base rate) | 0.0589% | 0.4200% |
| PR-AUC | 0.15098 | 0.44567 |
| PR-AUC / base rate (lift) | 256.39 | 106.123 |
| ROC-AUC | 0.98602 | 0.98607 |

> Note: PaySim is scored with raw margins (LightGBM raw_score=True) to prevent probability saturation at 1.0. Because rankings are monotonic with un-saturated probabilities, PR-AUC and ROC-AUC rankings are preserved.


### Threshold metrics (threshold = 5771827.608968, chosen on VAL)

| Metric | VAL | TEST |
| --- | --- | --- |
| Precision | 0.02494 | 0.15128 |
| Recall | 0.86833 | 0.86901 |
| F1 | 0.04849 | 0.25769 |
| Actual FP rate (FP / all legit) | 2.0000% | 2.0560% |
| % of all transactions flagged | 2.05% | 2.41% |
| Legit tied at threshold | 0 | 0 |

## Findings

- **IEEE-CIS generalization**: PR-AUC on test is 0.5324 vs 0.5975 on val (Δ = -0.0651); ROC-AUC is 0.9066 vs 0.9242 (Δ = -0.0176). Test performance is slightly **worse** than val, as expected on truly unseen data.

- **IEEE-CIS FP budget**: Target ≤ 2.00%. Val FP rate = 2.00% (met); Test FP rate = 2.46% (NOT met (exceeded budget by 0.45 pp)). Ties at threshold: 0 val, 0 test.

- **PaySim base-rate shift and generalization**: The fraud base rate shifted from 0.0589% on val to 0.4200% on test (7.1x difference, exceeding the 2x threshold). Because base rates differ by > 2x, PR-AUC is **NOT directly comparable** between val (0.1510) and test (0.4457). Evaluating via lift and ROC-AUC: PR-AUC lift is 106.123 on test vs 256.390 on val (Δ = -150.267); ROC-AUC is 0.9861 vs 0.9860 (Δ = +0.0000). Rankings are evaluated using raw margin scores (LightGBM raw_score=True), which avoid probability saturation at 1.0 and preserve monotonicity.

- **PaySim FP budget**: Target ≤ 2.00%. Val FP rate = 2.00% (met); Test FP rate = 2.06% (NOT met (exceeded budget by 0.06 pp)). Legit rows tied at threshold (5771827.608968): 0 val, 0 test; ties did not prevent meeting the 2% budget on val.

- **Calibration (IEEE-CIS)**: Across the 10 quantile bins, 2 have pred/obs > 1 and 8 have pred/obs < 1; overall mean predicted probability is 3.19% vs observed fraud rate 3.48%, confirming that the model **understates** true risk. The Brier score is 0.02213 compared to 0.03359 for a constant base-rate predictor (Brier skill score = 0.34115), confirming substantial predictive skill over baseline despite calibration differences (SMOTE was used during training as context).

- **Overall verdict (factual summary)**:
  - IEEE-CIS val->test change at fixed threshold (0.129365): PR-AUC changed from 0.5975 to 0.5324 (Δ = -0.0651); recall changed from 0.5786 to 0.5339 (Δ = -0.0447).
  - FP budget on test: Target ≤ 2.00% was NOT met on IEEE-CIS (actual 2.46%) and NOT met on PaySim (actual 2.06%).
  - PaySim test operating point at fixed threshold (5771827.608968): 2.41% of all transactions flagged, precision = 0.1513, recall = 0.8690.
