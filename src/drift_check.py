"""
src/drift_check.py

Phase 9 — Drift detection for FraudLens.

Computes Population Stability Index (PSI) for every modeling feature,
comparing the TRAIN distribution (reference) against the VAL distribution
(comparison) — since the Phase 3 split was time-based (train = earliest
70%, val = the next 15% chronologically), this is a genuine temporal
drift check, not a synthetic one: it answers "did real transaction
patterns shift between the training period and the following period?"

Deliberately implemented directly with pandas/numpy/scipy rather than a
drift-detection library (e.g. Evidently AI) — given this project's
repeated environment/dependency friction in earlier phases, a ~40-line
PSI implementation using packages already installed is both more
reliable and more demonstrably understood than an added dependency.

PSI interpretation (industry-standard thresholds):
    < 0.1   : no significant shift
    0.1-0.25: moderate shift, worth watching
    > 0.25  : significant shift, consider investigating / retraining

Usage:
    C:\\...\\Python313\\python.exe src\\drift_check.py

Required packages: pandas, numpy, scipy, matplotlib (all already
installed from earlier phases).
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE_DIR = os.path.abspath(os.path.join(os.getcwd(), ".."))
if not os.path.isdir(os.path.join(BASE_DIR, "data")):
    BASE_DIR = os.getcwd()
DATA_DIR = os.path.join(BASE_DIR, "data", "processed")
MODELS_DIR = os.path.join(BASE_DIR, "models")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
os.makedirs(REPORTS_DIR, exist_ok=True)


def psi(reference: np.ndarray, comparison: np.ndarray, bins: int = 10) -> float:
    """Population Stability Index between two 1D numeric arrays.
    Bins are built from the REFERENCE distribution's quantiles, then both
    arrays are binned the same way — this is the standard formulation."""
    reference = reference[~np.isnan(reference)]
    comparison = comparison[~np.isnan(comparison)]
    if len(reference) == 0 or len(comparison) == 0:
        return np.nan

    quantiles = np.linspace(0, 1, bins + 1)
    edges = np.unique(np.quantile(reference, quantiles))
    if len(edges) < 3:
        return 0.0  # reference is near-constant; not a meaningful PSI candidate

    edges[0], edges[-1] = -np.inf, np.inf
    ref_counts, _ = np.histogram(reference, bins=edges)
    comp_counts, _ = np.histogram(comparison, bins=edges)

    ref_pct = np.clip(ref_counts / max(len(reference), 1), 1e-4, None)
    comp_pct = np.clip(comp_counts / max(len(comparison), 1), 1e-4, None)

    return float(np.sum((comp_pct - ref_pct) * np.log(comp_pct / ref_pct)))


def classify(psi_value: float) -> str:
    if np.isnan(psi_value):
        return "n/a"
    if psi_value < 0.1:
        return "stable"
    if psi_value < 0.25:
        return "moderate shift"
    return "significant shift"


def main():
    print("Loading train and val data (train=earliest period, val=later period — "
          "a genuine temporal comparison given the Phase 3 time-based split)...")
    train = pd.read_parquet(os.path.join(DATA_DIR, "ieee_train.parquet"))
    val = pd.read_parquet(os.path.join(DATA_DIR, "ieee_val.parquet"))

    metadata_path = os.path.join(MODELS_DIR, "ieee_lightgbm_tuned_metadata.json")
    with open(metadata_path) as f:
        feats = json.load(f)["feature_columns"]
    print(f"Checking drift across {len(feats)} modeling features...")

    results = []
    for feat in feats:
        if feat not in train.columns or feat not in val.columns:
            continue
        score = psi(train[feat].values.astype(float), val[feat].values.astype(float))
        results.append({"feature": feat, "psi": score, "status": classify(score)})

    df = pd.DataFrame(results).sort_values("psi", ascending=False).reset_index(drop=True)

    n_stable = (df["status"] == "stable").sum()
    n_moderate = (df["status"] == "moderate shift").sum()
    n_significant = (df["status"] == "significant shift").sum()

    print(f"\n{'='*60}")
    print("DRIFT SUMMARY (train -> val, PSI)")
    print(f"{'='*60}")
    print(f"  Stable (PSI<0.1):          {n_stable}")
    print(f"  Moderate shift (0.1-0.25): {n_moderate}")
    print(f"  Significant shift (>0.25): {n_significant}")

    print(f"\nTop 15 features by PSI (most drifted first):")
    print(df.head(15).to_string(index=False))

    csv_path = os.path.join(REPORTS_DIR, "drift_report_train_vs_val.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nFull report saved -> {csv_path}")

    top20 = df.head(20)
    fig, ax = plt.subplots(figsize=(9, 7))
    colors = ["#c0533e" if s == "significant shift" else "#d4a24c" if s == "moderate shift" else "#4a9d6f"
              for s in top20["status"]]
    ax.barh(top20["feature"][::-1], top20["psi"][::-1], color=colors[::-1])
    ax.axvline(0.1, color="gray", linestyle="--", linewidth=1, label="0.1 (moderate)")
    ax.axvline(0.25, color="black", linestyle="--", linewidth=1, label="0.25 (significant)")
    ax.set_xlabel("PSI (train vs val)")
    ax.set_title("Top 20 Features by Population Stability Index")
    ax.legend()
    fig.tight_layout()
    plot_path = os.path.join(REPORTS_DIR, "drift_report_top20.png")
    fig.savefig(plot_path, dpi=120)
    plt.close(fig)
    print(f"Saved plot -> {plot_path}")

    print(f"\n{'='*60}")
    if n_significant == 0:
        print("ASSESSMENT: No features show significant drift between train and val. "
              "This is the expected, reassuring result for a train/val split only ~15 "
              "percentage-points apart in time — it's a useful sanity check that the "
              "split didn't accidentally straddle some real regime change in the data, "
              "and a baseline reading to compare against if this pipeline is ever run "
              "again on genuinely new data later.")
    else:
        print(f"ASSESSMENT: {n_significant} feature(s) show significant drift. Worth "
              "investigating whether this reflects a real behavioral shift over time "
              "or a feature-engineering artifact (e.g. a frequency-encoded feature "
              "whose encoding was fit on train only, which can show apparent 'drift' "
              "in val purely from being an out-of-sample encoding, not real behavior change).")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()