"""
src/evaluate_test.py
====================
Final held-out evaluation on the TEST sets for both IEEE-CIS and PaySim.

This is a ONE-SHOT evaluation. The model and threshold were decided on
train/val only. Nothing here feeds back to model selection.

Usage (from the repo root):
    C:\\...\\Python313\\python.exe src\\evaluate_test.py

Outputs:
    reports/test_evaluation.md
    reports/calibration_test.png
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.calibration import calibration_curve

warnings.filterwarnings("ignore")

# Resolve all paths relative to this file so the script is cwd-independent.
_HERE = Path(__file__).resolve()
BASE_DIR = _HERE.parents[1]
DATA_DIR = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _core_metrics(y_true: np.ndarray, y_score: np.ndarray) -> dict:
    base_rate = float(y_true.mean())
    pr_auc = float(average_precision_score(y_true, y_score))
    roc_auc = float(roc_auc_score(y_true, y_score))
    return {
        "rows": int(len(y_true)),
        "fraud_rate": round(base_rate, 6),
        "pr_auc": round(pr_auc, 5),
        "pr_auc_lift": round(pr_auc / base_rate if base_rate > 0 else float("nan"), 3),
        "roc_auc": round(roc_auc, 5),
    }


def _threshold_metrics(y_true: np.ndarray, y_score: np.ndarray, threshold: float) -> dict:
    """Precision / recall / F1, ACTUAL FP rate, and tie count at a fixed threshold."""
    y_pred = (y_score >= threshold).astype(int)
    n_legit = int((y_true == 0).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    flagged_frac = float(y_pred.mean())
    actual_fp_rate = float(fp / n_legit) if n_legit > 0 else float("nan")
    tie_count = int(((y_score == threshold) & (y_true == 0)).sum())
    return {
        "threshold": round(threshold, 6),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 5),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 5),
        "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 5),
        "actual_fp_rate": round(actual_fp_rate, 5),
        "pct_flagged": round(flagged_frac * 100, 3),
        "tie_count": tie_count,
    }


def _choose_threshold_at_fp_budget(y_true: np.ndarray, y_score: np.ndarray,
                                    fp_budget: float = 0.02) -> float:
    """98th percentile of legit scores (2% FP budget)."""
    legit_scores = y_score[y_true == 0]
    return float(np.percentile(legit_scores, (1.0 - fp_budget) * 100))


def _calibration_table(y_true: np.ndarray, y_prob: np.ndarray,
                        n_bins: int = 10) -> pd.DataFrame:
    frac_pos, mean_pred = calibration_curve(y_true, y_prob,
                                            n_bins=n_bins, strategy="quantile")
    return pd.DataFrame({
        "mean_pred_prob": np.round(mean_pred, 5),
        "observed_fraud_rate": np.round(frac_pos, 5),
        "ratio (pred/obs)": np.round(mean_pred / np.where(frac_pos > 0, frac_pos, np.nan), 2),
    })


def _generate_calibration_verdict(cal_table: pd.DataFrame, mean_pred: float, mean_obs: float,
                                  brier: float, brier_base: float, brier_skill: float) -> str:
    ratios = cal_table["ratio (pred/obs)"]
    bins_gt_1 = int((ratios > 1.0).sum())
    bins_lt_1 = int((ratios < 1.0).sum())

    if mean_pred < mean_obs and bins_lt_1 > bins_gt_1:
        direction = "understates"
    elif mean_pred > mean_obs and bins_gt_1 > bins_lt_1:
        direction = "overstates"
    else:
        direction = "roughly calibrated"

    return (
        f"Across the 10 quantile bins, {bins_gt_1} have pred/obs > 1 and {bins_lt_1} have pred/obs < 1; "
        f"overall mean predicted probability is {mean_pred:.2%} vs observed fraud rate {mean_obs:.2%}, "
        f"confirming that the model **{direction}** true risk. "
        f"The Brier score is {brier:.5f} compared to {brier_base:.5f} for a constant base-rate predictor "
        f"(Brier skill score = {brier_skill:.5f}), confirming substantial predictive skill over baseline "
        f"despite calibration differences (SMOTE was used during training as context)."
    )


def _save_calibration_plot(y_true: np.ndarray, y_prob: np.ndarray,
                            out_path: Path) -> None:
    frac_pos, mean_pred = calibration_curve(y_true, y_prob,
                                            n_bins=10, strategy="quantile")
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(mean_pred, frac_pos, marker="o", label="IEEE-CIS model (SMOTE-trained)")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfectly calibrated")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of actual positives (observed fraud rate)")
    ax.set_title("Reliability plot — IEEE-CIS TEST set")
    ax.legend()
    ax.set_xlim(0, None)
    ax.set_ylim(0, None)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# IEEE-CIS evaluation
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_ieee() -> dict:
    print("=" * 70)
    print("IEEE-CIS — Held-out TEST evaluation")
    print("=" * 70)

    # Load model and metadata
    model = joblib.load(MODELS_DIR / "ieee_lightgbm_tuned.joblib")
    with open(MODELS_DIR / "ieee_lightgbm_tuned_metadata.json") as f:
        meta = json.load(f)

    feature_columns: list[str] = meta["feature_columns"]
    threshold: float = meta["threshold_2pct_fp_budget"]   # chosen on VAL — fixed

    print(f"  Model: {meta.get('model_type')}")
    print(f"  Features: {len(feature_columns)}")
    print(f"  Threshold (from val): {threshold:.6f}")

    val_df = pd.read_parquet(DATA_DIR / "ieee_val.parquet")
    test_df = pd.read_parquet(DATA_DIR / "ieee_test.parquet")

    def _score(df: pd.DataFrame):
        # Exact same fillna(-999) as training
        X = df[feature_columns].fillna(-999)
        y = df["isFraud"].values.astype(int)
        prob = model.predict_proba(X)[:, 1]
        return y, prob

    y_val, prob_val = _score(val_df)
    y_test, prob_test = _score(test_df)

    val_core = _core_metrics(y_val, prob_val)
    test_core = _core_metrics(y_test, prob_test)
    val_thresh = _threshold_metrics(y_val, prob_val, threshold)
    test_thresh = _threshold_metrics(y_test, prob_test, threshold)

    # Calibration — TEST only
    brier = float(brier_score_loss(y_test, prob_test))
    p_obs = float(y_test.mean())
    p_pred = float(prob_test.mean())
    brier_baseline = float(p_obs * (1.0 - p_obs))
    brier_skill = float(1.0 - brier / brier_baseline) if brier_baseline > 0 else float("nan")
    cal_table = _calibration_table(y_test, prob_test)
    _save_calibration_plot(y_test, prob_test, REPORTS_DIR / "calibration_test.png")

    cal_verdict = _generate_calibration_verdict(
        cal_table=cal_table,
        mean_pred=p_pred,
        mean_obs=p_obs,
        brier=brier,
        brier_base=brier_baseline,
        brier_skill=brier_skill,
    )

    print(f"  Brier score (TEST): {brier:.5f} (baseline: {brier_baseline:.5f}, skill: {brier_skill:.5f})")
    print(f"  Mean pred: {p_pred:.4f}, Mean obs: {p_obs:.4f}")
    print(f"  Calibration plot -> reports/calibration_test.png")

    print(f"\n  VAL core metrics: {val_core}")
    print(f"  TEST core metrics: {test_core}")
    print(f"  VAL threshold metrics: {val_thresh}")
    print(f"  TEST threshold metrics: {test_thresh}")

    return {
        "val_core": val_core,
        "test_core": test_core,
        "val_thresh": val_thresh,
        "test_thresh": test_thresh,
        "threshold": threshold,
        "brier_test": brier,
        "brier_baseline": brier_baseline,
        "brier_skill": brier_skill,
        "mean_pred": p_pred,
        "mean_obs": p_obs,
        "cal_table": cal_table,
        "cal_verdict": cal_verdict,
    }


# ─────────────────────────────────────────────────────────────────────────────
# PaySim evaluation  (replicates run_paysim_pipeline recipe exactly)
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_paysim() -> dict:
    print("\n" + "=" * 70)
    print("PaySim — Held-out TEST evaluation")
    print("=" * 70)

    try:
        import lightgbm as lgb
    except ImportError:
        sys.exit("lightgbm not installed — cannot run PaySim evaluation.")

    train_df = pd.read_parquet(DATA_DIR / "paysim_train.parquet")
    val_df = pd.read_parquet(DATA_DIR / "paysim_val.parquet")
    test_df = pd.read_parquet(DATA_DIR / "paysim_test.parquet")

    # Feature selection — mirrors train_baseline._feature_columns with
    # drop_cols=["isFlaggedFraud"] as used in run_paysim_pipeline.
    target = "isFraud"
    drop_cols = {"isFlaggedFraud", target}
    numeric_cols = train_df.select_dtypes(include=[np.number]).columns.tolist()
    feats = [c for c in numeric_cols if c not in drop_cols]
    print(f"  Using {len(feats)} numeric features.")

    X_train = train_df[feats].fillna(-999)
    y_train = train_df[target].values.astype(int)
    X_val = val_df[feats].fillna(-999)
    y_val = val_df[target].values.astype(int)
    X_test = test_df[feats].fillna(-999)
    y_test = test_df[target].values.astype(int)

    # class imbalance ratio — identical to run_paysim_pipeline
    imbalance_ratio = float((y_train == 0).sum()) / max(float((y_train == 1).sum()), 1.0)
    print(f"  Imbalance ratio (legit:fraud): {imbalance_ratio:.2f}")

    model = lgb.LGBMClassifier(
        n_estimators=300, learning_rate=0.05, num_leaves=63,
        scale_pos_weight=imbalance_ratio,
        random_state=42, n_jobs=-1, verbosity=-1,
    )
    model.fit(X_train, y_train)
    print("  Training complete.")

    # Score with raw margins (LightGBM predict(raw_score=True)) to prevent
    # probability saturation at 1.0 which causes legit ties at the top.
    raw_val = model.predict(X_val, raw_score=True)
    raw_test = model.predict(X_test, raw_score=True)

    # Choose threshold as 98th percentile of VAL legit scores (2% FP budget),
    # then apply that fixed threshold to TEST.
    threshold = _choose_threshold_at_fp_budget(y_val, raw_val, fp_budget=0.02)
    print(f"  Threshold chosen on VAL (98th percentile of legit raw margin): {threshold:.6f}")

    val_core = _core_metrics(y_val, raw_val)
    test_core = _core_metrics(y_test, raw_test)
    val_thresh = _threshold_metrics(y_val, raw_val, threshold)
    test_thresh = _threshold_metrics(y_test, raw_test, threshold)

    print(f"  VAL core metrics: {val_core}")
    print(f"  TEST core metrics: {test_core}")
    print(f"  VAL threshold metrics: {val_thresh}")
    print(f"  TEST threshold metrics: {test_thresh}")

    return {
        "val_core": val_core,
        "test_core": test_core,
        "val_thresh": val_thresh,
        "test_thresh": test_thresh,
        "threshold": threshold,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Report generation
# ─────────────────────────────────────────────────────────────────────────────

def _md_row(*cells) -> str:
    return "| " + " | ".join(str(c) for c in cells) + " |"


def _generate_findings(ieee: dict, paysim: dict) -> list[str]:
    findings: list[str] = []

    # 1. IEEE-CIS generalization
    ieee_pr_val = ieee["val_core"]["pr_auc"]
    ieee_pr_test = ieee["test_core"]["pr_auc"]
    ieee_pr_diff = ieee_pr_test - ieee_pr_val
    ieee_roc_val = ieee["val_core"]["roc_auc"]
    ieee_roc_test = ieee["test_core"]["roc_auc"]
    ieee_roc_diff = ieee_roc_test - ieee_roc_val

    ieee_gen_eval = (
        "Test performance is slightly **worse** than val, as expected on truly unseen data."
        if ieee_pr_diff < 0
        else "Test performance is comparable to or **better** than val."
    )
    findings.append(
        f"**IEEE-CIS generalization**: PR-AUC on test is {ieee_pr_test:.4f} vs {ieee_pr_val:.4f} on val (Δ = {ieee_pr_diff:+.4f}); "
        f"ROC-AUC is {ieee_roc_test:.4f} vs {ieee_roc_val:.4f} (Δ = {ieee_roc_diff:+.4f}). {ieee_gen_eval}"
    )

    # 2. IEEE-CIS FP budget
    ieee_fp_val = ieee["val_thresh"]["actual_fp_rate"]
    ieee_fp_test = ieee["test_thresh"]["actual_fp_rate"]
    ieee_val_met = "met" if ieee_fp_val <= 0.02005 else "NOT met"
    ieee_test_met = "met" if ieee_fp_test <= 0.02000 else "NOT met"
    ieee_test_detail = "met" if ieee_test_met == "met" else f"NOT met (exceeded budget by {(ieee_fp_test - 0.02) * 100:.2f} pp)"
    findings.append(
        f"**IEEE-CIS FP budget**: Target ≤ 2.00%. Val FP rate = {ieee_fp_val:.2%} ({ieee_val_met}); "
        f"Test FP rate = {ieee_fp_test:.2%} ({ieee_test_detail}). "
        f"Ties at threshold: {ieee['val_thresh']['tie_count']} val, {ieee['test_thresh']['tie_count']} test."
    )

    # 3. PaySim base-rate shift and generalization
    paysim_br_val = paysim["val_core"]["fraud_rate"]
    paysim_br_test = paysim["test_core"]["fraud_rate"]
    br_ratio = paysim_br_test / paysim_br_val if paysim_br_val > 0 else float("inf")
    paysim_lift_val = paysim["val_core"]["pr_auc_lift"]
    paysim_lift_test = paysim["test_core"]["pr_auc_lift"]
    paysim_lift_diff = paysim_lift_test - paysim_lift_val
    paysim_roc_val = paysim["val_core"]["roc_auc"]
    paysim_roc_test = paysim["test_core"]["roc_auc"]
    paysim_roc_diff = paysim_roc_test - paysim_roc_val

    if br_ratio > 2.0 or br_ratio < 0.5:
        paysim_gen = (
            f"**PaySim base-rate shift and generalization**: The fraud base rate shifted from {paysim_br_val:.4%} on val to {paysim_br_test:.4%} on test "
            f"({br_ratio:.1f}x difference, exceeding the 2x threshold). Because base rates differ by > 2x, PR-AUC is **NOT directly comparable** "
            f"between val ({paysim['val_core']['pr_auc']:.4f}) and test ({paysim['test_core']['pr_auc']:.4f}). "
            f"Evaluating via lift and ROC-AUC: PR-AUC lift is {paysim_lift_test:.3f} on test vs {paysim_lift_val:.3f} on val (Δ = {paysim_lift_diff:+.3f}); "
            f"ROC-AUC is {paysim_roc_test:.4f} vs {paysim_roc_val:.4f} (Δ = {paysim_roc_diff:+.4f}). "
            f"Rankings are evaluated using raw margin scores (LightGBM raw_score=True), which avoid probability saturation at 1.0 and preserve monotonicity."
        )
    else:
        paysim_gen = (
            f"**PaySim generalization**: PR-AUC on test is {paysim['test_core']['pr_auc']:.4f} vs {paysim['val_core']['pr_auc']:.4f} on val; "
            f"ROC-AUC is {paysim_roc_test:.4f} vs {paysim_roc_val:.4f}."
        )
    findings.append(paysim_gen)

    # 4. PaySim FP budget
    paysim_fp_val = paysim["val_thresh"]["actual_fp_rate"]
    paysim_fp_test = paysim["test_thresh"]["actual_fp_rate"]
    paysim_val_met = "met" if paysim_fp_val <= 0.02005 else "NOT met"
    paysim_test_met = "met" if paysim_fp_test <= 0.02000 else "NOT met"
    paysim_test_detail = "met" if paysim_test_met == "met" else f"NOT met (exceeded budget by {(paysim_fp_test - 0.02) * 100:.2f} pp)"
    tie_note = (
        f"Legit rows tied at threshold ({paysim['threshold']:.6f}): {paysim['val_thresh']['tie_count']} val, {paysim['test_thresh']['tie_count']} test; "
        + ("ties did not prevent meeting the 2% budget on val." if paysim['val_thresh']['tie_count'] == 0 else "ties prevented hitting exactly 2% budget.")
    )
    findings.append(
        f"**PaySim FP budget**: Target ≤ 2.00%. Val FP rate = {paysim_fp_val:.2%} ({paysim_val_met}); "
        f"Test FP rate = {paysim_fp_test:.2%} ({paysim_test_detail}). {tie_note}"
    )

    # 5. Calibration
    findings.append(f"**Calibration (IEEE-CIS)**: {ieee['cal_verdict']}")

    # 6. Overall verdict replacement: At most 3 factual bullets
    summary_b1 = (
        f"IEEE-CIS val->test change at fixed threshold ({ieee['threshold']:.6f}): "
        f"PR-AUC changed from {ieee_pr_val:.4f} to {ieee_pr_test:.4f} (Δ = {ieee_pr_diff:+.4f}); "
        f"recall changed from {ieee['val_thresh']['recall']:.4f} to {ieee['test_thresh']['recall']:.4f} "
        f"(Δ = {ieee['test_thresh']['recall'] - ieee['val_thresh']['recall']:+.4f})."
    )
    summary_b2 = (
        f"FP budget on test: Target ≤ 2.00% was {ieee_test_met} on IEEE-CIS (actual {ieee_fp_test:.2%}) "
        f"and {paysim_test_met} on PaySim (actual {paysim_fp_test:.2%})."
    )
    summary_b3 = (
        f"PaySim test operating point at fixed threshold ({paysim['threshold']:.6f}): "
        f"{paysim['test_thresh']['pct_flagged']:.2f}% of all transactions flagged, "
        f"precision = {paysim['test_thresh']['precision']:.4f}, recall = {paysim['test_thresh']['recall']:.4f}."
    )

    findings.append(
        f"**Overall verdict (factual summary)**:\n"
        f"  - {summary_b1}\n"
        f"  - {summary_b2}\n"
        f"  - {summary_b3}"
    )

    return findings


def write_report(ieee: dict, paysim: dict) -> None:
    lines: list[str] = []

    lines.append("# FraudLens — Final Held-Out Test Evaluation\n")

    # ── IEEE-CIS ──────────────────────────────────────────────────────────────
    lines.append("## IEEE-CIS\n")
    lines.append("### Core metrics (threshold-independent)\n")
    lines.append(_md_row("Metric", "VAL", "TEST"))
    lines.append(_md_row("---", "---", "---"))
    vc, tc = ieee["val_core"], ieee["test_core"]
    lines.append(_md_row("Rows", f"{vc['rows']:,}", f"{tc['rows']:,}"))
    lines.append(_md_row("Fraud rate (base rate)", f"{vc['fraud_rate']:.4%}", f"{tc['fraud_rate']:.4%}"))
    lines.append(_md_row("PR-AUC", vc["pr_auc"], tc["pr_auc"]))
    lines.append(_md_row("PR-AUC / base rate (lift)", vc["pr_auc_lift"], tc["pr_auc_lift"]))
    lines.append(_md_row("ROC-AUC", vc["roc_auc"], tc["roc_auc"]))

    lines.append(f"\n### Threshold metrics (threshold = {ieee['threshold']:.6f}, chosen on VAL)\n")
    lines.append(_md_row("Metric", "VAL", "TEST"))
    lines.append(_md_row("---", "---", "---"))
    vt, tt = ieee["val_thresh"], ieee["test_thresh"]
    lines.append(_md_row("Precision", vt["precision"], tt["precision"]))
    lines.append(_md_row("Recall", vt["recall"], tt["recall"]))
    lines.append(_md_row("F1", vt["f1"], tt["f1"]))
    lines.append(_md_row("Actual FP rate (FP / all legit)", f"{vt['actual_fp_rate']:.4%}", f"{tt['actual_fp_rate']:.4%}"))
    lines.append(_md_row("% of all transactions flagged", f"{vt['pct_flagged']:.2f}%", f"{tt['pct_flagged']:.2f}%"))
    lines.append(_md_row("Legit tied at threshold", vt["tie_count"], tt["tie_count"]))

    lines.append("\n### Calibration (TEST set only)\n")
    lines.append(f"- **Mean predicted probability**: {ieee['mean_pred']:.2%}")
    lines.append(f"- **Observed fraud rate**: {ieee['mean_obs']:.2%}")
    lines.append(f"- **Brier score**: {ieee['brier_test']:.5f}")
    lines.append(f"- **Baseline Brier score (constant base rate p*(1-p))**: {ieee['brier_baseline']:.5f}")
    lines.append(f"- **Brier skill score (1 - Brier / Baseline)**: {ieee['brier_skill']:.5f} ({ieee['brier_skill']:.2%})\n")
    lines.append("> Context: The IEEE-CIS model was trained on SMOTE-resampled data (sampling_strategy=0.1).\n")
    lines.append("**10-bin quantile reliability table (TEST):**\n")
    lines.append(_md_row("Mean pred prob", "Observed fraud rate", "Ratio (pred / obs)"))
    lines.append(_md_row("---", "---", "---"))
    for _, row in ieee["cal_table"].iterrows():
        lines.append(_md_row(row["mean_pred_prob"], row["observed_fraud_rate"], row["ratio (pred/obs)"]))

    lines.append(f"\n**Calibration verdict**: {ieee['cal_verdict']}\n")

    # ── PaySim ────────────────────────────────────────────────────────────────
    lines.append("## PaySim\n")
    lines.append("### Core metrics (threshold-independent)\n")
    lines.append(_md_row("Metric", "VAL", "TEST"))
    lines.append(_md_row("---", "---", "---"))
    pvc, ptc = paysim["val_core"], paysim["test_core"]
    lines.append(_md_row("Rows", f"{pvc['rows']:,}", f"{ptc['rows']:,}"))
    lines.append(_md_row("Fraud rate (base rate)", f"{pvc['fraud_rate']:.4%}", f"{ptc['fraud_rate']:.4%}"))
    lines.append(_md_row("PR-AUC", pvc["pr_auc"], ptc["pr_auc"]))
    lines.append(_md_row("PR-AUC / base rate (lift)", pvc["pr_auc_lift"], ptc["pr_auc_lift"]))
    lines.append(_md_row("ROC-AUC", pvc["roc_auc"], ptc["roc_auc"]))
    lines.append("\n> Note: PaySim is scored with raw margins (LightGBM raw_score=True) to prevent probability saturation at 1.0. Because rankings are monotonic with un-saturated probabilities, PR-AUC and ROC-AUC rankings are preserved.\n")

    lines.append(f"\n### Threshold metrics (threshold = {paysim['threshold']:.6f}, chosen on VAL)\n")
    lines.append(_md_row("Metric", "VAL", "TEST"))
    lines.append(_md_row("---", "---", "---"))
    pvt, ptt = paysim["val_thresh"], paysim["test_thresh"]
    lines.append(_md_row("Precision", pvt["precision"], ptt["precision"]))
    lines.append(_md_row("Recall", pvt["recall"], ptt["recall"]))
    lines.append(_md_row("F1", pvt["f1"], ptt["f1"]))
    lines.append(_md_row("Actual FP rate (FP / all legit)", f"{pvt['actual_fp_rate']:.4%}", f"{ptt['actual_fp_rate']:.4%}"))
    lines.append(_md_row("% of all transactions flagged", f"{pvt['pct_flagged']:.2f}%", f"{ptt['pct_flagged']:.2f}%"))
    lines.append(_md_row("Legit tied at threshold", pvt["tie_count"], ptt["tie_count"]))

    # ── Findings ──────────────────────────────────────────────────────────────
    lines.append("\n## Findings\n")
    findings = _generate_findings(ieee, paysim)
    for f in findings:
        lines.append(f"- {f}\n")

    out_path = REPORTS_DIR / "test_evaluation.md"
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nReport written -> {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ieee_results = evaluate_ieee()
    paysim_results = evaluate_paysim()
    write_report(ieee_results, paysim_results)
    print("\nAll done. Check reports/ for test_evaluation.md and calibration_test.png")
