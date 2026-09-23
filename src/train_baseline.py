"""
src/train_baseline.py

Phase 4 — Baseline modeling for FraudLens.
Trains and honestly compares baseline models for IEEE-CIS and PaySim,
with every run logged to MLflow.

Usage (from a Jupyter cell in your existing 3.13 kernel, so it shares
the same environment that already has pandas/pyarrow installed):

    import sys
    sys.path.insert(0, "..")   # if running from notebooks/ folder
    from src.train_baseline import run_ieee_pipeline, run_paysim_pipeline

    run_ieee_pipeline()
    run_paysim_pipeline()

Required packages (install first, using the sys.executable trick to
target the correct kernel, same as we did for pyarrow):

    import sys
    !{sys.executable} -m pip install mlflow lightgbm imbalanced-learn scikit-learn matplotlib
"""

import os
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import mlflow
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    average_precision_score, roc_auc_score, f1_score,
    precision_score, recall_score, precision_recall_curve
)
from sklearn.calibration import calibration_curve

try:
    import lightgbm as lgb
    GBM_LIB = "lightgbm"
except ImportError:
    import xgboost as xgb
    GBM_LIB = "xgboost"

try:
    from imblearn.over_sampling import SMOTE
    HAS_SMOTE = True
except ImportError:
    HAS_SMOTE = False
    print("WARNING: imbalanced-learn not installed — SMOTE comparison run will be skipped.")

BASE_DIR = os.path.abspath(os.path.join(os.getcwd(), ".."))
if not os.path.isdir(os.path.join(BASE_DIR, "data")):
    BASE_DIR = os.getcwd()  # fallback if already at repo root
DATA_DIR = os.path.join(BASE_DIR, "data", "processed")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
os.makedirs(REPORTS_DIR, exist_ok=True)

mlflow.set_tracking_uri(f"sqlite:///{os.path.join(BASE_DIR, 'mlflow.db')}")


# ---------------------------------------------------------------- helpers

def _feature_columns(df, target_col, drop_cols):
    """Auto-select numeric feature columns, excluding target/id/known-drop columns."""
    exclude = set(drop_cols) | {target_col}
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    feats = [c for c in numeric_cols if c not in exclude]
    dropped_non_numeric = [c for c in df.columns if c not in numeric_cols and c not in exclude]
    if dropped_non_numeric:
        print(f"  Note: dropped {len(dropped_non_numeric)} non-numeric columns not used as features: "
              f"{dropped_non_numeric[:10]}{'...' if len(dropped_non_numeric) > 10 else ''}")
    return feats


def _metrics_at_fp_budget(y_true, y_prob, fp_budget=0.02):
    """Pick a threshold such that at most fp_budget fraction of legit transactions are flagged,
    then report precision/recall/F1 at that threshold. Returns dict + threshold used."""
    legit_scores = y_prob[y_true == 0]
    threshold = np.quantile(legit_scores, 1 - fp_budget)
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "threshold_at_fp_budget": float(threshold),
        "precision_at_fp_budget": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall_at_fp_budget": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_at_fp_budget": float(f1_score(y_true, y_pred, zero_division=0)),
    }, y_pred


def _core_metrics(y_true, y_prob):
    return {
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
    }


def _plot_calibration(y_true, y_prob, title, out_path):
    frac_pos, mean_pred = calibration_curve(y_true, y_prob, n_bins=10, strategy="quantile")
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(mean_pred, frac_pos, marker="o", label="Model")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfectly calibrated")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of actual positives")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def _log_run(run_name, experiment, params, metrics, artifact_paths=None):
    mlflow.set_experiment(experiment)
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(params)
        mlflow.log_metrics({k: v for k, v in metrics.items() if isinstance(v, (int, float))})
        if artifact_paths:
            for p in artifact_paths:
                mlflow.log_artifact(p)
    print(f"  Logged MLflow run: {run_name}  |  PR-AUC={metrics.get('pr_auc'):.4f}  "
          f"F1@budget={metrics.get('f1_at_fp_budget'):.4f}")


# ---------------------------------------------------------------- IEEE-CIS

def run_ieee_pipeline(fp_budget=0.02):
    print("=" * 70)
    print("IEEE-CIS — Baseline Modeling")
    print("=" * 70)

    train = pd.read_parquet(os.path.join(DATA_DIR, "ieee_train.parquet"))
    val = pd.read_parquet(os.path.join(DATA_DIR, "ieee_val.parquet"))

    target = "isFraud"
    drop_cols = ["TransactionID", "isFlaggedFraud"]
    feats = _feature_columns(train, target, drop_cols)
    print(f"Using {len(feats)} numeric features.")

    X_train, y_train = train[feats].fillna(-999), train[target].values
    X_val, y_val = val[feats].fillna(-999), val[target].values

    imbalance_ratio = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
    print(f"Computed class imbalance ratio (legit:fraud) = {imbalance_ratio:.2f} : 1")

    results = {}

    # --- (a) Logistic Regression baseline
    # Note: LR on 413k×545 is very slow; we use a stratified 50k-row subsample.
    # This is a rough baseline, not the primary model — GBM is the real comparison.
    print("\n[1/4] Logistic Regression baseline (stratified 50k subsample)...")
    t0 = time.time()
    rng_lr = np.random.default_rng(0)
    n_sub = min(50000, len(X_train))
    fraud_rows = np.where(y_train == 1)[0]
    legit_rows = np.where(y_train == 0)[0]
    # Stratified: keep all fraud + enough legit to reach n_sub
    n_legit_sub = min(n_sub - len(fraud_rows), len(legit_rows))
    legit_sub = rng_lr.choice(legit_rows, size=n_legit_sub, replace=False)
    sub_idx = np.concatenate([fraud_rows, legit_sub])
    X_lr, y_lr = X_train.iloc[sub_idx], y_train[sub_idx]
    lr = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            class_weight="balanced", max_iter=400,
            solver="lbfgs", random_state=42,
        )),
    ])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        lr.fit(X_lr, y_lr)
    y_prob = lr.predict_proba(X_val)[:, 1]
    m = _core_metrics(y_val, y_prob)
    fp_m, _ = _metrics_at_fp_budget(y_val, y_prob, fp_budget)
    m.update(fp_m)
    _log_run("ieee_logreg_baseline", "FraudLens_IEEE-CIS",
              {"model": "LogisticRegression", "class_weight": "balanced"}, m)
    results["LogisticRegression"] = m
    print(f"  ({time.time()-t0:.1f}s)")

    # --- (b) GBM with scale_pos_weight
    print(f"\n[2/4] {GBM_LIB} with scale_pos_weight={imbalance_ratio:.2f}...")
    t0 = time.time()
    gbm_spw = _train_gbm(X_train, y_train, imbalance_ratio)
    y_prob = _gbm_predict_proba(gbm_spw, X_val)
    m = _core_metrics(y_val, y_prob)
    fp_m, _ = _metrics_at_fp_budget(y_val, y_prob, fp_budget)
    m.update(fp_m)
    cal_path = _plot_calibration(y_val, y_prob, f"{GBM_LIB} (scale_pos_weight) — IEEE-CIS",
                                  os.path.join(REPORTS_DIR, "ieee_gbm_spw_calibration.png"))
    _log_run(f"ieee_{GBM_LIB}_scale_pos_weight", "FraudLens_IEEE-CIS",
              {"model": GBM_LIB, "scale_pos_weight": imbalance_ratio}, m, [cal_path])
    results[f"{GBM_LIB}_scale_pos_weight"] = m
    print(f"  ({time.time()-t0:.1f}s)")

    # --- (c) GBM + SMOTE comparison
    if HAS_SMOTE:
        print("\n[3/4] SMOTE + GBM comparison...")
        print("  Note: using sampling_strategy=0.1 + k_neighbors=3 + n_jobs=-1 for"
              " tractability on high-dimensional (545-feat) data.")
        t0 = time.time()
        # Pre-subsample minority class for kNN fitting — SMOTE only needs local neighborhood
        # structure; capping at 5000 minority points makes fit() fast while still generating
        # valid synthetic samples from representative regions of the fraud manifold.
        fraud_idx = np.where(y_train == 1)[0]
        if len(fraud_idx) > 5000:
            rng = np.random.default_rng(42)
            knn_idx = rng.choice(fraud_idx, size=5000, replace=False)
            legit_idx = np.where(y_train == 0)[0]
            fit_idx = np.concatenate([legit_idx, knn_idx])
            X_fit, y_fit = X_train.iloc[fit_idx], y_train[fit_idx]
        else:
            X_fit, y_fit = X_train, y_train
        from sklearn.neighbors import NearestNeighbors as _NN
        sm = SMOTE(random_state=42, sampling_strategy=0.1,
                   k_neighbors=_NN(n_neighbors=3, n_jobs=-1))
        X_res, y_res = sm.fit_resample(X_fit, y_fit)
        gbm_smote = _train_gbm(X_res, y_res, scale_pos_weight=1.0)  # already balanced by SMOTE
        y_prob = _gbm_predict_proba(gbm_smote, X_val)
        m = _core_metrics(y_val, y_prob)
        fp_m, _ = _metrics_at_fp_budget(y_val, y_prob, fp_budget)
        m.update(fp_m)
        _log_run(f"ieee_{GBM_LIB}_smote", "FraudLens_IEEE-CIS",
                  {"model": GBM_LIB, "resampling": "SMOTE"}, m)
        results[f"{GBM_LIB}_smote"] = m
        print(f"  ({time.time()-t0:.1f}s)")
    else:
        print("\n[3/4] Skipped SMOTE run (imbalanced-learn not installed).")

    print("\n[4/4] Comparison table:")
    _print_comparison(results)
    return results


# ---------------------------------------------------------------- PaySim

def run_paysim_pipeline(fp_budget=0.02):
    print("=" * 70)
    print("PaySim — Baseline Modeling (extreme ~769:1 imbalance)")
    print("=" * 70)

    train = pd.read_parquet(os.path.join(DATA_DIR, "paysim_train.parquet"))
    val = pd.read_parquet(os.path.join(DATA_DIR, "paysim_val.parquet"))

    target = "isFraud"
    drop_cols = ["isFlaggedFraud"]  # per EDA: nearly useless, explicitly excluded
    feats = _feature_columns(train, target, drop_cols)
    print(f"Using {len(feats)} numeric features.")

    X_train, y_train = train[feats].fillna(-999), train[target].values
    X_val, y_val = val[feats].fillna(-999), val[target].values

    imbalance_ratio = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
    print(f"Computed class imbalance ratio (legit:fraud) = {imbalance_ratio:.2f} : 1")

    results = {}

    # --- (a) Supervised GBM comparison point
    print(f"\n[1/2] {GBM_LIB} with scale_pos_weight={imbalance_ratio:.2f} (comparison point)...")
    t0 = time.time()
    gbm = _train_gbm(X_train, y_train, imbalance_ratio)
    y_prob = _gbm_predict_proba(gbm, X_val)
    m = _core_metrics(y_val, y_prob)
    fp_m, _ = _metrics_at_fp_budget(y_val, y_prob, fp_budget)
    m.update(fp_m)
    _log_run(f"paysim_{GBM_LIB}_scale_pos_weight", "FraudLens_PaySim",
              {"model": GBM_LIB, "scale_pos_weight": imbalance_ratio}, m)
    results[f"{GBM_LIB}_supervised"] = m
    print(f"  ({time.time()-t0:.1f}s)")

    # --- (b) Isolation Forest (unsupervised anomaly detection)
    print("\n[2/2] Isolation Forest (unsupervised)...")
    t0 = time.time()
    iso = IsolationForest(n_estimators=200, contamination=float(y_train.mean()), random_state=42, n_jobs=-1)
    iso.fit(X_train)
    anomaly_score = -iso.score_samples(X_val)  # higher = more anomalous = more "fraud-like"
    m = _core_metrics(y_val, anomaly_score)
    fp_m, _ = _metrics_at_fp_budget(y_val, anomaly_score, fp_budget)
    m.update(fp_m)
    _log_run("paysim_isolation_forest", "FraudLens_PaySim",
              {"model": "IsolationForest", "n_estimators": 200}, m)
    results["IsolationForest_unsupervised"] = m
    print(f"  ({time.time()-t0:.1f}s)")

    print("\nComparison table:")
    _print_comparison(results)
    return results


# ---------------------------------------------------------------- gbm wrappers

def _train_gbm(X, y, scale_pos_weight):
    if GBM_LIB == "lightgbm":
        model = lgb.LGBMClassifier(
            n_estimators=300, learning_rate=0.05, num_leaves=63,
            scale_pos_weight=scale_pos_weight, random_state=42, n_jobs=-1, verbosity=-1,
        )
    else:
        model = xgb.XGBClassifier(
            n_estimators=300, learning_rate=0.05, max_depth=6,
            scale_pos_weight=scale_pos_weight, random_state=42, n_jobs=-1,
            use_label_encoder=False, eval_metric="logloss",
        )
    model.fit(X, y)
    return model


def _gbm_predict_proba(model, X):
    return model.predict_proba(X)[:, 1]


def _print_comparison(results):
    df = pd.DataFrame(results).T[["pr_auc", "roc_auc", "precision_at_fp_budget",
                                   "recall_at_fp_budget", "f1_at_fp_budget"]]
    df.columns = ["PR-AUC", "ROC-AUC", "Precision@budget", "Recall@budget", "F1@budget"]
    print(df.round(4).to_string())


if __name__ == "__main__":
    run_ieee_pipeline()
    print()
    run_paysim_pipeline()