"""
src/train_advanced.py

Phase 5 — Advanced modeling for FraudLens (IEEE-CIS focus, since it's the
dataset where the winning model was LightGBM + SMOTE).

Does three things, in order:
1. Hyperparameter tuning (RandomizedSearchCV, PR-AUC scoring, on a subsample
   for tractability, then retrains the best config on the FULL training set).
2. SHAP explainability on the tuned model (summary plot + top-20 features).
3. Entity-graph feature evaluation: explicitly checks where the entity-graph
   features (card1_addr1_count, card1_addr1_email_count, amt_mean, amt_std)
   rank among all 576 features, to honestly answer "are they pulling weight?"

IMPORTANT: this script only touches ieee_train.parquet and ieee_val.parquet.
It deliberately does NOT touch ieee_test.parquet — the test set stays held
out for a single final evaluation later (Phase 11), so we don't overfit our
modeling decisions to it.

Usage (from a Jupyter cell, or as a script):
    C:\\...\\Python313\\python.exe src\\train_advanced.py

Required packages (install with the full Python 3.13 path, same as before):
    C:\\...\\Python313\\python.exe -m pip install shap
(lightgbm, scikit-learn, mlflow should already be installed from Phase 4)
"""

import os
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import mlflow
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import average_precision_score, make_scorer

try:
    import lightgbm as lgb
except ImportError:
    raise SystemExit("lightgbm not found — install it first (see Phase 4 setup).")

try:
    from imblearn.over_sampling import SMOTE
    HAS_SMOTE = True
except ImportError:
    HAS_SMOTE = False
    print("WARNING: imbalanced-learn not installed — proceeding WITHOUT SMOTE "
          "(will use scale_pos_weight only, which is a valid fallback).")

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False
    print("WARNING: shap not installed. Install it (see module docstring) "
          "before running — explainability is the core point of this phase.")

BASE_DIR = os.path.abspath(os.path.join(os.getcwd(), ".."))
if not os.path.isdir(os.path.join(BASE_DIR, "data")):
    BASE_DIR = os.getcwd()
DATA_DIR = os.path.join(BASE_DIR, "data", "processed")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
os.makedirs(REPORTS_DIR, exist_ok=True)

mlflow.set_tracking_uri(f"sqlite:///{os.path.join(BASE_DIR, 'mlflow.db')}")

# The entity-graph features we specifically want to evaluate (from Phase 3)
ENTITY_GRAPH_FEATURE_HINTS = ["card1_addr1_count", "card1_addr1_email_count",
                               "amt_mean", "amt_std"]


def _feature_columns(df, target_col, drop_cols):
    exclude = set(drop_cols) | {target_col}
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    return [c for c in numeric_cols if c not in exclude]


def _load_ieee_train_val():
    train = pd.read_parquet(os.path.join(DATA_DIR, "ieee_train.parquet"))
    val = pd.read_parquet(os.path.join(DATA_DIR, "ieee_val.parquet"))
    target = "isFraud"
    drop_cols = ["TransactionID", "isFlaggedFraud"]
    feats = _feature_columns(train, target, drop_cols)
    X_train, y_train = train[feats].fillna(-999), train[target].values
    X_val, y_val = val[feats].fillna(-999), val[target].values
    return X_train, y_train, X_val, y_val, feats


# ---------------------------------------------------------------- step 1

def tune_hyperparameters(X_train, y_train, subsample_size=100_000, n_iter=15, cv=3):
    """RandomizedSearchCV on a subsample for tractability, scored on PR-AUC."""
    print("=" * 70)
    print("STEP 1: Hyperparameter tuning (RandomizedSearchCV, PR-AUC scoring)")
    print("=" * 70)

    if len(X_train) > subsample_size:
        rng = np.random.RandomState(42)
        idx = rng.choice(len(X_train), size=subsample_size, replace=False)
        X_sub, y_sub = X_train.iloc[idx], y_train[idx]
        print(f"Using a {subsample_size:,}-row stratified-ish subsample for search "
              f"(full data used later for the final fit).")
    else:
        X_sub, y_sub = X_train, y_train

    imbalance_ratio = (y_sub == 0).sum() / max((y_sub == 1).sum(), 1)

    param_dist = {
        "num_leaves": [31, 63, 127],
        "learning_rate": [0.01, 0.03, 0.05, 0.1],
        "min_child_samples": [10, 20, 50, 100],
        "max_depth": [-1, 6, 8, 12],
        "n_estimators": [200, 300, 500],
        "reg_alpha": [0, 0.1, 1.0],
        "reg_lambda": [0, 0.1, 1.0],
    }

    base_model = lgb.LGBMClassifier(
        scale_pos_weight=imbalance_ratio, random_state=42, n_jobs=-1, verbosity=-1
    )

    pr_auc_scorer = "average_precision"  # built-in sklearn scorer, avoids make_scorer API differences across versions
    cv_splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)

    search = RandomizedSearchCV(
        base_model, param_distributions=param_dist, n_iter=n_iter,
        scoring=pr_auc_scorer, cv=cv_splitter, random_state=42, n_jobs=-1, verbose=1,
    )

    t0 = time.time()
    search.fit(X_sub, y_sub)
    print(f"Search complete in {time.time()-t0:.1f}s")
    print(f"Best CV PR-AUC: {search.best_score_:.4f}")
    print(f"Best params: {search.best_params_}")

    return search.best_params_, imbalance_ratio


def train_final_model(X_train, y_train, best_params, use_smote=True):
    print("\n" + "=" * 70)
    print("Training final model on FULL training set with tuned hyperparameters")
    print("=" * 70)

    if use_smote and HAS_SMOTE:
        print("Applying SMOTE on full training set (this may take a couple minutes "
              "given 576 features)...")
        t0 = time.time()
        sm = SMOTE(random_state=42, sampling_strategy=0.1, k_neighbors=3)
        X_res, y_res = sm.fit_resample(X_train, y_train)
        print(f"  SMOTE done in {time.time()-t0:.1f}s. Resampled shape: {X_res.shape}")
        model = lgb.LGBMClassifier(**best_params, scale_pos_weight=1.0,
                                    random_state=42, n_jobs=-1, verbosity=-1)
        model.fit(X_res, y_res)
    else:
        imbalance_ratio = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
        model = lgb.LGBMClassifier(**best_params, scale_pos_weight=imbalance_ratio,
                                    random_state=42, n_jobs=-1, verbosity=-1)
        model.fit(X_train, y_train)

    return model


def evaluate_and_log(model, X_val, y_val, best_params, fp_budget=0.02):
    y_prob = model.predict_proba(X_val)[:, 1]
    pr_auc = average_precision_score(y_val, y_prob)

    legit_scores = y_prob[y_val == 0]
    threshold = np.quantile(legit_scores, 1 - fp_budget)
    y_pred = (y_prob >= threshold).astype(int)
    from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
    metrics = {
        "pr_auc": float(pr_auc),
        "roc_auc": float(roc_auc_score(y_val, y_prob)),
        "precision_at_fp_budget": float(precision_score(y_val, y_pred, zero_division=0)),
        "recall_at_fp_budget": float(recall_score(y_val, y_pred, zero_division=0)),
        "f1_at_fp_budget": float(f1_score(y_val, y_pred, zero_division=0)),
    }

    print("\nTuned model — validation metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")
    print("\n(Compare to Phase 4 baseline: LightGBM+SMOTE scored PR-AUC=0.5568, "
          "F1@budget=0.5145 — this tells you whether tuning actually helped.)")

    mlflow.set_experiment("FraudLens_IEEE-CIS")
    with mlflow.start_run(run_name="ieee_lightgbm_tuned"):
        mlflow.log_params(best_params)
        mlflow.log_metrics(metrics)
    print("Logged to MLflow as 'ieee_lightgbm_tuned'.")

    return metrics, y_prob


# ---------------------------------------------------------------- step 2

def run_shap_analysis(model, X_val, feats, sample_size=5000):
    print("\n" + "=" * 70)
    print("STEP 2: SHAP explainability")
    print("=" * 70)

    if not HAS_SHAP:
        print("Skipping — shap not installed.")
        return None

    if len(X_val) > sample_size:
        X_sample = X_val.sample(n=sample_size, random_state=42)
    else:
        X_sample = X_val

    print(f"Computing SHAP values on a {len(X_sample):,}-row sample "
          f"(full val set would be slow with {len(feats)} features)...")
    t0 = time.time()
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)
    # LightGBM binary classifier: shap_values may be a list [class0, class1] or a single array
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    print(f"  Done in {time.time()-t0:.1f}s")

    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    importance_df = pd.DataFrame({
        "feature": feats,
        "mean_abs_shap": mean_abs_shap
    }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    importance_df["rank"] = importance_df.index + 1

    print("\nTop 20 features by mean |SHAP value|:")
    print(importance_df.head(20).to_string(index=False))

    # Summary plot
    fig = plt.figure(figsize=(9, 7))
    shap.summary_plot(shap_values, X_sample, feature_names=feats, show=False, max_display=20)
    plt.tight_layout()
    out_path = os.path.join(REPORTS_DIR, "ieee_shap_summary.png")
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved SHAP summary plot -> {out_path}")

    importance_csv = os.path.join(REPORTS_DIR, "ieee_shap_feature_importance.csv")
    importance_df.to_csv(importance_csv, index=False)
    print(f"Saved full feature importance ranking -> {importance_csv}")

    return importance_df


# ---------------------------------------------------------------- step 3

def evaluate_entity_graph_features(importance_df):
    print("\n" + "=" * 70)
    print("STEP 3: Entity-graph feature evaluation")
    print("=" * 70)

    if importance_df is None:
        print("Skipped — SHAP importance not available (shap not installed).")
        return

    total_features = len(importance_df)
    print(f"Checking rank of entity-graph features among {total_features} total features:\n")

    found_any = False
    for hint in ENTITY_GRAPH_FEATURE_HINTS:
        matches = importance_df[importance_df["feature"].str.contains(hint, case=False, na=False)]
        if len(matches) == 0:
            print(f"  '{hint}': NOT FOUND in feature set (check naming in src/features.py)")
            continue
        found_any = True
        for _, row in matches.iterrows():
            pct = 100 * (1 - row["rank"] / total_features)
            print(f"  '{row['feature']}': rank {row['rank']}/{total_features} "
                  f"(top {100-pct:.1f}% of features, mean|SHAP|={row['mean_abs_shap']:.5f})")

    if not found_any:
        print("\nNone of the expected entity-graph feature names were found — "
              "check src/features.py for the actual column names used and update "
              "ENTITY_GRAPH_FEATURE_HINTS at the top of this script.")
    else:
        print("\nHonest read: if these features rank in roughly the top third or "
              "better, they're pulling real weight. If they rank near the bottom, "
              "that's worth reporting honestly too — not every engineered feature "
              "has to be a winner, and saying so is more credible than pretending "
              "otherwise.")


# ---------------------------------------------------------------- main

def run_phase5_pipeline():
    X_train, y_train, X_val, y_val, feats = _load_ieee_train_val()
    print(f"Loaded IEEE-CIS: {X_train.shape[0]:,} train rows, {X_val.shape[0]:,} val rows, "
          f"{len(feats)} features.\n")

    best_params, imbalance_ratio = tune_hyperparameters(X_train, y_train)
    model = train_final_model(X_train, y_train, best_params, use_smote=True)
    metrics, y_prob = evaluate_and_log(model, X_val, y_val, best_params)
    importance_df = run_shap_analysis(model, X_val, feats)
    evaluate_entity_graph_features(importance_df)

    print("\n" + "=" * 70)
    print("Phase 5 complete.")
    print("=" * 70)
    return model, metrics, importance_df


if __name__ == "__main__":
    run_phase5_pipeline()