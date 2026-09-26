"""
src/persist_model.py

Retrains the Phase 5 tuned model using the hyperparameters already found
by the RandomizedSearchCV run, and saves it to disk so the FastAPI service
(Phase 7) can load it without retraining every time the server starts.

Uses the exact best_params printed by your actual Phase 5 run — no need
to re-run the 10-minute hyperparameter search again.

Usage:
    C:\\...\\Python313\\python.exe src\\persist_model.py

Required packages (should already be installed from Phase 4/5):
    scikit-learn, lightgbm, imbalanced-learn, joblib, pandas, numpy
"""

import os
import json
import numpy as np
import pandas as pd
import joblib
import lightgbm as lgb

try:
    from imblearn.over_sampling import SMOTE
    HAS_SMOTE = True
except ImportError:
    HAS_SMOTE = False

BASE_DIR = os.path.abspath(os.path.join(os.getcwd(), ".."))
if not os.path.isdir(os.path.join(BASE_DIR, "data")):
    BASE_DIR = os.getcwd()
DATA_DIR = os.path.join(BASE_DIR, "data", "processed")
MODELS_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

# These are the exact best_params from your actual Phase 5 RandomizedSearchCV run.
# If you re-tune later and get different params, update this dict to match.
BEST_PARAMS = {
    "reg_lambda": 0.1,
    "reg_alpha": 0,
    "num_leaves": 127,
    "n_estimators": 300,
    "min_child_samples": 50,
    "max_depth": 12,
    "learning_rate": 0.05,
}


def _feature_columns(df, target_col, drop_cols):
    exclude = set(drop_cols) | {target_col}
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    return [c for c in numeric_cols if c not in exclude]


def main():
    print("Loading training data...")
    train = pd.read_parquet(os.path.join(DATA_DIR, "ieee_train.parquet"))

    target = "isFraud"
    drop_cols = ["TransactionID", "isFlaggedFraud"]
    feats = _feature_columns(train, target, drop_cols)
    X_train = train[feats].fillna(-999)
    y_train = train[target].values

    print(f"Training on {len(X_train):,} rows, {len(feats)} features, "
          f"with known-best hyperparameters...")

    if HAS_SMOTE:
        print("Applying SMOTE (as in Phase 5)...")
        sm = SMOTE(random_state=42, sampling_strategy=0.1, k_neighbors=3)
        X_res, y_res = sm.fit_resample(X_train, y_train)
        model = lgb.LGBMClassifier(**BEST_PARAMS, scale_pos_weight=1.0,
                                    random_state=42, n_jobs=-1, verbosity=-1)
        model.fit(X_res, y_res)
    else:
        imbalance_ratio = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
        model = lgb.LGBMClassifier(**BEST_PARAMS, scale_pos_weight=imbalance_ratio,
                                    random_state=42, n_jobs=-1, verbosity=-1)
        model.fit(X_train, y_train)

    model_path = os.path.join(MODELS_DIR, "ieee_lightgbm_tuned.joblib")
    joblib.dump(model, model_path)
    print(f"Saved model -> {model_path}")

    # Save the feature list and a default scoring threshold alongside the model,
    # so the API knows exactly what columns to expect and how to interpret scores.
    val = pd.read_parquet(os.path.join(DATA_DIR, "ieee_val.parquet"))
    X_val = val[feats].fillna(-999)
    y_val = val[target].values
    y_prob_val = model.predict_proba(X_val)[:, 1]
    legit_scores = y_prob_val[y_val == 0]
    threshold_2pct_fp = float(np.quantile(legit_scores, 0.98))

    metadata = {
        "feature_columns": feats,
        "threshold_2pct_fp_budget": threshold_2pct_fp,
        "model_type": "LightGBM (tuned, Phase 5 hyperparameters, SMOTE-trained)",
    }
    metadata_path = os.path.join(MODELS_DIR, "ieee_lightgbm_tuned_metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved metadata (feature list + threshold) -> {metadata_path}")
    print(f"\nThreshold for a 2% false-positive budget: {threshold_2pct_fp:.4f}")
    print("Done. The API (Phase 7) will load these files at startup.")


if __name__ == "__main__":
    main()