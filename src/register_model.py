"""
src/register_model.py

Phase 9 — MLflow Model Registry for FraudLens.

Registers the persisted, tuned IEEE-CIS model (from src/persist_model.py)
as a named, versioned model in MLflow's Model Registry, and promotes it to
the "Production" stage — the actual MLOps practice of having a named,
queryable "this is the model currently serving" pointer, not just a pile
of experiment runs.

Usage:
    C:\\...\\Python313\\python.exe src\\register_model.py

Required packages: mlflow, lightgbm, joblib (all should already be
installed from Phases 4/5/7).
"""

import os
import json
import joblib
import mlflow
from mlflow.tracking import MlflowClient

BASE_DIR = os.path.abspath(os.path.join(os.getcwd(), ".."))
if not os.path.isdir(os.path.join(BASE_DIR, "data")):
    BASE_DIR = os.getcwd()
MODELS_DIR = os.path.join(BASE_DIR, "models")
REGISTERED_MODEL_NAME = "fraudlens-ieee-lightgbm"

mlflow.set_tracking_uri(f"sqlite:///{os.path.join(BASE_DIR, 'mlflow.db')}")


def main():
    model_path = os.path.join(MODELS_DIR, "ieee_lightgbm_tuned.joblib")
    metadata_path = os.path.join(MODELS_DIR, "ieee_lightgbm_tuned_metadata.json")

    print(f"Loading model from {model_path} ...")
    model = joblib.load(model_path)
    with open(metadata_path) as f:
        metadata = json.load(f)

    mlflow.set_experiment("FraudLens_IEEE-CIS")
    with mlflow.start_run(run_name="register_tuned_model") as run:
        mlflow.log_param("threshold_2pct_fp_budget", metadata["threshold_2pct_fp_budget"])
        mlflow.log_param("n_features", len(metadata["feature_columns"]))
        mlflow.log_param("model_type", metadata["model_type"])

        # Note: log_model's `name=` parameter replaces the older, now-deprecated
        # `artifact_path=` — using the current API so this doesn't break on a
        # future mlflow upgrade.
        mlflow.lightgbm.log_model(
            model,
            name="model",
            registered_model_name=REGISTERED_MODEL_NAME,
        )
        run_id = run.info.run_id

    print(f"Logged and registered as '{REGISTERED_MODEL_NAME}' (run_id={run_id})")

    client = MlflowClient()
    versions = client.search_model_versions(f"name='{REGISTERED_MODEL_NAME}'")
    latest_version = max(int(v.version) for v in versions)
    print(f"Latest registered version: {latest_version}")

    # Use the modern alias API ("production" pointer), not the deprecated
    # stages API (Production/Staging/Archived) — mlflow has been warning
    # since 2.9.0 that stages will be removed, and this project is already
    # on a version past that deprecation.
    client.set_registered_model_alias(REGISTERED_MODEL_NAME, "production", latest_version)
    print(f"Set alias 'production' -> version {latest_version}.")

    client.update_model_version(
        name=REGISTERED_MODEL_NAME,
        version=latest_version,
        description=(
            f"Tuned LightGBM, {metadata['model_type']}. "
            f"{len(metadata['feature_columns'])} features. "
            f"Threshold @ 2% FP budget: {metadata['threshold_2pct_fp_budget']:.4f}. "
            f"PR-AUC 0.598, F1@budget 0.541 on held-out validation (Phase 5)."
        ),
    )

    print("\nDone. View the registry with:")
    print(f"  C:\\...\\Python313\\python.exe -m mlflow ui --backend-store-uri "
          f"sqlite:///{os.path.join(BASE_DIR, 'mlflow.db')}")
    print("  Then open http://127.0.0.1:5000 and check the 'Models' tab.")


if __name__ == "__main__":
    main()