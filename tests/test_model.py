"""
tests/test_model.py
===================
Tests for model loading, scoring, and SHAP integration.

These tests load the REAL joblib model from models/ (no synthetic replacement)
and run on a tiny synthetic feature vector, so they require the model file to
be present but NOT the full data parquet files.

Skipped automatically in CI if the model file is absent (GitHub Actions does
not have the model committed to the repo — it lives in models/ which is
gitignored; CI tests only the agent/drift logic with synthetic data).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

_REPO = Path(__file__).resolve().parents[1]
_MODEL_PATH = _REPO / "models" / "ieee_lightgbm_tuned.joblib"
_META_PATH  = _REPO / "models" / "ieee_lightgbm_tuned_metadata.json"

MODEL_AVAILABLE = _MODEL_PATH.exists() and _META_PATH.exists()
skip_if_no_model = pytest.mark.skipif(
    not MODEL_AVAILABLE,
    reason="Model artifacts not present (expected in local env only, not CI)"
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def model_and_meta():
    """Load the real joblib model + metadata once for all tests in this module."""
    import joblib
    model = joblib.load(_MODEL_PATH)
    with open(_META_PATH) as f:
        meta = json.load(f)
    return model, meta


@pytest.fixture(scope="module")
def synthetic_row(model_and_meta):
    """One-row DataFrame with all feature columns set to safe fill value (-999)."""
    import pandas as pd
    _, meta = model_and_meta
    cols = meta["feature_columns"]
    return pd.DataFrame({c: [-999.0] for c in cols})


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

class TestModelLoading:
    @skip_if_no_model
    def test_model_file_exists(self):
        assert _MODEL_PATH.exists()

    @skip_if_no_model
    def test_metadata_file_exists(self):
        assert _META_PATH.exists()

    @skip_if_no_model
    def test_model_has_predict_proba(self, model_and_meta):
        model, _ = model_and_meta
        assert hasattr(model, "predict_proba"), "Model must expose predict_proba"

    @skip_if_no_model
    def test_metadata_has_required_keys(self, model_and_meta):
        _, meta = model_and_meta
        for key in ["model_type", "feature_columns", "threshold_2pct_fp_budget"]:
            assert key in meta, f"Missing metadata key: {key}"

    @skip_if_no_model
    def test_feature_count(self, model_and_meta):
        _, meta = model_and_meta
        assert len(meta["feature_columns"]) == 545, (
            f"Expected 545 features, got {len(meta['feature_columns'])}"
        )

    @skip_if_no_model
    def test_threshold_in_reasonable_range(self, model_and_meta):
        _, meta = model_and_meta
        t = meta["threshold_2pct_fp_budget"]
        assert 0.0 < t < 1.0, f"Threshold {t} is outside (0, 1)"


# ---------------------------------------------------------------------------
# Prediction / scoring
# ---------------------------------------------------------------------------

class TestModelPrediction:
    @skip_if_no_model
    def test_predict_proba_shape(self, model_and_meta, synthetic_row):
        model, _ = model_and_meta
        proba = model.predict_proba(synthetic_row)
        assert proba.shape == (1, 2), f"Expected (1,2), got {proba.shape}"

    @skip_if_no_model
    def test_predict_proba_sums_to_one(self, model_and_meta, synthetic_row):
        model, _ = model_and_meta
        proba = model.predict_proba(synthetic_row)
        assert abs(proba.sum(axis=1)[0] - 1.0) < 1e-6

    @skip_if_no_model
    def test_fraud_probability_in_range(self, model_and_meta, synthetic_row):
        model, _ = model_and_meta
        fraud_prob = float(model.predict_proba(synthetic_row)[:, 1][0])
        assert 0.0 <= fraud_prob <= 1.0

    @skip_if_no_model
    def test_scoring_deterministic(self, model_and_meta, synthetic_row):
        """Same input must produce same output (model is deterministic)."""
        model, _ = model_and_meta
        p1 = float(model.predict_proba(synthetic_row)[:, 1][0])
        p2 = float(model.predict_proba(synthetic_row)[:, 1][0])
        assert p1 == p2

    @skip_if_no_model
    def test_fillna_minus999_does_not_crash(self, model_and_meta):
        """The -999 NaN fill used in api/main.py must not crash the model."""
        import pandas as pd
        model, meta = model_and_meta
        cols = meta["feature_columns"]
        row_with_nans = pd.DataFrame({c: [float("nan")] for c in cols})
        filled = row_with_nans.fillna(-999)
        proba = model.predict_proba(filled)
        assert proba.shape == (1, 2)


# ---------------------------------------------------------------------------
# SHAP integration
# ---------------------------------------------------------------------------

class TestShapIntegration:
    @skip_if_no_model
    def test_tree_explainer_loads(self, model_and_meta):
        import shap
        model, _ = model_and_meta
        explainer = shap.TreeExplainer(model)
        assert explainer is not None

    @skip_if_no_model
    def test_shap_values_shape(self, model_and_meta, synthetic_row):
        import shap
        model, _ = model_and_meta
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(synthetic_row)
        # LightGBM may return list or array
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        assert shap_values.shape[1] == synthetic_row.shape[1]

    @skip_if_no_model
    def test_shap_top5_indexing(self, model_and_meta, synthetic_row):
        """Verify the top-5 index selection used in api/main.py doesn't crash."""
        import shap
        model, meta = model_and_meta
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(synthetic_row)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        contributions = shap_values[0]
        top_idx = np.argsort(np.abs(contributions))[::-1][:5]
        feats = meta["feature_columns"]
        results = [
            {"feature": feats[i], "shap_contribution": round(float(contributions[i]), 5)}
            for i in top_idx
        ]
        assert len(results) == 5
        for r in results:
            assert "feature" in r and "shap_contribution" in r

