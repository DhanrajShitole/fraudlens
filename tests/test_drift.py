"""
tests/test_drift.py
===================
Tests for src/drift_check.psi().
"""

import numpy as np
import pytest
import sys
from pathlib import Path

# Ensure src/ is importable.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from drift_check import psi


class TestPsi:
    def test_identical_distributions_stable(self):
        """Identical reference and comparison -> PSI should be < 0.1."""
        rng = np.random.default_rng(42)
        arr = rng.normal(0, 1, 1000)
        result = psi(arr, arr.copy())
        assert result < 0.1, f"Expected PSI < 0.1 for identical distributions, got {result}"

    def test_large_mean_shift_significant(self):
        """Large mean shift (reference N(0,1), comparison N(5,1)) -> PSI > 0.25."""
        rng = np.random.default_rng(0)
        reference = rng.normal(0, 1, 2000)
        comparison = rng.normal(5, 1, 2000)
        result = psi(reference, comparison)
        assert result > 0.25, f"Expected PSI > 0.25 for large mean shift, got {result}"

    def test_near_constant_column_does_not_crash(self):
        """Near-constant column (all same value except tiny jitter) -> returns 0.0 or nan, no crash."""
        reference = np.ones(500)
        comparison = np.ones(500) * 1.0001   # very slightly different
        result = psi(reference, comparison)
        assert result is not None
        assert isinstance(result, float)

    def test_nans_do_not_crash(self):
        """Arrays with NaN values -> psi() strips them and returns a float without crashing."""
        rng = np.random.default_rng(7)
        arr = rng.normal(0, 1, 1000).astype(float)
        arr[::10] = np.nan   # 10% NaN
        result = psi(arr, arr.copy())
        assert isinstance(result, float)
        # After stripping NaN, identical remaining values -> stable
        assert result < 0.1

    def test_all_nan_reference_returns_nan(self):
        """All-NaN reference -> result is nan (no crash)."""
        reference = np.full(100, np.nan)
        comparison = np.arange(100, dtype=float)
        result = psi(reference, comparison)
        assert np.isnan(result)

    def test_all_nan_comparison_returns_nan(self):
        """All-NaN comparison -> result is nan (no crash)."""
        reference = np.arange(100, dtype=float)
        comparison = np.full(100, np.nan)
        result = psi(reference, comparison)
        assert np.isnan(result)
