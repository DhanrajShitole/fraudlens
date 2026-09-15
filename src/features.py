"""
src/features.py
===============
Feature engineering library for FraudLens.

This module is designed to be imported by:
  - notebooks/02_feature_engineering.ipynb  (batch offline processing)
  - The FastAPI scoring service              (online inference)

Two datasets are handled independently and must NEVER be merged:
  - IEEE-CIS  (e-commerce transactions + identity)
  - PaySim    (synthetic mobile-money transactions)

All functions that "fit" on training data accept an explicit ``train_df``
argument so that the caller controls what constitutes training data.
This keeps leakage prevention explicit and testable.

Public API
----------
IEEE-CIS:
    decode_transaction_dt        - hour_of_day, day_of_week, day_of_month
    log_transform_amount         - log1p(TransactionAmt)
    FrequencyEncoder             - fit on train, transform val/test
    add_has_identity             - binary flag from identity table
    add_missingness_flags        - {col}_is_missing for >30% missing cols
    drop_high_missing_v_features - drop V-cols with >90% missing
    prune_correlated_v_features  - drop one of each pair with |r| > 0.95
    add_entity_graph_features    - aggregation-based card/addr/email counts
    time_split_ieee              - chronological 70/15/15 split

PaySim:
    add_paysim_features          - all PaySim-specific features in one pass
    time_split_paysim            - chronological 70/15/15 split

Utilities:
    get_v_cols                   - list V-feature column names
    get_id_cols                  - list identity column names
"""

from __future__ import annotations

import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Cyclic period used for PaySim step encoding.
#: ``step`` represents hours; 24 gives a within-day cycle.
_PAYSIM_STEP_PERIOD: int = 24

#: High-cardinality categorical columns to frequency-encode for IEEE-CIS.
IEEE_FREQ_ENCODE_COLS: List[str] = ["card1", "addr1", "P_emaildomain"]

#: Time column names.
IEEE_TIME_COL: str = "TransactionDT"
PAYSIM_TIME_COL: str = "step"

#: Label column.
LABEL_COL: str = "isFraud"


# ===========================================================================
# IEEE-CIS — TransactionDT decoding
# ===========================================================================

def decode_transaction_dt(df: pd.DataFrame) -> pd.DataFrame:
    """Decode ``TransactionDT`` into interpretable cyclic time components.

    ``TransactionDT`` is **seconds elapsed from an arbitrary reference point**,
    NOT a wall-clock timestamp.  Three features are derived purely from
    modular arithmetic so they remain meaningful even if the reference epoch
    is unknown:

    * ``hour_of_day``  — 0-23   (seconds mod 86400 divided by 3600)
    * ``day_of_week``  — 0-6    (days mod 7)
    * ``day_of_month`` — 1-31   (days mod 31, 1-indexed approximation)

    The original ``TransactionDT`` column is **preserved** (needed for
    chronological splitting).

    Parameters
    ----------
    df : pd.DataFrame
        Must contain the ``TransactionDT`` column.

    Returns
    -------
    pd.DataFrame
        A *copy* of ``df`` with three new columns appended.

    Notes
    -----
    ``day_of_month`` uses mod 31 which is an approximation (months differ
    in length), but still captures within-month position without knowing
    the true calendar month.
    """
    df = df.copy()

    seconds_in_day = 86_400   # 60 * 60 * 24
    seconds_in_hour = 3_600   # 60 * 60

    dt = df[IEEE_TIME_COL]
    df["hour_of_day"]  = (dt % seconds_in_day // seconds_in_hour).astype(np.int8)
    df["day_of_week"]  = (dt // seconds_in_day % 7).astype(np.int8)
    df["day_of_month"] = (dt // seconds_in_day % 31 + 1).astype(np.int8)

    return df


# ===========================================================================
# IEEE-CIS — TransactionAmt log transform
# ===========================================================================

def log_transform_amount(df: pd.DataFrame) -> pd.DataFrame:
    """Add a log1p-transformed copy of ``TransactionAmt``.

    The original ``TransactionAmt`` column is **kept** (not overwritten)
    so SHAP values remain interpretable on the original scale.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain ``TransactionAmt``.

    Returns
    -------
    pd.DataFrame
        A copy with a new ``TransactionAmt_log1p`` column.
    """
    df = df.copy()
    df["TransactionAmt_log1p"] = np.log1p(df["TransactionAmt"])
    return df


# ===========================================================================
# IEEE-CIS — Frequency Encoder
# ===========================================================================

class FrequencyEncoder:
    """Replace each categorical value with its frequency count in training data.

    Fitting is performed **only on training data** to prevent leakage.
    Unseen values in validation / test receive a count of 0.

    The fitted mapping dicts are stored as plain Python ``dict``s in
    ``self.freq_maps_`` so they can be serialised to JSON for the FastAPI
    scoring service.

    Parameters
    ----------
    cols : list of str, optional
        Columns to encode.  Defaults to ``IEEE_FREQ_ENCODE_COLS``
        (``card1``, ``addr1``, ``P_emaildomain``).

    Examples
    --------
    >>> enc = FrequencyEncoder()
    >>> train_fe = enc.fit_transform(train_df)
    >>> val_fe   = enc.transform(val_df)
    >>> test_fe  = enc.transform(test_df)
    """

    def __init__(self, cols: Optional[List[str]] = None) -> None:
        self.cols = cols if cols is not None else IEEE_FREQ_ENCODE_COLS
        self.freq_maps_: Dict[str, Dict] = {}

    def fit(self, df: pd.DataFrame) -> "FrequencyEncoder":
        """Compute frequency counts from ``df`` for each column in ``self.cols``.

        Parameters
        ----------
        df : pd.DataFrame
            Training data only (must NOT include validation or test rows).

        Returns
        -------
        FrequencyEncoder
            ``self`` (fluent interface).
        """
        self.freq_maps_ = {}
        for col in self.cols:
            if col not in df.columns:
                warnings.warn(
                    f"FrequencyEncoder.fit: column '{col}' not found; skipping.",
                    stacklevel=2,
                )
                continue
            self.freq_maps_[col] = df[col].value_counts().to_dict()
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply frequency-count encoding to ``df``.

        Columns are added as ``{col}_freq`` (original column preserved).
        Unseen values map to 0.

        Parameters
        ----------
        df : pd.DataFrame
            Any split (train, val, or test) to transform.

        Returns
        -------
        pd.DataFrame
            Copy of ``df`` with ``{col}_freq`` columns appended.

        Raises
        ------
        RuntimeError
            If ``fit()`` has not been called yet.
        """
        if not self.freq_maps_:
            raise RuntimeError(
                "FrequencyEncoder has not been fitted yet. Call fit() first."
            )
        df = df.copy()
        for col, freq_map in self.freq_maps_.items():
            if col not in df.columns:
                warnings.warn(
                    f"FrequencyEncoder.transform: column '{col}' not found; skipping.",
                    stacklevel=2,
                )
                continue
            df[f"{col}_freq"] = df[col].map(freq_map).fillna(0).astype(np.int32)
        return df

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convenience method: fit on ``df`` then transform ``df``."""
        return self.fit(df).transform(df)


# ===========================================================================
# IEEE-CIS — has_identity flag
# ===========================================================================

def add_has_identity(
    df: pd.DataFrame,
    identity_ids: pd.Index,
) -> pd.DataFrame:
    """Add a binary ``has_identity`` flag.

    Parameters
    ----------
    df : pd.DataFrame
        Transaction table containing ``TransactionID``.
    identity_ids : pd.Index or array-like
        ``TransactionID`` values present in the identity table.
        Should be derived **only from training identity rows**.

    Returns
    -------
    pd.DataFrame
        Copy with ``has_identity`` (int8, 0 or 1) appended.
    """
    df = df.copy()
    identity_set = set(identity_ids)
    df["has_identity"] = df["TransactionID"].isin(identity_set).astype(np.int8)
    return df


# ===========================================================================
# IEEE-CIS — missingness flags
# ===========================================================================

def add_missingness_flags(
    df: pd.DataFrame,
    v_cols: List[str],
    id_cols: List[str],
    threshold: float = 0.30,
    *,
    reference_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Add binary ``{col}_is_missing`` indicators for high-missingness columns.

    Only V-features and identity columns (``id_XX``) are considered.
    A flag is added for each column whose **training** missing rate exceeds
    ``threshold``.

    Parameters
    ----------
    df : pd.DataFrame
        The split to add flags to (train, val, or test).
    v_cols : list of str
        V-feature column names present in ``df``.
    id_cols : list of str
        Identity column names present in ``df``.
    threshold : float
        Missing-rate threshold (fraction, default 0.30 means >30%).
    reference_df : pd.DataFrame, optional
        If provided, missing rates are computed on this DataFrame (i.e. the
        training split) to avoid leakage when calling on val/test.

    Returns
    -------
    pd.DataFrame
        Copy of ``df`` with missingness flag columns appended.
    """
    df = df.copy()
    ref = reference_df if reference_df is not None else df
    candidates = [c for c in v_cols + id_cols if c in ref.columns and c in df.columns]
    miss_rates = ref[candidates].isnull().mean()
    flagged = miss_rates[miss_rates > threshold].index.tolist()
    for col in flagged:
        df[f"{col}_is_missing"] = df[col].isnull().astype(np.int8)
    print(
        f"  [missingness flags] Added {len(flagged)} flag columns "
        f"(threshold >{threshold:.0%})."
    )
    return df


# ===========================================================================
# IEEE-CIS — drop high-missing V-features
# ===========================================================================

def drop_high_missing_v_features(
    df: pd.DataFrame,
    v_cols: List[str],
    threshold: float = 0.90,
    *,
    reference_df: Optional[pd.DataFrame] = None,
) -> Tuple[pd.DataFrame, List[str]]:
    """Drop V-features whose missing rate exceeds ``threshold``.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to drop columns from.
    v_cols : list of str
        Candidate V-feature column names.
    threshold : float
        Missing-rate cut-off (fraction, default 0.90 means >90% missing).
    reference_df : pd.DataFrame, optional
        Compute missing rates on this DataFrame (training split) to avoid
        leakage when applying to val/test.

    Returns
    -------
    (pd.DataFrame, list of str)
        * Copy of ``df`` with high-missing V-features removed.
        * List of dropped column names.
    """
    ref = reference_df if reference_df is not None else df
    present_v = [c for c in v_cols if c in ref.columns]
    miss_rates = ref[present_v].isnull().mean()
    to_drop = miss_rates[miss_rates > threshold].index.tolist()
    df = df.copy().drop(columns=[c for c in to_drop if c in df.columns])

    print(f"\n  [V-feature drop >90% missing]  Dropped {len(to_drop)} V-features:")
    for name in sorted(to_drop):
        rate = miss_rates[name]
        print(f"    {name:>8s}  ({rate:.1%} missing)")
    if not to_drop:
        print("    (none)")

    return df, to_drop


# ===========================================================================
# IEEE-CIS — correlation-based V-feature pruning
# ===========================================================================

def prune_correlated_v_features(
    df: pd.DataFrame,
    v_cols: List[str],
    threshold: float = 0.95,
    *,
    reference_df: Optional[pd.DataFrame] = None,
) -> Tuple[pd.DataFrame, List[str]]:
    """Remove highly correlated V-features, keeping the one with fewer NaNs.

    For each pair of remaining V-features with Pearson |r| > ``threshold``,
    the column with *more* missing values is dropped.  Ties are broken
    lexicographically (keeps alphabetically earlier name).

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to prune.
    v_cols : list of str
        V-feature columns to consider (survivors after
        ``drop_high_missing_v_features`` was called).
    threshold : float
        Correlation magnitude cut-off (default 0.95).
    reference_df : pd.DataFrame, optional
        Compute correlations on this DataFrame (training split).

    Returns
    -------
    (pd.DataFrame, list of str)
        * Pruned copy of ``df``.
        * List of columns dropped by correlation pruning.
    """
    ref = reference_df if reference_df is not None else df
    present_v = [c for c in v_cols if c in ref.columns]

    print(
        f"\n  [V-feature correlation pruning]  Starting with {len(present_v)} V-features."
    )

    if len(present_v) < 2:
        print("  Too few V-features to compute correlations; skipping.")
        return df, []

    # Pairwise Pearson correlation (NaN pairs excluded by default).
    print("    Computing correlation matrix (may take a moment on large data)...")
    corr_matrix = ref[present_v].corr().abs()

    # Upper triangle only to avoid double-counting.
    upper = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape, dtype=bool), k=1)
    )

    miss_rates = ref[present_v].isnull().mean()
    to_drop: set = set()

    for col in upper.columns:
        if col in to_drop:
            continue
        partners = upper.index[upper[col] > threshold].tolist()
        for partner in partners:
            if partner in to_drop:
                continue
            col_miss = miss_rates[col]
            partner_miss = miss_rates[partner]
            if col_miss < partner_miss:
                to_drop.add(partner)
            elif partner_miss < col_miss:
                to_drop.add(col)
            else:
                # Tie: drop alphabetically later name.
                to_drop.add(max(col, partner))

    to_drop_list = sorted(to_drop)
    surviving = [c for c in present_v if c not in to_drop]

    df = df.copy().drop(columns=[c for c in to_drop_list if c in df.columns])

    print(
        f"    Dropped {len(to_drop_list)} correlated V-features "
        f"(|r| > {threshold})."
    )
    print(
        f"    Remaining V-features: {len(surviving)}  "
        f"(was {len(present_v)})."
    )

    return df, to_drop_list


# ===========================================================================
# IEEE-CIS — entity-graph aggregation features
# ===========================================================================

def add_entity_graph_features(
    df: pd.DataFrame,
    *,
    train_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Add aggregation-based entity-graph signal features.

    Features are derived from combinations of ``card1``, ``addr1``, and
    ``P_emaildomain``.  Device/IP columns are excluded per spec (too much
    missing data).

    Features added:

    * ``card1_addr1_count``       — transactions in training sharing same
      (card1, addr1) key.
    * ``card1_addr1_email_count`` — same but using (card1, addr1,
      P_emaildomain).
    * ``card1_addr1_amt_mean``    — mean TransactionAmt per (card1, addr1).
    * ``card1_addr1_amt_std``     — std TransactionAmt per (card1, addr1).

    Parameters
    ----------
    df : pd.DataFrame
        Split to add features to (train, val, or test).
    train_df : pd.DataFrame, optional
        **Training split only.**  Aggregations are fitted on this to prevent
        leakage.  If ``None``, ``df`` is used (correct only when ``df`` IS
        the training split).

    Returns
    -------
    pd.DataFrame
        Copy of ``df`` with entity-graph feature columns appended.

    Notes
    -----
    **Simplification:** Global group counts are used instead of time-windowed
    counts ("past N days").  The spec calls these features a *proxy* for
    entity-graph signal and notes that a real GNN layer is a later phase.
    Global counts are a correct and efficient approximation; the decision is
    documented in the notebook summary cell.
    """
    ref = train_df if train_df is not None else df
    df = df.copy()

    # --- (card1, addr1) group aggregations ---
    key1 = ["card1", "addr1"]
    if all(c in ref.columns for c in key1):
        grp1 = ref.groupby(key1, observed=True)["TransactionAmt"]

        agg1 = pd.concat(
            [
                grp1.count().rename("card1_addr1_count"),
                grp1.mean().rename("card1_addr1_amt_mean"),
                grp1.std().rename("card1_addr1_amt_std"),
            ],
            axis=1,
        ).reset_index()

        df = df.merge(agg1, on=key1, how="left")
        df["card1_addr1_count"]    = df["card1_addr1_count"].fillna(0).astype(np.int32)
        df["card1_addr1_amt_mean"] = df["card1_addr1_amt_mean"].astype(np.float32)
        df["card1_addr1_amt_std"]  = df["card1_addr1_amt_std"].astype(np.float32)
    else:
        warnings.warn(
            "add_entity_graph_features: 'card1' or 'addr1' not found; "
            "skipping (card1, addr1) aggregations.",
            stacklevel=2,
        )

    # --- (card1, addr1, P_emaildomain) count ---
    key2 = ["card1", "addr1", "P_emaildomain"]
    if all(c in ref.columns for c in key2):
        count2 = (
            ref.groupby(key2, observed=True)["TransactionAmt"]
            .count()
            .rename("card1_addr1_email_count")
            .reset_index()
        )
        df = df.merge(count2, on=key2, how="left")
        df["card1_addr1_email_count"] = (
            df["card1_addr1_email_count"].fillna(0).astype(np.int32)
        )
    else:
        warnings.warn(
            "add_entity_graph_features: card1/addr1/P_emaildomain missing; "
            "skipping triple-key count.",
            stacklevel=2,
        )

    print(
        "  [entity-graph features] Added: card1_addr1_count, "
        "card1_addr1_email_count, card1_addr1_amt_mean, card1_addr1_amt_std."
    )
    return df


# ===========================================================================
# PaySim — all feature engineering
# ===========================================================================

def add_paysim_features(df: pd.DataFrame) -> pd.DataFrame:
    """Engineer all PaySim features in one deterministic pass.

    PaySim is synthetically clean (zero missing values), so no fitting step
    is required.

    Features added:

    * ``balance_delta_orig``       — ``oldbalanceOrg - newbalanceOrig - amount``
    * ``is_transfer``              — 1 if type == TRANSFER, else 0
    * ``is_cashout``               — 1 if type == CASH_OUT, else 0
    * ``orig_balance_drained``     — 1 if newbalanceOrig == 0, else 0
    * ``dest_balance_zero``        — 1 if newbalanceDest == 0, else 0
    * ``amount_to_balance_ratio``  — amount / (oldbalanceOrg + 1)
    * ``step_sin``                 — sin(step * 2pi / 24)
    * ``step_cos``                 — cos(step * 2pi / 24)

    Parameters
    ----------
    df : pd.DataFrame
        Raw PaySim DataFrame.

    Returns
    -------
    pd.DataFrame
        Copy of ``df`` with all new feature columns appended.
    """
    df = df.copy()

    df["balance_delta_orig"] = (
        df["oldbalanceOrg"] - df["newbalanceOrig"] - df["amount"]
    ).astype(np.float64)

    df["is_transfer"] = (df["type"] == "TRANSFER").astype(np.int8)
    df["is_cashout"]  = (df["type"] == "CASH_OUT").astype(np.int8)

    df["orig_balance_drained"] = (df["newbalanceOrig"] == 0).astype(np.int8)
    df["dest_balance_zero"]    = (df["newbalanceDest"]  == 0).astype(np.int8)

    df["amount_to_balance_ratio"] = (
        df["amount"] / (df["oldbalanceOrg"] + 1)
    ).astype(np.float64)

    cycle = 2.0 * np.pi / _PAYSIM_STEP_PERIOD
    df["step_sin"] = np.sin(df["step"] * cycle).astype(np.float32)
    df["step_cos"] = np.cos(df["step"] * cycle).astype(np.float32)

    print(
        "  [paysim features] Added: balance_delta_orig, is_transfer, is_cashout, "
        "orig_balance_drained, dest_balance_zero, amount_to_balance_ratio, "
        "step_sin, step_cos."
    )
    return df


# ===========================================================================
# Splitting utilities
# ===========================================================================

def _chronological_split(
    df: pd.DataFrame,
    time_col: str,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict]:
    """Internal helper: sort by ``time_col`` and split 70 / 15 / 15."""
    df = df.sort_values(time_col).reset_index(drop=True)
    n = len(df)
    n_train = int(np.floor(n * train_frac))
    n_val   = int(np.floor(n * val_frac))
    n_test  = n - n_train - n_val

    train = df.iloc[:n_train].reset_index(drop=True)
    val   = df.iloc[n_train : n_train + n_val].reset_index(drop=True)
    test  = df.iloc[n_train + n_val :].reset_index(drop=True)

    info = {
        "n_total":            n,
        "n_train":            n_train,
        "n_val":              n_val,
        "n_test":             n_test,
        "train_time_min":     float(train[time_col].min()),
        "train_time_max":     float(train[time_col].max()),
        "val_time_min":       float(val[time_col].min()),
        "val_time_max":       float(val[time_col].max()),
        "test_time_min":      float(test[time_col].min()),
        "test_time_max":      float(test[time_col].max()),
        "train_boundary_idx": n_train,
        "val_boundary_idx":   n_train + n_val,
        "fractions": {
            "train": n_train / n,
            "val":   n_val   / n,
            "test":  n_test  / n,
        },
    }
    return train, val, test, info


def time_split_ieee(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict]:
    """Chronological 70 / 15 / 15 split for the IEEE-CIS dataset.

    Sorts by ``TransactionDT`` ascending — no shuffling.  This simulates
    real deployment where the model is always scored on future transactions.

    Parameters
    ----------
    df : pd.DataFrame
        Merged + feature-engineered IEEE-CIS DataFrame.

    Returns
    -------
    (train, val, test, split_info)
        ``split_info`` dict contains boundary ``TransactionDT`` values and
        row counts for reproducibility.
    """
    print(f"\n  [ieee split]  Sorting by {IEEE_TIME_COL} and splitting 70/15/15 ...")
    train, val, test, info = _chronological_split(df, IEEE_TIME_COL)
    _print_split_summary(info, IEEE_TIME_COL)
    return train, val, test, info


def time_split_paysim(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict]:
    """Chronological 70 / 15 / 15 split for the PaySim dataset.

    Sorts by ``step`` ascending — no shuffling.

    Parameters
    ----------
    df : pd.DataFrame
        Feature-engineered PaySim DataFrame.

    Returns
    -------
    (train, val, test, split_info)
        ``split_info`` dict contains boundary ``step`` values and row counts.
    """
    print(f"\n  [paysim split]  Sorting by {PAYSIM_TIME_COL} and splitting 70/15/15 ...")
    train, val, test, info = _chronological_split(df, PAYSIM_TIME_COL)
    _print_split_summary(info, PAYSIM_TIME_COL)
    return train, val, test, info


# ===========================================================================
# Internals & utilities
# ===========================================================================

def _print_split_summary(info: Dict, time_col: str) -> None:
    """Pretty-print split boundary info and assert no temporal overlap."""
    print(
        f"    Rows   -> train: {info['n_train']:>8,}  "
        f"val: {info['n_val']:>8,}  "
        f"test: {info['n_test']:>8,}  "
        f"(total: {info['n_total']:,})"
    )
    print(
        f"    {time_col:>15s} range -> "
        f"train [{info['train_time_min']:.0f} - {info['train_time_max']:.0f}]  "
        f"val [{info['val_time_min']:.0f} - {info['val_time_max']:.0f}]  "
        f"test [{info['test_time_min']:.0f} - {info['test_time_max']:.0f}]"
    )
    # Guard against temporal leakage.
    assert info["train_time_max"] <= info["val_time_min"], (
        "LEAKAGE DETECTED: train_time_max > val_time_min!"
    )
    assert info["val_time_max"] <= info["test_time_min"], (
        "LEAKAGE DETECTED: val_time_max > test_time_min!"
    )
    print("    Confirmed: no temporal overlap between splits.")


def get_v_cols(df: pd.DataFrame) -> List[str]:
    """Return sorted list of V-feature column names present in ``df``.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    list of str
        E.g. ['V1', 'V2', ..., 'V339'].
    """
    return sorted([c for c in df.columns if c.startswith("V") and c[1:].isdigit()])


def get_id_cols(df: pd.DataFrame) -> List[str]:
    """Return sorted list of identity column names (``id_XX``) in ``df``.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    list of str
        E.g. ['id_01', 'id_02', ...].
    """
    return sorted(
        [c for c in df.columns if c.lower().startswith("id_") and c[3:].isdigit()]
    )
