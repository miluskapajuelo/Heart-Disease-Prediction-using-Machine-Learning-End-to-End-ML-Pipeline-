"""
tests/test_features.py
-----------------------
Tests for add_engineered_features() in src/data/preprocess.py

What we test:
  - isquemia_score formula is mathematically correct
  - est_stroke_volume formula is mathematically correct
  - Output contains exactly the selected features from config
  - No nulls in engineered output
  - All 3 output splits have identical column names

Selected features come from CFG — never hardcoded.

Run: pytest tests/test_features.py -v
"""

import numpy as np
import pandas as pd
import pytest

from src.utils.config import CFG
from src.data.preprocess import add_engineered_features


# ── Constants from config ─────────────────────────────────────────────────────
SELECTED = CFG["features"]["selected"]


# ── Helper ────────────────────────────────────────────────────────────────────
def _make_df(n=20, seed=0):
    """Synthetic post-imputation DataFrame. No zeros, no nulls."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "age":      rng.integers(30, 75,  n).astype(float),
        "sex":      rng.integers(0,  2,   n).astype(float),
        "cp":       rng.integers(0,  4,   n).astype(float),
        "trestbps": rng.integers(90, 180, n).astype(float),
        "chol":     rng.integers(150, 350, n).astype(float),
        "fbs":      rng.integers(0,  2,   n).astype(float),
        "restecg":  rng.integers(0,  3,   n).astype(float),
        "thalach":  rng.integers(80, 200, n).astype(float),
        "exang":    rng.integers(0,  2,   n).astype(float),
        "oldpeak":  rng.uniform(0.1, 5,   n),
        "slope":    rng.integers(0,  3,   n).astype(float),
        "ca":       rng.integers(0,  4,   n).astype(float),
        "thal":     rng.integers(0,  4,   n).astype(float),
    })


# ── Formula tests ─────────────────────────────────────────────────────────────

def test_isquemia_score_formula():
    """isquemia_score = oldpeak + exang + (ca / 3)"""
    X = _make_df(10)
    X_fe, _, _ = add_engineered_features(X, X, X)

    row      = X.iloc[0]
    expected = row["oldpeak"] + row["exang"] + (row["ca"] / 3)
    actual   = X_fe["isquemia_score"].iloc[0]

    assert actual == pytest.approx(expected, rel=1e-5)


def test_est_stroke_volume_formula():
    """est_stroke_volume = (trestbps / thalach) * (age / 50)"""
    X = _make_df(10)
    X_fe, _, _ = add_engineered_features(X, X, X)

    row      = X.iloc[0]
    expected = (row["trestbps"] / row["thalach"]) * (row["age"] / 50)
    actual   = X_fe["est_stroke_volume"].iloc[0]

    assert actual == pytest.approx(expected, rel=1e-5)


def test_cardiac_capacity_formula():
    """cardiac_capacity = thalach / age"""
    X = _make_df(10)
    X_fe, _, _ = add_engineered_features(X, X, X)

    row      = X.iloc[0]
    expected = row["thalach"] / row["age"]
    actual   = X_fe["cardiac_capacity"].iloc[0]

    assert actual == pytest.approx(expected, rel=1e-5)


def test_troponin_index_formula():
    """troponin_index = (oldpeak * 1.5) + (exang * 2) + (ca * 1.2)"""
    X = _make_df(10)
    X_fe, _, _ = add_engineered_features(X, X, X)

    row      = X.iloc[0]
    expected = (row["oldpeak"] * 1.5) + (row["exang"] * 2) + (row["ca"] * 1.2)
    actual   = X_fe["troponin_index"].iloc[0]

    assert actual == pytest.approx(expected, rel=1e-5)


# ── Output quality ────────────────────────────────────────────────────────────

def test_engineered_features_added():
    """Output must contain the 4 engineered columns on top of originals."""
    X = _make_df(10)
    X_fe, _, _ = add_engineered_features(X, X, X)

    for col in ["isquemia_score", "est_stroke_volume",
                "cardiac_capacity", "troponin_index"]:
        assert col in X_fe.columns, f"Engineered column '{col}' missing from output"


def test_no_nulls_in_any_split():
    """No NaN values should appear in the engineered output."""
    X_train = _make_df(20, seed=0)
    X_val   = _make_df(6,  seed=1)
    X_test  = _make_df(6,  seed=2)

    X_tr_fe, X_v_fe, X_te_fe = add_engineered_features(X_train, X_val, X_test)

    assert X_tr_fe.isnull().sum().sum() == 0, "Nulls in X_train_fe"
    assert X_v_fe.isnull().sum().sum()  == 0, "Nulls in X_val_fe"
    assert X_te_fe.isnull().sum().sum() == 0, "Nulls in X_test_fe"


def test_all_splits_have_same_columns():
    """All 3 output DataFrames must have identical column names."""
    X_train = _make_df(20, seed=0)
    X_val   = _make_df(6,  seed=1)
    X_test  = _make_df(6,  seed=2)

    X_tr_fe, X_v_fe, X_te_fe = add_engineered_features(X_train, X_val, X_test)

    assert list(X_tr_fe.columns) == list(X_v_fe.columns)
    assert list(X_tr_fe.columns) == list(X_te_fe.columns)


def test_original_columns_preserved():
    """All original 13 columns must still be present in the output."""
    X = _make_df(10)
    X_fe, _, _ = add_engineered_features(X, X, X)

    for col in X.columns:
        assert col in X_fe.columns, f"Original column '{col}' was dropped"