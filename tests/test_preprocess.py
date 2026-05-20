
"""
tests/test_preprocess.py
-------------------------
Tests for src/data/preprocess.py
 
Functions tested:
  - replace_zero_with_nan()  — replaces impossible zeros with NaN
  - fit_imputer()            — fits imputer on train only, transforms all splits
 
What we test:
  - Zeros in zero_as_nan_cols become NaN
  - No nulls remain in any split after imputation
  - Val and test use TRAIN medians (leakage prevention)
  - Output column names and index are preserved
  - Returned imputer is fitted
 
Columns come from CFG — never hardcoded.
 
Run: pytest tests/test_preprocess.py -v
"""
 
import numpy as np
import pandas as pd
import pytest
from sklearn.impute import SimpleImputer
 
from src.utils.config import CFG
from src.data.preprocess import replace_zero_with_nan, fit_imputer
 
 
# ── Constants from config ─────────────────────────────────────────────────────
ZERO_COLS = CFG["preprocessing"]["zero_as_nan_col"]
 
 
# ── Helpers ───────────────────────────────────────────────────────────────────
def _make_df(n=20, seed=0):
    """Synthetic DataFrame with all 13 input columns. No zeros — clean data."""
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
 
 
# ── replace_zero_with_nan() tests ─────────────────────────────────────────────
 
@pytest.mark.parametrize("col", ZERO_COLS)
def test_zero_replaced_in_each_col(col):
    """
    Each zero_as_nan column must not retain 0 after replacement.
    Tests each column individually so failures are easy to identify.
    """
    df = _make_df(10)
    df.loc[0, col] = 0   # inject zero
 
    result = replace_zero_with_nan(df)
 
    assert pd.isna(result.loc[0, col]), (
        f"{col}=0 was not replaced with NaN"
    )
 
 
def test_replace_zero_does_not_modify_original():
    """replace_zero_with_nan must return a copy — original unchanged."""
    df = _make_df(10)
    df.loc[0, "chol"] = 0
 
    _ = replace_zero_with_nan(df)
 
    # Original must still have 0
    assert df.loc[0, "chol"] == 0, "Original DataFrame was modified — missing .copy()"
 
 
def test_replace_zero_only_affects_specified_columns():
    """Columns NOT in zero_as_nan_cols must be untouched even if they have zeros."""
    df = _make_df(10)
    df.loc[0, "fbs"] = 0   # fbs=0 is valid (fasting blood sugar <= 120)
 
    result = replace_zero_with_nan(df)
 
    # fbs is not in ZERO_COLS so it must remain 0
    assert result.loc[0, "fbs"] == 0, (
        "fbs=0 was incorrectly replaced — only ZERO_COLS should be affected"
    )
 
 
# ── fit_imputer() tests ───────────────────────────────────────────────────────
 
def test_no_nulls_after_imputation():
    """All 3 splits must have zero nulls after fit_imputer()."""
    X_train = _make_df(20)
    X_val   = _make_df(6)
    X_test  = _make_df(6)
 
    # Inject zeros then replace with NaN (simulates real pipeline)
    for df in [X_train, X_val, X_test]:
        for col in ZERO_COLS:
            df.loc[0, col] = 0
 
    X_train = replace_zero_with_nan(X_train)
    X_val   = replace_zero_with_nan(X_val)
    X_test  = replace_zero_with_nan(X_test)
 
    X_tr, X_te, X_v, _ = fit_imputer(X_train, X_val, X_test)
 
    assert X_tr.isnull().sum().sum() == 0, "Nulls in X_train after imputation"
    assert X_v.isnull().sum().sum()  == 0, "Nulls in X_val after imputation"
    assert X_te.isnull().sum().sum() == 0, "Nulls in X_test after imputation"
 
 
def test_val_and_test_use_train_medians():
    """
    Val and test must be imputed with TRAIN medians — not their own.
    Key leakage prevention test.
    """
    # Train: chol forced high (median ~300)
    X_train = _make_df(20)
    X_train["chol"] = 300.0
 
    # Val: chol=0 (missing) — must be filled with 300 from train
    X_val = _make_df(6)
    X_val["chol"] = 0.0
    X_val = replace_zero_with_nan(X_val)
 
    X_test = _make_df(6)
 
    X_tr, X_te, X_v, _ = fit_imputer(X_train, X_val, X_test)
 
    assert X_v["chol"].iloc[0] == pytest.approx(300.0), (
        f"Val imputed with {X_v['chol'].iloc[0]} — expected train median 300.0"
    )
 
 
def test_output_columns_match_input():
    """Output column names must exactly match input."""
    X_train = _make_df(20)
    X_val   = _make_df(6)
    X_test  = _make_df(6)
 
    X_tr, X_te, X_v, _ = fit_imputer(X_train, X_val, X_test)
 
    assert list(X_tr.columns) == list(X_train.columns)
    assert list(X_v.columns)  == list(X_val.columns)
    assert list(X_te.columns) == list(X_test.columns)
 
 
def test_output_index_preserved():
    """DataFrame index must survive imputation unchanged."""
    X_train = _make_df(20)
    X_val   = _make_df(6)
    X_test  = _make_df(6)
 
    X_tr, X_te, X_v, _ = fit_imputer(X_train, X_val, X_test)
 
    pd.testing.assert_index_equal(X_tr.index, X_train.index)
    pd.testing.assert_index_equal(X_v.index,  X_val.index)
    pd.testing.assert_index_equal(X_te.index, X_test.index)
 
 
def test_returns_fitted_imputer():
    """Returned imputer must be a fitted SimpleImputer with correct statistics."""
    X_train = _make_df(20)
    X_val   = _make_df(6)
    X_test  = _make_df(6)
 
    _, _, _, imputer = fit_imputer(X_train, X_val, X_test)
 
    assert isinstance(imputer, SimpleImputer)
    assert hasattr(imputer, "statistics_"), "Imputer is not fitted"
    assert len(imputer.statistics_) == X_train.shape[1]