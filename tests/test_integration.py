"""
tests/test_integration.py
--------------------------
Integration tests — runs the full pipeline on the REAL heart.csv dataset.

Unlike unit tests (synthetic data, test logic in isolation),
these tests use actual patient data to verify the entire pipeline
produces results consistent with the notebook.

Key values verified here come directly from notebook output:
  - 302 rows after deduplication (1025 raw - 723 duplicates)
  - 180 / 61 / 61 train/val/test split
  - chol median  = 241.5
  - trestbps median = 130.0
  - thalach median  = 154.5

Columns are read from heart.csv — never hardcoded.

Skipped automatically if heart.csv is not present.

Run: pytest tests/test_integration.py -v
"""

import pytest
import pandas as pd
from pathlib import Path

from src.utils.config import CFG

# ── Skip entire file if heart.csv not present ─────────────────────────────────
DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "raw" / "heart.csv"

pytestmark = pytest.mark.skipif(
    not DATA_PATH.exists(),
    reason=f"heart.csv not found at {DATA_PATH} — skipping integration tests"
)

from src.data.load_data  import load_data
from src.data.split_data import split_data
from src.data.preprocess import preprocess, build_features


# ── Run pipeline once — reuse across all tests in this file ──────────────────
@pytest.fixture(scope="module")
def pipeline():
    """
    Full pipeline fixture — runs once per file, not once per test.
    Returns all intermediate outputs for individual tests to inspect.
    """
    df = load_data()

    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df)

    X_tr_imp, X_v_imp, X_te_imp, imputer = preprocess(X_train, X_val, X_test)

    X_tr_fe, X_v_fe, X_te_fe = build_features(X_tr_imp, X_v_imp, X_te_imp)

    return {
        "df":       df,
        "X_train":  X_train,  "y_train": y_train,
        "X_val":    X_val,    "y_val":   y_val,
        "X_test":   X_test,   "y_test":  y_test,
        "X_tr_imp": X_tr_imp,
        "X_tr_fe":  X_tr_fe,
        "X_v_fe":   X_v_fe,
        "X_te_fe":  X_te_fe,
        "imputer":  imputer,
    }


# ── Load and shape checks ─────────────────────────────────────────────────────

def test_raw_columns_include_target(pipeline):
    """
    Columns are read from heart.csv directly — not hardcoded.
    We verify count and target column presence.
    """
    df     = pipeline["df"]
    target = CFG["split"]["target_column"]

    assert target in df.columns, f"Target column '{target}' not found in dataset"
    assert df.shape[1] == 14, f"Expected 14 columns, got {df.shape[1]}"


def test_dataset_shape_after_dedup(pipeline):
    """
    After dropping duplicates: 302 rows x 14 columns.
    (1025 raw - 723 duplicates = 302 clean rows)
    """
    assert pipeline["df"].shape == (302, 14), (
        f"Expected (302, 14), got {pipeline['df'].shape}"
    )


def test_no_nulls_in_raw_data(pipeline):
    """Raw data has no true nulls — zeros are encoded missing values."""
    assert pipeline["df"].isnull().sum().sum() == 0


# ── Split checks ──────────────────────────────────────────────────────────────

def test_split_sizes(pipeline):
    """60/20/20 of 302 rows = 180 train, 61 val, 61 test."""
    p     = pipeline
    total = len(p["X_train"]) + len(p["X_val"]) + len(p["X_test"])

    assert total == 302
    assert len(p["X_train"]) == 180
    assert len(p["X_val"])   == 61
    assert len(p["X_test"])  == 61


def test_class_balance_preserved(pipeline):
    """
    Stratification must preserve ~54% positive class across all splits.
    Tolerance: ±5%
    """
    p = pipeline
    for name, y in [("train", p["y_train"]), ("val", p["y_val"]), ("test", p["y_test"])]:
        rate = y.mean()
        assert 0.49 <= rate <= 0.59, (
            f"{name} positive rate = {rate:.1%} — expected ~54% ± 5%"
        )


# ── Imputation checks ─────────────────────────────────────────────────────────

def test_no_nulls_after_imputation(pipeline):
    """No nulls should survive imputation in any split."""
    assert pipeline["X_tr_imp"].isnull().sum().sum() == 0
    assert pipeline["X_v_fe"].isnull().sum().sum()   == 0
    assert pipeline["X_te_fe"].isnull().sum().sum()  == 0


def test_train_medians_match_notebook(pipeline):
    """
    Medians learned from real training data must match notebook values.
    Tolerance: ±5 units.
    """
    imputer = pipeline["imputer"]
    cols    = pipeline["X_train"].columns.tolist()
    medians = dict(zip(cols, imputer.statistics_))

    assert medians["chol"]     == pytest.approx(241.5, abs=5.0), (
        f"chol median={medians['chol']} — expected ~241.5"
    )
    assert medians["trestbps"] == pytest.approx(130.0, abs=5.0), (
        f"trestbps median={medians['trestbps']} — expected ~130.0"
    )
    assert medians["thalach"]  == pytest.approx(154.5, abs=5.0), (
        f"thalach median={medians['thalach']} — expected ~154.5"
    )


# ── Feature engineering checks ────────────────────────────────────────────────

def test_final_features_match_config(pipeline):
    """
    Final feature columns must match CFG['features']['selected'] exactly.
    If someone changes the config, this test catches the mismatch.
    """
    expected = CFG["features"]["selected"]
    actual   = list(pipeline["X_tr_fe"].columns)

    assert actual == expected, (
        f"\nExpected: {expected}"
        f"\nGot     : {actual}"
    )


def test_no_nulls_in_engineered_features(pipeline):
    """No NaN values in final feature-engineered DataFrames."""
    for name, X in [
        ("X_tr_fe", pipeline["X_tr_fe"]),
        ("X_v_fe",  pipeline["X_v_fe"]),
        ("X_te_fe", pipeline["X_te_fe"]),
    ]:
        nulls = X.isnull().sum().sum()
        assert nulls == 0, f"{name} has {nulls} nulls after full pipeline"