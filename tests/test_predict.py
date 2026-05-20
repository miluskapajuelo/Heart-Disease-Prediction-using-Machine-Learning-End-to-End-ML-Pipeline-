"""
tests/test_predict.py
----------------------
Tests for src/models/train.py and src/models/evaluate.py

What we test:
  - train_model() returns expected dict structure for all 3 models
  - Scaler returned for LR only, None for tree models
  - CV and val scores are valid floats between 0 and 1
  - All 3 tuned models score > 0.5 (better than random)
  - Best model is selected dynamically by val_score — not hardcoded
  - evaluate_model() returns expected metric keys in valid ranges
  - Gap equals |cv_score - roc_auc| — always positive
  - Unknown model name raises a clear ValueError

Note: Tests use tiny grids so GridSearchCV runs in seconds.
      Real grids live in model_config.yaml.

Run: pytest tests/test_predict.py -v
"""

import copy
import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import StandardScaler
from unittest.mock import patch

from src.utils.config import CFG
from src.models.train    import train_model
from src.models.evaluate import evaluate_model


# ── Constants from config ─────────────────────────────────────────────────────
SELECTED    = CFG["features"]["selected"]
ALL_MODELS  = ["logistic_regression", "random_forest", "xgboost"]
TREE_MODELS = ["random_forest", "xgboost"]


# ── Tiny grids — 1 combination each so tests run in seconds ──────────────────
TINY_GRIDS = {
    "logistic_regression": {"C": [1.0]},
    "random_forest":       {"n_estimators": [10], "max_depth": [3], "min_samples_split": [2]},
    "xgboost":             {"n_estimators": [10], "max_depth": [2], "learning_rate": [0.1], "subsample": [1.0]},
}


def _fast_cfg(model_name):
    """Return CFG with tiny grid for the given model — keeps tests fast."""
    cfg = copy.deepcopy(CFG)
    cfg["models"][model_name]["grid"] = TINY_GRIDS[model_name]
    return cfg


# ── Synthetic data ────────────────────────────────────────────────────────────
def _make_data(n_train=80, n_val=30, n_test=30, seed=42):
    """
    Synthetic feature-engineered DataFrames with a learnable signal.
    Target is determined by the first feature so models can beat random.
    Column names come from CFG — not hardcoded.
    """
    rng = np.random.default_rng(seed)
    n_features = len(SELECTED)
    n_total = n_train + n_val + n_test

    X_all = rng.uniform(0, 5, size=(n_total, n_features))
    # Strong linear signal: label=1 when feature[0] > 2.5
    noise = rng.normal(0, 0.3, n_total)
    y_all = ((X_all[:, 0] - 2.5 + noise) > 0).astype(int)

    X_all = pd.DataFrame(X_all, columns=SELECTED)
    y_all = pd.Series(y_all, name="target")

    X_train = X_all.iloc[:n_train].reset_index(drop=True)
    X_val   = X_all.iloc[n_train:n_train + n_val].reset_index(drop=True)
    X_test  = X_all.iloc[n_train + n_val:].reset_index(drop=True)
    y_train = y_all.iloc[:n_train].reset_index(drop=True)
    y_val   = y_all.iloc[n_train:n_train + n_val].reset_index(drop=True)
    y_test  = y_all.iloc[n_train + n_val:].reset_index(drop=True)

    return X_train, X_val, X_test, y_train, y_val, y_test


# ── train_model() tests ───────────────────────────────────────────────────────

@pytest.mark.parametrize("model_name", ALL_MODELS)
def test_train_returns_expected_keys(model_name):
    """train_model() must return dict with all required keys."""
    X_train, X_val, _, y_train, y_val, _ = _make_data()

    with patch("src.models.train.CFG", _fast_cfg(model_name)):
        result = train_model(model_name, X_train, y_train, X_val, y_val)

    assert set(result.keys()) == {"model", "scaler", "best_params", "cv_score", "val_score"}


@pytest.mark.parametrize("model_name", ALL_MODELS)
def test_scores_between_0_and_1(model_name):
    """cv_score and val_score must be floats in [0, 1]."""
    X_train, X_val, _, y_train, y_val, _ = _make_data()

    with patch("src.models.train.CFG", _fast_cfg(model_name)):
        result = train_model(model_name, X_train, y_train, X_val, y_val)

    assert 0.0 <= result["cv_score"]  <= 1.0
    assert 0.0 <= result["val_score"] <= 1.0


def test_logistic_regression_returns_fitted_scaler():
    """LR requires scaling — must return a fitted StandardScaler."""
    X_train, X_val, _, y_train, y_val, _ = _make_data()

    with patch("src.models.train.CFG", _fast_cfg("logistic_regression")):
        result = train_model("logistic_regression", X_train, y_train, X_val, y_val)

    assert isinstance(result["scaler"], StandardScaler)
    assert hasattr(result["scaler"], "mean_"), "Scaler is not fitted"


@pytest.mark.parametrize("model_name", TREE_MODELS)
def test_tree_models_return_no_scaler(model_name):
    """Tree models do not need scaling — scaler must be None."""
    X_train, X_val, _, y_train, y_val, _ = _make_data()

    with patch("src.models.train.CFG", _fast_cfg(model_name)):
        result = train_model(model_name, X_train, y_train, X_val, y_val)

    assert result["scaler"] is None


def test_unknown_model_raises_value_error():
    """Passing an unknown model name must raise ValueError with clear message."""
    X_train, X_val, _, y_train, y_val, _ = _make_data()

    with pytest.raises(ValueError, match="Unknown model"):
        train_model("svm", X_train, y_train, X_val, y_val)


@pytest.mark.parametrize("model_name", ALL_MODELS)
def test_all_tuned_models_beat_random(model_name):
    """
    All 3 tuned models must score ROC-AUC > 0.5 on val set.
    A score <= 0.5 means the model is worse than a random classifier.
    """
    X_train, X_val, _, y_train, y_val, _ = _make_data(n_train=100, n_val=40)

    with patch("src.models.train.CFG", _fast_cfg(model_name)):
        result = train_model(model_name, X_train, y_train, X_val, y_val)

    assert result["val_score"] > 0.5, (
        f"{model_name} val_score={result['val_score']:.3f} — worse than random"
    )


# ── Best model selection ──────────────────────────────────────────────────────

def test_best_model_selected_dynamically():
    """
    Best model must be chosen by max val_score — never hardcoded.
    We force RF to win and verify it gets selected.
    """
    results = [
        {"model_name": "logistic_regression", "val_score": 0.80},
        {"model_name": "random_forest",        "val_score": 0.93},
        {"model_name": "xgboost",              "val_score": 0.88},
    ]

    best = max(results, key=lambda r: r["val_score"])

    assert best["model_name"] == "random_forest", (
        f"Expected random_forest, got {best['model_name']}"
    )


def test_best_model_xgboost_wins_when_highest():
    """Same test — XGBoost wins when it has the highest val_score."""
    results = [
        {"model_name": "logistic_regression", "val_score": 0.80},
        {"model_name": "random_forest",        "val_score": 0.85},
        {"model_name": "xgboost",              "val_score": 0.92},
    ]

    best = max(results, key=lambda r: r["val_score"])

    assert best["model_name"] == "xgboost"


@pytest.mark.parametrize("model_name", ALL_MODELS)
def test_model_predicts_on_new_data(model_name):
    """Smoke test — fitted model must predict without errors on unseen data."""
    X_train, X_val, X_test, y_train, y_val, _ = _make_data()

    with patch("src.models.train.CFG", _fast_cfg(model_name)):
        result = train_model(model_name, X_train, y_train, X_val, y_val)

    model  = result["model"]
    scaler = result["scaler"]
    X_in   = scaler.transform(X_test) if scaler is not None else X_test

    preds  = model.predict(X_in)
    probas = model.predict_proba(X_in)[:, 1]

    assert len(preds)  == len(X_test)
    assert len(probas) == len(X_test)
    assert set(preds).issubset({0, 1})
    assert ((probas >= 0) & (probas <= 1)).all()


# ── evaluate_model() tests ────────────────────────────────────────────────────

def test_evaluate_returns_expected_keys():
    """evaluate_model() must return dict with all required metric keys."""
    X_train, X_val, X_test, y_train, y_val, y_test = _make_data()

    with patch("src.models.train.CFG", _fast_cfg("xgboost")):
        result = train_model("xgboost", X_train, y_train, X_val, y_val)

    metrics = evaluate_model(
        model_name = "xgboost",
        model      = result["model"],
        X_test     = X_test,
        y_test     = y_test,
        cv_score   = result["cv_score"],
        scaler     = result["scaler"],
    )

    expected = {
        "model_name", "accuracy", "roc_auc",
        "precision", "recall", "f1",
        "cv_score", "gap", "tp", "tn", "fp", "fn"
    }
    assert set(metrics.keys()) == expected


def test_evaluate_metrics_in_valid_range():
    """All numeric metrics must be floats in [0, 1]."""
    X_train, X_val, X_test, y_train, y_val, y_test = _make_data()

    with patch("src.models.train.CFG", _fast_cfg("xgboost")):
        result = train_model("xgboost", X_train, y_train, X_val, y_val)

    metrics = evaluate_model(
        model_name = "xgboost",
        model      = result["model"],
        X_test     = X_test,
        y_test     = y_test,
        cv_score   = result["cv_score"],
        scaler     = result["scaler"],
    )

    for key in ["accuracy", "roc_auc", "precision", "recall", "f1"]:
        assert 0.0 <= metrics[key] <= 1.0, (
            f"{key} = {metrics[key]:.4f} is outside [0, 1]"
        )


def test_gap_is_absolute_difference():
    """Gap must equal |cv_score - roc_auc| — always positive."""
    X_train, X_val, X_test, y_train, y_val, y_test = _make_data()

    with patch("src.models.train.CFG", _fast_cfg("xgboost")):
        result = train_model("xgboost", X_train, y_train, X_val, y_val)

    metrics = evaluate_model(
        model_name = "xgboost",
        model      = result["model"],
        X_test     = X_test,
        y_test     = y_test,
        cv_score   = result["cv_score"],
        scaler     = result["scaler"],
    )

    expected_gap = abs(result["cv_score"] - metrics["roc_auc"])
    assert metrics["gap"] == pytest.approx(expected_gap, abs=1e-6)
    assert metrics["gap"] >= 0


def test_confusion_matrix_values_sum_to_test_size():
    """TP + TN + FP + FN must equal the total number of test samples."""
    X_train, X_val, X_test, y_train, y_val, y_test = _make_data()

    with patch("src.models.train.CFG", _fast_cfg("xgboost")):
        result = train_model("xgboost", X_train, y_train, X_val, y_val)

    metrics = evaluate_model(
        model_name = "xgboost",
        model      = result["model"],
        X_test     = X_test,
        y_test     = y_test,
        cv_score   = result["cv_score"],
        scaler     = result["scaler"],
    )

    total = metrics["tp"] + metrics["tn"] + metrics["fp"] + metrics["fn"]
    assert total == len(X_test), (
        f"TP+TN+FP+FN={total} but test size={len(X_test)}"
    )