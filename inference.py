import json
import pandas as pd
import numpy as np
from src.utils.config  import CFG, PROJECT_ROOT, RISK_BANDS
from src.utils.helpers import _explain_patient, load_pickle
import shap

models_dir = PROJECT_ROOT / CFG["paths"]["models_dir"]

model = load_pickle(models_dir / "best_model.pkl")
imputer = load_pickle(models_dir / "preprocessor.pkl")
explainer = shap.TreeExplainer(model)

# Model metadata (name, version, metrics, etc.) saved during training
metadata_path = models_dir / "model_metadata.json"
if metadata_path.exists():
    with open(metadata_path) as f:
        metadata = json.load(f)
    best_model_name = metadata["model_name"]

else:
    metadata = {}
    best_model_name = type(model).__name__

print(f"  Model loaded   : {best_model_name}")

def classify(probability: float) -> tuple[str, str]:
    """Devuelve (level, label) según la banda en la que cae la probabilidad."""
    for band in CFG["risk_bands"]:
        if probability < band["max"]:
            return band["level"], band["label"]
    last = CFG["risk_bands"][-1]
    return last["level"], last["label"]
 
def _preprocess_patient(X: pd.DataFrame, imputer) -> pd.DataFrame:
    """
    Apply the same preprocessing used during training:
      1. Replace impossible zeros with NaN
      2. Impute using training medians (from fitted imputer)
    """
    zero_cols = CFG["preprocessing"]["zero_as_nan_col"]   # ["chol", "trestbps", "thalach"]
 
    X_proc = X.copy()
    # Only replace zeros in columns that exist in the input
    cols_to_replace = [c for c in zero_cols if c in X_proc.columns]
    X_proc[cols_to_replace] = X_proc[cols_to_replace].replace(0, np.nan)
 
    X_imp = pd.DataFrame(
        imputer.transform(X_proc),
        columns=X_proc.columns,
    )
    return X_imp
 
 
def _build_features_patient(X: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer the 2 composite features and select the 6 final features.
    Must mirror src/data/preprocess.py build_features() exactly.
    """
    df = X.copy()
 
    # Engineer composite features
    df["isquemia_score"]    = df["oldpeak"] + df["exang"] + (df["ca"] / 3)
    df["est_stroke_volume"] = (df["trestbps"] / df["thalach"]) * (df["age"] / 50)
 
    # Select exactly the 6 final features
    selected = CFG["features"]["selected"]
    return df[selected]

def classify(probability: float) -> tuple:
    """
    Map a probability to (risk_level, risk_label, recommendations) using
    RISK_BANDS loaded from configs/risk_bands.yaml.
 
    Walks the bands in order and returns the first one where
    probability < band["max"]. Bands must be in strictly ascending `max`
    order for this to work — enforced at import time in
    src/utils/config.py, so a misordered config fails fast instead of
    silently misclassifying patients.
 
    The fallback (`bands[-1]`) covers probability == 1.0 exactly, which
    fails the `<` check on the last band even though it should always
    land there.
    """
    for band in RISK_BANDS:
        if probability < band["max"]:
            return band["level"], band["label"], band["recommendations"]
    last = RISK_BANDS[-1]
    return last["level"], last["label"], last["recommendations"]
 

 
 
# ── Main predict function ─────────────────────────────────────────────────────
 
def predict(patient: dict) -> dict:
    """
    Predict heart disease risk for a single patient.
 
    Args:
        patient: dict with 13 raw clinical measurements.
                 Keys must match the original feature names.
                 Missing values should be 0 for physiologically
                 impossible columns (chol, trestbps, thalach),
                 or np.nan for truly unknown values.
 
    Returns:
        dict with:
            probability  → float [0.0, 1.0] — P(heart disease)
            prediction   → int   [0 or 1]   — 0=No Disease, 1=Disease
            risk_label   → str              — "Low Risk", "High Risk"
            risk_level   → str              — "LOW", "MODERATE", "HIGH"
    """
 
    X = pd.DataFrame([patient])
 
    X_imp = _preprocess_patient(X, imputer)
 
    X_fe = _build_features_patient(X_imp)
 
    X_in =  X_fe
 
    prediction  = int(model.predict(X_in)[0])
    probability = float(model.predict_proba(X_in)[0, 1])
    model_confidence, confidence_std = _model_confidence(model, X_in)

    risk_level, risk_label,recommendations = classify(probability)

    feature_contributions = _explain_patient(X_in, explainer)
 
    return {
        "probability": round(probability, 4),
        "prediction":  prediction,
        "risk_label":  risk_label,
        "risk_level":  risk_level,
        "recommendations" : recommendations,
        "model_confidence": model_confidence,
        "confidence_std" : confidence_std,
        "feature_contributions": feature_contributions,
    }

def _model_confidence(model, X_in: pd.DataFrame) -> tuple:
    """
    Model confidence based on agreement between the Random Forest's trees.

    """
    if not hasattr(model, "estimators_"):
        return None, None
 
    tree_probs = np.array([
        tree.predict_proba(X_in.to_numpy())[0, 1] for tree in model.estimators_
    ])
    std = float(tree_probs.std())
 
    if std < 0.155:
        level = "High"
    elif std < 0.423:
        level = "Moderate"
    else:
        level = "Low"
 
    return level, round(std, 4)
 
 
def print_result(patient: dict, result: dict) -> None:
    """Print a formatted prediction result."""
    print(f"\n{'─' * 50}")
    print(f"  HEART DISEASE RISK ASSESSMENT")
    print(f"{'─' * 50}")
    print(f"  Model          : {best_model_name}")
    print(f"  P(disease)     : {result['probability']:.2%}")
    print(f"  Prediction     : {'Disease' if result['prediction'] == 1 else 'No Disease'}")
    print(f"  Risk Level     : {result['risk_level']}")
    print(f"  Risk Label     : {result['risk_label']}")
    print(f"{'─' * 50}")
    print()
    print("  Input features:")
    for k, v in patient.items():
        print(f"    {k:<12}: {v}")
    print()
    print("  ⚠ This is a research tool only.")
    print("  ⚠ Not validated for clinical use.")
    print(f"{'─' * 50}\n")
 
 
# ── Example patients ──────────────────────────────────────────────────────────
 
if __name__ == "__main__":
 
    # Patient 1 — high risk profile
    # Male, 63, typical angina, high ST depression, blocked vessels
    patient_high_risk = {
        "age":      63,
        "sex":       1,    # male
        "cp":        3,    # asymptomatic (paradoxically high risk in this dataset)
        "trestbps": 145,
        "chol":     233,
        "fbs":       1,
        "restecg":   0,
        "thalach":  150,
        "exang":     0,
        "oldpeak":  2.3,
        "slope":     0,
        "ca":        0,
        "thal":      1,
    }
 
    # Patient 2 — low risk profile
    # Female, 45, no angina, normal ECG, good heart rate
    patient_low_risk = {
        "age":      45,
        "sex":       0,    # female
        "cp":        2,    # non-anginal pain
        "trestbps": 110,
        "chol":     264,
        "fbs":       0,
        "restecg":   1,
        "thalach":  132,
        "exang":     0,
        "oldpeak":  1.2,
        "slope":     1,
        "ca":        0,
        "thal":      2,
    }
 
    print("\n" + "=" * 50)
    print("  Running inference on 2 example patients")
    print("=" * 50)
 
    result_1 = predict(patient_high_risk)
    print_result(patient_high_risk, result_1)
 
    result_2 = predict(patient_low_risk)
    print_result(patient_low_risk, result_2)
    