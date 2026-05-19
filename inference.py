import pandas as pd
import numpy as np
from src.utils.config  import CFG, PROJECT_ROOT
from src.utils.helpers import load_pickle

models_dir = PROJECT_ROOT / CFG["paths"]["models_dir"]


model = load_pickle(models_dir / "best_model.pkl")
imputer = load_pickle(models_dir / "preprocessor.pkl")



# Best model name for display
best_model_name_path = models_dir / "best_model_name.txt"
if best_model_name_path.exists():
    best_model_name = best_model_name_path.read_text().strip()
else:
    best_model_name = type(model).__name__
 
print(f"  Model loaded   : {best_model_name}")

 
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
 
    threshold = CFG["evaluation"]["decision_threshold"]   # 0.5
 
    if probability < 0.30:
        risk_level = "LOW"
        risk_label = "Low Risk"
    elif probability < threshold:
        risk_level = "MODERATE"
        risk_label = "Moderate Risk"
    else:
        risk_level = "HIGH"
        risk_label = "High Risk"
 
    return {
        "probability": round(probability, 4),
        "prediction":  prediction,
        "risk_label":  risk_label,
        "risk_level":  risk_level,
    }
 
 
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
    