import pandas as pd
from src.utils.config import CFG

def build_features(
    X_train: pd.DataFrame,
    X_val:   pd.DataFrame,
    X_test:  pd.DataFrame,
):
    """
    Engineer 2 composite clinical features from imputed data,
    then select the 6 final features validated by ablation tests.
 
    Features engineered:
        isquemia_score    = oldpeak + exang + (ca / 3)
            → Composite ischemia risk signal
            → Note: no normalization by oldpeak_max — redundant because
              tree models are scale-invariant and LR has StandardScaler
 
        est_stroke_volume = (trestbps / thalach) * (age / 50)
            → Proxy for heart work efficiency
            → SHAP confirmed 15.1% importance — non-linear signal
 
    Features dropped (see module docstring for full justification):
        troponin_index, cardiac_capacity, oldpeak, thalach, age,
        trestbps, fbs, ca, exang, chol, restecg
 
    Final 6 features (validated by correlation + heatmap + VIF + ablation):
        sex, cp, thal, isquemia_score, est_stroke_volume, slope
 
    Args:
        X_train, X_val, X_test: Imputed DataFrames from preprocess().
 
    Returns:
        X_train_fe, X_val_fe, X_test_fe : DataFrames with 6 selected features
    """
 
    def _add_features(X):
        df = X.copy()
        df["isquemia_score"]    = df["oldpeak"] + df["exang"] + (df["ca"] / 3)
        df["est_stroke_volume"] = (df["trestbps"] / df["thalach"]) * (df["age"] / 50)
        return df
 
    X_train_fe = _add_features(X_train)
    X_val_fe   = _add_features(X_val)
    X_test_fe  = _add_features(X_test)
 
    print(f"  Features before engineering : {X_train.shape[1]}")
    print(f"  Features after engineering  : {X_train_fe.shape[1]}")
 
    # ── Select only final validated feature set ───────────────────────────────
    selected = CFG["features"]["selected"]
    # ["sex", "cp", "thal", "isquemia_score", "est_stroke_volume", "slope"]
 
    X_train_fe = X_train_fe[selected]
    X_val_fe   = X_val_fe[selected]
    X_test_fe  = X_test_fe[selected]
 
    print(f"  Features selected for models: {selected}")
 
    # ── Confirm no nulls in output ────────────────────────────────────────────
    for name, X in [("X_train_fe", X_train_fe), ("X_val_fe", X_val_fe), ("X_test_fe", X_test_fe)]:
        nulls = X.isnull().sum().sum()
        status = "✔" if nulls == 0 else f"✘  {nulls} nulls!"
        print(f"  Nulls in {name}: {status}")
 
    return X_train_fe, X_val_fe, X_test_fe