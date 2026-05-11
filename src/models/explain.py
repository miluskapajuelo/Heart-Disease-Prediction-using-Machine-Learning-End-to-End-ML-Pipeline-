import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
 
from src.utils.config import CFG, PROJECT_ROOT
import shap
 
def explain_model(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    patient_idx: int = 0,
) -> pd.DataFrame:
    """
    Run SHAP analysis on the best model and save all explanation figures.
 
    Args:
        model       : Fitted XGBoost estimator (best model from compare_models)
        X_test      : Test features (after build_features, selected columns)
        y_test      : Test target
        patient_idx : Index of the patient to explain with a force plot (default 0)
 
    Returns:
        shap_importance : DataFrame with mean |SHAP| and share of total per feature
    """
 
    figures_dir = PROJECT_ROOT / CFG["paths"]["figures_dir"]
    figures_dir.mkdir(parents=True, exist_ok=True)
 
    # ── Compute SHAP values ───────────────────────────────────────────────────
    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
 
    # ── 1. Bar chart — global feature importance ──────────────────────────────
    print("  Generating SHAP bar chart (global importance)...")
    fig_bar, ax_bar = plt.subplots(figsize=(8, 5))
    shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
    plt.tight_layout()
    _save(fig_bar, figures_dir / "shap_bar.png")
 
    # ── 2. Beeswarm — direction and magnitude per patient ─────────────────────
    print("  Generating SHAP beeswarm plot...")
    fig_bee, ax_bee = plt.subplots(figsize=(8, 5))
    shap.summary_plot(shap_values, X_test, show=False)
    plt.tight_layout()
    _save(fig_bee, figures_dir / "shap_beeswarm.png")
 
    # ── 3. SHAP importance table ──────────────────────────────────────────────
    mean_abs = np.abs(shap_values).mean(axis=0)
    total    = mean_abs.sum()
 
    shap_importance = (
        pd.DataFrame({
            "feature":       X_test.columns,
            "mean_abs_shap": mean_abs,
        })
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )
    shap_importance["share"] = (shap_importance["mean_abs_shap"] / total).map("{:.1%}".format)
 
    print("\n  SHAP Importance Table:")
    print(shap_importance.to_string(index=False))
 
    top = shap_importance.iloc[0]
    print(f"\n  Top feature '{top['feature']}' accounts for {top['share']} of total SHAP importance.")
 
    # ── 4. Force plot — single patient ────────────────────────────────────────
    patient    = X_test.iloc[patient_idx]
    true_label = y_test.iloc[patient_idx]
    pred_label = model.predict(X_test.iloc[[patient_idx]])[0]
    pred_proba = model.predict_proba(X_test.iloc[[patient_idx]])[0, 1]
 
    print(f"\n  Force plot — patient {patient_idx}:")
    print(f"    True label     : {true_label}")
    print(f"    Predicted      : {pred_label}")
    print(f"    P(disease)     : {pred_proba:.2%}")
 
    shap.force_plot(
        explainer.expected_value,
        shap_values[patient_idx],
        patient,
        matplotlib=True,
        show=False,
    )
    plt.tight_layout()
    _save(plt.gcf(), figures_dir / "shap_force_patient_0.png")
 
    return shap_importance
 
def _save(fig, path: Path, dpi: int = 150):
    """Save a figure and close it."""
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✔ Saved → {path}")