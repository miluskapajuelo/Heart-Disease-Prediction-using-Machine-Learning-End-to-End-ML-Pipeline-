import yaml
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import RocCurveDisplay

from src.utils.config import CFG, PROJECT_ROOT

def _build_table(
        all_metrics: list, 
) -> pd.DataFrame:
    """
    Build and print a sorted comparison DataFrame metrics dicts.
    """
    rows = []
    for m in all_metrics:
        rows.append({
            "model": m["model_name"],
            "Accuracy": round(m["accuracy"]),
            "ROC-AUC": round(m["roc-auc"]),
            "Precision": round(m["precision"]),
            "Recall": round(m["recall"]),
            "F1": round(m["F1"]),
            "CV Score": round(m["CV Score"]),
            "Gap": round(m["gap"]),
        })

        comparison_df = (
            pd.DataFrame(rows)
            .set_index("Model")
            .sort_values("ROC-AUC", ascending=False)
        )

        print("\n" + "=" * 65)
        print("  MODEL COMPARISON — TEST SET")
        print("=" * 65)
        print(comparison_df.to_string())
        print("=" * 65)
        print(f"\n  Best model by ROC-AUC: {comparison_df.index[0]}")


def _plot_roc_curves(
        all_metrics: list,
        all_results: list,
        X_test: pd.DataFrame,
        Y_test: pd.Series
        ) -> None:
    """
    Plot all ROC curves on the same axes and save to reports/figures
    """
    fig, ax = plt.subplots(figsize=(7,6))

    for metrics, result in zip(all_metrics, all_results):
        model = result["model"]
        scaler = result["scaler"]

        X_in = scaler.transform(X_test) if scaler is not None else X_test
        Y_prob = model.predict_proba(X_in)[:,1]
        label = f"{metrics["model_name"]}(AUC = {metrics["roc_auc"]:3f})"

        RocCurveDisplay.from_predictions(Y_test, Y_prob, ax=ax, name=label)

        ax.plot([0, 1], [0, 1], "k--", label="Random classifier")
        ax.set_title("ROC Curves — All Models (Test Set)")
        ax.legend(loc="lower right", fontsize=8)
        plt.tight_layout()
    
        out_path = PROJECT_ROOT / CFG["paths"]["figures_dir"] / "roc_curves_all_models.png"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  ✔ ROC curves saved → {out_path}")
    
def compare_models(
    all_metrics: list,
    all_results: list,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> pd.DataFrame:
    """
    print comparison table and save overlaid ROC curves for all models.
 
    Args:
        all_metrics : list of dicts from evaluate_model()
        all_results : list of dicts from train_model()
        X_test      : test features
        y_test      : test target
 
    Returns:
        comparison_df : DataFrame sorted by ROC-AUC descending
    """
    df = _build_table(all_metrics)
    _plot_roc_curves(all_metrics, all_results, X_test, y_test)
    return df
   