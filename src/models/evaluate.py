import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    roc_auc_score
)

def evaluate_model(
        model_name: str,
        model, 
        x_test: pd.DataFrame,
        y_test: pd.Series,
        cv_score: float,
        scaler=None
) -> dict:
    """
    evaluates a trained classification model on the test dataset.

    Applies scaling if a scaler is provided, generates predictions,
    computes evaluation metrics, prints a classification report,
    and checks for possible overfitting by comparing the test ROC-AUC
    score against the cross-validation ROC-AUC score.

    Args:
        model_name (str): Name of the model being evaluated.
        model: Trained classification model implementing
            predict() and predict_proba().
        x_test (pd.DataFrame): Test feature dataset.
        y_test (pd.Series): True labels for the test dataset.
        cv_score (float): Cross-validation ROC-AUC score obtained during training.
        scaler (optional): Fitted scaler used to transform the test
            data before evaluation. Defaults to None.

    Returns:
        dict: Dictionary containing evaluation metrics including:
            - model_name
            - accuracy
            - roc-auc
            - precision
            - recall
            - f1
            - cv-score
            - gap
    """
    
    # apply scaling if needed
    x_in = scaler.transform(x_test) if scaler is not None else x_test

    # predictions
    y_pred = model.predict(x_in)
    y_proba = model.predict_proba(x_in)[:,1]

    # metrics
    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)
    report = classification_report(y_test, y_pred, output_dict=True)
    gap = abs(cv_score - roc_auc)

    precision = report["1"]["precision"]
    recall = report["1"]["recall"]
    f1 = report["1"]["f1-score"]

    # overfitting flag
    threshold = 0.05
    gap_label = "good generalization" if gap<threshold else "possible overfitting"

    # print report
    print(f"\n{'─' * 55}")
    print(f"{model_name}")
    print(f"\n{'─' * 55}")
    print(f" Accuracy : {accuracy:.4f}")
    print(f" ROC-AUC : {roc_auc:.4f}")
    print(f" Precision : {precision:.4f}")
    print(f" Recall : {recall:.4f}")
    print(f" F1 : {f1:.4f}")
    print(f" CV ROC-AUC : {cv_score:.4f} (train)")
    print(f" Gap : {gap:.4f} -> {gap_label}")
    print()
    print( classification_report(
        y_test, y_pred,
        target_names=["No Disease", "Disease"]
    ))

    return {
        "model_name": model_name,
        "accuracy": accuracy,
        "roc-auc": roc_auc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "cv-score": cv_score,
        "gap": gap,
    }



