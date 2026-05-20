import pandas as pd
from src.utils.config import CFG
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
import xgboost as xgb

def _get_base_estimator(model_name: str):
    """
    """
    random_state = CFG["split"]["random_state"]

    if model_name == "logistic_regression":
        return LogisticRegression(
            # solver= "lbfgs", bigger datasets
            max_iter=1000,
            random_state=random_state
        )
    elif model_name == "random_forest":
        return RandomForestClassifier(
            n_estimators=100,
            max_depth= 10,
            random_state= random_state,
            n_jobs=1
        )
    elif model_name=="xgboost":
        return xgb.XGBClassifier(
            n_estimators=100,
            max_depth= 3,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=random_state,
            verbosity=0,
        )
    else:
        raise ValueError(
            f"unknown model {model_name}"
        )


def train_model(
        model_name,
        x_train,
        y_train,
        x_val,
        y_val 
)->dict:
    """
    Train a model with GridSearchCV and evaluate on the validation set.
 
    Args:
        model_name : "logistic_regression", "random_forest", or "xgboost"
        X_train    : Training features (after build_features)
        y_train    : Training target
        X_val      : Validation features
        y_val      : Validation target
 
    Returns:
        dict with keys:
            model       → best fitted estimator
            scaler      → fitted StandardScaler if LR, else None
            best_params → winning hyperparameter combination
            cv_score    → best CV ROC-AUC scored on training set
            val_score   → ROC-AUC on the held-out validation set
    """
    
    if model_name not in CFG["models"]:
        raise ValueError(f"Unknown model '{model_name}'")

    cv_folds = CFG["training"]["cv_folds"]
    scoring = CFG["training"]["scoring"]
    n_jobs = CFG["training"]["n_jobs"]
    grid = CFG["models"][model_name]["grid"]
    needs_scaling = CFG["models"][model_name]["needs_scaling"]

    scaler = None

    if needs_scaling:
        scaler= StandardScaler()
        x_train_in = scaler.fit_transform(x_train)
        x_val_in = scaler.transform(x_val)
    else:
        x_train_in = x_train
        x_val_in = x_val

    
    cv_strategy = StratifiedKFold(
        n_splits=cv_folds,
        shuffle=True,
        random_state=CFG['split']["random_state"]
    )
    
# GridSearchCV on training set only

    grid_search = GridSearchCV(
        estimator= _get_base_estimator(model_name),
        param_grid= grid,
        cv= cv_strategy,
        scoring=scoring,
        n_jobs=n_jobs,
        verbose=0
    )

    grid_search.fit(x_train_in, y_train)

    best_model = grid_search.best_estimator_
    cv_score = grid_search.best_score_
    best_params = grid_search.best_params_

    # Validation score

    y_proba = best_model.predict_proba(x_val_in)[:,1]
    val_score = roc_auc_score(y_val, y_proba)
    
    # print(f"  Best params : {best_params}")
    # print(f"  CV ROC-AUC  : {cv_score:.4f}  (train)")
    # print(f"  Val ROC-AUC : {val_score:.4f}  (validation)")

    return{
        "model": best_model,
        "scaler": scaler,
        "best_params": best_params,
        "cv_score": cv_score,
        "val_score": val_score
    }


def _count_combinations(grid: dict) -> int:
    """
    count total number of hyperparameter combinations in a grid
    """

    total = 1
    for values in grid.values():
        total *= len(values)
        
    return total