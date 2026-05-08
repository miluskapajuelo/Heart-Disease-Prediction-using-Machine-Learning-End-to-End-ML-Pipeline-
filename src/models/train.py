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
        model_name: str,
        x_train_fe : pd.Dataframe,
        y_train_fe : pd.Dataframe,
        x_val_fe : pd.Dataframe,
        y_val_fe : pd.Dataframe,
)->dict:
    """
    """
    cv_folds = CFG["training"]["cv_folds"]
    scoring = CFG["training"]["scoring"]
    n_jobs = CFG["training"]["n_jobs"]
    grid = CFG["models"][model_name]["grid"]
    needs_scaling = CFG["models"][model_name]["needs_scaling"]

    scaler = None

    if needs_scaling:
        scaler= StandardScaler()
        x_train_in = scaler.fit_transform(x_train_fe)
        x_val_in = scaler.fit_transform(x_val_fe)
    else:
        x_train_in = x_train_fe
        x_val_in = x_train_fe

    
    cv_strategy = StratifiedKFold(
        n_splits=cv_folds,
        shuffle=True,
        random_state=CFG["random_state"]
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

    grid_search.fit(x_train_in, y_train_fe)

    best_model = grid_search.best_estimator_
    cv_score = grid_search.score
    best_params = grid_search.best_params_

    y_proba = best_model.predict_proba(x_val_in)[:,1]
    val_score = roc_auc_score(y_val_fe, y_proba)
    
    print(f"  Best params : {best_params}")
    print(f"  CV ROC-AUC  : {cv_score:.4f}  (train)")
    print(f"  Val ROC-AUC : {val_score:.4f}  (validation)")

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