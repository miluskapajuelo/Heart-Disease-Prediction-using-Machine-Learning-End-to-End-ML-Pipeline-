import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer

from src.utils.config import CFG



ZERO_AS_NAN_COLS = CFG["preprocessing"]["zero_as_nan_col"]
IMPUTER_STRATEGY = CFG["preprocessing"]["imputer_strategy"]
ENGINEERING_FEATURES = CFG["features"]["engineering_inputs"]

def replace_zero_with_nan(
        df:pd.DataFrame,
        columns: list[str] = ZERO_AS_NAN_COLS
) -> pd.DataFrame:
    """
    replace impossible zero values with NaN

    Args:
        X: dataframe
        columns: features which their values can not contain zeros
    Returns:
        df: dataframe with zero values replaced by NaN

    """
    df = df.copy()
    existing_cols = [col for col in columns if col in df.columns]
    df[existing_cols] = df[existing_cols].replace(0, np.nan)

    return df



def fit_imputer(
        X_train: pd.DataFrame,
        X_val: pd.DataFrame,
        X_test: pd.DataFrame,
        strategy:str = IMPUTER_STRATEGY
):
    """
    fit imputer on training data only

    Args:
        X_train: train data used to fit the imputer
        strategy: imputation strategy to use

    Returns:
        X_train_imp: imputed training features
        X_val_imp: imputed validation features (using train medians)
        X_test_imp: imputed test features (using train medians)
        imputer: a fitted SimpleImputer instance

    """

    X_train_proc = X_train.copy()
    X_test_proc = X_test.copy()
    X_val_proc =  X_val.copy()

    imputer = SimpleImputer(strategy=strategy)
    imputer.fit(X_train_proc)

    # medians learned from training set
    print(" medians learned from training set \n")
    for col, median in zip(X_train.columns, imputer.statistics_):
        print(f" {col}: {median: .2f}")

    # tranform all splits with the same fitted imputed

    X_train_imp = pd.DataFrame(
        imputer.transform(X_train_proc),
        columns=X_train.columns,
        index=X_train.index
    )

    X_val_imp = pd.DataFrame(
        imputer.transform(X_val_proc),
        columns=X_val.columns,
        index=X_val.index
    )

    X_test_imp = pd.DataFrame(
        imputer.transform(X_test_proc),
        columns=X_test.columns,
        index=X_test.index
    )

    return X_train_imp, X_test_imp, X_val_imp, imputer



def add_engineered_features(
    X_train_imp: pd.DataFrame,
    X_val_imp:   pd.DataFrame,
    X_test_imp:  pd.DataFrame,
) -> pd.DataFrame:
    """
    engineer 4 composite clinical features from imputed data

    Args:
        X_train_imp, X_val_imp, X_test_imp: imputed dataframe

    Returns:
         X_train_fe, X_val_fe, X_test_fe: dataframes with engineered features added
         oldpeak_max(float): compute from train
    
    """
    oldpeak_max = X_train_imp["oldpeak"].max()

    def _add_features(x, oldpeak_max):
        df = x.copy()
        df['cardiac_capacity']  = df['thalach'] / df['age']
        df['isquemia_score']    = (df['oldpeak'] / oldpeak_max) + df['exang'] + (df['ca'] / 3)
        df['est_stroke_volume'] = (df['trestbps'] / df['thalach']) * (df['age'] / 50)
        df['troponin_index']    = (df['oldpeak'] * 1.5) + (df['exang'] * 2) + (df['ca'] * 1.2)

        return df
    
    X_train_fe = _add_features(X_train_imp, oldpeak_max)
    X_val_fe = _add_features(X_val_imp, oldpeak_max)
    X_test_fe = _add_features(X_test_imp, oldpeak_max)


    features_selected = CFG["features"]["selected"]

    X_train_fe = X_train_fe[features_selected]
    X_val_fe = X_val_fe[features_selected]
    X_test_fe = X_test_fe[features_selected]

    return X_train_fe, X_val_fe, X_test_fe, oldpeak_max

    