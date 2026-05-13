import pandas as pd
from sklearn.model_selection import train_test_split

from src.utils.config import CFG

def split_data(df: pd.DataFrame):
    """
    Split  df into stratified train/val/test sets

    Arg:
        df: Clean Dataframe from load_data()

    Returns: 
        X_train, X-val, X-test: Feature Dataframe
        Y_train, Y-val, Y-test: Target series
    """

    target = CFG["split"]["target_column"] #target
    test_size = CFG["split"]["test_size"] # 20%
    val_size = CFG["split"]["val_size"] # 25%
    random_state = CFG["split"]["random_state"] #42

    df_no_target = df.drop(columns=[target])
    df_target = df[target]


    # first cut: isolate test set until final evaluation
    X_temp, X_test, Y_temp, Y_test = train_test_split(
        df_no_target, 
        df_target,
        test_size= test_size,
        stratify= df_target,
        random_state=random_state

    )

    #X_temp : all the df without test 0.2, all the features no target
    #Y_temp : all the df without test 0.2, just target

    # second cut: split remainder into train and val
    X_train, X_val, Y_train, Y_val = train_test_split(
        X_temp, 
        Y_temp,
        test_size= val_size, # 0.25
        stratify=Y_temp,
        random_state= random_state
    )

    return X_train, X_val, X_test, Y_train, Y_val, Y_test