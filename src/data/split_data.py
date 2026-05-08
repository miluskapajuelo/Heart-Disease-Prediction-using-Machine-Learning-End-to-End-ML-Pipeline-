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

    target = CFG["split"]["target_column"]
    test_size = CFG["split"]["test_size"]
    val_size = CFG["split"]["val_size"]
    ramdom_state = CFG["split"]["random_state"]

    X = df.drop(columns=[target])
    Y = df[target]


    # first cut: isolate test set until final evaluation

    X_temp, X_test, Y_temp, Y_test = train_test_split(
        X, Y,
        test_size= test_size,
        stratify= Y,
        random_state=ramdom_state

    )

    # second cut: split remainder into train and val

    X_train, X_val, Y_train, Y_val = train_test_split(
        X_temp, Y_temp,
        test_size= val_size,
        stratify=Y_temp,
        random_state= ramdom_state
    )

    splits = {
        "train":(X_train, Y_train),
        "val":(X_val, Y_val),
        "test":(X_test, Y_test),
    }

    total = sum(len(X) for X,_ in splits.values())

    for name, x, y in splits.values():
        print(f"{name}:      {x.shape[0]:>4} samples  ({y.mean():.1%} positive)")