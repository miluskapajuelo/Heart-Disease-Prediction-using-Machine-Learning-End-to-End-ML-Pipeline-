import json
import pickle
from pathlib import Path

import matplotlib.pyplot as plt

# Serialization

def save_json(obj, path):
    """
    Save a dict/list to disk as a .json file.

    Args:
        obj:
            the python object to serialize (must be JSON-serializable).

        path(str | Path):
            destination file path where the json file will be stored.

    Returns:
        None

    Notes:
        - Creates parent directories automatically if they do not exist.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
    print(f"  ✔ Saved  → {path}")


def save_pickle(obj, path):
    """
    save any python object to disk as a .pkl file.

    Args: 
        obj: 
            the python object to serialize and save.

        path(str | Path):
            destination file path where the pickle file will be stored.

    Returns: 
        None
    
    Notes:
        - Creates parent directories automatically if they do not exist.
        - The object is saved in binary write mode using the pickle module
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)
    print(f"saved {path}")


def load_pickle(path):
    """
    Load and return a python object from a .pkl file.

    Args: 
       path(str | Path):
           File path where the pickle file will be retrieved.

    Raises: 
        FileNotFoundError:
            if the specific file does not exist.
    
    Returns: 
        None
    
    Notes: 
    - Opens the file in binary read mode
    - Uses the 'pickle' module to deserialize the object 
        

    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"no file found at: {path}")
    with open(path, 'rb') as f:
        return pickle.load(f)
    
# Dataframes checks

def check_nulls(df):
    """
    check null values in the dataframe.

    Arg:
        df(Dataframe): 
            The Dataframe to analyze for missing values.
        
    Returns: 
        None.

    Note: 
    - Prints a message indicating whether missing values exist.
    - Null values may include NaN or missing entries.
    """
    
    null_counts = df.isnull().sum()
    total = int(null_counts)
    if total == 0:
        print("there is not null values")
    else:
        print(f"total: {total}")
        
def check_duplicates(df):
    """
    Check duplicate rows in a pandas dataframe.

    Arg:
        df(Dataframe): 
            The Dataframe to analyze for duplicate records.
    
    Returns: 
        None

    Note: 
        - Prints a message indicating whether duplicates exist
    """
    duplicated_counts = df.duplicated().sum()
    total = int(duplicated_counts)
    if total == 0:
        print("there is not null values")
    else:
        total
        print(f"total: {total}")


def save_figure(fig, path, dpi=150):
    """
    save a matplotlib figure to disck and close it to free memory.

    Args:
        fig: 
        path:
        dpi=150
    Returns: 
        None
    """

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    print(f"figure saves in {path}")

def save_pickle(obj, path):
    """
    Save any object to disk as a .pkl file.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)
    print(f"  ✔ Saved  → {path}")