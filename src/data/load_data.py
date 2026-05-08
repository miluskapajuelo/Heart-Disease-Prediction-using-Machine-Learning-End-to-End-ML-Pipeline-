from pathlib import Path
import pandas as pd
from src.utils.config import get_path
from src.utils.helpers import check_duplicates




def load_data() -> pd.DataFrame:
    """
    Load a dataset from the heart disease CSV file.

    Returns: 
        df: A pandas Dataframe containing the loaded data.

    """
    path = get_path("raw_data")
    if not path.exists():
        raise FileNotFoundError(
            f"Raw data not found at {path}"
           )

    if path.suffix.lower() != ".csv":
        raise ValueError(
            f"Expected a .csv file, got {path.suffix}"
        )
    
    df = pd.read_csv(path)

    if df.empty:
        raise ValueError(
            f"File loaded but contains no rows {path}"
        )
    return df


def drop_duplicated(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove duplicated rows from dataset.

    Args: 
        df: dataframe from .csv.

    Returns: 
        A pandas Dataframe containing the loaded data.

    Raises:
        FileNotFoundError: if the file does not exist.
        ValueError: if the loaded dataframe is empty.
    """
   
    duplicates = check_duplicates(df)
    if duplicates:
        df = df.drop_duplicates()
        return df

