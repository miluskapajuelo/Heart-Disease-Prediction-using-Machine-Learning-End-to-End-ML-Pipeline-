from pathlib import Path

import yaml

# works regarless of where the script is run from
PROJECT_ROOT = Path(__file__).resolve().parents[2]

_config_path = PROJECT_ROOT / "configs" / "model_config_yaml"

with open(_config_path, "r") as f:
    CFG: dict = yaml.safe_load(f)

def get_path(key:str) -> Path:
    """
    Shortcut to run a relative path from CFG into an absolute path
    
    Args: 
        key(str): key name that is used as a key in a dict
    
    Returns: 
        path: the directory of the key
    
    Example:
        get_path("raw_data") -> PROJECT_ROOT / "data/raw/heart.csv"
    """

    path = PROJECT_ROOT / CFG["paths"][key]
    return path