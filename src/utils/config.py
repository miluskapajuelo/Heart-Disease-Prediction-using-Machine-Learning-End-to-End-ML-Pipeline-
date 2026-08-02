from pathlib import Path

import yaml

# works regarless of where the script is run from
PROJECT_ROOT = Path(__file__).resolve().parents[2]

_config_path = PROJECT_ROOT / "configs" / "model_config.yaml"

with open(_config_path, "r") as f:
    CFG: dict = yaml.safe_load(f)

_risk_bands_path = PROJECT_ROOT / "configs" / "risk_bands.yaml"

with open(_risk_bands_path, "r") as f:
    _risk_bands_dfg : dict = yaml.safe_load(f)

RISK_BANDS: list = _risk_bands_dfg["risk_bands"]

for _band in RISK_BANDS:
    if _band["level"] == "MODERATE" and "max" not in _band:
        _band["max"] = CFG["evaluation"]["decision_threshold"]

_maxes = [b["max"] for b in RISK_BANDS]
assert _maxes == sorted(_maxes), (f"risk_bands.yaml: bands must have ascending `max` values, got {_maxes}")

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