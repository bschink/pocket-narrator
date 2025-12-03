# config_utils.py
from pathlib import Path
from typing import Union
import yaml


def load_yaml(path: Union[str, Path]) -> dict:
    path = Path(path)
    with path.open("r") as f:
        return yaml.safe_load(f)
