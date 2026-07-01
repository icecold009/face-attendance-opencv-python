# config.py
import yaml
from pathlib import Path

DEFAULT_CONFIG_PATH = Path(__file__).parent / "config.yaml"

def load_config(path: str | None = None) -> dict:
    """
    Load YAML configuration for the app.
    """
    cfg_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    with cfg_path.open("r") as f:
        return yaml.safe_load(f)