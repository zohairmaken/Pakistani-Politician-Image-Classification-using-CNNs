import yaml
import os
from pathlib import Path

def load_config(config_path: str = "configs/config.yaml") -> dict:
    """Load configuration from a YAML file."""
    # Find config file relative to project root
    # Try absolute path first
    if os.path.isabs(config_path):
        path = Path(config_path)
    else:
        # Search in current dir and its parents
        path = Path(os.getcwd()) / config_path
        if not path.exists():
            # Fallback to absolute base_dir from environment if needed
            # For now assume it's in the current directory or relative
            pass

    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found at {path}")

    with open(path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config

def get_config_val(config: dict, key_path: str, default=None):
    """Get a nested configuration value using dot notation (e.g., 'paths.data_dir')."""
    keys = key_path.split('.')
    val = config
    for key in keys:
        if isinstance(val, dict) and key in val:
            val = val[key]
        else:
            return default
    return val
