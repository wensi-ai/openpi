"""Robot configuration registry.

Import this module to access robot configurations through ROBOT_REGISTRY.

Example:
    from openpi.configs.robots import ROBOT_REGISTRY
    robot_config = ROBOT_REGISTRY["i3l/A1"]
"""

from pathlib import Path
import importlib

from .base_config import ROBOT_REGISTRY

# Automatically import all robot config modules to trigger registration
_current_dir = Path(__file__).parent
_robot_modules = [
    f.stem for f in _current_dir.glob("*.py")
    if f.stem not in ("__init__", "base_config")
]

for _module_name in _robot_modules:
    importlib.import_module(f".{_module_name}", package=__name__)

__all__ = ["ROBOT_REGISTRY"]
