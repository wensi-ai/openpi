"""Task configuration registry."""

from pathlib import Path
import importlib

# Global registry for task configurations
TASK_REGISTRY: dict[str, dict[str, str]] = {}

# Automatically import all task config modules to trigger registration
_current_dir = Path(__file__).parent
_task_modules = [
    f.stem for f in _current_dir.glob("*.py")
    if f.stem != "__init__"
]

for _module_name in _task_modules:
    importlib.import_module(f".{_module_name}", package=__name__)

__all__ = ["TASK_REGISTRY"]
