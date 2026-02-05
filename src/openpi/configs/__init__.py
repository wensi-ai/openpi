"""OpenPI configuration modules."""

from .robots import ROBOT_REGISTRY
from .tasks import TASK_REGISTRY

__all__ = ["ROBOT_REGISTRY", "TASK_REGISTRY"]