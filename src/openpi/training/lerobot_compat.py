"""Compatibility imports for LeRobot dataset APIs."""

from typing import Any

try:
    from lerobot.datasets import LeRobotDataset
    from lerobot.datasets import LeRobotDatasetMetadata
    from lerobot.datasets import MultiLeRobotDataset
except ImportError:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
    from lerobot.datasets.lerobot_dataset import MultiLeRobotDataset

__all__ = ["LeRobotDataset", "LeRobotDatasetMetadata", "MultiLeRobotDataset", "tasks_from_metadata"]


def tasks_from_metadata(metadata: Any) -> dict[int, str]:
    return metadata.tasks
