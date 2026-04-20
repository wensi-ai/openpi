"""Base configuration dataclasses for robot configs."""

from dataclasses import dataclass
from typing import Dict, List, Optional


# Global registry for robot configurations
ROBOT_REGISTRY: Dict[str, "RobotConfig"] = {}


def register_robot(name: str, config: "RobotConfig") -> "RobotConfig":
    """Register a robot config in the global registry.
    
    Args:
        name: Robot identifier in format 'bucket/robot_type' (e.g., 'i3l/A1')
        config: RobotConfig instance to register
        
    Returns:
        The same config instance (for chaining)
    """
    ROBOT_REGISTRY[name] = config
    return config


@dataclass
class ObservationConfig:
    """Configuration for a single camera observation view."""
    name: str               # unique identifier for the observation
    obs_key: str            # full name in the obsevation dict during inference
    dataset_key: str        # this is the name of the key in LeRobot datasets
    resolution: List[int]   # [width, height]


@dataclass
class StateActionConfig:
    """Configuration for action / proprioception data."""
    name: str                           # Name for this action/proprio component
    indices: Optional[List[int]] = None # Indices in the full action/proprio array
    is_eef: bool = False                # Whether this corresponds to an end-effector command (gripper, dexhand, etc.)
    needs_delta_comp: bool = False      # [Action Only] Whether to compute delta from previous step


@dataclass
class RobotConfig:
    """Base configuration for a robot.
    
    This matches the structure defined in i3l.yaml robot configurations.
    """
    name: str
    robot_type: str
    observations: Dict[str, ObservationConfig]
    action_key: str
    action_dim: int
    action: List[StateActionConfig]
    proprio: List[StateActionConfig]
    
    def __post_init__(self):
        """Convert dictionaries to proper dataclass instances if needed."""
        # Convert observation configs if they're still dicts
        if self.observations:
            self.observations = {
                key: ObservationConfig(**val) if isinstance(val, dict) else val
                for key, val in self.observations.items()
            }
        
        # Convert action and proprio configs if they're still dicts
        self.proprio = [
            StateActionConfig(**item) if isinstance(item, dict) else item
            for item in self.proprio
        ]
        self.action = [
            StateActionConfig(**item) if isinstance(item, dict) else item
            for item in self.action
        ]
        
        # Validate action_dim matches sum of action indices
        total_dim = sum([len(a.indices) for a in self.action])
        if total_dim != self.action_dim:
            raise ValueError(
                f"Sum of action indices ({total_dim}) doesn't match "
                f"action_dim ({self.action_dim})"
            )

