import numpy as np
import random
import logging
import lerobot.datasets.lerobot_dataset as lerobot_dataset
from typing import Dict, Callable

logging.basicConfig(level=logging.INFO)


def cotrain(data_ratio: np.ndarray) -> np.ndarray:
    """Keep the data ratio as is for cotraining"""
    assert data_ratio.size == 4 and np.isclose(np.sum(data_ratio), 1.0), "Data ratio should have 4 elements summing to 1.0, got {}".format(data_ratio)
    return data_ratio / data_ratio.sum()

def hg_daggar(data_ratio: np.ndarray) -> np.ndarray:
    """Remove on-policy data for HG-DAGGER"""
    assert data_ratio.size == 4 and np.isclose(np.sum(data_ratio), 1.0), "Data ratio should have 4 elements summing to 1.0, got {}".format(data_ratio)
    adjusted_ratio = data_ratio.copy()
    adjusted_ratio[1] = 0.0  # pre-intervention data
    adjusted_ratio[3] = 0.0  # policy data
    return adjusted_ratio / adjusted_ratio.sum()

def sirius(data_ratio: np.ndarray) -> np.ndarray:
    """For SIRIUS, We keep human demo as is, set pre-intervention to 0, intervention to 0.5, and policy as the res"""
    assert data_ratio.size == 4 and np.isclose(np.sum(data_ratio), 1.0), "Data ratio should have 4 elements summing to 1.0, got {}".format(data_ratio)
    adjusted_ratio = np.zeros_like(data_ratio)
    adjusted_ratio[0] = min(data_ratio[0], 0.5)  # human demo
    adjusted_ratio[2] = 0.0 if data_ratio[2] == 0.0 else 0.5  # intervention
    adjusted_ratio[3] = 0.0 if data_ratio[3] == 0.0 else 1.0 - adjusted_ratio[0] - adjusted_ratio[2]  # policy
    return adjusted_ratio / adjusted_ratio.sum()

def i3l(data_ratio: np.ndarray) -> np.ndarray:
    """For I3l, We keep human demo as is, set pre-intervention to 0, intervention to 0.25, and policy as the res"""
    assert data_ratio.size == 4 and np.isclose(np.sum(data_ratio), 1.0), "Data ratio should have 4 elements summing to 1.0, got {}".format(data_ratio)
    adjusted_ratio = np.zeros_like(data_ratio)
    adjusted_ratio[0] = min(data_ratio[0], 0.70)  # human demo
    adjusted_ratio[2] = 0.0 if data_ratio[2] == 0.0 else 0.30  # intervention
    adjusted_ratio[3] = 0.0 if data_ratio[3] == 0.0 else 1.0 - adjusted_ratio[0] - adjusted_ratio[2]  # policy
    return adjusted_ratio / adjusted_ratio.sum()


REWEIGHTING_STRATEGIES: Dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "cotrain": cotrain,
    "hg_daggar": hg_daggar,
    "sirius": sirius,
    "i3l": i3l,
}


class I3LLeRobotDataset(lerobot_dataset.MultiLeRobotDataset):
    def __init__(
        self, 
        sampling_mode: bool = False,
        reweight_strategy: str | None = "sirius",
        *args, 
        **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        # ====== Sampling specific initialization ======
        self.sampling_mode = sampling_mode
        if self.sampling_mode:
            # set take over data ratio
            self.reweight_strategy = reweight_strategy
            # Load sampling indices
            self._human_demo_data, self._pre_int_data, self._int_data, self._policy_data = [], [], [], []
            idx_offset = 0
            for dataset in self._datasets:
                for idx in range(dataset.num_frames):
                    if dataset.hf_dataset[idx]["int_state"] == 0:
                        self._human_demo_data.append(idx + idx_offset)
                    elif dataset.hf_dataset[idx]["int_state"] == 1:
                        self._pre_int_data.append(idx + idx_offset)
                    elif dataset.hf_dataset[idx]["int_state"] == 2:
                        self._int_data.append(idx + idx_offset)
                    else:
                        self._policy_data.append(idx + idx_offset)
                idx_offset += dataset.num_frames

            # Now, set up the data sampling ratio
            data_ratio = np.array([
                len(self._human_demo_data), len(self._pre_int_data), len(self._int_data), len(self._policy_data)
            ])
            total_data_points = data_ratio.sum()
            data_ratio = data_ratio / np.sum(data_ratio)
            if self.reweight_strategy is not None and self.reweight_strategy in REWEIGHTING_STRATEGIES:
                adjusted_ratio = REWEIGHTING_STRATEGIES[self.reweight_strategy](data_ratio)
                self._intervention_data_ratio = adjusted_ratio
            else:
                logging.warning(f"Reweighting strategy {self.reweight_strategy} not recognized! Using original data ratio.")
                self._intervention_data_ratio = data_ratio
            self._intervention_data_divider = np.cumsum(self._intervention_data_ratio)
            logging.info(
                f"""\n
                Reweight strategy: {self.reweight_strategy},
                Total data points: {total_data_points}, 
                Initial data ratio: {np.round(data_ratio, 3)},
                Data sampling ratio: {np.round(self._intervention_data_ratio, 3)}\n
                """
            )

    def __getitem__(self, index: int):
        if self.sampling_mode:
            # randomly sample a data point
            data_seed = random.random()
            if 0 <= data_seed < self._intervention_data_divider[0] and len(self._human_demo_data) > 0:
                index = random.choice(self._human_demo_data)
            elif self._intervention_data_divider[0] <= data_seed < self._intervention_data_divider[1] and len(self._pre_int_data) > 0:
                index = random.choice(self._pre_int_data)
            elif self._intervention_data_divider[1] <= data_seed < self._intervention_data_divider[2] and len(self._int_data) > 0:
                index = random.choice(self._int_data)
            else:
                index = random.choice(self._policy_data)
        return super().__getitem__(index)