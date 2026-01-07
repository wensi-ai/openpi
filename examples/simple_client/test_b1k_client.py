import numpy as np
from openpi_client import websocket_client_policy as _websocket_client_policy

import numpy as np
from openpi_client.image_tools import resize_with_pad
from collections import deque
import logging
from openpi.shared.eval_b1k_wrapper import OpenPIWrapper

RESIZE_SIZE = 224


openpi_policy = OpenPIWrapper(
    host='10.79.12.37',
    port=8000,
    text_prompt="pick up the green mug",
    # control_mode="receeding_horizon",
)

import h5py
path = "/svl/u/mengdixu/b1k-datagen/mimicgen/datasets/demo_450.hdf5"
data = h5py.File(path, "r")
demo_0 = data["data/demo_449"]
pred_action = []
gt_actions =  demo_0["actions"]
traj_len = gt_actions.shape[0]
for idx in range(350,400):
    obs_ego = demo_0["obs/robot_r1::robot_r1:eyes:Camera:0::rgb"][idx,:,:,:3]
    obs_wrist_left = demo_0["obs/robot_r1::robot_r1:left_eef_link:Camera:0::rgb"][idx,:,:,:3]
    obs_wrist_right = demo_0["obs/robot_r1::robot_r1:right_eef_link:Camera:0::rgb"][idx,:,:,:3]
    obs = np.stack([obs_ego, obs_wrist_left, obs_wrist_right], axis=0)  #(num_cameras, H, W, C) 
    obs = obs[None, None]  #(B, T, num_cameras, H, W, C) 
    proprio = demo_0["obs/prop_state"][idx,:][None,None] # (B, T, 21)
    example = {
        "observation": obs,
        "proprio": proprio,
    }
    action = openpi_policy.act(example)
    first_action = {key: value[0] for key, value in action.items()}
    first_action = np.concatenate([v for v in first_action.values()])
    pred_action.append(first_action)

pred_action = np.stack(pred_action, axis=0)
print(pred_action[0])

#plot the first action and gt action
import matplotlib.pyplot as plt
fig, axes = plt.subplots(4, 5, figsize=(10, 10))
for i in range(4):
    for j in range(5):
        axes[i, j].plot(pred_action[:50,i*5+j], label='pred action')
        axes[i, j].plot(gt_actions[350:400,i*5+j], label='gt action')
        axes[i, j].legend()
        #set ylim to [-1, 1]
        # axes[i, j].set_ylim([-1, 1])
plt.savefig("pred_action.png")