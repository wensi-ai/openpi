"""Convert molmospaces drawer datagen output into a LeRobot dataset for
pi0.5-droid fine-tuning.

Input layout (from scripts/datagen/data_gen.sh in molmospaces):

    <data_dir>/
        trajectories_batch_1_of_1.h5                     # multiple traj_i groups
        episode_{NNNNNNNN}_exo_camera_1_batch_1_of_1.mp4
        episode_{NNNNNNNN}_wrist_camera_batch_1_of_1.mp4
        episode_{NNNNNNNN}_..._{depth,seg,robot_rgb}.mp4  # unused here
        experiment_config_*.pkl
        running_log.log

Each traj_i stores per-step dicts as JSON encoded in uint8 (T, N) blobs:

    obs/agent/qpos            -> {"arm": [7], "base": [], "gripper": [2]}
    actions/commanded_action  -> {"arm": [7], "base": [], "gripper": [1]}
    obs_scene                 -> JSON with task_description, policy_dt_ms, ...

The corresponding video frames live in the per-episode mp4 files.

Output: a LeRobot dataset with the schema expected by our
pi05_droid_renderscale training config (LeRobotMolmospacesDroidDataConfig).
The action target is the raw planner joint-position command (same command
mode as FrankaRobotConfig in molmospaces, which is what the eval_policy
runs), shifted by one so obs[t] is paired with commanded_action[t+1] (the
molmospaces LastActionSensor stores the *last* action taken, so index 0 is
empty and index t is what was applied between obs[t-1] and obs[t]).

    exterior_image_1_left  (180, 320, 3) uint8      from exo_camera_1
    wrist_image_left       (180, 320, 3) uint8      from wrist_camera
    joint_position         (7,)  float32            qpos.arm[:7]  (radians)
    gripper_position       (1,)  float32            qpos.gripper[0] / 0.824033 clipped [0,1]
    actions                (8,)  float32            [7D commanded arm joint pos, 1D gripper in {0,1}]

Usage:
    uv run examples/molmospaces/convert_molmodata_to_lerobot.py \\
        --data_dir /viscam/projects/egodex/renderscale/molmodata/drawer \\
        --repo_id renderscale/drawer_open \\
        --overwrite
"""

from __future__ import annotations

import json
import re
import shutil
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import h5py
import numpy as np
import tyro
from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME, LeRobotDataset
from tqdm import tqdm


DEFAULT_REPO_ID = "renderscale/drawer_open"
IMG_H, IMG_W = 180, 320
# Matches pi_policy.py:obs_to_model_input — gripper state is normalized by
# this constant and clipped to [0, 1] at both train and eval time.
GRIPPER_OBS_SCALE = 0.824033
# Raw commanded gripper action is 0 or 255; we normalize to 0 or 1 so the
# model's sigmoid-style output matches the binary threshold used at eval.
GRIPPER_ACTION_SCALE = 255.0


def _decode_jsons(ds: h5py.Dataset) -> list[dict]:
    out = []
    for row in ds[:]:
        b = bytes(row).rstrip(b"\x00")
        out.append(json.loads(b.decode("utf-8")) if b else {})
    return out


def _load_mp4_frames(path: Path, n_expected: int) -> np.ndarray:
    """Read up to n_expected frames from path. Generated-dataset variant:
    if the mp4 is short (sim2real inference may have been killed mid-clip,
    leaving a truncated mp4), return only the frames that exist — caller is
    responsible for truncating other per-step arrays to the same length.
    """
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {path}")
    buf = []
    while len(buf) < n_expected:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (IMG_W, IMG_H), interpolation=cv2.INTER_AREA)
        buf.append(frame)
    cap.release()
    return np.stack(buf, axis=0) if buf else np.zeros((0, IMG_H, IMG_W, 3), dtype=np.uint8)


def _episode_id_from_traj_name(name: str) -> int:
    m = re.match(r"traj_(\d+)$", name)
    if not m:
        raise ValueError(name)
    return int(m.group(1))


# Matches trajectories_batch_<i>_of_<N>.h5 (single- or multi-shard) plus an
# optional alphanumeric/underscore suffix used to keep differently-configured
# runs in the same output dir (e.g. trajectories_batch_1_of_4_cam_rand.h5).
# The captured batch tag includes the suffix so it flows through to the mp4
# filename lookup (episode_*_<camera>_<batch_tag>.mp4) without further changes.
_H5_NAME_RE = re.compile(r"trajectories_(batch_\d+_of_\d+(?:_[A-Za-z0-9_]+)?)\.h5$")


def _batch_tag_from_h5(h5_path: Path) -> str:
    m = _H5_NAME_RE.search(h5_path.name)
    if not m:
        raise ValueError(
            f"h5 filename does not match trajectories_batch_<i>_of_<N>.h5: {h5_path}"
        )
    return m.group(1)


def _episode_mp4(data_dir: Path, ep: int, camera: str, batch_tag: str) -> Path:
    return data_dir / f"episode_{ep:08d}_{camera}_{batch_tag}.mp4"


def _load_episode(job: tuple[str, str, int, str]) -> dict:
    """Worker: read one traj's videos + per-step JSON and return arrays ready
    to be streamed into LeRobotDataset in the main process. Kept as a
    top-level function so ProcessPoolExecutor can pickle it.
    """
    h5_path, traj_key, ep_id, batch_tag = job
    data_dir = Path(h5_path).parent

    with h5py.File(h5_path, "r", swmr=True) as f:
        t = f[traj_key]
        qpos = _decode_jsons(t["obs/agent/qpos"])
        cmd = _decode_jsons(t["actions/commanded_action"])
        scene = json.loads(bytes(t["obs_scene"][()]).rstrip(b"\x00").decode("utf-8"))
        T = len(qpos)

    exo = _load_mp4_frames(_episode_mp4(data_dir, ep_id, "exo_camera_1", batch_tag), T)
    wrist = _load_mp4_frames(_episode_mp4(data_dir, ep_id, "wrist_camera", batch_tag), T)
    # Sim2real inference may have written a truncated mp4 if it was killed
    # mid-clip. Cap T at the shorter of the two videos so the per-step arrays
    # below align with the actual frames available.
    n_video = min(len(exo), len(wrist))
    if n_video == 0:
        raise RuntimeError(f"{traj_key}: zero frames in generated mp4(s)")
    if n_video < T:
        qpos = qpos[:n_video]
        cmd = cmd[: n_video + 1]  # cmd[1:n_video+1] feeds the n_video-1 valid pairs below
        T = n_video

    arm_qpos = np.asarray([q["arm"] for q in qpos], dtype=np.float32)        # (T, 7)
    grip_obs_raw = np.asarray([q["gripper"][0] for q in qpos], dtype=np.float32)
    grip_obs = np.clip(grip_obs_raw / GRIPPER_OBS_SCALE, 0.0, 1.0)           # (T,)

    # commanded_action stores the LAST action applied to the robot:
    #     obs[t-1] --[commanded_action[t]]--> obs[t]
    # so index 0 is empty (no action has been taken at reset). The training
    # pair for obs[t] is commanded_action[t+1] (the action executed from
    # obs[t]). This shifts the action by one and drops obs[T-1] (no future
    # action). The arm command is joint-position target, matching the
    # FrankaRobotConfig command_mode used at both datagen and eval time.
    #
    # Some new-format trajectories include a final planner-status sentinel
    # like {"success": [0.0]} with no "arm" / "gripper" keys. We truncate
    # each trajectory at the first such index so the kept portion stays
    # temporally contiguous (needed for action-horizon chunking).
    cmd_tail = cmd[1:]                                                        # (T-1,)
    n_valid = len(cmd_tail)
    for i, c in enumerate(cmd_tail):
        if "arm" not in c or "gripper" not in c:
            n_valid = i
            break
    if n_valid == 0:
        raise RuntimeError(f"{traj_key}: no valid actions after reset")
    cmd_tail = cmd_tail[:n_valid]

    arm_target = np.asarray([c["arm"] for c in cmd_tail], dtype=np.float32)  # (n_valid, 7)
    grip_cmd = np.asarray([c["gripper"][0] for c in cmd_tail], dtype=np.float32)  # (n_valid,)
    grip_target = grip_cmd / GRIPPER_ACTION_SCALE                             # (n_valid,)

    actions = np.concatenate([arm_target, grip_target[:, None]], axis=1)      # (n_valid, 8)
    # Keep obs indices 0..n_valid-1 so pair (obs[t], action[t+1]) is well-defined
    # and the episode has exactly n_valid frames.
    exo = exo[:n_valid]
    wrist = wrist[:n_valid]
    arm_qpos = arm_qpos[:n_valid]
    grip_obs = grip_obs[:n_valid]

    prompt = (scene.get("task_description") or "open the drawer").lower()

    return {
        "traj_key": traj_key,
        "exo": exo,
        "wrist": wrist,
        "arm_qpos": arm_qpos,
        "grip_obs": grip_obs[:, None].astype(np.float32),
        "actions": actions.astype(np.float32),
        "prompt": prompt,
    }


def _list_h5_files(data_dir: Path, shard_pattern: str | None = None) -> list[Path]:
    """All trajectories_batch_<i>_of_<N>.h5 files in data_dir, sorted so the
    conversion output is deterministic across runs.

    If `shard_pattern` is given, only keep files whose batch tag (e.g.
    "batch_3_of_5") matches the regex — useful when a directory contains
    h5 shards from multiple datagen runs and only one set is complete."""
    files = sorted(
        p for p in data_dir.iterdir() if _H5_NAME_RE.search(p.name) is not None
    )
    if shard_pattern is None:
        return files
    pat = re.compile(shard_pattern)
    return [p for p in files if pat.search(_batch_tag_from_h5(p))]


def _list_jobs(h5_files: list[Path]) -> list[tuple[str, str, int, str]]:
    """Same as the parent script but skip trajectories whose paired ext+wrist
    mp4s don't exist in the staging dir. The generated runs only covered 1000
    of the 1000+ source trajectories (962 from `_of_5` + first 38 of
    `_of_11_cam_rand_b6`); the rest of the of_11cr_b6 h5's traj_* keys lack
    mp4 outputs and would crash _load_episode."""
    jobs: list[tuple[str, str, int, str]] = []
    skipped = 0
    for h5_path in h5_files:
        batch_tag = _batch_tag_from_h5(h5_path)
        data_dir = h5_path.parent
        with h5py.File(h5_path, "r", swmr=True) as f:
            keys = sorted(
                (k for k in f.keys() if k.startswith("traj_")),
                key=_episode_id_from_traj_name,
            )
        for k in keys:
            ep_id = _episode_id_from_traj_name(k)
            ext_mp4 = _episode_mp4(data_dir, ep_id, "exo_camera_1", batch_tag)
            wrist_mp4 = _episode_mp4(data_dir, ep_id, "wrist_camera", batch_tag)
            if not (ext_mp4.exists() and wrist_mp4.exists()):
                skipped += 1
                continue
            jobs.append((str(h5_path), k, ep_id, batch_tag))
    if skipped:
        print(f"[generated] skipped {skipped} trajectories with missing mp4s")
    return jobs


def _peek_dt_seconds(h5_files: list[Path]) -> float:
    for h5_path in h5_files:
        with h5py.File(h5_path, "r", swmr=True) as f:
            first = next((k for k in f.keys() if k.startswith("traj_")), None)
            if first is None:
                continue
            scene = json.loads(
                bytes(f[first]["obs_scene"][()]).rstrip(b"\x00").decode("utf-8")
            )
        return float(scene.get("policy_dt_ms", 66.0)) / 1000.0
    raise RuntimeError(f"No trajectories found in any of {h5_files}")


def main(
    data_dir: str,
    *,
    repo_id: str = DEFAULT_REPO_ID,
    num_workers: int = 24,
    image_writer_processes: int = 10,
    overwrite: bool = False,
    append: bool = False,
    shard_pattern: str | None = None,
) -> None:
    data_dir_p = Path(data_dir)
    h5_files = _list_h5_files(data_dir_p, shard_pattern)
    if not h5_files:
        raise SystemExit(
            f"No trajectories_batch_*_of_*.h5 files found in {data_dir_p}"
        )
    print(f"Found {len(h5_files)} h5 shard(s):")
    for p in h5_files:
        print(f"  {p.name}")

    dt = _peek_dt_seconds(h5_files)
    fps = int(round(1.0 / dt))
    print(f"policy_dt = {dt * 1000:.1f} ms  ->  fps = {fps}")

    out_dir = HF_LEROBOT_HOME / repo_id
    if append and overwrite:
        raise SystemExit("--append and --overwrite are mutually exclusive")

    if append:
        if not out_dir.exists():
            raise SystemExit(
                f"--append requires {out_dir} to already exist. Run without --append first."
            )
        dataset = LeRobotDataset(repo_id=repo_id)
        if int(round(dataset.fps)) != fps:
            raise SystemExit(
                f"fps mismatch: existing dataset is {dataset.fps}, new data is {fps}"
            )
        dataset.start_image_writer(
            num_processes=image_writer_processes, num_threads=10
        )
        dataset.episode_buffer = dataset.create_episode_buffer()
        print(
            f"Appending to existing dataset: {dataset.meta.total_episodes} episodes, "
            f"{dataset.meta.total_frames} frames"
        )
    else:
        if out_dir.exists():
            if not overwrite:
                raise SystemExit(
                    f"{out_dir} exists. Pass --overwrite to recreate or --append to extend."
                )
            shutil.rmtree(out_dir)

        dataset = LeRobotDataset.create(
            repo_id=repo_id,
            robot_type="panda",
            fps=fps,
            features={
                "exterior_image_1_left": {
                    "dtype": "image",
                    "shape": (IMG_H, IMG_W, 3),
                    "names": ["height", "width", "channel"],
                },
                "wrist_image_left": {
                    "dtype": "image",
                    "shape": (IMG_H, IMG_W, 3),
                    "names": ["height", "width", "channel"],
                },
                "joint_position": {
                    "dtype": "float32",
                    "shape": (7,),
                    "names": ["joint_position"],
                },
                "gripper_position": {
                    "dtype": "float32",
                    "shape": (1,),
                    "names": ["gripper_position"],
                },
                "actions": {
                    "dtype": "float32",
                    "shape": (8,),
                    "names": ["actions"],
                },
            },
            image_writer_threads=10,
            image_writer_processes=image_writer_processes,
        )

    jobs = _list_jobs(h5_files)
    if append:
        # In --append mode, skip jobs that were already saved on a prior run.
        # Jobs are processed in deterministic order (shard order × traj order),
        # so the first N jobs correspond to the first N saved episodes.
        already_done = dataset.meta.total_episodes
        if already_done > 0:
            if already_done > len(jobs):
                raise SystemExit(
                    f"existing dataset has {already_done} episodes but only "
                    f"{len(jobs)} jobs available — cannot resume coherently"
                )
            print(f"--append resume: skipping first {already_done} of {len(jobs)} jobs (already saved)")
            jobs = jobs[already_done:]
    print(f"Converting {len(jobs)} trajectories with {num_workers} workers")

    with ProcessPoolExecutor(max_workers=num_workers) as pool:
        for ep in tqdm(pool.map(_load_episode, jobs, chunksize=1),
                       total=len(jobs), desc="convert"):
            T = len(ep["actions"])
            for t in range(T):
                dataset.add_frame({
                    "exterior_image_1_left": ep["exo"][t],
                    "wrist_image_left": ep["wrist"][t],
                    "joint_position": ep["arm_qpos"][t],
                    "gripper_position": ep["grip_obs"][t],
                    "actions": ep["actions"][t],
                    "task": ep["prompt"],
                })
            dataset.save_episode()

    print(f"Done. LeRobot dataset at {out_dir}")


if __name__ == "__main__":
    tyro.cli(main)
