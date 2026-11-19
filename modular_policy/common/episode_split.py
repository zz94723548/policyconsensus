"""Reproducible, task-stratified episode allocation without copying sensor data."""

import hashlib
import json
from pathlib import Path

import numpy as np


SPLIT_NAMES = ("expert_train", "expert_val", "risk_train", "risk_calibration", "test")
DEFAULT_COUNTS = (30, 5, 7, 4, 4)
MT4_TASKS = (
    "open_box", "toilet_seat_up", "open_drawer",
    "take_umbrella_out_of_umbrella_stand",
)


def canonical_hash(value):
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def dataset_identity(root):
    """Bind a manifest to episode boundaries and array schemas, not just a path.

    This is a structural fingerprint, not a checksum of every sensor chunk.
    """
    ends = np.asarray(root["meta/episode_ends"][:], dtype=np.int64)
    if ends.ndim != 1 or not len(ends) or np.any(np.diff(np.r_[0, ends]) <= 0):
        raise ValueError("episode_ends must be nonempty and strictly increasing")
    arrays = {}
    for key, array in root["data"].arrays():
        if array.shape[0] != ends[-1]:
            raise ValueError(f"Inconsistent frame count for {key}")
        arrays[key] = {"shape": list(array.shape), "dtype": str(array.dtype)}
    return {"episode_ends": ends.tolist(), "arrays": arrays}


def make_manifest(root, zarr_path, tasks=MT4_TASKS, episodes_per_task=50,
                  counts=DEFAULT_COUNTS, seed=42):
    tasks, counts = list(tasks), list(counts)
    if not tasks or len(set(tasks)) != len(tasks):
        raise ValueError("Task names must be nonempty and unique")
    if len(counts) != len(SPLIT_NAMES) or any(type(n) is not int or n <= 0 for n in counts):
        raise ValueError("Provide five positive integer split counts")
    if sum(counts) != episodes_per_task:
        raise ValueError("Split counts must sum to episodes_per_task")
    identity = dataset_identity(root)
    ends = identity["episode_ends"]
    if len(ends) != len(tasks) * episodes_per_task:
        raise ValueError("Episode count does not match contiguous task blocks")
    rng = np.random.default_rng(seed)
    splits = {name: [] for name in SPLIT_NAMES}
    task_by_episode = []
    by_task = {}
    for task_index, task in enumerate(tasks):
        ids = rng.permutation(np.arange(episodes_per_task) + task_index * episodes_per_task)
        task_by_episode.extend([task] * episodes_per_task)
        by_task[task] = {}
        offset = 0
        for name, count in zip(SPLIT_NAMES, counts):
            selected = sorted(ids[offset:offset + count].tolist())
            splits[name].extend(selected)
            by_task[task][name] = selected
            offset += count
    manifest = {
        "version": 1,
        "seed": int(seed),
        "source": {
            "zarr_path": str(Path(zarr_path).resolve()),
            "structural_sha256": canonical_hash(identity),
            "n_episodes": len(ends),
            "n_frames": ends[-1],
            "episode_ends": ends,
        },
        "task_assignment": "contiguous_blocks",
        "task_order": tasks,
        "episodes_per_task": episodes_per_task,
        "counts_per_task": dict(zip(SPLIT_NAMES, counts)),
        "task_by_episode": task_by_episode,
        "splits": splits,
        "by_task": by_task,
    }
    validate_manifest(manifest, root)
    return manifest


def validate_manifest(manifest, root):
    if manifest.get("version") != 1:
        raise ValueError("Unsupported split manifest version")
    identity = dataset_identity(root)
    source = manifest["source"]
    if canonical_hash(identity) != source["structural_sha256"]:
        raise ValueError("Split manifest does not match this dataset's episode boundaries/schema")
    n = len(identity["episode_ends"])
    if source["n_episodes"] != n or source["episode_ends"] != identity["episode_ends"]:
        raise ValueError("Inconsistent source episode metadata")
    if set(manifest["splits"]) != set(SPLIT_NAMES):
        raise ValueError("Split manifest must contain exactly the five named partitions")
    all_ids = []
    for name in SPLIT_NAMES:
        ids = manifest["splits"][name]
        if not ids or any(type(i) is not int or i < 0 or i >= n for i in ids):
            raise ValueError(f"Invalid episode IDs in {name}")
        all_ids.extend(ids)
    if sorted(all_ids) != list(range(n)):
        raise ValueError("Episode partitions overlap or do not cover the dataset")
    tasks = manifest["task_order"]
    block_size = manifest["episodes_per_task"]
    expected_tasks = [task for task in tasks for _ in range(block_size)]
    if manifest["task_by_episode"] != expected_tasks or len(expected_tasks) != n:
        raise ValueError("Invalid contiguous task assignment")
    for task in tasks:
        for name in SPLIT_NAMES:
            ids = [i for i in manifest["splits"][name] if expected_tasks[i] == task]
            if (len(ids) != manifest["counts_per_task"][name]
                    or sorted(ids) != manifest["by_task"][task][name]):
                raise ValueError(f"Invalid task allocation: {task}/{name}")


def load_manifest(path, root):
    manifest = json.loads(Path(path).read_text())
    validate_manifest(manifest, root)
    return manifest
