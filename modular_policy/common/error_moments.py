"""Paired uncentered error moments and provenance checks (no simulator imports)."""

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from modular_policy.common.episode_split import canonical_hash


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def validate_risk_split(split, original):
    if split.get("version") != 1 or split.get("purpose") != "risk_statistics_model_train_val":
        raise ValueError("Unsupported risk statistics split")
    if split["source_manifest_sha256"] != canonical_hash(original):
        raise ValueError("Risk split and original manifest hashes differ")
    roles = ("risk_model_train", "risk_model_val")
    if set(split["splits"]) != set(roles) or split["task_order"] != original["task_order"]:
        raise ValueError("Risk split roles/task order differ")
    combined = []
    for role, expected in zip(roles, (5, 2)):
        ids = split["splits"][role]
        if any(type(i) is not int or i < 0 or i >= len(original["task_by_episode"]) for i in ids):
            raise ValueError("Invalid episode ID")
        combined.extend(ids)
        for task in split["task_order"]:
            selected = sorted(i for i in ids if original["task_by_episode"][i] == task)
            if len(selected) != expected or selected != split["by_task"][task][role]:
                raise ValueError(f"Expected {expected} episodes per task in {role}")
    if len(set(combined)) != len(combined) or set(combined) != set(original["splits"]["risk_train"]):
        raise ValueError("Risk partitions overlap or do not cover risk_train")
    for role in ("expert_train", "expert_val", "risk_calibration", "test"):
        if set(combined) & set(original["splits"][role]):
            raise ValueError(f"Risk data leaks into {role}")


def check_moments(value):
    value = np.asarray(value, dtype=np.float64)
    if value.shape[-1] != 3 or not np.isfinite(value).all():
        raise ValueError("Moments must be finite [..., A/B/C]")
    a, b, c = np.moveaxis(value, -1, 0)
    if np.any(a < 0) or np.any(b < 0) or np.any(c * c > a * b + 1e-6 * np.maximum(a * b, 1e-12)):
        raise ValueError("Error moment matrix is not positive semidefinite")


def quadratic_risk(moments, weight):
    a, b, c = np.moveaxis(np.asarray(moments, dtype=np.float64), -1, 0)
    return weight ** 2 * a + (1 - weight) ** 2 * b + 2 * weight * (1 - weight) * c


def optimal_weight(moments, fallback=0.5, tolerance=1e-8):
    check_moments(moments)
    a, b, c = np.moveaxis(np.asarray(moments, dtype=np.float64), -1, 0)
    denominator = a + b - 2 * c
    good = denominator > tolerance * np.maximum(a + b, 1e-12)
    ratio = np.divide(b - c, denominator, out=np.full_like(denominator, fallback), where=good)
    return np.clip(ratio, 0, 1), ~good


def paired_moments(rgb_error, pcd_error):
    if rgb_error.shape != pcd_error.shape:
        raise ValueError("Paired errors must have the same shape")
    rgb = rgb_error.double().flatten(1)
    pcd = pcd_error.double().flatten(1)
    result = torch.stack([(rgb * rgb).mean(1), (pcd * pcd).mean(1), (rgb * pcd).mean(1)], -1)
    if not torch.isfinite(result).all():
        raise ValueError("Nonfinite paired errors")
    return result


def context_noise(seed, episode, anchor, timestep, repeat, role, shape):
    # CPU draws keyed per context make batching independent of the random samples.
    words = [int(seed), int(episode), int(anchor), int(timestep), int(repeat), int(role)]
    noise_seed = int(np.random.SeedSequence(words).generate_state(1, dtype=np.uint64)[0]) % (2 ** 63 - 1)
    return torch.randn(shape, generator=torch.Generator().manual_seed(noise_seed))


def bootstrap_delta(values, labels, repeats=2000, seed=2026):
    values, labels = np.asarray(values), np.asarray(labels)
    rng = np.random.default_rng(seed)
    estimates = np.zeros(repeats)
    for task in np.unique(labels):
        group = values[labels == task]
        draws = rng.integers(len(group), size=(repeats, len(group)))
        estimates += group[draws].sum(1) / len(values)
    return {"mean": float(values.mean()), "ci95": np.quantile(estimates, [0.025, 0.975]).tolist()}
