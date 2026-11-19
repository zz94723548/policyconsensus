"""Episode conformal radii and exact robust weights for PSD Frobenius balls."""

import math

import numpy as np

from modular_policy.common.episode_split import canonical_hash
from modular_policy.common.error_moments import check_moments, optimal_weight, quadratic_risk


def calibration_ids(original):
    return heldout_ids(original, "risk_calibration")


def heldout_ids(original, role):
    if role not in ('risk_calibration', 'test'):
        raise ValueError('Expected a held-out partition')
    ids = original["splits"][role]
    if len(ids) != len(set(ids)) or not ids:
        raise ValueError("Invalid calibration episode IDs")
    for other in ["expert_train", "expert_val", "risk_train", "risk_calibration", "test"]:
        if other != role and set(ids) & set(original["splits"][other]):
            raise ValueError(f"Held-out partition overlaps {other}")
    if any(type(i) is not int or not 0 <= i < len(original["task_by_episode"]) for i in ids):
        raise ValueError("Calibration episode outside source dataset")
    for task in original["task_order"]:
        if sum(original["task_by_episode"][i] == task for i in ids) != 4:
            raise ValueError("Expected exactly four calibration episodes per task")
    return ids


def audit_calibration_identity(training, calibration, partition="risk_calibration"):
    if partition not in ('risk_calibration', 'test') or calibration.get("collection_partition") != partition:
        raise ValueError("A dedicated calibration cache is required")
    for key in ["source_manifest", "risk_split", "timesteps", "lambda", "context_stride",
                "sampler_sha256", "dataset_sha256", "policy_sha256"]:
        if canonical_hash(training[key]) != canonical_hash(calibration[key]):
            raise ValueError(f"Training/calibration provenance mismatch: {key}")
    for modality in ["rgb", "pcd"]:
        for key in ["sha256", "weights", "policy_config"]:
            if canonical_hash(training["experts"][modality][key]) != canonical_hash(calibration["experts"][modality][key]):
                raise ValueError(f"Different frozen expert: {modality}/{key}")
    repeat_key = 'test_noise_repeats' if partition == 'test' else 'calibration_noise_repeats'
    if calibration[repeat_key] != training["val_noise_repeats"]:
        raise ValueError("Calibration labels must use the frozen validation MC repeat count")


def normalized_frobenius_residual(prediction, target, scale):
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("Output scale must be fixed and positive")
    prediction, target = np.asarray(prediction, dtype=np.float64), np.asarray(target, dtype=np.float64)
    if prediction.shape != target.shape:
        raise ValueError("Prediction and target moment shapes differ")
    error = (prediction - target) / scale
    if error.shape[-1] != 3 or not np.isfinite(error).all():
        raise ValueError("Invalid moment residuals")
    return np.sqrt(error[..., 0] ** 2 + error[..., 1] ** 2 + 2 * error[..., 2] ** 2)


def episode_conformal_threshold(scores, alpha):
    scores = np.asarray(scores, dtype=np.float64)
    if scores.ndim != 1 or not len(scores) or np.any(scores < 0) or not np.isfinite(scores).all():
        raise ValueError("Expected finite nonnegative episode scores")
    if not 0 < alpha < 1:
        raise ValueError("alpha must lie in (0,1)")
    rank = max(1, math.ceil((len(scores) + 1) * (1 - alpha) - 1e-12))
    if rank > len(scores):
        raise ValueError(f"{len(scores)} episodes cannot yield a finite {1-alpha:.0%} threshold; need more calibration data")
    return float(np.sort(scores)[rank - 1]), rank


def fit_shared_radius(episodes, task_order, alpha, scale):
    """Per-task episode quantiles, then their maximum: conservative task-free use.

    No exchangeability across different tasks is assumed. The coverage statement
    requires exchangeability of complete episodes WITHIN each known task.
    """
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("Output scale must be positive")
    if len({e["episode_id"] for e in episodes}) != len(episodes):
        raise ValueError("Duplicate calibration episodes")
    if set(e["task"] for e in episodes) != set(task_order):
        raise ValueError("Calibration task coverage differs")
    groups = {}
    for task in task_order:
        group = [e for e in episodes if e["task"] == task]
        q, rank = episode_conformal_threshold([e["score"] for e in group], alpha)
        groups[task] = {"episodes": len(group), "rank": rank, "normalized_radius": q}
    q = max(g["normalized_radius"] for g in groups.values())
    return {"normalized_radius": q, "radius": float(q * scale), "scale": float(scale),
            "alpha": alpha, "nominal_episode_coverage": 1 - alpha, "per_task": groups}


def robust_weight(moments, radius, fallback=0.5, tolerance=1e-8):
    """Exact min-max for {Gamma >= 0: ||Gamma-Gamma_hat||_F <= radius}.

    max v.T Gamma v = v.T Gamma_hat v + radius * ||v||^2. The
    maximizing perturbation is PSD, so the PSD constraint is already satisfied.
    """
    check_moments(moments)
    radius = np.asarray(radius, dtype=np.float64)
    if not np.isfinite(radius).all() or np.any(radius < 0):
        raise ValueError("Radius must be finite and nonnegative")
    adjusted = np.asarray(moments, dtype=np.float64).copy()
    adjusted[..., 0] += radius
    adjusted[..., 1] += radius
    return optimal_weight(adjusted, fallback, tolerance)


def worst_case_risk(moments, weight, radius):
    return quadratic_risk(moments, weight) + radius * (weight ** 2 + (1 - weight) ** 2)
