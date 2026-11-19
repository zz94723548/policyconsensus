"""Causal histories over frozen paired-error caches; no image/Numba imports."""

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from modular_policy.common.episode_split import canonical_hash
from modular_policy.common.error_moments import check_moments, file_hash, validate_risk_split
from modular_policy.common.moment_calibration import calibration_ids, heldout_ids


class ErrorMomentsDataset(Dataset):
    def __init__(self, cache_dir, role, history_contexts=4, risk_mode='shared'):
        if risk_mode not in ('shared', 'per_step'):
            raise ValueError('Unknown risk mode')
        self.risk_mode = risk_mode
        if role not in ("risk_model_train", "risk_model_val", "risk_calibration", 'test','expert_train','expert_val') or history_contexts < 1:
            raise ValueError("Invalid role or history length")
        root = Path(cache_dir).resolve()
        self.metadata = json.loads((root / "metadata.json").read_text())
        if self.metadata["status"] != "complete":
            raise ValueError("Paired error collection has not completed")
        identity = self.metadata["identity"]
        if canonical_hash(identity) != self.metadata["signature"]:
            raise ValueError("Cache identity hash mismatch")
        split, original = identity["risk_split"], identity["source_manifest"]
        validate_risk_split(split, original)
        partition = identity.get("collection_partition", "risk_train")
        if partition == "risk_calibration":
            roles = {"risk_calibration": calibration_ids(original)}
        elif partition == "risk_train":
            roles = split["splits"]
        elif partition == 'test':
            roles = {'test': heldout_ids(original, 'test')}
        elif partition=='expert_router':
            from modular_policy.common.frozen_router_data import router_cache_roles
            roles=router_cache_roles(identity)
        else:
            raise ValueError("Unsupported cache partition")
        if role not in roles:
            raise ValueError("Requested role is not present in this dedicated cache")
        records = self.metadata["episodes"]
        all_ids = [e["episode_id"] for e in records]
        if len(all_ids) != len(set(all_ids)) or set(all_ids) != {i for ids in roles.values() for i in ids}:
            raise ValueError("Cache episode coverage differs from declared partition")
        for e in records:
            if e["role"] not in roles or e["episode_id"] not in roles[e["role"]]:
                raise ValueError("Cached episode has the wrong role")
        expected = roles[role]
        self.role, self.history_contexts = role, history_contexts
        self.task_order = split["task_order"]
        self.timesteps = np.asarray(identity["timesteps"])
        self.step_weights = np.asarray(identity["lambda"], dtype=np.float64)
        if len(self.step_weights) != len(self.timesteps) or np.any(self.step_weights < 0) or not np.isclose(self.step_weights.sum(), 1):
            raise ValueError("Invalid diffusion step weighting")
        self.episodes, self.index, self.sample_weights = [], [], []
        for record in sorted((r for r in records if r["role"] == role), key=lambda r: r["episode_id"]):
            path = (root / record["file"]).resolve()
            if root not in path.parents or file_hash(path) != record["sha256"]:
                raise ValueError(f"Invalid cached episode file: {path}")
            with np.load(path, allow_pickle=False) as data:
                item = {k: data[k] for k in ["rgb_features", "pcd_features", "moments", "moments_by_step", "anchor_frames"]}
                if int(data["episode_id"]) != record["episode_id"] or not np.array_equal(data["timesteps"], self.timesteps):
                    raise ValueError("Episode ID/timestep mismatch in cache")
            n = len(item["moments"])
            if n != record["contexts"] or n < 1 or any(len(v) != n for v in item.values()):
                raise ValueError("Inconsistent cached context lengths")
            if item["moments"].shape != (n, 3) or item["moments_by_step"].shape != (n, len(self.timesteps), 3):
                raise ValueError("Invalid moment tensor shape")
            for key, dim in [("rgb_features", record["rgb_dim"]), ("pcd_features", record["pcd_dim"])]:
                if item[key].shape != (n, dim) or not np.isfinite(item[key]).all():
                    raise ValueError("Invalid frozen condition features")
            check_moments(item["moments_by_step"])
            check_moments(item["moments"])
            if not np.allclose(item["moments"], np.einsum("k,nkc->nc", self.step_weights, item["moments_by_step"]), rtol=1e-6, atol=1e-12):
                raise ValueError("Shared labels differ from weighted step moments")
            if np.any(np.diff(item["anchor_frames"]) != identity["context_stride"]):
                raise ValueError("Context anchors are not in chronological stride order")
            eid = record["episode_id"]
            ends = original["source"]["episode_ends"]
            begin, end = (ends[eid - 1] if eid else 0), ends[eid]
            if np.any(item["anchor_frames"] < begin) or np.any(item["anchor_frames"] >= end):
                raise ValueError("Context anchors cross episode boundaries")
            if record["task"] != original["task_by_episode"][eid]:
                raise ValueError("Wrong task label")
            item.update(episode_id=eid, task=record["task"])
            self.index.extend((len(self.episodes), j) for j in range(n))
            self.sample_weights.extend([1.0 / (len(expected) * n)] * n)
            self.episodes.append(item)
        if [e["episode_id"] for e in self.episodes] != sorted(expected):
            raise ValueError("Missing role episodes")
        dims = {(e["rgb_features"].shape[1], e["pcd_features"].shape[1]) for e in self.episodes}
        if len(dims) != 1:
            raise ValueError("Feature dimensions differ between episodes")
        self.rgb_dim, self.pcd_dim = dims.pop()

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):
        episode, position = self.index[index]
        item = self.episodes[episode]
        begin = max(0, position - self.history_contexts + 1)
        length = position - begin + 1
        result = {}
        for key in ["rgb_features", "pcd_features"]:
            history = np.zeros((self.history_contexts, item[key].shape[1]), dtype=np.float32)
            # Valid chronological prefix followed by padding; packing ignores padding.
            history[:length] = item[key][begin:position + 1]
            result[key] = torch.from_numpy(history)
        label_key = 'moments_by_step' if self.risk_mode == 'per_step' else 'moments'
        result.update(length=torch.tensor(length), target=torch.tensor(item[label_key][position], dtype=torch.float32),
                      index=torch.tensor(index), weight=torch.tensor(self.sample_weights[index], dtype=torch.float64))
        return result

    def training_statistics(self):
        if self.role not in ("risk_model_train",'expert_train'):
            raise ValueError("Statistics must be fitted on the 20 training risk episodes or expert_train only")
        result = {}
        for key in ["rgb_features", "pcd_features"]:
            mean = np.mean([e[key].astype(np.float64).mean(0) for e in self.episodes], axis=0)
            square = np.mean([(e[key].astype(np.float64) ** 2).mean(0) for e in self.episodes], axis=0)
            result[key] = {"mean": mean.astype(np.float32),
                           "std": np.maximum(np.sqrt(np.maximum(square - mean ** 2, 0)), 1e-4).astype(np.float32)}
        result["global_moments"] = np.mean([e["moments"].mean(0) for e in self.episodes], axis=0)
        result["output_scale"] = max(float(result["global_moments"][:2].mean()), 1e-8)
        if self.risk_mode == 'per_step':
            result['global_moments_by_step'] = np.mean([e['moments_by_step'].mean(0) for e in self.episodes], axis=0)
            result['step_output_scale'] = np.maximum(result['global_moments_by_step'][:, :2].mean(-1), 1e-8)
        return result
