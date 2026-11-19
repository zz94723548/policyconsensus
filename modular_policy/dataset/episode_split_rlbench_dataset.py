"""Lazy, read-only episode partitions with expert-train-only normalization."""

import copy

import numpy as np
import torch
import zarr

from modular_policy.common.episode_split import load_manifest
from modular_policy.common.replay_buffer import ReplayBuffer
from modular_policy.common.seq_sampler import SequenceSampler
from modular_policy.dataset.base_dataset import BaseDataset
from modular_policy.dataset.rlbench_dataset import RlbenchDataset
from modular_policy.model.common.normalizer import LinearNormalizer, SingleFieldLinearNormalizer


class EpisodeSplitRlbenchDataset(RlbenchDataset):
    def __init__(self, zarr_path, split_manifest, obs_keys, action_key="action",
                 split="expert_train", validation_split="expert_val", horizon=16,
                 pad_before=1, pad_after=7, image_scale=1.0):
        BaseDataset.__init__(self)
        if image_scale != 1.0:
            raise ValueError("Expert split datasets use RGB in [0, 1]")
        self.zarr_path = str(zarr_path)
        self.split_manifest = str(split_manifest)
        root = zarr.open_group(self.zarr_path, mode="r")
        self.manifest = load_manifest(self.split_manifest, root)
        self.obs_keys, self.action_key = list(obs_keys), action_key
        keys = [action_key, *self.obs_keys]
        if "fused_dino" in keys or "fused_dino_raw" in keys:
            raise ValueError("This RGB/PCD pipeline does not use fitted DINO/PCA features")
        self.replay_buffer = ReplayBuffer({
            "data": {k: root["data"][k] for k in keys},
            "meta": {"episode_ends": root["meta/episode_ends"][:]},
        })
        self.horizon, self.pad_before, self.pad_after = horizon, pad_before, pad_after
        self.image_scale, self.validation_split = image_scale, validation_split
        self._normalizer = None
        self._select_split(split)

    def _select_split(self, split):
        if split not in self.manifest["splits"]:
            raise ValueError(f"Unknown episode split: {split}")
        self.split = split
        self.episode_ids = np.array(self.manifest["splits"][split], dtype=np.int64)
        self.train_mask = np.zeros(self.replay_buffer.n_episodes, dtype=bool)
        self.train_mask[self.episode_ids] = True
        self.seq_sampler = SequenceSampler(
            self.replay_buffer, sequence_length=self.horizon,
            pad_before=self.pad_before, pad_after=self.pad_after,
            episode_mask=self.train_mask, keys=[self.action_key, *self.obs_keys],
        )
        if not len(self.seq_sampler):
            raise ValueError(f"No usable action sequences in {split}")

    def get_validation_dataset(self):
        result = copy.copy(self)
        result._select_split(self.validation_split)
        return result

    def get_normalizer(self, **kwargs):
        if kwargs:
            raise ValueError("This pipeline uses fixed train-only limits normalization")
        if self._normalizer is not None:
            return self._normalizer
        result = LinearNormalizer()
        for key in [self.action_key, *self.obs_keys]:
            if "rgb" in key:
                # Known uint8 sensor range: no held-out frames are inspected.
                stats = {"min": [0.0], "max": [1.0], "mean": [0.5], "std": [0.5]}
                scale, offset = np.array([2.0], np.float32), np.array([-1.0], np.float32)
            else:
                stats = self._training_stats(key)
                input_range = stats["max"] - stats["min"]
                constant = input_range < 1e-4
                input_range[constant] = 2.0
                scale = (2.0 / input_range).astype(np.float32)
                offset = (-1.0 - scale * stats["min"]).astype(np.float32)
                offset[constant] = -stats["min"][constant]
            stats = {k: np.asarray(v, dtype=np.float32) for k, v in stats.items()}
            field = SingleFieldLinearNormalizer.create_manual(scale, offset, stats)
            field.requires_grad_(False)
            result["action" if key == self.action_key else key] = field
        self._normalizer = result
        return result

    def _training_stats(self, key):
        """Merge per-block moments, reading ONLY expert_train episodes."""
        array = self.replay_buffer[key]
        ends = self.replay_buffer.episode_ends
        count, mean, m2, low, high = 0, None, None, None, None
        for episode in self.manifest["splits"]["expert_train"]:
            begin = int(ends[episode - 1]) if episode else 0
            end = int(ends[episode])
            for start in range(begin, end, 128):
                block = np.asarray(array[start:min(start + 128, end)], dtype=np.float64)
                block = block.reshape(-1, array.shape[-1])
                if not np.isfinite(block).all():
                    raise ValueError(f"Nonfinite expert training data in {key}")
                n = len(block)
                block_mean = block.mean(axis=0)
                block_m2 = ((block - block_mean) ** 2).sum(axis=0)
                if count == 0:
                    mean, m2 = block_mean, block_m2
                    low, high = block.min(axis=0), block.max(axis=0)
                else:
                    delta = block_mean - mean
                    m2 += block_m2 + delta ** 2 * count * n / (count + n)
                    mean += delta * n / (count + n)
                    low = np.minimum(low, block.min(axis=0))
                    high = np.maximum(high, block.max(axis=0))
                count += n
        return {"min": low, "max": high, "mean": mean,
                "std": np.sqrt(m2 / max(count - 1, 1))}

    def get_all_actions(self):
        ends = self.replay_buffer.episode_ends
        chunks = []
        for i in self.episode_ids:
            start = int(ends[i - 1]) if i else 0
            chunks.append(self.replay_buffer[self.action_key][start:int(ends[i])])
        return torch.from_numpy(np.concatenate(chunks))
