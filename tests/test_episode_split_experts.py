import copy
import json
from pathlib import Path

import numpy as np
import pytest
import torch
import zarr
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from modular_policy.common.episode_split import make_manifest, validate_manifest
from modular_policy.dataset.episode_split_rlbench_dataset import EpisodeSplitRlbenchDataset


@pytest.fixture
def source(tmp_path):
    path = tmp_path / "demos.zarr"
    root = zarr.open_group(str(path), mode="w")
    # Two task blocks, with one episode for each of the five roles per task.
    ends = np.arange(1, 11) * 12
    root.create_dataset("meta/episode_ends", data=ends)
    rng = np.random.default_rng(7)
    root.create_dataset("data/action", data=rng.normal(size=(120, 2)).astype("float32"))
    root.create_dataset("data/joint_positions", data=rng.normal(size=(120, 2)).astype("float32"))
    root.create_dataset("data/fused_pointcloud", data=rng.normal(size=(120, 8, 3)).astype("float32"))
    root.create_dataset("data/front_rgb", data=rng.integers(0, 256, size=(120, 3, 8, 8), dtype="uint8"))
    manifest = make_manifest(root, str(path), tasks=["a", "b"], episodes_per_task=5,
                             counts=[1, 1, 1, 1, 1], seed=42)
    manifest_path = tmp_path / "split.json"
    manifest_path.write_text(json.dumps(manifest))
    return path, root, manifest, manifest_path


def dataset(source, **kwargs):
    path, _, _, split_path = source
    return EpisodeSplitRlbenchDataset(
        str(path), str(split_path), ["front_rgb", "fused_pointcloud", "joint_positions"],
        horizon=8, pad_before=1, pad_after=3, **kwargs,
    )


def test_reproducible_stratified_partition_and_no_overlap(source):
    path, root, manifest, _ = source
    assert manifest == make_manifest(root, str(path), ["a", "b"], 5, [1] * 5, 42)
    all_ids = [i for ids in manifest["splits"].values() for i in ids]
    assert sorted(all_ids) == list(range(10))
    for task in ["a", "b"]:
        assert all(len(ids) == 1 for ids in manifest["by_task"][task].values())


def test_rejects_overlap_and_wrong_episode_boundaries(source):
    _, root, manifest, _ = source
    bad = copy.deepcopy(manifest)
    bad["splits"]["expert_val"][0] = bad["splits"]["expert_train"][0]
    with pytest.raises(ValueError, match="overlap"):
        validate_manifest(bad, root)
    root["meta/episode_ends"][0] = 11
    with pytest.raises(ValueError, match="does not match"):
        validate_manifest(manifest, root)


def test_validation_is_explicit_and_sequences_never_cross_episodes(source):
    train = dataset(source)
    val = train.get_validation_dataset()
    manifest = source[2]
    assert val.episode_ids.tolist() == manifest["splits"]["expert_val"]
    assert len(val.episode_ids) == 2  # ~train would incorrectly include eight.
    for ds in [train, val]:
        ends = ds.replay_buffer.episode_ends
        for begin, end, _, _ in ds.seq_sampler.indices:
            episode = np.searchsorted(ends, begin, side="right")
            assert episode == np.searchsorted(ends, end - 1, side="right")
            assert episode in ds.episode_ids
    assert set(train[0]["obs"]) == set(train.obs_keys)


def test_held_out_values_do_not_change_any_fitted_normalizer(source):
    train = dataset(source)
    expected = {k: v.clone() for k, v in train.get_normalizer().state_dict().items()}
    _, root, manifest, _ = source
    for name, ids in manifest["splits"].items():
        if name == "expert_train":
            continue
        for episode in ids:
            for key in ["action", "joint_positions", "fused_pointcloud"]:
                root[f"data/{key}"][episode * 12:(episode + 1) * 12] = 1e6
    updated = dataset(source)
    for key, value in updated.get_normalizer().state_dict().items():
        torch.testing.assert_close(value, expected[key], rtol=0, atol=0)
    assert updated.get_validation_dataset()[0]["action"].min() == 1e6


def test_streamed_training_moments_match_direct_fit_and_rgb_range(source):
    train = dataset(source)
    _, root, manifest, _ = source
    for key in ["action", "fused_pointcloud", "joint_positions"]:
        direct = np.concatenate([root[f"data/{key}"][i * 12:(i + 1) * 12]
                                 for i in manifest["splits"]["expert_train"]])
        direct = direct.reshape(-1, direct.shape[-1])
        stats = train.get_normalizer()[key].get_input_stats()
        np.testing.assert_allclose(stats["min"].numpy(), direct.min(0), atol=1e-6)
        np.testing.assert_allclose(stats["max"].numpy(), direct.max(0), atol=1e-6)
        np.testing.assert_allclose(stats["mean"].numpy(), direct.mean(0), atol=1e-6)
        np.testing.assert_allclose(stats["std"].numpy(), direct.std(0, ddof=1), atol=1e-6)
    torch.testing.assert_close(train.get_normalizer()["front_rgb"].normalize(torch.tensor([0., 1.])),
                               torch.tensor([-1., 1.]))


def test_rgb_pcd_configs_have_identical_diffusion_spaces():
    OmegaConf.register_new_resolver("eval", eval, replace=True)
    config_dir = str(Path(__file__).resolve().parents[1] / "modular_policy/config")
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        rgb = compose(config_name="train_rgb_expert")
        pcd = compose(config_name="train_pcd_expert")
    for key in ["noise_scheduler", "horizon", "n_obs_steps", "n_action_steps",
                "num_inference_steps", "diffusion_step_embed_dim", "down_dims"]:
        assert rgb.policy[key] == pcd.policy[key]
    assert rgb.task.dataset.split_manifest == pcd.task.dataset.split_manifest
    assert "fused_pointcloud" not in rgb.expert_obs_keys
    assert not any("rgb" in key for key in pcd.expert_obs_keys)


def test_validation_uses_passed_policy_and_preserves_rng():
    from modular_policy.workspace.train_modality_expert import TrainModalityExpertWorkspace

    class EvalPolicy:
        def eval(self):
            return self

        def compute_loss(self, batch):
            return torch.rand(()) + batch["action"].mean()

    cfg = OmegaConf.create({"training": {"validation_seed": 100, "max_val_steps": None}})
    loader = [{"action": torch.full((2, 8, 2), 3.0)}]
    rng = torch.random.get_rng_state().clone()
    a = TrainModalityExpertWorkspace._validate(EvalPolicy(), loader, torch.device("cpu"), cfg)
    b = TrainModalityExpertWorkspace._validate(EvalPolicy(), loader, torch.device("cpu"), cfg)
    assert a == b and 3 <= a < 4
    assert torch.equal(rng, torch.random.get_rng_state())
