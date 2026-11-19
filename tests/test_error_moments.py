"""Synthetic math/cache tests. No trained expert or simulator is loaded/run."""

import copy
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from modular_policy.common.episode_split import canonical_hash
from modular_policy.common.error_moments import (
    check_moments, context_noise, file_hash, optimal_weight, paired_moments,
    quadratic_risk, validate_risk_split,
)
from modular_policy.dataset.error_moments_dataset import ErrorMomentsDataset
from modular_policy.model.error_moments import matrix_regression_loss, moments_from_raw
from modular_policy.workspace.train_error_moments import mean_fusion_risk, risk_report, should_stop


def test_uncentered_paired_moments_keep_bias_and_negative_cross_term():
    er, ep = torch.full((2, 16, 9), 2.0), torch.full((2, 16, 9), -1.0)
    moments = paired_moments(er, ep).numpy()
    np.testing.assert_allclose(moments, [[4, 1, -2], [4, 1, -2]])
    check_moments(moments)
    for weight in [0, 0.2, 0.5, 1]:
        expected = (weight * er + (1 - weight) * ep).square().flatten(1).mean(1).numpy()
        np.testing.assert_allclose(quadratic_risk(moments, weight), expected, atol=1e-7)


def test_optimal_weight_endpoints_and_degenerate_fallback():
    w, fallback = optimal_weight([4, 1, -2])
    assert float(w) == pytest.approx(1 / 3) and not fallback
    w, fallback = optimal_weight([1, 1, 1], fallback=0.7)
    assert float(w) == 0.7 and fallback
    assert float(optimal_weight([1, 4, 2])[0]) == 1.0
    with pytest.raises(ValueError, match="positive semidefinite"):
        optimal_weight([1, 1, 2])


def test_noise_is_keyed_to_context_not_batching_and_validation_is_independent():
    a = context_noise(2026, 10, 100, 99, 0, 0, (16, 9))
    b = context_noise(2026, 10, 100, 99, 0, 0, (16, 9))
    other = context_noise(2026, 10, 100, 99, 0, 1, (16, 9))
    assert torch.equal(a, b)
    assert not torch.equal(a, other)


def test_psd_transform_and_fixed_scale_matrix_loss():
    raw = torch.tensor([[0., 0., -2.], [10., -10., 20.], [-10., 10., -20.]])
    moments = moments_from_raw(raw, torch.tensor(0.1))
    check_moments(moments.numpy())
    value = matrix_regression_loss(torch.tensor([[2., 3., 4.]]), torch.tensor([[1., 1., 1.]]), 2.)
    torch.testing.assert_close(value, torch.tensor([(1 + 4 + 18) / 4]))


@pytest.fixture
def cache(tmp_path):
    repo = Path(__file__).resolve().parents[1]
    original = json.loads((repo / "splits/mt4_seed42.json").read_text())
    split = json.loads((repo / "splits/mt4_risk_model_seed142.json").read_text())
    identity = {"source_manifest": original, "risk_split": split,
                "timesteps": [99, 0], "lambda": [0.5, 0.5], "context_stride": 8}
    metadata = {"status": "complete", "identity": identity, "signature": canonical_hash(identity), "episodes": []}
    for role, ids in split["splits"].items():
        (tmp_path / role).mkdir()
        for eid in ids:
            first = original["source"]["episode_ends"][eid - 1] if eid else 0
            rgb = np.arange(6, dtype=np.float32).reshape(2, 3) + eid
            pcd = np.arange(4, dtype=np.float32).reshape(2, 2) + eid
            if role == "risk_model_val":
                rgb += 1e6
                pcd += 1e6
            steps = np.array([[[1, 4, 0.5], [3, 2, -0.5]]] * 2, np.float64)
            relative = f"{role}/episode_{eid:04d}.npz"
            path = tmp_path / relative
            np.savez_compressed(path, episode_id=eid, timesteps=[99, 0], rgb_features=rgb,
                                pcd_features=pcd, moments=steps.mean(1), moments_by_step=steps,
                                anchor_frames=[first, first + 8])
            metadata["episodes"].append({"episode_id": eid, "role": role,
                "task": original["task_by_episode"][eid], "contexts": 2,
                "rgb_dim": 3, "pcd_dim": 2, "file": relative, "sha256": file_hash(path)})
    (tmp_path / "metadata.json").write_text(json.dumps(metadata))
    return tmp_path, metadata


def test_split_is_exactly_twenty_eight_and_rejects_overlap(cache):
    _, meta = cache
    identity = meta["identity"]
    validate_risk_split(identity["risk_split"], identity["source_manifest"])
    bad = copy.deepcopy(identity["risk_split"])
    bad["splits"]["risk_model_val"][0] = bad["splits"]["risk_model_train"][0]
    with pytest.raises(ValueError):
        validate_risk_split(bad, identity["source_manifest"])


def test_causal_history_never_crosses_episode_and_episode_weights_sum_to_one(cache):
    path, _ = cache
    ds = ErrorMomentsDataset(path, "risk_model_train", history_contexts=4)
    assert len(ds.episodes) == 20 and len(ds) == 40
    assert sum(ds.sample_weights) == pytest.approx(1)
    for index in [0, 2, 4, 6]:
        first, second = ds[index], ds[index + 1]
        assert int(first["length"]) == 1 and int(second["length"]) == 2
        assert torch.count_nonzero(first["rgb_features"][1:]) == 0
        torch.testing.assert_close(second["rgb_features"][0], first["rgb_features"][0])
        torch.testing.assert_close(second["rgb_features"][1], torch.tensor(ds.episodes[index // 2]["rgb_features"][1]))


def test_training_normalization_never_fits_extreme_validation_features(cache):
    path, _ = cache
    train = ErrorMomentsDataset(path, "risk_model_train")
    val = ErrorMomentsDataset(path, "risk_model_val")
    stats = train.training_statistics()
    assert stats["rgb_features"]["mean"].max() < 1000
    assert val.episodes[0]["rgb_features"].min() > 1e5
    assert stats["output_scale"] == pytest.approx(2.5)
    with pytest.raises(ValueError, match="20 training"):
        val.training_statistics()


def test_cache_integrity_must_match_bytes(cache):
    path, metadata = cache
    target = path / metadata["episodes"][0]["file"]
    target.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="Invalid cached"):
        ErrorMomentsDataset(path, "risk_model_train")


def test_incomplete_cache_cannot_be_trained(cache):
    path, metadata = cache
    metadata["status"] = "running"
    (path / "metadata.json").write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="not completed"):
        ErrorMomentsDataset(path, "risk_model_train")


def test_baseline_risks_use_same_validation_moments(cache):
    path, _ = cache
    ds = ErrorMomentsDataset(path, "risk_model_val")
    report = risk_report(ds, {"rgb": 1., "pcd": 0., "equal": 0.5})
    assert report["episode_count"] == 8
    assert report["mean_risk"] == {"rgb": 2., "pcd": 3., "equal": 1.25}
    assert mean_fusion_risk(ds, 0.5) == pytest.approx(report["mean_risk"]["equal"])


def test_disabled_early_stopping_does_not_stop_even_after_long_plateau():
    from omegaconf import OmegaConf
    training = OmegaConf.create({"early_stopping": False, "min_epochs": 20, "patience": 30})
    assert not should_stop(training, 60, 30)
    assert not should_stop(training, 500, 490)
    training.early_stopping = True
    assert should_stop(training, 60, 30)
    assert not should_stop(training, 19, 30)
    assert not should_stop(training, 60, 29)


def test_collector_reads_only_causal_observation_prefix_with_episode_padding():
    from types import SimpleNamespace
    from modular_policy.workspace.collect_error_moments import read_contexts

    class Array:
        def __init__(self, data):
            self.shape = data.shape
            self.oindex = data

    class Buffer(dict):
        episode_ends = np.array([10, 20])

    pixels = np.broadcast_to(np.arange(20)[:, None, None, None], (20, 3, 2, 2)).copy()
    buffer = Buffer(front_rgb=Array(pixels))
    sampler = SimpleNamespace(indices=np.array([[10, 18, 1, 9]]),
                              sample_sequence=lambda _: {"action": np.zeros((16, 9), np.float32)})
    ds = SimpleNamespace(replay_buffer=buffer, seq_sampler=sampler, action_key="action", obs_keys=["front_rgb"])
    obs, actions, frames = read_contexts(ds, [0], 1, 2, torch.device("cpu"))
    np.testing.assert_array_equal(frames, [[10, 10]])
    assert obs["front_rgb"].shape == (1, 2, 3, 2, 2)
    torch.testing.assert_close(obs["front_rgb"], torch.full((1, 2, 3, 2, 2), 10 / 255))
    assert actions.shape == (1, 16, 9)


def test_per_step_target_scales_and_risk_aggregation(cache):
    from modular_policy.model.error_moments import HistoryErrorMoments
    path, _ = cache
    dataset = ErrorMomentsDataset(path, 'risk_model_train', risk_mode='per_step')
    assert dataset[0]['target'].shape == (2, 3)
    statistics = dataset.training_statistics()
    np.testing.assert_allclose(statistics['step_output_scale'], [2.5, 2.5])
    np.testing.assert_allclose(statistics['global_moments_by_step'], [[1,4,.5],[3,2,-.5]])
    model = HistoryErrorMoments(3, 2, num_steps=2)
    model.set_statistics(statistics)
    assert model.output_scale.shape == (2,)
    weights = np.broadcast_to([1., 0.], (len(dataset), 2))
    # First diffusion step chooses RGB (1), second PCD (2): mean=1.5.
    report = risk_report(dataset, {'stepwise': weights, 'equal': .5})
    assert report['mean_risk']['stepwise'] == pytest.approx(1.5)
    assert mean_fusion_risk(dataset, weights) == pytest.approx(1.5)
    np.testing.assert_allclose(report['risk_by_step']['stepwise'], [1,2])
    repeated = np.full((len(dataset), 2), .5)
    assert mean_fusion_risk(dataset, repeated) == pytest.approx(mean_fusion_risk(dataset, .5))


def test_three_k_outputs_are_psd_scaled_per_step_and_differentiable():
    from modular_policy.model.error_moments import HistoryErrorMoments
    model = HistoryErrorMoments(3, 2, projection_dim=2, hidden_dim=3, head_dim=2, num_steps=100)
    with torch.no_grad():
        model.output_scale.copy_(torch.logspace(-5, 0, 100))
    result = model(torch.zeros(2, 4, 3), torch.ones(2, 4, 2), torch.tensor([1,4]))
    assert result.shape == (2, 100, 3)
    assert model.head[-1].out_features == 300
    check_moments(result.detach().numpy())
    loss = matrix_regression_loss(result, result.detach() * 1.1, model.output_scale)
    assert loss.shape == (2,100) and torch.isfinite(loss).all()
    loss.mean().backward()
    assert model.head[-1].weight.grad is not None
    # Old A architecture/state dict retains exactly three outputs and scalar scale.
    shared = HistoryErrorMoments(3, 2)
    clone = HistoryErrorMoments(3, 2)
    clone.load_state_dict(shared.state_dict(), strict=True)
    assert clone.output_scale.ndim == 0 and clone.head[-1].out_features == 3


@pytest.mark.parametrize('risk_mode',['shared','per_step'])
def test_per_step_training_checkpoint_reports_and_resume_on_synthetic_cache(cache, tmp_path, monkeypatch,risk_mode):
    from omegaconf import OmegaConf
    from modular_policy.workspace.train_error_moments import TrainErrorMomentsWorkspace
    path, metadata = cache
    metadata['identity']['experts'] = {key: {'sha256': key} for key in ['rgb', 'pcd']}
    metadata['signature'] = canonical_hash(metadata['identity'])
    (path/'metadata.json').write_text(json.dumps(metadata))
    cfg = OmegaConf.create({'cache_dir': str(path), 'risk_mode': risk_mode, 'history_contexts': 4,
        'weight_tolerance': 1e-8, 'resume': False,
        'model': {'projection_dim': 2, 'hidden_dim': 3, 'head_dim': 2},
        'optimizer': {'_target_': 'torch.optim.AdamW', 'lr': 3e-4, 'weight_decay': 1e-4},
        'training': {'device': 'cpu', 'seed': 42, 'batch_size': 64, 'num_epochs': 2,
            'early_stopping': False, 'min_epochs': 1, 'patience': 1, 'snapshot_every': 1,
            'gradient_clip_norm': 1.}})
    output = tmp_path / 'synthetic_training'
    workspace = TrainErrorMomentsWorkspace(cfg, output_dir=str(output))
    workspace.run()
    audit = json.loads((output/'training_audit.json').read_text())
    report = json.loads((output/'best_risk_validation.json').read_text())
    assert audit['risk_mode'] == report['risk_mode'] == risk_mode
    assert audit['timesteps'] == [99,0]
    assert np.shape(audit['output_scale']) == ((2,) if risk_mode=='per_step' else ())
    with np.load(output/'best_risk_validation_predictions.npz') as data:
        assert data['moments'].shape == data['targets'].shape == ((16,2,3) if risk_mode=='per_step' else (16,3))
        assert data['rgb_weights'].shape == ((16,2) if risk_mode=='per_step' else (16,))
    for name in ['best.ckpt','best_risk.ckpt','latest.ckpt','epoch_0001.ckpt','epoch_0002.ckpt']:
        assert (output/'checkpoints'/name).exists()
    before = file_hash(output/'checkpoints/latest.ckpt')
    cfg.resume = True
    resumed = TrainErrorMomentsWorkspace(cfg, output_dir=str(output))
    resumed.run()
    assert resumed.epoch == 2
    assert file_hash(output/'checkpoints/latest.ckpt') == before
    # Exercise B's actual checkpoint loader with synthetic experts, not RLBench models.
    from types import SimpleNamespace
    from modular_policy.policy.error_moment_fusion import load_step_fusion_policy
    class TinyExpert(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.parameter = torch.nn.Parameter(torch.ones(1))
            self.n_action_steps = 8
            self.num_inference_steps = 2
            self.noise_scheduler = SimpleNamespace(timesteps=torch.tensor([99,0]), set_timesteps=lambda _:None)
    monkeypatch.setattr('modular_policy.policy.error_moment_fusion.load_frozen_expert',
        lambda path, modality, device: (TinyExpert(), None, {'sha256': modality}))
    monkeypatch.setattr('modular_policy.policy.error_moment_fusion.audit_experts',
        lambda *args: np.array([99,0]))
    policy, provenance = load_step_fusion_policy('rgb.ckpt', 'pcd.ckpt', output/'checkpoints/best_risk.ckpt',
                                                42, torch.device('cpu'))
    assert policy.statistics.num_steps == (2 if risk_mode=='per_step' else None)
    assert policy.timestep_ids == ([99,0] if risk_mode=='per_step' else None)
    assert provenance['scheme'] == ('B' if risk_mode=='per_step' else 'A') and provenance['radius'] is None
