"""Calibration algebra and synthetic caches only; no model inference or simulator."""

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from modular_policy.common.episode_split import canonical_hash
from modular_policy.common.error_moments import file_hash, optimal_weight, quadratic_risk
from modular_policy.common.moment_calibration import (
    audit_calibration_identity, calibration_ids, episode_conformal_threshold,
    fit_shared_radius, normalized_frobenius_residual, robust_weight, worst_case_risk, heldout_ids,
)
from modular_policy.dataset.error_moments_dataset import ErrorMomentsDataset


def test_frobenius_counts_cross_entry_twice_and_uses_fixed_scale():
    value = normalized_frobenius_residual([[2, 3, 4]], [[1, 1, 1]], 2.)
    np.testing.assert_allclose(value, [np.sqrt(23) / 2])
    with pytest.raises(ValueError, match="shapes"):
        normalized_frobenius_residual([[1, 1, 0]], [1, 1, 0], 1.)


def test_finite_sample_rank_does_not_pretend_four_episodes_support_ninety_percent():
    assert episode_conformal_threshold([4, 1, 3, 2], 0.2) == (4., 4)
    with pytest.raises(ValueError, match="finite 90%"):
        episode_conformal_threshold([1, 2, 3, 4], 0.1)
    assert episode_conformal_threshold([1, 2, 3, 4], 0.9999999999999) == (1., 1)


def test_shared_radius_is_conservative_for_each_task():
    episodes = [{"episode_id": i, "task": "a" if i < 4 else "b", "score": float(i + 1)} for i in range(8)]
    fitted = fit_shared_radius(episodes, ["a", "b"], 0.2, 0.1)
    assert fitted["normalized_radius"] == 8
    assert fitted["radius"] == pytest.approx(0.8)
    assert fitted["per_task"]["a"]["normalized_radius"] == 4
    assert fitted["per_task"]["b"]["normalized_radius"] == 8
    with pytest.raises(ValueError, match="Duplicate"):
        fit_shared_radius(episodes + [episodes[0]], ["a", "b"], 0.2, 1.)


def test_zero_radius_recovers_point_weight_and_large_radius_shrinks_to_equal():
    moments = np.array([[1., 4., 0.2], [4., 1., -1.]])
    point = optimal_weight(moments)[0]
    np.testing.assert_allclose(robust_weight(moments, 0)[0], point)
    robust = robust_weight(moments, 100)[0]
    assert np.all(np.abs(robust - 0.5) < np.abs(point - 0.5))
    with pytest.raises(ValueError):
        robust_weight(moments, -1)


def test_robust_formula_matches_grid_minimum_and_psd_worst_case_direction():
    moments = np.array([1., 4., 0.2])
    radius = 0.7
    weight = float(robust_weight(moments, radius)[0])
    grid = np.linspace(0, 1, 10001)
    assert worst_case_risk(moments, weight, radius) <= worst_case_risk(moments, grid, radius).min() + 1e-10
    v = np.array([weight, 1 - weight])
    delta = radius * np.outer(v, v) / np.dot(v, v)
    assert np.linalg.norm(delta, "fro") == pytest.approx(radius)
    perturbed = moments + [delta[0, 0], delta[1, 1], delta[0, 1]]
    assert quadratic_risk(perturbed, weight) == pytest.approx(worst_case_risk(moments, weight, radius))
    assert np.linalg.eigvalsh(delta).min() >= -1e-12


def make_heldout_cache(tmp_path, partition='risk_calibration'):
    repo = Path(__file__).resolve().parents[1]
    original = json.loads((repo / "splits/mt4_seed42.json").read_text())
    split = json.loads((repo / "splits/mt4_risk_model_seed142.json").read_text())
    training = {"source_manifest": original, "risk_split": split, "timesteps": [99, 0],
                "lambda": [0.5, 0.5], "context_stride": 8, "val_noise_repeats": 8,
                "sampler_sha256": "sampler", "dataset_sha256": "dataset", "policy_sha256": "policy",
                "experts": {key: {"sha256": key, "weights": "ema_model", "policy_config": {}} for key in ["rgb", "pcd"]}}
    repeat_key = 'test_noise_repeats' if partition == 'test' else 'calibration_noise_repeats'
    identity = {**copy.deepcopy(training), "collection_partition": partition, repeat_key: 8}
    meta = {"status": "complete", "identity": identity, "signature": canonical_hash(identity), "episodes": []}
    (tmp_path / partition).mkdir()
    for eid in heldout_ids(original, partition):
        first = original["source"]["episode_ends"][eid - 1] if eid else 0
        relative = f"{partition}/episode_{eid:04d}.npz"
        steps = np.array([[[1., 2., 0.2], [3., 4., -0.2]]] * 2)
        np.savez_compressed(tmp_path / relative, episode_id=eid, timesteps=[99, 0], moments_by_step=steps,
                            moments=steps.mean(1), rgb_features=np.ones((2, 3)), pcd_features=np.ones((2, 2)),
                            anchor_frames=[first, first + 8])
        meta["episodes"].append({"episode_id": eid, "role": partition, "file": relative,
             "task": original["task_by_episode"][eid], "contexts": 2, "rgb_dim": 3, "pcd_dim": 2,
             "sha256": file_hash(tmp_path / relative)})
    (tmp_path / "metadata.json").write_text(json.dumps(meta))
    return tmp_path, training, meta


@pytest.fixture
def calibration_cache(tmp_path):
    return make_heldout_cache(tmp_path)


def test_dedicated_calibration_cache_has_sixteen_episodes_and_no_training_statistics(calibration_cache):
    path, training, meta = calibration_cache
    audit_calibration_identity(training, meta["identity"])
    data = ErrorMomentsDataset(path, "risk_calibration", history_contexts=4)
    assert len(data.episodes) == 16 and len(data) == 32
    assert sum(data.sample_weights) == pytest.approx(1.)
    with pytest.raises(ValueError, match="20 training"):
        data.training_statistics()
    with pytest.raises(ValueError, match="not present"):
        ErrorMomentsDataset(path, "risk_model_train")


def test_calibration_cannot_silently_use_different_experts_or_monte_carlo_protocol(calibration_cache):
    _, training, meta = calibration_cache
    bad = copy.deepcopy(meta["identity"])
    bad["experts"]["rgb"]["sha256"] = "different weights"
    with pytest.raises(ValueError, match="Different frozen expert"):
        audit_calibration_identity(training, bad)
    bad = copy.deepcopy(meta["identity"])
    bad["calibration_noise_repeats"] = 4
    with pytest.raises(ValueError, match="MC repeat"):
        audit_calibration_identity(training, bad)


def test_calibration_rejects_missing_or_mixed_training_episodes(calibration_cache):
    path, _, meta = calibration_cache
    meta["episodes"].pop()
    (path / "metadata.json").write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="coverage"):
        ErrorMomentsDataset(path, "risk_calibration")


def test_final_test_cache_is_isolated_and_cannot_fit_statistics(tmp_path):
    path, training, meta = make_heldout_cache(tmp_path, 'test')
    audit_calibration_identity(training, meta['identity'], partition='test')
    dataset = ErrorMomentsDataset(path, 'test')
    assert len(dataset.episodes) == 16
    assert set(e['episode_id'] for e in dataset.episodes) == set(training['source_manifest']['splits']['test'])
    with pytest.raises(ValueError, match='20 training'):
        dataset.training_statistics()
    with pytest.raises(ValueError, match='not present'):
        ErrorMomentsDataset(path, 'risk_calibration')
    bad = copy.deepcopy(meta['identity'])
    bad['test_noise_repeats'] = 4
    with pytest.raises(ValueError, match='MC repeat'):
        audit_calibration_identity(training, bad, partition='test')


def test_final_heldout_ids_reject_overlap(calibration_cache):
    _, training, _ = calibration_cache
    original = copy.deepcopy(training['source_manifest'])
    original['splits']['test'][0] = original['splits']['risk_calibration'][0]
    with pytest.raises(ValueError, match='overlaps'):
        heldout_ids(original, 'test')


def test_final_protocol_rejects_changed_frozen_files(tmp_path):
    from modular_policy.common.final_error_moments import read_final_protocol
    protocol = {'purpose': 'frozen_final_offline_error_moments', 'risk_mode': 'shared',
                'fixed_rgb_weight': .6, 'files': {}}
    names = ['rgb_checkpoint', 'pcd_checkpoint', 'statistics_checkpoint', 'calibration', 'source_manifest', 'risk_split']
    for name in names:
        path = tmp_path / name
        path.write_text(name)
        protocol['files'][name] = {'path': str(path), 'sha256': file_hash(path)}
    path = tmp_path / 'protocol.json'
    path.write_text(json.dumps(protocol))
    assert read_final_protocol(path)['fixed_rgb_weight'] == .6
    (tmp_path / 'calibration').write_text('modified radius')
    with pytest.raises(ValueError, match='calibration'):
        read_final_protocol(path)


def test_final_evaluation_reuses_frozen_radius_and_preserves_files(tmp_path):
    import dill
    import torch
    from omegaconf import OmegaConf
    from modular_policy.model.error_moments import HistoryErrorMoments
    from modular_policy.workspace.calibrate_error_moments import CalibrateErrorMomentsWorkspace
    cache, training, meta = make_heldout_cache(tmp_path, 'test')
    run = tmp_path / 'model_run'
    (run / 'checkpoints').mkdir(parents=True)
    checkpoint = run / 'checkpoints/frozen.ckpt'
    model_cfg = {'projection_dim': 2, 'hidden_dim': 3, 'head_dim': 2}
    model = HistoryErrorMoments(3, 2, **model_cfg)
    torch.save({'cfg': OmegaConf.create({'risk_mode': 'shared', 'history_contexts': 4, 'model': model_cfg}),
                'state_dicts': {'model': model.state_dict()},
                'pickles': {'signature': dill.dumps('training')}}, checkpoint, pickle_module=dill)
    training_signature = canonical_hash(training)
    (run / 'cache_metadata.json').write_text(json.dumps({'identity': training, 'signature': training_signature}))
    (run / 'training_audit.json').write_text(json.dumps({'signature': 'training',
        'train_episode_ids': training['risk_split']['splits']['risk_model_train'],
        'val_episode_ids': training['risk_split']['splits']['risk_model_val']}))
    calibration = tmp_path / 'frozen_calibration.json'
    artifact = {'checkpoint_sha256': file_hash(checkpoint), 'training_cache_signature': training_signature,
        'training_signature': 'training', 'history_contexts': 4, 'global_fallback_rgb_weight': .5,
        'weight_tolerance': 1e-8, 'fitted': {'radius': .1, 'normalized_radius': .1},
        'episode_scores': [{'episode_id': i} for i in calibration_ids(training['source_manifest'])]}
    calibration.write_text(json.dumps(artifact))
    files = {'statistics_checkpoint': checkpoint, 'calibration': calibration}
    for name in ['rgb_checkpoint', 'pcd_checkpoint', 'source_manifest', 'risk_split']:
        path = tmp_path / name
        path.write_text(name)
        files[name] = path
    protocol = {'purpose': 'frozen_final_offline_error_moments', 'risk_mode': 'shared',
        'fixed_rgb_weight': .6, 'seed': 2026, 'test_noise_repeats': 8,
        'files': {name: {'path': str(path), 'sha256': file_hash(path)} for name, path in files.items()}}
    protocol_path = tmp_path / 'protocol.json'
    protocol_path.write_text(json.dumps(protocol))
    meta['identity'].update(final_protocol=protocol, final_protocol_sha256=file_hash(protocol_path), seed=2026)
    meta['signature'] = canonical_hash(meta['identity'])
    (cache / 'metadata.json').write_text(json.dumps(meta))
    cfg = OmegaConf.create({'evaluation_only': True, 'final_protocol': str(protocol_path),
                           'test_cache_dir': str(cache), 'batch_size': 7, 'device': 'cpu'})
    output = tmp_path / 'report'
    workspace = CalibrateErrorMomentsWorkspace(cfg, output_dir=str(output))
    workspace.run()
    report = json.loads((output / 'final_test_report.json').read_text())
    assert report['episode_count'] == 16
    assert report['fixed_rgb_weight'] == .6 and report['frozen_radius'] == .1
    assert set(report['mean_risk']) == {'rgb', 'pcd', 'equal', 'fixed', 'point', 'robust'}
    for name, path in files.items():
        assert file_hash(path) == protocol['files'][name]['sha256']
    with pytest.raises(FileExistsError, match='already exists'):
        workspace.run()
