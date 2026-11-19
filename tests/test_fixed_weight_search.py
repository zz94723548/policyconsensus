import json
import pytest
from modular_policy.common.fixed_weight_search import summarize_fixed_weights, TASKS


def make_logs(root):
    for percent in range(0, 101, 10):
        folder = root / f'rgb_{percent/100:.1f}'
        folder.mkdir()
        count = round(percent / 100 * 80)
        log = {'fusion': {'mode': 'fixed', 'fixed_rgb_weight': percent / 100,
            'seed': 42, 'test_start_seed': 1000, 'n_parallel_envs': 8, 'n_test': 80,
            'calibration_sha256': 'calibration', 'rgb': {'sha256': 'rgb'}, 'pcd': {'sha256': 'pcd'}},
            'episode_results': {str(1000 + i): {'task': TASKS[i % 4], 'seed': 1000 + i,
                'success': i < count} for i in range(80)}, 'mean_success_rate': count / 80}
        (folder / 'eval_log.json').write_text(json.dumps(log))


def test_grid_summary_contains_all_candidates_and_task_rates(tmp_path):
    make_logs(tmp_path)
    report = summarize_fixed_weights(tmp_path)
    assert len(report['rows']) == 11 and not report['missing_rgb_percent']
    assert report['rows'][5]['success_rate'] == .5
    assert report['rows'][5]['open_box'] == .5
    assert report['best_rgb_percent'] == [100]


def test_grid_summary_reports_missing_without_inventing_results(tmp_path):
    make_logs(tmp_path)
    (tmp_path / 'rgb_0.5' / 'eval_log.json').unlink()
    report = summarize_fixed_weights(tmp_path)
    assert report['missing_rgb_percent'] == [50]
    assert len(report['rows']) == 10


def test_grid_summary_rejects_changed_expert_or_seed(tmp_path):
    make_logs(tmp_path)
    path = tmp_path / 'rgb_0.5' / 'eval_log.json'
    log = json.loads(path.read_text())
    log['fusion']['seed'] = 43
    path.write_text(json.dumps(log))
    with pytest.raises(ValueError, match='protocol'):
        summarize_fixed_weights(tmp_path)
