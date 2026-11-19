"""Summarize completed native eval logs; never starts a simulation."""
import json
from pathlib import Path


TASKS = ['open_box', 'toilet_seat_up', 'open_drawer', 'take_umbrella_out_of_umbrella_stand']


def summarize_fixed_weights(root):
    rows, missing, reference = [], [], None
    for percent in range(0, 101, 10):
        weight = percent / 100
        path = Path(root) / f'rgb_{weight:.1f}' / 'eval_log.json'
        if not path.exists():
            missing.append(percent)
            continue
        log = json.loads(path.read_text())
        info = log['fusion']
        if info['mode'] != 'fixed' or abs(info['fixed_rgb_weight'] - weight) > 1e-8:
            raise ValueError(f'Unexpected weight/mode in {path}')
        episodes = log['episode_results']
        protocol = {k: info[k] for k in ['seed', 'test_start_seed', 'n_parallel_envs', 'n_test', 'calibration_sha256']}
        protocol['expert_hashes'] = [info[m]['sha256'] for m in ['rgb', 'pcd']]
        protocol['episode_tasks'] = sorted((k, e['task'], e['seed']) for k, e in episodes.items())
        if len(episodes) != info['n_test'] or reference is not None and protocol != reference:
            raise ValueError(f'Inconsistent evaluation protocol/episodes in {path}')
        reference = protocol
        row = {'rgb_percent': percent, 'pcd_percent': 100 - percent,
               'successes': sum(bool(e['success']) for e in episodes.values()), 'episodes': len(episodes)}
        if not episodes:
            raise ValueError(f'No episodes in {path}')
        row['success_rate'] = row['successes'] / row['episodes']
        if abs(row['success_rate'] - log['mean_success_rate']) > 1e-8:
            raise ValueError(f'Overall metric/count mismatch in {path}')
        for task in TASKS:
            group = [e for e in episodes.values() if e['task'] == task]
            if not group or len(group) * 4 != len(episodes):
                raise ValueError(f'Task imbalance in {path}')
            row[task] = sum(bool(e['success']) for e in group) / len(group)
        rows.append(row)
    best = max((r['success_rate'] for r in rows), default=None)
    return {'rows': rows, 'missing_rgb_percent': missing,
            'best_rgb_percent': [r['rgb_percent'] for r in rows if r['success_rate'] == best],
            'best_success_rate': best,
            'scope': 'Search-set results; freeze selected weight before independent evaluation.'}
