"""Read existing simulation/offline reports; never evaluates a model."""
import json
from pathlib import Path

from modular_policy.common.fixed_weight_search import TASKS


def compare_matched(root):
    root=Path(root)
    locations={m:root/m/'eval_log.json' for m in ['rgb','pcd','equal','point','robust','router',
               'point_latest','robust_latest','per_step_latest','router_expert_data']}
    local_fixed=root/'fixed_0.6/eval_log.json'
    locations['fixed']=local_fixed if local_fixed.exists() else root.parent/'eval_fixed_grid_seed42/rgb_0.6/eval_log.json'
    locations['per_step']=root.parent/'eval_scheme_b_seed42/point/eval_log.json'
    offline_path=root/'offline_errors/router/offline_error_report.json'
    offline=json.loads(offline_path.read_text()) if offline_path.exists() else None
    rows,missing,reference=[],[],None
    for method in ['rgb','pcd','equal','fixed','point','robust','per_step','router',
                   'point_latest','robust_latest','per_step_latest','router_expert_data']:
        path=locations[method]
        if not path.exists():
            missing.append(method)
            continue
        log=json.loads(path.read_text())
        info=log['fusion']
        if method=='fixed' and (info.get('mode')!='fixed' or abs(info.get('fixed_rgb_weight',-1)-.6)>1e-8):
            raise ValueError(f'Expected fixed RGB weight 0.6: {path}')
        episodes=log['episode_results']
        protocol={k:info[k] for k in ['seed','test_start_seed','n_parallel_envs','n_test']}
        protocol['experts']=[info[m]['sha256'] for m in ['rgb','pcd']]
        protocol['tasks']=sorted((key,e['task'],e['seed']) for key,e in episodes.items())
        if reference is not None and protocol!=reference:
            raise ValueError(f'Mismatched experts/seed/task protocol: {path}')
        reference=protocol
        successes=sum(bool(e['success']) for e in episodes.values())
        if len(episodes)!=info['n_test'] or not episodes or abs(successes/len(episodes)-log['mean_success_rate'])>1e-8:
            raise ValueError(f'Inconsistent success count: {path}')
        if method=='router' and offline and info['router_sha256']!=offline['router']['router_sha256']:
            raise ValueError('Router simulation and offline reports use different checkpoints')
        row={'method':method,'successes':successes,'episodes':len(episodes),
             'success_rate':log['mean_success_rate'],
             'offline_epsilon_mse':offline['mean_risk'].get(method) if offline else None,
             'simulation_source':str(path)}
        for task in TASKS:
            selected=[e for e in episodes.values() if e['task']==task]
            if not selected:
                raise ValueError('Missing task in simulation log')
            row[task]=sum(bool(e['success']) for e in selected)/len(selected)
        rows.append(row)
    return {'rows':rows,'missing_simulation_methods':missing,
        'offline_source':str(offline_path) if offline else None,
        'scope':'Existing batches with matched seed protocol, not guaranteed identical physical initial states; offline MSE and success rate are distinct metrics'}
