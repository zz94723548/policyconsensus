"""RLBench branch-state restoration and weight-effect validation."""
import copy
import json
import os
import pathlib
import random

import click
import dill
import hydra
import numpy as np
import torch
import tqdm

from modular_policy.workspace.base_workspace import BaseWorkspace


def _json_safe(value):
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.floating, float)):
        return float(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    return value


def _write_json(path, value):
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(value), indent=2, sort_keys=True))


def _write_jsonl(path, rows):
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as stream:
        for row in rows:
            stream.write(json.dumps(_json_safe(row), sort_keys=True) + '\n')


def _stable_seed(base_seed, task, environment_seed, decision_index):
    import hashlib
    identity = f'{base_seed}|{task}|{environment_seed}|{decision_index}'
    digest = hashlib.blake2b(identity.encode(), digest_size=8).digest()
    return int.from_bytes(digest, 'little') % (2**63 - 1)


def _set_deterministic(seed):
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def _load_policy(checkpoint, device):
    with open(checkpoint, 'rb') as stream:
        payload = torch.load(stream, pickle_module=dill, map_location='cpu')
    cfg = payload['cfg']
    workspace_cls = hydra.utils.get_class(cfg._target_)
    workspace: BaseWorkspace = workspace_cls(cfg, output_dir=None)
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)
    policy = workspace.ema_model if cfg.training.get('use_ema', False) else workspace.model
    policy.eval().to(torch.device(device))
    return policy, cfg


def _make_env(cfg, task_name, seed):
    from modular_policy.env.rlbench.env import RlbenchEnv
    from modular_policy.gymnasium_util.multistep_wrapper import MultiStepWrapper
    runner = cfg.task.env_runner
    raw_env = RlbenchEnv(
        task_name=task_name,
        image_size=runner.get('image_size', 128),
        num_points=runner.get('num_points'),
        enable_depth=runner.get('enable_depth', False),
        enable_dino=runner.get('enable_dino', False),
        image_scale=runner.get('image_scale', 255.0),
        seed=seed,
        camera_names=list(runner.get('camera_names', ['front'])),
        robot_state_ports=list(runner.get('robot_state_ports', [])),
        max_episode_steps=runner.get('max_episode_steps', 250),
        zarr_path=runner.get('zarr_path'),
    )
    return MultiStepWrapper(
        raw_env,
        n_obs_steps=runner.get('n_obs_steps', cfg.n_obs_steps),
        n_action_steps=runner.get('n_action_steps', cfg.n_action_steps),
        max_episode_steps=runner.get('max_episode_steps', 250),
        reward_agg_method='sum',
    )


def _policy_action(policy, obs, diffusion_seed, fixed_rgb_weight=None):
    if fixed_rgb_weight is None:
        policy.set_inference_fixed_weights(None)
    else:
        policy.set_inference_fixed_weights([
            float(fixed_rgb_weight), 1.0 - float(fixed_rgb_weight)
        ])
    policy.set_inference_diffusion_seeds([diffusion_seed])
    obs_tensor = {
        key: torch.from_numpy(value).to(
            device=policy.device, dtype=policy.dtype
        ).unsqueeze(0)
        for key, value in obs.items()
        if key in policy.get_observation_ports()
    }
    with torch.inference_mode():
        output = policy.predict_action(obs_tensor)
    return output['action'][0].detach().cpu().numpy(), output


def _observation_diff(before, after):
    result = {}
    for key in sorted(before.keys() & after.keys()):
        lhs, rhs = np.asarray(before[key]), np.asarray(after[key])
        if lhs.shape != rhs.shape:
            result[key] = {'shape_before': lhs.shape, 'shape_after': rhs.shape}
            continue
        delta = np.abs(lhs.astype(np.float64) - rhs.astype(np.float64))
        result[key] = {
            'mae': float(delta.mean()),
            'max_abs': float(delta.max(initial=0.0)),
        }
        if ('pointcloud' in key and lhs.ndim == 2 and rhs.ndim == 2
                and lhs.shape[-1] >= 3 and rhs.shape[-1] >= 3):
            lhs_xyz = torch.as_tensor(lhs[:, :3], dtype=torch.float32)
            rhs_xyz = torch.as_tensor(rhs[:, :3], dtype=torch.float32)
            distances = torch.cdist(lhs_xyz[None], rhs_xyz[None])[0]
            result[key].update({
                'chamfer_l2': float(
                    distances.min(dim=1).values.square().mean()
                    + distances.min(dim=0).values.square().mean()
                ),
                'mean_nearest_distance': float(
                    0.5 * (
                        distances.min(dim=1).values.mean()
                        + distances.min(dim=0).values.mean()
                    )
                ),
                'centroid_l2': float(torch.linalg.vector_norm(
                    lhs_xyz.mean(dim=0) - rhs_xyz.mean(dim=0)
                )),
            })
    return result


def _object_diff(before, after):
    lhs = {item['handle']: item for item in before}
    rhs = {item['handle']: item for item in after}
    rows = []
    for handle in sorted(lhs.keys() & rhs.keys()):
        rows.append({
            'handle': handle,
            'name': lhs[handle]['name'],
            'pose_max_abs': float(np.max(np.abs(
                lhs[handle]['pose'] - rhs[handle]['pose']
            ))),
            'linear_velocity_max_abs': float(np.max(np.abs(
                lhs[handle]['linear_velocity'] - rhs[handle]['linear_velocity']
            ))),
            'angular_velocity_max_abs': float(np.max(np.abs(
                lhs[handle]['angular_velocity'] - rhs[handle]['angular_velocity']
            ))),
        })
    return rows


def _capture_at_decision(env, policy, task, environment_seed, diffusion_seed,
                         decision_index):
    obs, _ = env.reset(seed=environment_seed)
    policy.reset()
    policy.set_inference_fixed_weights(None)
    for current_decision in range(decision_index):
        seed = _stable_seed(
            diffusion_seed, task, environment_seed, current_decision
        )
        action, _ = _policy_action(policy, obs, seed)
        obs, _, done, _, _ = env.step(action)
        if done:
            raise RuntimeError(
                f'Episode ended before decision {decision_index} '
                f'(ended at {current_decision})'
            )
    return copy.deepcopy(env.get_current_observation()), env.get_branch_state()


def _run_branch(env, policy, snapshot, task, environment_seed, decision_index,
                diffusion_seed, weight, prefix_actions, continuation):
    env.set_branch_state(snapshot)
    obs = env.get_current_observation()
    branch_seed = _stable_seed(
        diffusion_seed, task, environment_seed, decision_index
    )
    action, output = _policy_action(policy, obs, branch_seed, weight)
    action = action[:prefix_actions]
    obs, reward, done, _, _ = env.step(action)
    prefix_return = float(reward)
    success = bool(reward >= 1)
    continuation_decision = decision_index + 1
    while continuation == 'router' and not done:
        seed = _stable_seed(
            diffusion_seed, task, environment_seed, continuation_decision
        )
        action, _ = _policy_action(policy, obs, seed, None)
        obs, reward, done, _, _ = env.step(action)
        success = success or bool(reward >= 1)
        continuation_decision += 1
    applied = output.get('applied_weights', output.get('weights'))
    return {
        'task': task,
        'environment_seed': environment_seed,
        'decision_index': decision_index,
        'weight_rgb': weight,
        'weight_pointcloud': 1.0 - weight,
        'decision_diffusion_seed': branch_seed,
        'prefix_return': prefix_return,
        'terminal_success': success,
        'applied_weights': applied[0].detach().cpu().tolist(),
        'continuation_decisions': continuation_decision - decision_index - 1,
    }


def _task_for_episode(task_group, episode_index):
    from modular_policy.env.rlbench.factory import get_subtasks
    subtasks = get_subtasks(task_group)
    return subtasks[episode_index % len(subtasks)]


def run_restore_validation(checkpoint, task, episodes, test_start_seed,
                           diffusion_seed, decision_index, repeats, output,
                           device):
    _set_deterministic(diffusion_seed)
    policy, cfg = _load_policy(checkpoint, device)
    first_task = _task_for_episode(task, 0)
    env = _make_env(cfg, first_task, test_start_seed)
    output = pathlib.Path(output)
    rows = []
    try:
        for episode_index in tqdm.tqdm(
                range(episodes), desc='B3 restore episodes', dynamic_ncols=True):
            this_task = _task_for_episode(task, episode_index)
            environment_seed = test_start_seed + episode_index
            if env.env.task_name != this_task:
                env.env.set_task(this_task, seed=environment_seed)
            before_obs, snapshot = _capture_at_decision(
                env, policy, this_task, environment_seed, diffusion_seed,
                decision_index,
            )
            before_objects = snapshot['env_state']['object_states']
            for repeat in range(repeats):
                restored_obs = env.set_branch_state(snapshot)
                after_objects = env.get_branch_object_state()
                rows.append({
                    'task': this_task,
                    'environment_seed': environment_seed,
                    'decision_index': decision_index,
                    'repeat': repeat,
                    'observation_diff': _observation_diff(
                        before_obs[-1] if not isinstance(before_obs, dict) else {
                            key: value[-1] for key, value in before_obs.items()
                        },
                        restored_obs,
                    ),
                    'object_diff': _object_diff(before_objects, after_objects),
                })
            _write_jsonl(output / 'restore_trials.jsonl', rows)
    finally:
        env.close()
    _write_jsonl(output / 'restore_trials.jsonl', rows)
    summary = {
        'episodes': episodes,
        'repeats': repeats,
        'trials': len(rows),
        'checkpoint': checkpoint,
    }
    _write_json(output / 'restore_summary.json', summary)
    return summary


def run_branch_validation(checkpoint, task, episodes, test_start_seed,
                          diffusion_seed, decision_index, weights, repeats,
                          prefix_actions, continuation, output, device,
                          episode_indices, shuffle_branches, order_seed):
    _set_deterministic(diffusion_seed)
    policy, cfg = _load_policy(checkpoint, device)
    selected_indices = (
        list(range(episodes)) if episode_indices is None
        else list(episode_indices)
    )
    invalid_indices = [
        index for index in selected_indices if index < 0 or index >= episodes
    ]
    if invalid_indices:
        raise click.BadParameter(
            f'episode indices must be in [0, {episodes - 1}]; '
            f'invalid values: {invalid_indices}',
            param_hint='--episode-indices',
        )
    first_index = selected_indices[0]
    first_task = _task_for_episode(task, first_index)
    env = _make_env(cfg, first_task, test_start_seed + first_index)
    output = pathlib.Path(output)
    rows = []
    progress = tqdm.tqdm(
        total=len(selected_indices) * len(weights) * repeats,
        desc='B3 branch rollouts', dynamic_ncols=True,
    )
    execution_index = 0
    try:
        for episode_index in selected_indices:
            this_task = _task_for_episode(task, episode_index)
            environment_seed = test_start_seed + episode_index
            if env.env.task_name != this_task:
                env.env.set_task(this_task, seed=environment_seed)
            _, snapshot = _capture_at_decision(
                env, policy, this_task, environment_seed, diffusion_seed,
                decision_index,
            )
            branch_specs = [
                (weight, repeat)
                for weight in weights
                for repeat in range(repeats)
            ]
            state_order_seed = _stable_seed(
                order_seed, this_task, environment_seed, decision_index
            )
            if shuffle_branches:
                random.Random(state_order_seed).shuffle(branch_specs)
            for branch_order, (weight, repeat) in enumerate(branch_specs):
                row = _run_branch(
                    env, policy, snapshot, this_task, environment_seed,
                    decision_index, diffusion_seed, weight,
                    prefix_actions, continuation,
                )
                row.update({
                    'episode_index': episode_index,
                    'repeat': repeat,
                    'branch_order': branch_order,
                    'execution_index': execution_index,
                    'order_seed': order_seed,
                    'state_order_seed': state_order_seed,
                    'branches_shuffled': shuffle_branches,
                })
                rows.append(row)
                execution_index += 1
                progress.update(1)
            _write_jsonl(output / 'branches.jsonl', rows)
    finally:
        progress.close()
        env.close()
    _write_jsonl(output / 'branches.jsonl', rows)
    groups = {}
    for row in rows:
        key = (row['task'], row['environment_seed'], row['weight_rgb'])
        groups.setdefault(key, []).append(float(row['terminal_success']))
    stats = []
    state_weight_means = {}
    for (task_name, seed, weight), values in sorted(groups.items()):
        success_mean = float(np.mean(values))
        stats.append({
            'task': task_name,
            'environment_seed': seed,
            'weight_rgb': weight,
            'repeats': len(values),
            'success_mean': success_mean,
            'success_variance': float(np.var(values, ddof=1)) if len(values) > 1 else 0.0,
        })
        state_weight_means.setdefault((task_name, seed), []).append(
            (weight, success_mean)
        )
    _write_json(output / 'within_weight_variance.json', stats)
    effects = []
    for (task_name, seed), values in sorted(state_weight_means.items()):
        ordered = sorted(values, key=lambda item: item[1], reverse=True)
        best_value = ordered[0][1]
        best_weights = [weight for weight, value in ordered if value == best_value]
        second_value = next(
            (value for _, value in ordered if value < best_value), best_value
        )
        effects.append({
            'task': task_name,
            'environment_seed': seed,
            'best_weights_rgb': best_weights,
            'best_success_mean': best_value,
            'second_distinct_success_mean': second_value,
            'best_second_gap': best_value - second_value,
            'between_weight_range': (
                max(value for _, value in values)
                - min(value for _, value in values)
            ),
        })
    _write_json(output / 'between_weight_effect.json', effects)
    summary = {
        'states': len(selected_indices),
        'episodes': episodes,
        'episode_indices': selected_indices,
        'weights': list(weights),
        'repeats': repeats,
        'branches': len(rows),
        'checkpoint': checkpoint,
        'continuation': continuation,
        'branches_shuffled': shuffle_branches,
        'order_seed': order_seed,
    }
    _write_json(output / 'summary.json', summary)
    return summary


def _common_options(function):
    options = [
        click.option('--checkpoint', required=True, type=click.Path(exists=True, dir_okay=False)),
        click.option('--task', default='mt4', show_default=True),
        click.option('--episodes', type=click.IntRange(min=1), required=True),
        click.option('--test-start-seed', type=int, required=True),
        click.option('--diffusion-seed', type=int, required=True),
        click.option('--decision-index', type=click.IntRange(min=0), default=4, show_default=True),
        click.option('--output', required=True, type=click.Path(file_okay=False)),
        click.option('--device', default='cuda:0', show_default=True),
    ]
    for option in reversed(options):
        function = option(function)
    return function


@click.group()
def validate_branch_state():
    """Validate RLBench state restoration and candidate-weight branches."""


@validate_branch_state.command()
@_common_options
def smoke(**kwargs):
    kwargs['repeats'] = 1
    summary = run_restore_validation(**kwargs)
    click.echo(json.dumps(summary, indent=2))


@validate_branch_state.command()
@_common_options
@click.option('--repeats', type=click.IntRange(min=1), default=5, show_default=True)
def restore(repeats, **kwargs):
    summary = run_restore_validation(repeats=repeats, **kwargs)
    click.echo(json.dumps(summary, indent=2))


def _parse_weights(_, __, value):
    try:
        weights = tuple(float(item.strip()) for item in value.split(','))
    except ValueError as exc:
        raise click.BadParameter('use comma-separated numeric RGB weights') from exc
    if not weights or any(not 0 <= weight <= 1 for weight in weights):
        raise click.BadParameter('every RGB weight must be in [0, 1]')
    return weights


def _parse_episode_indices(_, __, value):
    if value is None:
        return None
    try:
        indices = tuple(int(item.strip()) for item in value.split(','))
    except ValueError as exc:
        raise click.BadParameter(
            'use comma-separated zero-based episode indices'
        ) from exc
    if not indices or any(index < 0 for index in indices):
        raise click.BadParameter('episode indices must be non-negative')
    if len(indices) != len(set(indices)):
        raise click.BadParameter('episode indices must not contain duplicates')
    return indices


def _branch_options(function):
    function = _common_options(function)
    function = click.option('--weights', callback=_parse_weights, required=True)(function)
    function = click.option('--repeats', type=click.IntRange(min=1), default=5, show_default=True)(function)
    function = click.option('--prefix-actions', type=click.IntRange(min=1), default=8, show_default=True)(function)
    function = click.option('--continuation', type=click.Choice(['router', 'none']), default='router', show_default=True)(function)
    function = click.option(
        '--episode-indices', callback=_parse_episode_indices,
        help='Comma-separated zero-based indices selected from --episodes.',
    )(function)
    function = click.option(
        '--shuffle-branches/--no-shuffle-branches', default=False,
        show_default=True,
        help='Shuffle weight/repeat execution order independently per state.',
    )(function)
    function = click.option(
        '--order-seed', type=int, default=0, show_default=True,
        help='Seed controlling deterministic branch-order shuffling.',
    )(function)
    return function


@validate_branch_state.command()
@_branch_options
def repeat(**kwargs):
    summary = run_branch_validation(**kwargs)
    click.echo(json.dumps(summary, indent=2))


@validate_branch_state.command()
@_branch_options
def compare(**kwargs):
    summary = run_branch_validation(**kwargs)
    click.echo(json.dumps(summary, indent=2))

