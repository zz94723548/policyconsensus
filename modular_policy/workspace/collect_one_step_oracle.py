"""Collect independent one-step-deviation branches for RGB/PCD policies."""
import json
import hashlib
import os
import pathlib
import subprocess
import sys

import click
import numpy as np


def _json_safe(value):
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value)
    return value


def _append_jsonl(path, row):
    with pathlib.Path(path).open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(_json_safe(row), sort_keys=True) + '\n')
        stream.flush()
        os.fsync(stream.fileno())


def _worker(args):
    import torch

    # These helpers follow the repository's established checkpoint, RLBench,
    # seed, and fixed-weight conventions.
    from modular_policy.workspace.validate_branch_state import (
        _load_policy, _make_env, _policy_action, _stable_seed, _task_for_episode,
    )

    policy, cfg = _load_policy(args.checkpoint, args.device)
    task_name = _task_for_episode(args.task, args.episode_index)
    # Each repeat is an independent episode seed. Candidate weights within a
    # repeat share this seed, while different repeats do not reuse RNG streams.
    environment_seed = (
        args.test_start_seed
        + args.episode_index
        + args.repeat * args.episodes
    )
    env = _make_env(cfg, task_name, environment_seed)
    try:
        obs, _ = env.reset(seed=environment_seed)
        policy.reset()
        observation_history = [env.get_current_observation()]
        # Reach the decision point with the frozen Router.
        for decision in range(args.decision_index):
            seed = _stable_seed(args.diffusion_seed, task_name,
                                environment_seed, decision)
            action, _ = _policy_action(policy, obs, seed)
            obs, _, done, _, _ = env.step(action)
            if done:
                return {
                    'state_id': f'{task_name}_{environment_seed}_d{args.decision_index}',
                    'task': task_name,
                    'environment_seed': environment_seed,
                    'decision_index': args.decision_index,
                    'candidate_rgb_weight': args.weight,
                    'candidate_pcd_weight': 1.0 - args.weight,
                    'continuation': 'router',
                    'prefix_actions': args.prefix_actions,
                    'repeat': args.repeat,
                    'independent_process': True,
                    'branch_valid': False,
                    'skip_reason': f'episode ended at decision {decision} before target decision',
                }
            observation_history.append(env.get_current_observation())

        state_id = f'{task_name}_{environment_seed}_d{args.decision_index}'
        state_path = pathlib.Path(args.output) / 'features' / f'{state_id}_w{args.weight:g}_r{args.repeat}.pt'
        state_path.parent.mkdir(parents=True, exist_ok=True)
        current_obs = env.get_current_observation()
        cpu_history = [
            {k: torch.as_tensor(v).cpu() for k, v in obs_at_step.items()}
            for obs_at_step in observation_history
        ]
        cpu_bundle = {'current': cpu_history[-1], 'history': cpu_history}
        torch.save(cpu_bundle, state_path)
        feature_digest = hashlib.sha256()
        for history_index, obs_at_step in enumerate(cpu_history):
            for key in sorted(obs_at_step):
                value = obs_at_step[key].contiguous()
                feature_digest.update(f'{history_index}:{key}'.encode('utf-8'))
                feature_digest.update(str(value.dtype).encode('ascii'))
                feature_digest.update(str(tuple(value.shape)).encode('ascii'))
                feature_digest.update(value.numpy().tobytes())

        branch_seed = _stable_seed(args.diffusion_seed, task_name,
                                   environment_seed, args.decision_index)
        action, output = _policy_action(policy, current_obs, branch_seed, args.weight)
        if len(action) < args.prefix_actions:
            raise RuntimeError(
                f'predicted action sequence has {len(action)} steps, but '
                f'--prefix-actions={args.prefix_actions}')
        # The candidate policy controls exactly one chunk.
        obs, reward, done, _, _ = env.step(action[:args.prefix_actions])
        chunk_return = float(reward)
        total_return = chunk_return
        success = bool(reward >= 1)
        continuation_decisions = 0
        decision = args.decision_index + 1
        while not done:
            seed = _stable_seed(args.diffusion_seed, task_name,
                                environment_seed, decision)
            continuation_action, _ = _policy_action(policy, obs, seed, None)
            obs, reward, done, _, _ = env.step(continuation_action)
            total_return += float(reward)
            success = success or bool(reward >= 1)
            continuation_decisions += 1
            decision += 1

        applied = output.get('applied_weights', output.get('weights'))
        applied = applied[0].detach().cpu().tolist()
        return {
            'state_id': state_id,
            'task': task_name,
            'environment_seed': environment_seed,
            'decision_index': args.decision_index,
            'candidate_rgb_weight': args.weight,
            'candidate_pcd_weight': 1.0 - args.weight,
            'applied_weights': applied,
            'continuation': 'router',
            'prefix_actions': args.prefix_actions,
            'chunk_return': chunk_return,
            'episode_return': total_return,
            'discount_factor': 1.0,
            'discounted_return': total_return,  # gamma=1, so equal to episode_return
            'terminal_success': success,
            'repeat': args.repeat,
            'independent_process': True,
            'branch_valid': True,
            'decision_diffusion_seed': branch_seed,
            'continuation_decisions': continuation_decisions,
            'state_feature_path': str(state_path),
            'state_feature_sha256': feature_digest.hexdigest(),
        }
    finally:
        env.close()


@click.command()
@click.option('--checkpoint', required=True, type=click.Path(exists=True, dir_okay=False))
@click.option('--task', required=True)
@click.option('--episodes', type=click.IntRange(min=1), required=True)
@click.option('--test-start-seed', type=int, required=True)
@click.option('--diffusion-seed', type=int, required=True)
@click.option('--decision-index', type=click.IntRange(min=0), required=True)
@click.option('--weights', required=True, help='Comma-separated RGB weights in [0,1].')
@click.option('--prefix-actions', type=click.IntRange(min=1), default=8, show_default=True)
@click.option('--continuation', type=click.Choice(['router']), default='router', show_default=True)
@click.option('--num-repeats', type=click.IntRange(min=1), default=1, show_default=True)
@click.option('--independent-process/--no-independent-process', default=True, show_default=True)
@click.option('--output', required=True, type=click.Path(file_okay=False))
@click.option('--device', default='cuda:0', show_default=True)
@click.option('--_worker', is_flag=True, hidden=True)
@click.option('--_episode-index', type=int, hidden=True)
@click.option('--_weight', type=float, hidden=True)
@click.option('--_repeat', type=int, hidden=True)
def collect_one_step_oracle(checkpoint, task, episodes, test_start_seed,
                            diffusion_seed, decision_index, weights,
                            prefix_actions, continuation, num_repeats,
                            independent_process, output, device, _worker,
                            _episode_index, _weight, _repeat):
    """Run one weighted action chunk, then Router to episode termination."""
    if _worker:
        class Args:
            pass
        args = Args()
        args.checkpoint, args.task, args.test_start_seed = checkpoint, task, test_start_seed
        args.diffusion_seed, args.decision_index = diffusion_seed, decision_index
        args.prefix_actions, args.device, args.output = prefix_actions, device, output
        args.episodes = episodes
        args.episode_index, args.weight, args.repeat = _episode_index, _weight, _repeat
        row = _worker_run(args)
        _append_jsonl(pathlib.Path(output) / 'branches.jsonl', row)
        return

    try:
        weight_values = [float(value.strip()) for value in weights.split(',')]
    except ValueError as exc:
        raise click.BadParameter('use comma-separated values, for example 0,1') from exc
    if (not weight_values or any(not np.isfinite(w) or w < 0 or w > 1 for w in weight_values)
            or len(weight_values) != len(set(weight_values))):
        raise click.BadParameter('weights must be unique finite values in [0,1]')
    if not independent_process:
        raise click.UsageError('B4 collection requires --independent-process')
    output_path = pathlib.Path(output)
    if output_path.exists() and any(output_path.iterdir()):
        raise click.ClickException(f'output directory is not empty: {output_path}')
    output_path.mkdir(parents=True, exist_ok=True)
    script = pathlib.Path(__file__).resolve().parents[2] / 'scripts' / 'collect_one_step_oracle.py'
    branch_count = episodes * len(weight_values) * num_repeats
    branches_path = output_path / 'branches.jsonl'
    states_path = output_path / 'states.jsonl'
    branches_path.write_text('', encoding='utf-8')
    states_path.write_text('', encoding='utf-8')
    for episode_index in range(episodes):
        for weight in weight_values:
            for repeat in range(num_repeats):
                command = [sys.executable, str(script), '--checkpoint', str(pathlib.Path(checkpoint).resolve()),
                           '--task', task, '--episodes', str(episodes), '--test-start-seed', str(test_start_seed),
                           '--diffusion-seed', str(diffusion_seed), '--decision-index', str(decision_index),
                           '--weights', str(weight), '--prefix-actions', str(prefix_actions),
                           '--continuation', continuation, '--num-repeats', '1', '--independent-process',
                           '--output', str(output_path.resolve()), '--device', device, '--_worker',
                           '--_episode-index', str(episode_index), '--_weight', str(weight), '--_repeat', str(repeat)]
                click.echo(f'[{episode_index + 1}/{episodes}, weight={weight:g}, repeat={repeat}] starting isolated simulator process')
                subprocess.run(command, check=True, env=os.environ.copy())

    commit = None
    try:
        commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    except Exception:
        pass
    with branches_path.open(encoding='utf-8') as stream:
        branch_rows = [json.loads(line) for line in stream if line.strip()]
    state_groups = {}
    valid_branch_rows = [row for row in branch_rows if row.get('branch_valid', True)]
    for row in valid_branch_rows:
        state_groups.setdefault(row['state_id'], []).append(row)
    state_rows = []
    for state_id, rows in sorted(state_groups.items()):
        hashes = {row['state_feature_sha256'] for row in rows}
        exact_match = len(hashes) == 1
        state_rows.append({
            'state_id': state_id,
            'task': rows[0]['task'],
            'environment_seed': rows[0]['environment_seed'],
            'decision_index': rows[0]['decision_index'],
            'source_policy': 'router',
            'state_feature_path': rows[0]['state_feature_path'],
            'canonical_observation_match': exact_match,
            'branch_features': [
                {'weight_rgb': row['candidate_rgb_weight'],
                 'repeat': row['repeat'],
                 'path': row['state_feature_path'],
                 'sha256': row['state_feature_sha256']}
                for row in rows
            ],
        })
    states_path.write_text(
        ''.join(json.dumps(row, sort_keys=True) + '\n' for row in state_rows),
        encoding='utf-8',
    )
    summary = {
        'checkpoint': str(pathlib.Path(checkpoint).resolve()), 'git_commit': commit,
        'task': task, 'episodes': episodes, 'test_start_seed': test_start_seed,
        'diffusion_seed': diffusion_seed, 'decision_index': decision_index,
        'weights_rgb': weight_values, 'continuation': continuation,
        'prefix_actions': prefix_actions, 'num_repeats': num_repeats,
        'independent_process': True, 'branch_count': branch_count,
        'completed_branch_count': len(branch_rows),
        'valid_branch_count': len(valid_branch_rows),
        'skipped_branch_count': len(branch_rows) - len(valid_branch_rows),
        'device': device, 'success_ci': None,
        'discount_factor': 1.0,
        'canonical_observations_verified': all(
            row['canonical_observation_match'] for row in state_rows
        ),
        'observation_mismatch_groups': sum(
            not row['canonical_observation_match'] for row in state_rows
        ),
        'repeat_environment_seed_scheme': (
            'test_start_seed + episode_index + repeat * episodes'
        ),
        'note': 'Raw branch collection only; aggregate success rates and confidence intervals separately.',
    }
    (output_path / 'summary.json').write_text(json.dumps(summary, indent=2, sort_keys=True), encoding='utf-8')
    click.echo(json.dumps(summary, indent=2))


def _worker_run(args):
    return _worker(args)


if __name__ == '__main__':
    collect_one_step_oracle()
