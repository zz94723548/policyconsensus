"""
Usage:
python experiments/eval_policy_sim.py --checkpoint path/to/ckpt -o path/to/output_dir
"""

if __name__ == "__main__":
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)

import sys
# use line-buffering for both stdout and stderr
sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode='w', buffering=1)

import os
import random
import re
import pathlib
import click
import hydra
import omegaconf
import torch
import dill
import wandb
import json
import numpy as np
from modular_policy.env_runner.base_runner import BaseRunner
from modular_policy.workspace.base_workspace import BaseWorkspace
from modular_policy.policy.base_policy import BasePolicy
from typing import List


def make_json_safe(obj):
    if isinstance(obj, dict):
        return {k: make_json_safe(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_json_safe(v) for v in obj]
    elif isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.floating, float)):
        return float(obj)
    elif isinstance(obj, (np.integer, int)):
        return int(obj)
    else:
        return obj

@click.command()
@click.option('-c', '--checkpoint', required=True, help="either a .ckpt file or a directory containing .ckpt files")
@click.option('-o', '--output_dir', required=True, help="output directory for eval info dump")
@click.option('-n', '--num_exp', default=3, help="num experiments to run")
@click.option('--num-videos', type=click.IntRange(min=0), default=None,
              help="number of episodes to record videos for (defaults to checkpoint config)")
@click.option('-d', '--device', default='cuda:0', help="device to run on")
@click.option('-u', '--update', is_flag=True, help="weather to update `success_rate` in ckpt file name")
@click.option('-s', '--dropout_obs', default=(), multiple=True, help="a list of sensor ports to drop i.e. set to 0s")
@click.option('--expert', type=click.Choice(['rgb', 'pointcloud']), default=None,
              help="run only the selected expert (DiffusionUnetsPolicy checkpoints)")
@click.option('--fixed-weights', default=None, metavar='RGB_WEIGHT,PCD_WEIGHT',
              help="exact inference mixture probabilities; values must sum to 1")
@click.option('--test-start-seed', type=int, default=1000, show_default=True,
              help="first RLBench episode seed")
@click.option('--diffusion-seed', type=int, default=None,
              help="base seed for per-episode, per-decision diffusion RNG streams")
@click.option('--deterministic', is_flag=True,
              help="enable deterministic random seeds and Torch backends")
@click.option('--episode-log', type=click.Path(dir_okay=False), default=None,
              help="write one JSON object per episode")
@click.option('--decision-log', type=click.Path(dir_okay=False), default=None,
              help="write one JSON object per policy replanning decision")
def eval_policy_sim(checkpoint, output_dir, num_exp, num_videos, device, update,
                    dropout_obs, expert, fixed_weights, test_start_seed,
                    diffusion_seed, deterministic, episode_log, decision_log):
    if expert is not None and fixed_weights is not None:
        raise click.UsageError("--expert and --fixed-weights are mutually exclusive")

    parsed_fixed_weights = None
    if fixed_weights is not None:
        try:
            parsed_fixed_weights = [float(x.strip()) for x in fixed_weights.split(',')]
        except ValueError as exc:
            raise click.BadParameter(
                "use comma-separated numbers, for example 0.75,0.25",
                param_hint='--fixed-weights',
            ) from exc
        if len(parsed_fixed_weights) != 2:
            raise click.BadParameter(
                "RGB+PCD evaluation requires exactly two values",
                param_hint='--fixed-weights',
            )
        if (not all(np.isfinite(x) for x in parsed_fixed_weights)
                or any(x < 0 for x in parsed_fixed_weights)
                or not np.isclose(sum(parsed_fixed_weights), 1.0, atol=1e-6)):
            raise click.BadParameter(
                "weights must be finite, non-negative, and sum to 1",
                param_hint='--fixed-weights',
            )

    if deterministic:
        global_seed = diffusion_seed if diffusion_seed is not None else test_start_seed
        os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
        random.seed(global_seed)
        np.random.seed(global_seed % (2**32))
        torch.manual_seed(global_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(global_seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True)
        print(f"Deterministic evaluation enabled (global seed {global_seed})")

    if os.path.exists(output_dir):
        click.confirm(f"Output path {output_dir} already exists! Overwrite?", abort=True)
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # grab all checkpoints
    ckpts: List[str]    # file paths to checkpoints to evaluate
    if os.path.isdir(checkpoint):
        ckpts = [
            os.path.join(checkpoint, f) 
            for f in os.listdir(checkpoint) 
            if f.endswith('.ckpt') and f != 'latest.ckpt'
        ]
    else:
        ckpts = [checkpoint,]

    base_output_dir = output_dir
    for ckpt in ckpts:
        # format output dir
        if len(ckpts) > 1:
            if (match := re.match(
                r'^ep-(\d{4})_sr-(\d{1}\.\d{3})\.ckpt$', 
                os.path.basename(ckpt)
            )):
                output_dir = os.path.join(base_output_dir, f"ep-{match.group(1)}")
            else:
                output_dir = os.path.join(
                    base_output_dir, 
                    os.path.basename(ckpt).replace('.ckpt', '')
                )
            pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
        else:
            output_dir = base_output_dir
        
        # load checkpoint
        payload = torch.load(open(ckpt, 'rb'), pickle_module=dill)
        cfg = payload['cfg']
        
        # Fix configuration issue: num_points must be None if enable_depth is False
        if hasattr(cfg.task.env_runner, 'num_points') and hasattr(cfg.task.env_runner, 'enable_depth'):
            if not cfg.task.env_runner.enable_depth and cfg.task.env_runner.num_points is not None:
                cfg.task.env_runner.num_points = None
        
        cls = hydra.utils.get_class(cfg._target_)
        workspace: BaseWorkspace = cls(cfg, output_dir=output_dir)
        workspace.load_payload(payload, exclude_keys=None, include_keys=None)
        
        # get policy from workspace
        policy: BasePolicy = workspace.model
        try:
            if cfg.training.use_ema:
                policy = workspace.ema_model
        except omegaconf.errors.ConfigAttributeError:
            # compositional policy does not have ema_model
            pass
        
        # Fix crop randomizers for deterministic evaluation (eval_fixed_crop behavior)
        try:
            import robomimic.models.base_nets as rmbn
            
            # Find and directly modify crop randomizers
            print("Searching for crop randomizers in policy...")
            fixed_count = 0
            
            def fix_crop_randomizers(module, prefix=""):
                nonlocal fixed_count
                for name, child in module.named_children():
                    if isinstance(child, rmbn.CropRandomizer):
                        full_name = f"{prefix}.{name}" if prefix else name
                        print(f"Found CropRandomizer at: {full_name}")
                        print(f"  Original: crop_height={child.crop_height}, crop_width={child.crop_width}")
                        
                        # Simply set the randomizer to eval mode for deterministic cropping
                        child.eval()
                        fixed_count += 1
                        print(f"  Fixed: Set to eval mode for deterministic center cropping")
                    else:
                        # Recursively check children
                        fix_crop_randomizers(child, f"{prefix}.{name}" if prefix else name)
            
            fix_crop_randomizers(policy)
            
            if fixed_count > 0:
                print(f"Applied eval_fixed_crop fix: modified {fixed_count} crop randomizers for deterministic evaluation")
            else:
                print("No CropRandomizer modules found - policy may not use cropping")
                
        except Exception as e:
            print(f"Warning: Could not apply crop randomizer fix: {e}")
            import traceback
            traceback.print_exc()
        
        device = torch.device(device)
        policy.to(device)
        policy.eval()

        if not hasattr(policy, 'set_inference_fixed_weights'):
            if expert is not None or parsed_fixed_weights is not None:
                raise click.ClickException(
                    "This checkpoint policy does not support fixed expert weights"
                )
        elif expert is not None:
            modalities = [m for m in policy.modalities if m != 'robot_state']
            if expert not in modalities:
                raise click.ClickException(
                    f"Expert {expert!r} is unavailable; checkpoint modalities: {modalities}"
                )
            if getattr(policy, 'num_modules', 1) != 1:
                raise click.ClickException(
                    "--expert currently requires exactly one expert per modality"
                )
            expert_index = modalities.index(expert)
            weights = [0.0] * len(policy.models)
            weights[expert_index] = 1.0
            policy.set_inference_fixed_weights(weights)
            print(f"Forced single-expert inference: {expert} "
                  f"(model index {expert_index}, applied weights {weights})")
        elif parsed_fixed_weights is not None:
            if getattr(policy, 'num_modules', 1) != 1:
                raise click.ClickException(
                    "--fixed-weights currently requires one expert per modality"
                )
            modalities = [m for m in policy.modalities if m != 'robot_state']
            if modalities != ['rgb', 'pointcloud']:
                raise click.ClickException(
                    f"Expected RGB+PCD modalities, got {modalities}"
                )
            policy.set_inference_fixed_weights(parsed_fixed_weights)
            print("Forced fixed inference mixture: "
                  f"RGB={parsed_fixed_weights[0]}, "
                  f"PointCloud={parsed_fixed_weights[1]}")

        # === WEIGHT CONSISTENCY CHECK ===
        # This verifies that model weights are loaded correctly and remain constant during evaluation
        try:
            # Get all model parameters and compute consistent checksums
            param_sums = []
            param_counts = []
            param_names = []
            
            for name, param in policy.named_parameters():
                if param.requires_grad:  # Only check trainable parameters
                    param_sum = param.sum().item()
                    param_count = param.numel()
                    param_sums.append(param_sum)
                    param_counts.append(param_count)
                    param_names.append(name)
            
            # Compute overall checksums
            total_param_sum = sum(param_sums)
            total_param_count = sum(param_counts)
            mean_param_value = total_param_sum / total_param_count if total_param_count > 0 else 0.0
            
            # Print weight consistency check
            print("=" * 60)
            print("[WEIGHT CONSISTENCY CHECK]")
            print(f"Total trainable parameters: {total_param_count:,}")
            print(f"Sum of all weights: {total_param_sum:.10f}")
            print(f"Mean weight value: {mean_param_value:.10f}")
            
            # Also print first few parameter details for extra verification
            print("First 3 parameter details:")
            for i in range(min(3, len(param_names))):
                print(f"  {param_names[i]}: sum={param_sums[i]:.10f}, count={param_counts[i]:,}")
            
            # Print checksum for easy comparison
            import hashlib
            checksum_str = f"{total_param_sum:.10f}_{total_param_count}_{mean_param_value:.10f}"
            checksum = hashlib.md5(checksum_str.encode()).hexdigest()[:8]
            print(f"Weight checksum: {checksum}")
            print("=" * 60)
            
        except Exception as e:
            print(f"[WEIGHT CHECK ERROR] Failed to compute weight checksums: {e}")

        # dropout obs ports
        dropout_obs_str = ','.join(dropout_obs)
        print(f"Dropout obs ports: {dropout_obs_str}")
        
        # run eval
        print(f"Running evaluation on {ckpt}")
        env_runner: BaseRunner = hydra.utils.instantiate(
            cfg.task.env_runner,
            output_dir=output_dir,
            obs_to_drop=dropout_obs,
            test_start_seed=test_start_seed,
            diffusion_seed_base=diffusion_seed,
            episode_log_path=episode_log,
            decision_log_path=decision_log,
            checkpoint_path=ckpt,
            # n_parallel_envs = 25, 
            n_test=num_exp,
            n_parallel_envs=min(
                num_exp,
                # cfg.task.env_runner.get('n_parallel_envs', num_exp)
                4,
            ),
            n_test_vis=min(
                num_exp,
                num_videos if num_videos is not None
                else cfg.task.env_runner.get('n_test_vis', num_exp)
            )
        )
        runner_log = env_runner.run(policy)
        env_runner.close()
        
        for key, value in runner_log.items():
            if isinstance(value, wandb.sdk.data_types.video.Video):
                runner_log[key] = [value]
                
        print(f"Success rate = {runner_log['mean_success_rate']}")
        
        json_log = {
            'checkpoint': ckpt,
            'num_exp': num_exp,
            'dropout_obs': ','.join(dropout_obs),
            'test_start_seed': test_start_seed,
            'diffusion_seed': diffusion_seed,
            'deterministic': deterministic,
        }
        
        for key, value in runner_log.items():
            if isinstance(value, list):
                for i, video in enumerate(value):
                    json_log[f'{key}_{i}'] = video._path
            else:
                json_log[key] = value

        # json.dump(json_log, open(os.path.join(output_dir, 'eval_log.json'), 'w'), indent=2, sort_keys=True)
        json_safe_log = make_json_safe(json_log)
        json.dump(json_safe_log, open(os.path.join(output_dir, 'eval_log.json'), 'w'), indent=2, sort_keys=True)

        # update the checkpoint name
        if update:
            new_ckpt_path = os.path.join(
                os.path.dirname(ckpt), 
                re.sub(
                    r'\b\d\.\d{3}\.ckpt$',
                    f"{runner_log['mean_success_rate']:.3f}.ckpt", 
                    os.path.basename(ckpt)
                )
            )
            os.rename(ckpt, new_ckpt_path)
            print(f"{ckpt} -> {new_ckpt_path}")


if __name__ == '__main__':
    eval_policy_sim()
