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
@click.option('-d', '--device', default='cuda:0', help="device to run on")
@click.option('-u', '--update', is_flag=True, help="weather to update `success_rate` in ckpt file name")
@click.option('-s', '--dropout_obs', default=(), multiple=True, help="a list of sensor ports to drop i.e. set to 0s")
@click.option('--fusion', type=click.Choice(['rgb', 'pcd', 'equal', 'fixed', 'point', 'robust', 'router']), default=None,
              help='Matched MT4 expert/fusion comparison using the same sampler and environment settings.')
@click.option('--pcd-checkpoint', type=click.Path(exists=True, dir_okay=False), default=None)
@click.option('--calibration', type=click.Path(exists=True, dir_okay=False), default=None)
@click.option('--statistics-checkpoint', type=click.Path(exists=True, dir_okay=False), default=None,
              help='A/shared or B/per-step statistics checkpoint for --fusion point; omit --calibration.')
@click.option('--router-checkpoint', type=click.Path(exists=True, dir_okay=False), default=None,
              help='Frozen-expert native MMPC router checkpoint for --fusion router.')
@click.option('--eval-seed', type=click.IntRange(min=0), default=42, show_default=True)
@click.option('--test-start-seed', type=click.IntRange(min=0), default=1000, show_default=True)
@click.option('--rgb-weight', type=click.FloatRange(min=0, max=1), default=None,
              help='Constant RGB weight for --fusion fixed; PCD weight is 1 minus this value.')
def eval_policy_sim(checkpoint, output_dir, num_exp, device, update, dropout_obs,
                    fusion, pcd_checkpoint, calibration, eval_seed, test_start_seed, rgb_weight, statistics_checkpoint, router_checkpoint):
    if (fusion == 'fixed') != (rgb_weight is not None):
        raise click.UsageError('--fusion fixed requires --rgb-weight; other modes must omit --rgb-weight.')
    if statistics_checkpoint and (fusion != 'point' or calibration):
        raise click.UsageError('Direct statistics loading requires --fusion point; omit --calibration.')
    if (fusion == 'router') != (router_checkpoint is not None) or router_checkpoint and (calibration or statistics_checkpoint):
        raise click.UsageError('Use --fusion router with --router-checkpoint only; omit calibration/statistics options.')
    if fusion and (os.path.isdir(checkpoint) or not pcd_checkpoint or
                   not (calibration or statistics_checkpoint or router_checkpoint) or update or dropout_obs):
        raise click.UsageError('Fusion requires one RGB checkpoint, --pcd-checkpoint and calibration or B statistics; omit -u/-s.')
    if fusion and (num_exp < 4 or num_exp % 4):
        raise click.UsageError('MT4 fusion evaluation requires a positive multiple of four episodes.')
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
        payload = torch.load(ckpt, map_location='cpu', pickle_module=dill)
        cfg = payload['cfg']

        # Split-only expert checkpoints predate an embedded runner configuration.
        # Reuse the released MT4 runner; keep the original evaluation procedure.
        if ('env_runner' not in cfg.task and cfg._target_ ==
                'modular_policy.workspace.train_modality_expert.TrainModalityExpertWorkspace'):
            task_cfg = omegaconf.OmegaConf.load(
                pathlib.Path(__file__).resolve().parents[1] / 'config/task/rlbench/mt4.yaml'
            )
            # Hydra checkpoints retain struct mode: explicitly allow this new key.
            with omegaconf.open_dict(cfg.task):
                cfg.task.env_runner = task_cfg.env_runner
            cfg.task.env_runner.enable_dino = False
            cfg.task.env_runner.image_scale = cfg.task.dataset.image_scale
        
        # Fix configuration issue: num_points must be None if enable_depth is False
        if hasattr(cfg.task.env_runner, 'num_points') and hasattr(cfg.task.env_runner, 'enable_depth'):
            if not cfg.task.env_runner.enable_depth and cfg.task.env_runner.num_points is not None:
                cfg.task.env_runner.num_points = None
        
        fusion_provenance = None
        if fusion:
            del payload
            import random
            random.seed(eval_seed)
            np.random.seed(eval_seed)
            torch.manual_seed(eval_seed)
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
            from modular_policy.policy.error_moment_fusion import load_fusion_policy, load_step_fusion_policy, load_native_router_policy
            if router_checkpoint:
                policy, fusion_provenance = load_native_router_policy(
                    ckpt, pcd_checkpoint, router_checkpoint, eval_seed, torch.device(device))
            elif statistics_checkpoint:
                policy, fusion_provenance = load_step_fusion_policy(
                    ckpt, pcd_checkpoint, statistics_checkpoint, eval_seed, torch.device(device))
            else:
                policy, fusion_provenance = load_fusion_policy(
                    ckpt, pcd_checkpoint, fusion, calibration, eval_seed, torch.device(device),
                    fixed_rgb_weight=rgb_weight)
            # The RGB checkpoint's native runner already observes RGB and pointcloud.
            cfg.task.env_runner.enable_depth = True
            cfg.task.env_runner.num_points = 512
            cfg.task.env_runner.enable_dino = False
            cfg.task.env_runner.n_parallel_envs = 8
            cfg.task.env_runner.n_test_vis = 8
            fusion_provenance['test_start_seed'] = test_start_seed
            fusion_provenance['n_parallel_envs'] = 8
            fusion_provenance['n_test'] = num_exp
        else:
            cls = hydra.utils.get_class(cfg._target_)
            workspace: BaseWorkspace = cls(cfg, output_dir=output_dir)
            workspace.load_payload(
                payload, exclude_keys=['optimizer', 'lr_scheduler'], include_keys=[]
            )
            policy: BasePolicy = workspace.model
            try:
                if cfg.training.use_ema:
                    policy = workspace.ema_model
            except omegaconf.errors.ConfigAttributeError:
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

        # === WEIGHT CONSISTENCY CHECK ===
        # This verifies that model weights are loaded correctly and remain constant during evaluation
        try:
            # Get all model parameters and compute consistent checksums
            param_sums = []
            param_counts = []
            param_names = []
            
            for name, param in policy.named_parameters():
                if param.numel():  # Frozen evaluation models still need a real checksum.
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
            print(f"Total model parameters: {total_param_count:,}")
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
        # OpenCV may set its own Qt plugin directory during import.
        if os.environ.get('COPPELIASIM_ROOT'):
            os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = os.environ['COPPELIASIM_ROOT']
        env_runner: BaseRunner = hydra.utils.instantiate(
            cfg.task.env_runner,
            output_dir=output_dir,
            obs_to_drop=dropout_obs,
            # n_parallel_envs = 25, 
            n_test=num_exp,
            test_start_seed=test_start_seed
        )
        try:
            runner_log = env_runner.run(policy)
        finally:
            env_runner.close()
        if fusion:
            json.dump(fusion_provenance, open(os.path.join(output_dir, 'fusion_config.json'), 'w'), indent=2)
            json.dump(policy.trace, open(os.path.join(output_dir, 'fusion_trace.json'), 'w'), indent=2)
        
        for key, value in runner_log.items():
            if isinstance(value, wandb.sdk.data_types.video.Video):
                runner_log[key] = [value]
                
        print(f"Success rate = {runner_log['mean_success_rate']}")
        
        json_log = {
            'checkpoint': ckpt,
            'num_exp': num_exp,
            'dropout_obs': ','.join(dropout_obs),
        }
        if fusion:
            json_log['fusion'] = fusion_provenance
        
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
