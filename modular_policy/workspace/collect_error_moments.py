"""Offline paired expert errors; no RLBench runner or simulator is constructed."""

import json
from pathlib import Path

import dill
import hydra
import numpy as np
from omegaconf import OmegaConf
import torch
from tqdm import tqdm

from modular_policy.common.episode_split import canonical_hash
from modular_policy.common.error_moments import (
    check_moments, context_noise, file_hash, paired_moments, validate_risk_split, write_json,
)
from modular_policy.workspace.base_workspace import BaseWorkspace
from modular_policy.common.moment_calibration import calibration_ids, heldout_ids


def load_frozen_expert(path, modality, device):
    from modular_policy.policy.dp_unet import DiffusionUnetPolicy

    payload = torch.load(path, map_location="cpu", pickle_module=dill)
    cfg = payload["cfg"]
    if cfg.expert_modality != modality or not cfg.training.use_ema:
        raise ValueError(f"Expected an EMA {modality} expert")
    policy = hydra.utils.instantiate(cfg.policy)
    if not isinstance(policy, DiffusionUnetPolicy):
        raise TypeError("Only the existing single-modality DiffusionUnetPolicy is supported")
    policy.load_state_dict(payload["state_dicts"]["ema_model"], strict=True)
    info = {"path": str(Path(path).resolve()), "sha256": file_hash(path), "weights": "ema_model",
            "epoch": dill.loads(payload["pickles"]["epoch"]) - 1,
            "policy_config": OmegaConf.to_container(cfg.policy, resolve=True)}
    del payload
    return policy.eval().requires_grad_(False).to(device), cfg, info


def audit_experts(rgb, pcd, rcfg, pcfg, manifest):
    for policy, cfg, modality in [(rgb, rcfg, "rgb"), (pcd, pcfg, "pointcloud")]:
        if cfg.split_manifest_sha256 != canonical_hash(manifest):
            raise ValueError("Expert checkpoint and source manifest differ")
        if set(policy.modalities) - {"robot_state"} != {modality}:
            raise ValueError("Expert modalities differ from the requested pair")
        if policy.noise_scheduler.config.prediction_type != "epsilon":
            raise ValueError("A shared epsilon prediction target is required")
    for key in ["horizon", "action_dim", "n_obs_steps", "n_action_steps", "num_inference_steps"]:
        if getattr(rgb, key) != getattr(pcd, key):
            raise ValueError(f"Expert mismatch: {key}")
    for key in ["action_key", "pad_before", "pad_after", "image_scale"]:
        if rcfg.task.dataset[key] != pcfg.task.dataset[key]:
            raise ValueError(f"Dataset alignment mismatch: {key}")
    if rcfg.task.dataset.image_scale != 1.0:
        raise ValueError("This collector expects the existing [0,1] RGB input scale")
    if OmegaConf.to_container(rcfg.policy.noise_scheduler) != OmegaConf.to_container(pcfg.policy.noise_scheduler):
        raise ValueError("Noise scheduler settings differ")
    for key in ["betas", "alphas_cumprod"]:
        if not torch.equal(getattr(rgb.noise_scheduler, key), getattr(pcd.noise_scheduler, key)):
            raise ValueError(f"Scheduler arrays differ: {key}")
    shared = set(rgb.obs_ports) & set(pcd.obs_ports)
    for key in shared:
        if rgb.obs_key_shapes[key] != pcd.obs_key_shapes[key]:
            raise ValueError(f"Shared observation shape differs: {key}")
    for key in shared | {"action"}:
        for name in ["scale", "offset"]:
            if not torch.equal(rgb.normalizer[key].params_dict[name], pcd.normalizer[key].params_dict[name]):
                raise ValueError(f"Normalizer mismatch: {key}.{name}")
    for policy in [rgb, pcd]:
        policy.noise_scheduler.set_timesteps(policy.num_inference_steps)
    if not torch.equal(rgb.noise_scheduler.timesteps, pcd.noise_scheduler.timesteps):
        raise ValueError("Inference timestep mappings differ")
    return rgb.noise_scheduler.timesteps.cpu().numpy()


def encode_condition(policy, observations):
    n = policy.n_obs_steps
    inputs = policy.normalizer.normalize({k: observations[k] for k in policy.obs_ports})
    size = len(next(iter(inputs.values())))
    features = policy.obs_encoder({k: x[:, :n].reshape(-1, *x.shape[2:]) for k, x in inputs.items()})
    return torch.cat([x.reshape(size, -1) for x in features.values()], dim=-1)


def read_contexts(dataset, sequence_ids, episode, n_obs, device):
    ends = dataset.replay_buffer.episode_ends
    first = int(ends[episode - 1]) if episode else 0
    last = int(ends[episode])
    frames, actions = [], []
    for index in sequence_ids:
        begin, end, sample_start, _ = dataset.seq_sampler.indices[index]
        if begin < first or end > last:
            raise ValueError("Action window crosses episode boundary")
        frames.append(np.clip(begin - sample_start + np.arange(n_obs), first, last - 1))
        actions.append(dataset.seq_sampler.sample_sequence(index)[dataset.action_key])
    frames = np.stack(frames)
    observations = {}
    for key in dataset.obs_keys:
        array = dataset.replay_buffer[key]
        value = np.asarray(array.oindex[frames.reshape(-1)], dtype=np.float32)
        value = value.reshape(len(frames), n_obs, *array.shape[1:])
        if "rgb" in key:
            value /= 255.0
        observations[key] = torch.from_numpy(value).to(device)
    return observations, torch.from_numpy(np.stack(actions).astype(np.float32)).to(device), frames


@torch.inference_mode()
def collect_episode(dataset, ids, episode, role, rgb, pcd, timesteps, cfg, device):
    repeats = (cfg.test_noise_repeats if role == 'test' else
               cfg.get("calibration_noise_repeats", cfg.val_noise_repeats) if role == "risk_calibration"
               else cfg.train_noise_repeats if role in ("risk_model_train",'expert_train') else cfg.val_noise_repeats)
    n, k = len(ids), len(timesteps)
    moments = np.zeros((n, k, 3), dtype=np.float64)
    errors = np.empty((n, k, repeats, 2, rgb.horizon, rgb.action_dim), np.float32) if cfg.save_errors else None
    rfeatures, pfeatures, frame_indices = [], [], []
    for start in range(0, n, cfg.batch_size):
        subset = ids[start:start + cfg.batch_size]
        obs, actions, frames = read_contexts(dataset, subset, episode, rgb.n_obs_steps, device)
        rc, pc = encode_condition(rgb, obs), encode_condition(pcd, obs)
        rfeatures.append(rc.cpu().numpy())
        pfeatures.append(pc.cpu().numpy())
        frame_indices.append(frames)
        actions = rgb.normalizer["action"].normalize(actions)
        for j, step in enumerate(timesteps):
            t = torch.full((len(subset),), int(step), device=device, dtype=torch.long)
            for repeat in range(repeats):
                noise = torch.stack([
                    context_noise(cfg.seed, episode, int(anchor), step, repeat,
                                  {"risk_model_train": 0, "risk_model_val": 1, "risk_calibration": 2, "test": 3,
                                   'expert_train':4,'expert_val':5}[role],
                                  (rgb.horizon, rgb.action_dim))
                    for anchor in frames[:, -1]
                ]).to(device)
                noisy = rgb.noise_scheduler.add_noise(actions, noise, t)
                er = rgb.model(noisy, t, global_cond=rc) - noise
                ep = pcd.model(noisy, t, global_cond=pc) - noise
                moments[start:start + len(subset), j] += paired_moments(er, ep).cpu().numpy() / repeats
                if errors is not None:
                    errors[start:start + len(subset), j, repeat, 0] = er.cpu().numpy()
                    errors[start:start + len(subset), j, repeat, 1] = ep.cpu().numpy()
    check_moments(moments)
    frames = np.concatenate(frame_indices)
    result = {"episode_id": np.int64(episode), "sequence_indices": np.asarray(ids),
              "observation_frame_indices": frames, "anchor_frames": frames[:, -1],
              "timesteps": timesteps, "moments_by_step": moments, "moments": moments.mean(1),
              "rgb_features": np.concatenate(rfeatures), "pcd_features": np.concatenate(pfeatures)}
    if errors is not None:
        result["paired_errors"] = errors
    return result


class CollectErrorMomentsWorkspace(BaseWorkspace):
    def run(self):
        # Numba/dataset dependencies are loaded only for this offline operation.
        from modular_policy.dataset.episode_split_rlbench_dataset import EpisodeSplitRlbenchDataset

        cfg = self.cfg
        for key in ["batch_size", "context_stride", "train_noise_repeats", "val_noise_repeats"]:
            if cfg[key] <= 0:
                raise ValueError(f"{key} must be positive")
        if cfg.seed < 0:
            raise ValueError("seed must be nonnegative")
        original = json.loads(Path(cfg.source_manifest).read_text())
        split = json.loads(Path(cfg.risk_split).read_text())
        validate_risk_split(split, original)
        partition = cfg.get("partition", "risk_train")
        if partition == "risk_train":
            roles = split["splits"]
        elif partition == 'expert_router':
            from modular_policy.common.frozen_router_data import router_cache_roles
            roles=router_cache_roles({'collection_partition':partition,'source_manifest':original})
        elif partition == "risk_calibration":
            roles = {"risk_calibration": calibration_ids(original)}
            if cfg.get("calibration_noise_repeats", cfg.val_noise_repeats) <= 0:
                raise ValueError("Calibration MC repeat count must be positive")
        elif partition == 'test':
            from modular_policy.common.final_error_moments import read_final_protocol
            protocol = read_final_protocol(cfg.final_protocol)
            for key in ['rgb_checkpoint', 'pcd_checkpoint', 'source_manifest', 'risk_split']:
                if Path(cfg[key]).resolve() != Path(protocol['files'][key]['path']).resolve():
                    raise ValueError(f'Final collection differs from frozen {key}')
            for key in ['seed', 'context_stride', 'test_noise_repeats']:
                if cfg[key] != protocol[key]:
                    raise ValueError(f'Final collection differs from frozen {key}')
            if cfg.test_noise_repeats <= 0:
                raise ValueError('Final-test noise repeats must be positive')
            roles = {'test': heldout_ids(original, 'test')}
        else:
            raise ValueError("Unsupported paired-error collection partition")
        output = Path(self.output_dir)
        output.mkdir(parents=True, exist_ok=True)
        device = torch.device(cfg.device)
        rgb, rcfg, rinfo = load_frozen_expert(cfg.rgb_checkpoint, "rgb", device)
        pcd, pcfg, pinfo = load_frozen_expert(cfg.pcd_checkpoint, "pcd", device)
        timesteps = audit_experts(rgb, pcd, rcfg, pcfg, original)
        obs_keys = list(dict.fromkeys([*rgb.obs_ports, *pcd.obs_ports]))
        dataset = EpisodeSplitRlbenchDataset(
            cfg.zarr_path or original["source"]["zarr_path"], cfg.source_manifest,
            obs_keys, action_key=rcfg.task.dataset.action_key, split='expert_train' if partition=='expert_router' else partition,
            horizon=rgb.horizon, pad_before=rcfg.task.dataset.pad_before,
            pad_after=rcfg.task.dataset.pad_after,
        )
        for key in obs_keys:
            if list(dataset.replay_buffer[key].shape[1:]) != list(rgb.obs_key_shapes.get(key, pcd.obs_key_shapes.get(key))):
                raise ValueError(f"Source/checkpoint observation mismatch: {key}")
        if dataset.replay_buffer[dataset.action_key].shape[1:] != (rgb.action_dim,):
            raise ValueError("Source/checkpoint action dimensions differ")
        dataset.seq_sampler.keys = [dataset.action_key]
        identity = {"version": 1, "source_manifest": original, "risk_split": split,
                    "experts": {"rgb": rinfo, "pcd": pinfo},
                    "timesteps": timesteps.tolist(), "lambda": (np.ones(len(timesteps)) / len(timesteps)).tolist(),
                    "seed": cfg.seed, "context_stride": cfg.context_stride,
                    "train_noise_repeats": cfg.train_noise_repeats, "val_noise_repeats": cfg.val_noise_repeats,
                    "save_errors": cfg.save_errors, "device": str(device),
                    "software": {"torch": str(torch.__version__), "numpy": np.__version__},
                    "collector_sha256": file_hash(__file__),
                    "moment_helpers_sha256": file_hash(Path(__file__).parents[1] / "common/error_moments.py"),
                    "sampler_sha256": file_hash(Path(__file__).parents[1] / "common/seq_sampler.py"),
                    "dataset_sha256": file_hash(Path(__file__).parents[1] / "dataset/episode_split_rlbench_dataset.py"),
                    "policy_sha256": file_hash(Path(__file__).parents[1] / "policy/dp_unet.py")}
        if partition == "risk_calibration":
            identity.update(collection_partition=partition,
                            calibration_noise_repeats=cfg.get("calibration_noise_repeats", cfg.val_noise_repeats))
        elif partition == 'test':
            identity.update(collection_partition='test', test_noise_repeats=cfg.test_noise_repeats,
                            final_protocol=protocol, final_protocol_sha256=file_hash(cfg.final_protocol))
        elif partition == 'expert_router':
            identity.update(collection_partition=partition)
        signature = canonical_hash(identity)
        metadata_path = output / "metadata.json"
        if metadata_path.exists():
            metadata = json.loads(metadata_path.read_text())
            if not cfg.resume or metadata["signature"] != signature:
                raise ValueError("Existing cache: enable resume with identical provenance or use a new cache_dir")
        else:
            metadata = {"identity": identity, "signature": signature, "status": "running", "episodes": [],
                        "moment_columns": ["A_rgb", "B_pcd", "C_cross"],
                        "scope": "demonstration forward-noising epsilon MSE; not task success",
                        "errors_axes": ["context", "step", "repeat", "rgb_pcd", "horizon", "action"]}
        previous = {e["episode_id"]: e for e in metadata["episodes"]}
        write_json(metadata_path, metadata)
        ends = dataset.replay_buffer.episode_ends
        rows = dataset.seq_sampler.indices
        row_episodes = np.searchsorted(ends, rows[:, 0], side="right")
        for role in roles:
            if partition=='expert_router':
                dataset._select_split(role)
                dataset.seq_sampler.keys=[dataset.action_key]
                rows=dataset.seq_sampler.indices
                row_episodes=np.searchsorted(ends,rows[:,0],side='right')
            (output / role).mkdir(exist_ok=True)
            for episode in tqdm(roles[role], desc=role):
                if episode in previous:
                    record = previous[episode]
                    if record["role"] != role or file_hash(output / record["file"]) != record["sha256"]:
                        raise ValueError(f"Corrupt cached episode {episode}")
                    continue
                first = int(ends[episode - 1]) if episode else 0
                ids = np.flatnonzero(row_episodes == episode)
                anchors = rows[ids, 0] - rows[ids, 2] + rgb.n_obs_steps - 1
                ids = ids[(anchors - first) % cfg.context_stride == 0]
                if not len(ids):
                    raise ValueError(f"No valid contexts for episode {episode}")
                values = collect_episode(dataset, ids, episode, role, rgb, pcd, timesteps, cfg, device)
                relative = f"{role}/episode_{episode:04d}.npz"
                temporary = output / (relative + ".tmp.npz")
                np.savez_compressed(temporary, **values)
                temporary.replace(output / relative)
                metadata["episodes"].append({"episode_id": episode, "role": role,
                    "task": original["task_by_episode"][episode], "contexts": len(ids),
                    "rgb_dim": values["rgb_features"].shape[-1], "pcd_dim": values["pcd_features"].shape[-1],
                    "file": relative, "sha256": file_hash(output / relative)})
                write_json(metadata_path, metadata)
        expected = [i for ids in roles.values() for i in ids]
        if (len(metadata["episodes"]) != len(expected)
                or {e["episode_id"] for e in metadata["episodes"]} != set(expected)):
            raise ValueError("Incomplete or unexpected cache episodes")
        metadata["status"] = "complete"
        write_json(metadata_path, metadata)
        print(f"Paired error cache complete: {output.resolve()}")
