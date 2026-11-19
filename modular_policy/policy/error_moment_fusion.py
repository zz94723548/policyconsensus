"""Shared (A) or per-step (B) fusion using the existing DDPM sampler/runner."""

import json
from collections import deque
from pathlib import Path
from types import SimpleNamespace

import dill
import numpy as np
import torch

from modular_policy.common.error_moments import file_hash, optimal_weight
from modular_policy.common.episode_split import canonical_hash
from modular_policy.common.moment_calibration import robust_weight
from modular_policy.model.error_moments import HistoryErrorMoments
from modular_policy.policy.base_policy import BasePolicy
from modular_policy.policy.dp_unet import DiffusionUnetPolicy
from modular_policy.workspace.collect_error_moments import (
    audit_experts, encode_condition, load_frozen_expert,
)


class PlanningHistory:
    """Batch lanes stay separate; chronological valid prefixes match training."""
    def __init__(self, length):
        self.length = length
        self.reset()

    def reset(self):
        self.rgb = deque(maxlen=self.length)
        self.pcd = deque(maxlen=self.length)

    def append(self, rgb, pcd):
        if self.rgb and len(self.rgb[-1]) != len(rgb):
            raise ValueError("Batch changed without policy.reset()")
        self.rgb.append(rgb.detach())
        self.pcd.append(pcd.detach())
        length = len(self.rgb)
        def padded(values):
            result = values[-1].new_zeros((len(rgb), self.length, values[-1].shape[-1]))
            result[:, :length] = torch.stack(list(values), dim=1)
            return result
        return padded(self.rgb), padded(self.pcd), torch.full((len(rgb),), length, dtype=torch.long)


class ErrorMomentFusionPolicy(BasePolicy):
    weights_are_probabilities = True

    def __init__(self, rgb, pcd, mode, seed=42, statistics=None,
                 history_contexts=4, radius=0., fallback=0.5, tolerance=1e-8,
                 fixed_rgb_weight=None, timestep_ids=None):
        super().__init__()
        if mode not in ("rgb", "pcd", "equal", "fixed", "point", "robust", 'router'):
            raise ValueError("Unknown fusion mode")
        if mode == 'fixed' and (fixed_rgb_weight is None or not np.isfinite(fixed_rgb_weight)
                                or not 0 <= fixed_rgb_weight <= 1):
            raise ValueError('Fixed fusion requires an RGB weight in [0,1]')
        self.fixed_rgb_weight = fixed_rgb_weight
        if mode in ("point", "robust", 'router') and statistics is None:
            raise ValueError("Conditional fusion requires a frozen statistics model")
        self.rgb, self.pcd, self.statistics = rgb, pcd, statistics
        self.timestep_ids = None if timestep_ids is None else [int(t) for t in timestep_ids]
        self.step_indices = None
        if self.timestep_ids is not None:
            if mode != 'point':
                raise ValueError('Scheme B currently supports point weights; do not reuse scheme A calibration')
            if (len(set(self.timestep_ids)) != len(self.timestep_ids) or
                    getattr(statistics, 'num_steps', None) != len(self.timestep_ids)):
                raise ValueError('Scheme B output count and unique timestep IDs must match')
            rgb.noise_scheduler.set_timesteps(rgb.num_inference_steps)
            if rgb.noise_scheduler.timesteps.cpu().tolist() != self.timestep_ids:
                raise ValueError('Scheme B timestep order differs from the deployment scheduler')
            self.step_indices = {step: i for i, step in enumerate(self.timestep_ids)}
        elif getattr(statistics, 'num_steps', None) is not None:
            raise ValueError('Per-step statistics require explicit deployment timestep IDs')
        self.mode, self.seed = mode, seed
        self.radius, self.fallback, self.tolerance = radius, fallback, tolerance
        self.history = PlanningHistory(history_contexts)
        self.chunk = -1
        self.generator = None
        self.trace = []

    def reset(self):
        self.history.reset()
        self.chunk += 1
        self.generator = torch.Generator(device=self.device).manual_seed(self.seed + self.chunk)

    def get_policy_name(self):
        if self.mode == 'router':
            return 'rgb_pcd_native_frozen_router'
        return "rgb_pcd_scheme_" + ('b_' if self.step_indices is not None else 'a_') + self.mode

    def get_observation_ports(self):
        return list(dict.fromkeys([*self.rgb.obs_ports, *self.pcd.obs_ports]))

    @torch.inference_mode()
    def predict_action(self, obs_dict):
        if self.generator is None:
            raise RuntimeError("Call policy.reset() at episode batch start")
        rc, pc = encode_condition(self.rgb, obs_dict), encode_condition(self.pcd, obs_dict)
        moments = None
        if self.mode in ("rgb", "pcd", "equal", "fixed"):
            value = self.fixed_rgb_weight if self.mode == 'fixed' else {"rgb": 1., "pcd": 0., "equal": .5}[self.mode]
            weights = np.full(len(rc), value)
        else:
            rh, ph, lengths = self.history.append(rc, pc)
            if self.mode == 'router':
                probabilities = self.statistics(rh, ph, lengths)
                if probabilities.shape != (len(rc),2) or not torch.isfinite(probabilities).all():
                    raise ValueError('Invalid native router probabilities')
                weights = probabilities[:,0].cpu().numpy()
            else:
                moments = self.statistics(rh, ph, lengths).cpu().numpy()
            if self.mode == "robust":
                weights, _ = robust_weight(moments, self.radius, self.fallback, self.tolerance)
            elif self.mode == 'point':
                weights, _ = optimal_weight(moments, self.fallback, self.tolerance)
        weight_tensor = torch.as_tensor(weights, device=rc.device, dtype=rc.dtype)
        if self.step_indices is not None and weights.shape != (len(rc), len(self.timestep_ids)):
            raise ValueError('Scheme B must predict one RGB weight per context and timestep')
        w = weight_tensor.reshape(-1, 1, 1) if self.step_indices is None else None

        # Reuse the original sampling loop: ONE shared latent, ONE scheduler step.
        def paired_prediction(sample, timestep, global_cond):
            if self.mode == "rgb" or (self.mode == 'fixed' and self.fixed_rgb_weight == 1):
                return self.rgb.model(sample, timestep, global_cond=global_cond)
            if self.mode == "pcd" or (self.mode == 'fixed' and self.fixed_rgb_weight == 0):
                return self.pcd.model(sample, timestep, global_cond=pc)
            er = self.rgb.model(sample, timestep, global_cond=global_cond)
            ep = self.pcd.model(sample, timestep, global_cond=pc)
            step_w = w
            if self.step_indices is not None:
                key = int(timestep.item()) if isinstance(timestep, torch.Tensor) else int(timestep)
                step_w = weight_tensor[:, self.step_indices[key], None, None]
            return step_w * er + (1 - step_w) * ep

        sampler = SimpleNamespace(model=paired_prediction, noise_scheduler=self.rgb.noise_scheduler,
            horizon=self.rgb.horizon, action_dim=self.rgb.action_dim,
            num_inference_steps=self.rgb.num_inference_steps)
        normalized = DiffusionUnetPolicy.conditional_sample(
            sampler, rc, generator=self.generator, **self.rgb.kwargs)
        action_pred = self.rgb.normalizer["action"].unnormalize(normalized)
        start = self.rgb.n_obs_steps - 1
        self.trace.append({"chunk": self.chunk, "rgb_weights": weights.tolist(),
                           "moments": moments.tolist() if moments is not None else None})
        pairs = torch.stack([weight_tensor, 1 - weight_tensor], dim=-1)
        result = {"action": action_pred[:, start:start + self.rgb.n_action_steps],
                  "action_pred": action_pred, "weights": pairs.mean(1) if pairs.ndim == 3 else pairs}
        if pairs.ndim == 3:
            result['weights_by_step'] = pairs
        return result


def load_fusion_policy(rgb_path, pcd_path, mode, calibration_path, seed, device, fixed_rgb_weight=None):
    """Bind all three comparisons to the same frozen expert/model provenance."""
    artifact = json.loads(Path(calibration_path).read_text())
    training_run = Path(artifact["checkpoint"]).resolve().parents[1]
    metadata = json.loads((training_run / "cache_metadata.json").read_text())
    if (metadata["signature"] != artifact["training_cache_signature"] or
            canonical_hash(metadata["identity"]) != metadata["signature"]):
        raise ValueError("Calibration/training metadata mismatch")
    rgb, rcfg, rinfo = load_frozen_expert(rgb_path, "rgb", device)
    pcd, pcfg, pinfo = load_frozen_expert(pcd_path, "pcd", device)
    timesteps = audit_experts(rgb, pcd, rcfg, pcfg, metadata["identity"]["source_manifest"])
    for modality, info in [("rgb", rinfo), ("pcd", pinfo)]:
        if info["sha256"] != metadata["identity"]["experts"][modality]["sha256"]:
            raise ValueError(f"{modality} expert differs from calibrated expert")
    if not np.array_equal(timesteps, artifact["timesteps"]):
        raise ValueError("Calibrated and deployed diffusion steps differ")
    if rgb.n_action_steps != metadata["identity"]["context_stride"]:
        raise ValueError("Planning interval differs from training history interval")
    if artifact["risk_mode"] != "shared" or artifact["history_contexts"] < 1:
        raise ValueError("Only scheme A is supported")
    radius = float(artifact["fitted"]["radius"])
    if not np.isfinite(radius) or radius < 0:
        raise ValueError("Invalid calibrated radius")
    if file_hash(artifact["checkpoint"]) != artifact["checkpoint_sha256"]:
        raise ValueError("Statistics checkpoint changed after calibration")
    statistics = None
    if mode in ("point", "robust"):
        payload = torch.load(artifact["checkpoint"], map_location="cpu", pickle_module=dill)
        if dill.loads(payload["pickles"]["signature"]) != artifact["training_signature"]:
            raise ValueError("Statistics training signature differs")
        if (payload["cfg"].risk_mode != 'shared' or
                payload["cfg"].history_contexts != artifact["history_contexts"]):
            raise ValueError("Statistics model history/scheme differs from calibration")
        state = payload["state_dicts"]["model"]
        statistics = HistoryErrorMoments(state["rgb_mean"].numel(), state["pcd_mean"].numel(),
                                        **payload["cfg"].model)
        statistics.load_state_dict(state, strict=True)
        statistics.eval().requires_grad_(False).to(device)
        del payload
    policy = ErrorMomentFusionPolicy(rgb, pcd, mode, seed, statistics, artifact["history_contexts"],
        radius, artifact["global_fallback_rgb_weight"], artifact["weight_tolerance"],
        fixed_rgb_weight=fixed_rgb_weight).to(device).eval()
    provenance = {"mode": mode, "seed": seed, "rgb": rinfo, "pcd": pinfo,
                  "calibration": str(Path(calibration_path).resolve()),
                  "calibration_sha256": file_hash(calibration_path),
                  "statistics_checkpoint": artifact["checkpoint"],
                  "statistics_sha256": artifact["checkpoint_sha256"], "radius": radius,
                  "history_contexts": artifact["history_contexts"], "scheme": "A",
                  "initial_noise_seed_rule": "eval_seed + episode_batch_index",
                  "weight_order": ["rgb", "pcd"], "fixed_rgb_weight": fixed_rgb_weight}
    return policy, provenance


def load_step_fusion_policy(rgb_path, pcd_path, statistics_path, seed, device):
    """Load A/shared or B/per-step point weights directly, without calibration."""
    run = Path(statistics_path).resolve().parents[1]
    metadata = json.loads((run / 'cache_metadata.json').read_text())
    audit = json.loads((run / 'training_audit.json').read_text())
    baseline_path = run / 'baselines.json'
    baseline = json.loads(baseline_path.read_text())
    risk_mode=audit.get('risk_mode')
    if (metadata['status'] != 'complete' or canonical_hash(metadata['identity']) != metadata['signature'] or
            audit['cache_signature'] != metadata['signature'] or risk_mode not in ('shared','per_step')):
        raise ValueError('Invalid statistics training/cache provenance')
    identity = metadata['identity']
    if identity.get('collection_partition', 'risk_train') != 'risk_train':
        raise ValueError('Scheme B must be trained from the original development cache')
    for key, role in [('train_episode_ids', 'risk_model_train'), ('val_episode_ids', 'risk_model_val')]:
        if sorted(audit[key]) != sorted(identity['risk_split']['splits'][role]):
            raise ValueError('Scheme B training/validation episodes differ from cached partition')
    rgb, rcfg, rinfo = load_frozen_expert(rgb_path, 'rgb', device)
    pcd, pcfg, pinfo = load_frozen_expert(pcd_path, 'pcd', device)
    timesteps = audit_experts(rgb, pcd, rcfg, pcfg, identity['source_manifest'])
    for modality, info in [('rgb', rinfo), ('pcd', pinfo)]:
        if info['sha256'] != identity['experts'][modality]['sha256']:
            raise ValueError(f'{modality} expert differs from scheme B training expert')
    if not np.array_equal(timesteps, identity['timesteps']) or audit.get('timesteps',identity['timesteps']) != timesteps.tolist():
        raise ValueError('Scheme B timestep mapping differs from expert scheduler/cache')
    if rgb.n_action_steps != identity['context_stride']:
        raise ValueError('Scheme B planning interval differs from cached history')
    payload = torch.load(statistics_path, map_location='cpu', pickle_module=dill)
    cfg = payload['cfg']
    if (cfg.risk_mode != risk_mode or dill.loads(payload['pickles']['signature']) != audit['signature'] or
            cfg.history_contexts != audit['history_contexts']):
        raise ValueError('Scheme B checkpoint/config and training audit differ')
    state = payload['state_dicts']['model']
    model = HistoryErrorMoments(state['rgb_mean'].numel(), state['pcd_mean'].numel(),
                               num_steps=len(timesteps) if risk_mode=='per_step' else None, **cfg.model)
    model.load_state_dict(state, strict=True)
    if not np.allclose(model.output_scale.numpy(), audit['output_scale'], rtol=1e-6, atol=0):
        raise ValueError('Scheme B per-step scales differ from train-only audit')
    if not torch.isfinite(model.output_scale).all() or torch.any(model.output_scale <= 0):
        raise ValueError('Invalid per-step output scales')
    fallback = float(baseline['global_rgb_weight'])
    expected = float(optimal_weight(baseline['global_moments'], tolerance=cfg.weight_tolerance)[0])
    if not 0 <= fallback <= 1 or not np.isclose(fallback, expected):
        raise ValueError('Invalid train-only global fallback')
    model.eval().requires_grad_(False).to(device)
    policy = ErrorMomentFusionPolicy(rgb, pcd, 'point', seed, model, cfg.history_contexts,
        fallback=fallback, tolerance=cfg.weight_tolerance,
        timestep_ids=timesteps if risk_mode=='per_step' else None).to(device).eval()
    provenance = {'mode': 'point', 'scheme': 'B' if risk_mode=='per_step' else 'A', 'risk_mode': risk_mode, 'seed': seed,
        'rgb': rinfo, 'pcd': pinfo, 'statistics_checkpoint': str(Path(statistics_path).resolve()),
        'statistics_sha256': file_hash(statistics_path), 'training_signature': audit['signature'],
        'training_cache_signature': metadata['signature'], 'baselines_sha256': file_hash(baseline_path),
        'history_contexts': cfg.history_contexts, 'timesteps': timesteps.tolist(), 'radius': None,
        'initial_noise_seed_rule': 'eval_seed + episode_batch_index', 'weight_order': ['rgb', 'pcd'],
        'logged_avg_weights': 'mean over denoising steps; full weights are in fusion_trace.json' if risk_mode=='per_step' else 'one shared weight per planning context'}
    return policy, provenance


def load_native_router_checkpoint(router_path, device):
    """Load the frozen small router alone, shared by offline and simulation eval."""
    from modular_policy.model.frozen_expert_router import FrozenExpertRouter, shared_state_dimension
    run = Path(router_path).resolve().parents[1]
    metadata = json.loads((run/'cache_metadata.json').read_text())
    audit = json.loads((run/'training_audit.json').read_text())
    if (metadata['status'] != 'complete' or canonical_hash(metadata['identity']) != metadata['signature'] or
            audit['cache_signature'] != metadata['signature'] or
            audit.get('model_kind') != 'native_mmpc_frozen_router'):
        raise ValueError('Invalid native router provenance')
    identity = metadata['identity']
    from modular_policy.common.frozen_router_data import router_cache_roles
    roles=router_cache_roles(identity)
    train_role,val_role=('expert_train','expert_val') if 'expert_train' in roles else ('risk_model_train','risk_model_val')
    for key,role in [('train_episode_ids',train_role),('val_episode_ids',val_role)]:
        if sorted(audit[key]) != sorted(roles[role]):
            raise ValueError('Router training/validation IDs differ')
    for key,path in [('router_source_sha256','policy/dp_unets_spec.py'),
                     ('router_model_sha256','model/frozen_expert_router.py')]:
        if audit[key] != file_hash(Path(__file__).parents[1]/path):
            raise ValueError('Native router implementation changed since training')
    payload = torch.load(router_path,map_location='cpu',pickle_module=dill)
    if (dill.loads(payload['pickles']['signature']) != audit['signature'] or
            payload['cfg']._target_ != 'modular_policy.workspace.train_frozen_router.TrainFrozenRouterWorkspace' or
            payload['cfg'].history_contexts != 1):
        raise ValueError('Router checkpoint/config differs from audit')
    model = FrozenExpertRouter(audit['feature_dimensions']['rgb'],audit['feature_dimensions']['pcd'],
                              shared_state_dimension(identity))
    model.load_state_dict(payload['state_dicts']['model'],strict=True)
    model.eval().requires_grad_(False).to(device)
    provenance = {'mode':'router','scheme':'native_mmpc_router',
        'router_checkpoint':str(Path(router_path).resolve()),'router_sha256':file_hash(router_path),
        'training_signature':audit['signature'],'training_cache_signature':metadata['signature'],
        'checkpoint_epoch':dill.loads(payload['pickles']['epoch'])-1,
        'history_contexts':1,'radius':None,'weight_order':['rgb','pcd'],
        'initial_noise_seed_rule':'eval_seed + episode_batch_index',
        'objective':'direct frozen-expert mixture epsilon MSE'}
    provenance['training_data_protocol']=identity.get('collection_partition','risk_train')
    return model,provenance,metadata,audit


def load_native_router_policy(rgb_path, pcd_path, router_path, seed, device):
    model,provenance,metadata,audit = load_native_router_checkpoint(router_path,device)
    identity = metadata['identity']
    rgb,rcfg,rinfo = load_frozen_expert(rgb_path,'rgb',device)
    pcd,pcfg,pinfo = load_frozen_expert(pcd_path,'pcd',device)
    steps = audit_experts(rgb,pcd,rcfg,pcfg,identity['source_manifest'])
    for modality,info in [('rgb',rinfo),('pcd',pinfo)]:
        if info['sha256'] != identity['experts'][modality]['sha256']:
            raise ValueError('Router expert checkpoint differs from cached training expert')
    if not np.array_equal(steps,identity['timesteps']) or rgb.n_action_steps != identity['context_stride']:
        raise ValueError('Router deployment diffusion/planning protocol differs')
    policy = ErrorMomentFusionPolicy(rgb,pcd,'router',seed,model,history_contexts=1).to(device).eval()
    provenance.update(seed=seed,rgb=rinfo,pcd=pinfo)
    return policy,provenance
