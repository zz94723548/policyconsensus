import copy
import tempfile
import unittest
from pathlib import Path

import dill
import torch
import torch.nn as nn
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from omegaconf import OmegaConf

from modular_policy.model.common.normalizer import LinearNormalizer
from modular_policy.policy.dp_unets_spec import DiffusionUnetsPolicy
from modular_policy.workspace.train_two_stage_policy import TrainTwoStagePolicyWorkspace


class TinyEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.rgb = nn.Linear(2, 2)
        self.pcd = nn.Linear(2, 2)

    def forward(self, obs):
        return {'rgb': self.rgb(obs['rgb']), 'pcd': self.pcd(obs['pcd']),
                'q': obs['q']}


class TinyExpert(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(6, 2)

    def forward(self, trajectory, timestep, global_cond):
        return self.linear(torch.cat((trajectory, global_cond[:, None, :]
                                      .expand(-1, trajectory.shape[1], -1)), dim=-1))


def make_policy(stage):
    policy = DiffusionUnetsPolicy.__new__(DiffusionUnetsPolicy)
    nn.Module.__init__(policy)
    policy.training_stage = stage
    policy.num_modules = 1
    policy.modalities = ['robot_state', 'rgb', 'pointcloud']
    policy.obs_config = {'rgb': ['rgb'], 'pointcloud': ['pcd'], 'robot_state': ['q']}
    policy.obs_encoder = TinyEncoder()
    policy.models = nn.ModuleList([TinyExpert(), TinyExpert()])
    policy.weight_predictor = nn.Sequential(nn.Linear(6, 8), nn.ReLU(), nn.Linear(8, 2))
    policy.noise_scheduler = DDPMScheduler(num_train_timesteps=10, prediction_type='epsilon')
    policy.normalizer = LinearNormalizer()
    policy.normalizer.fit({key: torch.tensor([[-1., -1.], [1., 1.]])
                           for key in ('rgb', 'pcd', 'q', 'action')})
    policy.n_obs_steps = 1
    policy.composition_strategy = 'soft_gating'
    policy.last_expert_losses = {}
    return policy


def batch():
    return {'obs': {k: torch.randn(4, 1, 2) for k in ('rgb', 'pcd', 'q')},
            'action': torch.randn(4, 3, 2)}


class TwoStageTests(unittest.TestCase):
    def test_experts_get_individual_gradients_without_router(self):
        policy = make_policy('experts')
        policy.compute_loss(batch()).backward()
        self.assertEqual(set(policy.last_expert_losses), {'rgb', 'pointcloud'})
        for module in (policy.models[0], policy.models[1], policy.obs_encoder.rgb,
                       policy.obs_encoder.pcd):
            self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0
                                for p in module.parameters()))
        self.assertTrue(all(p.grad is None for p in policy.weight_predictor.parameters()))

    def test_checkpoint_transfer_and_router_updates_only_router(self):
        experts = make_policy('experts')
        source_cfg = OmegaConf.create({'policy': {'training_stage': 'experts', 'num_modules': 1}})
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'expert.ckpt'
            torch.save({'cfg': source_cfg, 'state_dicts': {'ema_model': experts.state_dict()}},
                       path, pickle_module=dill)
            workspace = TrainTwoStagePolicyWorkspace.__new__(TrainTwoStagePolicyWorkspace)
            workspace.cfg = OmegaConf.create({
                'policy': {'training_stage': 'router', 'num_modules': 1},
                'training': {'expert_checkpoint': str(path), 'expert_weights': 'ema_model'},
            })
            workspace.model = make_policy('router')
            workspace.ema_model = copy.deepcopy(workspace.model)
            workspace.stage = 'router'
            workspace._initialize_router()
            workspace.configure_normalizer(experts.normalizer)
            for key, value in experts.models.state_dict().items():
                self.assertTrue(torch.equal(value, workspace.model.models.state_dict()[key]))
            frozen_before = copy.deepcopy(workspace.model.models.state_dict())
            router_before = copy.deepcopy(workspace.model.weight_predictor.state_dict())
            opt = torch.optim.AdamW(workspace.model.weight_predictor.parameters(), lr=0.01)
            workspace.model.compute_loss(batch()).backward()
            opt.step()
            self.assertTrue(all(torch.equal(v, workspace.model.models.state_dict()[k])
                                for k, v in frozen_before.items()))
            self.assertTrue(any(not torch.equal(v, workspace.model.weight_predictor.state_dict()[k])
                                for k, v in router_before.items()))
            self.assertTrue(all(p.grad is None for p in workspace.model.obs_encoder.parameters()))
            altered = copy.deepcopy(experts.normalizer)
            next(iter(altered.parameters())).data.add_(1)
            with self.assertRaisesRegex(ValueError, 'normalizer'):
                workspace.configure_normalizer(altered)


if __name__ == '__main__':
    unittest.main()
