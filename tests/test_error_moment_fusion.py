"""Synthetic adapters only: no real checkpoints, GPU, RLBench or experiments."""

import numpy as np
import pytest
import torch
from torch import nn
from diffusers import DDPMScheduler

from modular_policy.policy.error_moment_fusion import ErrorMomentFusionPolicy, PlanningHistory


def test_history_lane_isolation_prefix_and_reset():
    history = PlanningHistory(3)
    for step in range(5):
        rgb = torch.tensor([[float(step)], [100. + step]])
        r, p, lengths = history.append(rgb, rgb + 10)
        assert lengths.tolist() == [min(step + 1, 3)] * 2
        valid = lengths[0]
        assert torch.equal(r[1, :valid] - r[0, :valid], torch.full((valid, 1), 100.))
        assert torch.equal(p[:, :valid] - r[:, :valid], torch.full((2, valid, 1), 10.))
        assert torch.all(r[:, valid:] == 0)
    assert r[0, :, 0].tolist() == [2., 3., 4.]
    with pytest.raises(ValueError, match='reset'):
        history.append(torch.zeros(1, 1), torch.zeros(1, 1))
    history.reset()
    r, _, lengths = history.append(torch.tensor([[9.]]), torch.tensor([[8.]]))
    assert lengths.tolist() == [1] and r[0, :, 0].tolist() == [9., 0., 0.]


class Probe(nn.Module):
    def __init__(self, value):
        super().__init__()
        self.value, self.samples = value, []

    def forward(self, sample, timestep, global_cond):
        self.samples.append(sample.clone())
        return torch.full_like(sample, self.value)


class Identity:
    def unnormalize(self, value):
        return value


class Expert(nn.Module):
    def __init__(self, port, prediction):
        super().__init__()
        self.model = Probe(prediction)
        self.port, self.obs_ports = port, [port]
        self.noise_scheduler = DDPMScheduler(num_train_timesteps=3, clip_sample=False)
        self.horizon, self.action_dim, self.n_obs_steps, self.n_action_steps = 4, 2, 2, 2
        self.num_inference_steps, self.kwargs = 3, {}
        self.normalizer = {'action': Identity()}


class Statistics(nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def forward(self, rgb, pcd, length):
        self.calls += 1
        return rgb.new_tensor([4., 1., 0.]).expand(len(rgb), -1)


@pytest.mark.parametrize('mode,expected,fixed', [
    ('rgb', 1., None), ('pcd', 0., None), ('equal', .5, None),
    ('point', .2, None), ('robust', 2/7, None),
    *[('fixed', percent / 100, percent / 100) for percent in range(0, 101, 10)]])
def test_native_sampler_shared_latent_and_once_per_plan(monkeypatch, mode, expected, fixed):
    monkeypatch.setattr('modular_policy.policy.error_moment_fusion.encode_condition',
                        lambda expert, obs: obs[expert.port])
    rgb, pcd, stats = Expert('rgb', 2.), Expert('pcd', -1.), Statistics()
    policy = ErrorMomentFusionPolicy(rgb, pcd, mode, statistics=stats, radius=1., fixed_rgb_weight=fixed)
    policy.reset()
    result = policy.predict_action({'rgb': torch.zeros(2, 3), 'pcd': torch.ones(2, 2)})
    assert stats.calls == (1 if mode in ('point', 'robust') else 0)
    assert result['action'].shape == (2, 2, 2)
    assert torch.allclose(result['weights'][:, 0], torch.full((2,), expected))
    assert torch.allclose(result['weights'].sum(-1), torch.ones(2))
    assert len(rgb.model.samples) == (0 if mode == 'pcd' or mode == 'fixed' and fixed == 0 else 3)
    assert len(pcd.model.samples) == (0 if mode == 'rgb' or mode == 'fixed' and fixed == 1 else 3)
    for r, p in zip(rgb.model.samples, pcd.model.samples):
        assert torch.equal(r, p)
    # Compare against native sampling with the analytically combined prediction.
    from types import SimpleNamespace
    from modular_policy.policy.dp_unet import DiffusionUnetPolicy
    reference = SimpleNamespace(model=Probe(expected * 2 - (1 - expected)),
        noise_scheduler=DDPMScheduler(num_train_timesteps=3, clip_sample=False),
        horizon=4, action_dim=2, num_inference_steps=3)
    action = DiffusionUnetPolicy.conditional_sample(reference, torch.zeros(2, 3),
        generator=torch.Generator().manual_seed(42))
    assert torch.allclose(result['action_pred'], action, atol=1e-6)
    policy.reset()
    assert not policy.history.rgb
    assert policy.generator.initial_seed() == 43


def test_explicit_seed_reproduces_and_none_preserves_rng():
    import random
    from modular_policy.common.evaluation_reproducibility import seed_environment
    np_state, py_state = np.random.get_state(), random.getstate()
    try:
        seed_environment(1000)
        first = (np.random.random(), random.random())
        states = np.random.get_state(), random.getstate()
        expected = (np.random.random(), random.random())
        np.random.set_state(states[0])
        random.setstate(states[1])
        seed_environment(None)
        assert (np.random.random(), random.random()) == expected
        seed_environment(1000)
        assert (np.random.random(), random.random()) == first
    finally:
        np.random.set_state(np_state)
        random.setstate(py_state)


def test_initial_observation_audit_tracks_each_lane_and_port():
    from modular_policy.common.evaluation_reproducibility import observation_snapshot
    obs = {'rgb': np.zeros((2, 2, 3, 4, 4), dtype=np.float32),
           'joint_positions': np.arange(28, dtype=np.float32).reshape(2, 2, 7)}
    before = observation_snapshot(obs, 0)
    obs['rgb'][1] += 1
    assert observation_snapshot(obs, 0) == before
    obs['rgb'][0, 0, 0, 0, 0] = 1
    after = observation_snapshot(obs, 0)
    assert after['sha256_by_port']['rgb'] != before['sha256_by_port']['rgb']
    assert after['robot_state'] == before['robot_state']


@pytest.mark.parametrize('container', ['list', 'tuple', 'object_array'])
def test_initial_observation_audit_accepts_parallel_reset_format(container):
    from modular_policy.common.evaluation_reproducibility import observation_snapshot
    stacked = {'rgb': np.arange(96, dtype=np.float32).reshape(2, 2, 3, 2, 4),
               'joint_positions': np.arange(28, dtype=np.float32).reshape(2, 2, 7)}
    lanes = [{port: values[lane] for port, values in stacked.items()} for lane in range(2)]
    if container == 'tuple':
        lanes = tuple(lanes)
    elif container == 'object_array':
        array = np.empty(2, dtype=object)
        array[:] = lanes
        lanes = array
    for lane in range(2):
        assert observation_snapshot(lanes, lane) == observation_snapshot(stacked, lane)
    before = observation_snapshot(lanes, 0)
    lanes[1]['rgb'][0, 0, 0, 0] += 1
    assert observation_snapshot(lanes, 0) == before


def test_scheme_b_once_per_plan_and_exact_timestep_weight_lookup(monkeypatch):
    from types import SimpleNamespace
    from modular_policy.policy.dp_unet import DiffusionUnetPolicy
    monkeypatch.setattr('modular_policy.policy.error_moment_fusion.encode_condition',
                        lambda expert, obs: obs[expert.port])

    class StepStatistics(nn.Module):
        num_steps = 3
        def __init__(self):
            super().__init__()
            self.calls = 0
        def forward(self, rgb, pcd, length):
            self.calls += 1
            return rgb.new_tensor([[1.,4.,0.],[4.,1.,0.],[1.,1.,0.]]).expand(len(rgb),-1,-1)

    rgb, pcd, stats = Expert('rgb', 2.), Expert('pcd', -1.), StepStatistics()
    policy = ErrorMomentFusionPolicy(rgb, pcd, 'point', statistics=stats, timestep_ids=[2,1,0])
    policy.reset()
    result = policy.predict_action({'rgb': torch.zeros(2,3), 'pcd': torch.ones(2,2)})
    assert stats.calls == 1
    expected = torch.tensor([[.8,.2,.5]]).expand(2,-1)
    torch.testing.assert_close(result['weights_by_step'][...,0], expected)
    assert result['weights'].shape == (2,2)
    assert len(rgb.model.samples) == len(pcd.model.samples) == 3
    for r,p in zip(rgb.model.samples,pcd.model.samples):
        assert torch.equal(r,p)

    class Reference(nn.Module):
        def forward(self, sample, timestep, global_cond):
            w = {2:.8,1:.2,0:.5}[int(timestep)]
            return torch.full_like(sample, 2*w-(1-w))

    reference = SimpleNamespace(model=Reference(), noise_scheduler=DDPMScheduler(num_train_timesteps=3,clip_sample=False),
        horizon=4, action_dim=2, num_inference_steps=3)
    expected_actions = DiffusionUnetPolicy.conditional_sample(reference, torch.zeros(2,3),
        generator=torch.Generator().manual_seed(42))
    torch.testing.assert_close(result['action_pred'],expected_actions)
    assert len(policy.trace[0]['rgb_weights'][0]) == 3
    with pytest.raises(ValueError, match='timestep order'):
        ErrorMomentFusionPolicy(rgb,pcd,'point',statistics=stats,timestep_ids=[0,1,2])
    with pytest.raises(ValueError, match='do not reuse'):
        ErrorMomentFusionPolicy(rgb,pcd,'robust',statistics=stats,timestep_ids=[2,1,0])


def test_native_router_probability_is_used_in_existing_shared_sampler(monkeypatch):
    from modular_policy.model.frozen_expert_router import FrozenExpertRouter
    monkeypatch.setattr('modular_policy.policy.error_moment_fusion.encode_condition',lambda expert,obs:obs[expert.port])
    router = FrozenExpertRouter(3,2,1)
    with torch.no_grad():
        for parameter in router.parameters():
            parameter.zero_()
    policy = ErrorMomentFusionPolicy(Expert('rgb',2.),Expert('pcd',-1.),'router',statistics=router,history_contexts=1)
    policy.reset()
    result = policy.predict_action({'rgb':torch.zeros(2,3),'pcd':torch.zeros(2,2)})
    torch.testing.assert_close(result['weights'],torch.full((2,2),.5))
    assert len(policy.rgb.model.samples) == len(policy.pcd.model.samples) == 3
    assert policy.trace[0]['moments'] is None
