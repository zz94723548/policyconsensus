"""Direct MMPC MLP routing with frozen expert features and cached epsilon loss."""
import torch
from torch import nn

from modular_policy.policy.dp_unets_spec import make_weight_predictor, soft_gating_weights


def shared_state_dimension(identity):
    configs = [identity['experts'][m]['policy_config'] for m in ['rgb', 'pcd']]
    if configs[0]['n_obs_steps'] != configs[1]['n_obs_steps']:
        raise ValueError('Expert observation histories differ')
    states = [{k: v['shape'] for k,v in cfg['shape_meta']['obs'].items() if v['type']=='robot_state'}
              for cfg in configs]
    if states[0] != states[1] or not states[0]:
        raise ValueError('Shared robot-state definitions differ')
    return configs[0]['n_obs_steps'] * sum(shape[-1] for shape in states[0].values())


def direct_router_loss(probabilities, moments):
    """Exactly E||w*e_rgb+(1-w)*e_pcd||² for the frozen paired MC samples."""
    w = probabilities[..., 0]
    a,b,c = moments.unbind(-1)
    return w.square()*a + (1-w).square()*b + 2*w*(1-w)*c


class FrozenExpertRouter(nn.Module):
    def __init__(self, rgb_dim, pcd_dim, shared_state_dim):
        super().__init__()
        if not 0 < shared_state_dim < min(rgb_dim, pcd_dim):
            raise ValueError('Invalid shared robot-state feature size')
        self.rgb_dim, self.pcd_dim, self.shared_state_dim = rgb_dim, pcd_dim, shared_state_dim
        self.input_dim = rgb_dim + pcd_dim - shared_state_dim
        self.weight_predictor = make_weight_predictor(self.input_dim, 2)
        # Keeps inherited loss/report interface compatible; no normalization of inputs.
        self.register_buffer('output_scale', torch.ones(()))

    def router_input(self, rgb, pcd):
        if rgb.shape[-1] != self.rgb_dim or pcd.shape[-1] != self.pcd_dim:
            raise ValueError('Frozen router feature dimensions differ')
        n = self.shared_state_dim
        if not torch.equal(rgb[..., -n:], pcd[..., -n:]):
            raise ValueError('RGB/PCD shared robot states differ')
        # Native joint encoder order: RGB features, PCD features, robot state once.
        return torch.cat([rgb[..., :-n], pcd[..., :-n], rgb[..., -n:]], dim=-1)

    def forward(self, rgb_features, pcd_features, length):
        if rgb_features.shape[1] != 1 or pcd_features.shape[1] != 1 or not torch.all(length == 1):
            raise ValueError('Native router uses current planning context only (history_contexts=1)')
        value = self.router_input(rgb_features[:,0], pcd_features[:,0])
        return soft_gating_weights(self.weight_predictor(value), dim=-1)
