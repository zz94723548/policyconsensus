"""One shared PSD matrix (A), or K PSD matrices from one forward pass (B)."""

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence


def moments_from_raw(raw, scale):
    a, b = (F.softplus(raw[..., :2]) + 1e-6).unbind(-1)
    c = torch.tanh(raw[..., 2]) * torch.sqrt(a * b)
    if isinstance(scale, torch.Tensor) and scale.ndim == 1:
        scale = scale[:, None]
    return torch.stack([a, b, c], -1) * scale


def matrix_regression_loss(prediction, target, scale):
    if isinstance(scale, torch.Tensor) and scale.ndim == 1:
        scale = scale[:, None]
    error = (prediction - target) / scale
    return error[..., 0].square() + error[..., 1].square() + 2 * error[..., 2].square()


class HistoryErrorMoments(nn.Module):
    def __init__(self, rgb_dim, pcd_dim, projection_dim=64, hidden_dim=64, head_dim=32, num_steps=None):
        super().__init__()
        if num_steps is not None and (not isinstance(num_steps, int) or num_steps < 1):
            raise ValueError('num_steps must be a positive integer')
        self.num_steps = num_steps
        for name, dim in [("rgb", rgb_dim), ("pcd", pcd_dim)]:
            self.register_buffer(name + "_mean", torch.zeros(dim))
            self.register_buffer(name + "_std", torch.ones(dim))
            setattr(self, name + "_projection", nn.Sequential(
                nn.Linear(dim, projection_dim), nn.LayerNorm(projection_dim), nn.SiLU()))
        self.register_buffer("output_scale", torch.ones(()) if num_steps is None else torch.ones(num_steps))
        self.temporal = nn.GRU(projection_dim * 2, hidden_dim, batch_first=True)
        self.head = nn.Sequential(nn.Linear(hidden_dim, head_dim), nn.SiLU(),
                                  nn.Linear(head_dim, 3 if num_steps is None else 3 * num_steps))

    @torch.no_grad()
    def set_statistics(self, statistics):
        for name in ["rgb", "pcd"]:
            for key in ["mean", "std"]:
                getattr(self, name + "_" + key).copy_(torch.as_tensor(statistics[name + "_features"][key]))
        if self.num_steps is None:
            self.output_scale.fill_(statistics["output_scale"])
        else:
            self.output_scale.copy_(torch.as_tensor(statistics['step_output_scale']))

    def forward(self, rgb_features, pcd_features, length):
        features = []
        for name, x in [("rgb", rgb_features), ("pcd", pcd_features)]:
            x = (x - getattr(self, name + "_mean")) / getattr(self, name + "_std")
            features.append(getattr(self, name + "_projection")(x))
        sequence = torch.cat(features, -1)
        packed = pack_padded_sequence(sequence, length.cpu(), batch_first=True, enforce_sorted=False)
        _, hidden = self.temporal(packed)
        raw = self.head(hidden[-1])
        if self.num_steps is not None:
            raw = raw.reshape(-1, self.num_steps, 3)
        return moments_from_raw(raw, self.output_scale)
