"""Two-stage training for one independently supervised expert per modality."""
import copy
import pathlib

import dill
import hydra
import torch
from omegaconf import OmegaConf

from modular_policy.workspace.train_policy import TrainPolicyWorkspace


class TrainTwoStagePolicyWorkspace(TrainPolicyWorkspace):
    def __init__(self, cfg, output_dir=None):
        super().__init__(cfg, output_dir=output_dir)
        self.stage = cfg.policy.training_stage
        if self.stage not in ('experts', 'router'):
            raise ValueError('Two-stage workspace requires experts or router stage')
        if self.stage == 'experts':
            self.model.weight_predictor.requires_grad_(False)
            params = [p for p in self.model.parameters() if p.requires_grad]
        else:
            self.model.adapt('weight_predictor')
            params = list(self.model.weight_predictor.parameters())
        # Each stage has its own optimizer and progress counters.
        self.optimizer = hydra.utils.instantiate(cfg.optimizer, params=params)

    def configure_training_mode(self):
        if self.stage == 'router':
            # The parent evaluation loop calls train(); enforce frozen eval mode
            # again at the beginning of every epoch.
            self.model.adapt('weight_predictor')
        else:
            self.model.train()
            self.model.weight_predictor.requires_grad_(False)

    def configure_normalizer(self, normalizer):
        if self.stage == 'experts':
            return super().configure_normalizer(normalizer)
        # Frozen experts must receive precisely the normalization used for
        # their training, including on a resumed router run.
        expected = self.model.normalizer.state_dict()
        actual = normalizer.state_dict()
        if (not expected or expected.keys() != actual.keys() or any(
            not torch.equal(expected[key].cpu(), actual[key].cpu()) for key in expected
        )):
            raise ValueError('Router dataset normalizer differs from expert checkpoint')
        if self.ema_model is not None:
            self.ema_model.set_normalizer(self.model.normalizer)

    def _initialize_router(self):
        checkpoint = self.cfg.training.get('expert_checkpoint')
        if not checkpoint:
            raise ValueError('Set training.expert_checkpoint to an experts-stage checkpoint')
        path = pathlib.Path(hydra.utils.to_absolute_path(checkpoint))
        with path.open('rb') as stream:
            payload = torch.load(stream, pickle_module=dill, map_location='cpu')
        source_cfg = payload['cfg']
        if source_cfg.policy.get('training_stage') != 'experts':
            raise ValueError('expert_checkpoint must come from the experts training stage')
        # Refuse silent changes to conditioning, diffusion, and architecture.
        source_policy = OmegaConf.to_container(source_cfg.policy, resolve=True)
        target_policy = OmegaConf.to_container(self.cfg.policy, resolve=True)
        source_policy.pop('training_stage', None)
        target_policy.pop('training_stage', None)
        if source_policy != target_policy:
            raise ValueError('Expert and router policy configurations do not match')
        source_key = self.cfg.training.get('expert_weights', 'ema_model')
        if source_key not in ('model', 'ema_model') or source_key not in payload['state_dicts']:
            raise ValueError(f'Expert checkpoint has no requested weights: {source_key}')
        self.model.load_state_dict(payload['state_dicts'][source_key], strict=True)
        # The stage-one router was untrained. Start with a uniform mixture.
        torch.nn.init.zeros_(self.model.weight_predictor[-1].weight)
        torch.nn.init.zeros_(self.model.weight_predictor[-1].bias)
        self.model.adapt('weight_predictor')
        if self.ema_model is not None:
            self.ema_model = copy.deepcopy(self.model)
        print(f'Initialized frozen experts from {path} ({source_key})')

    def run(self):
        is_resume = self.cfg.training.resume and self.get_checkpoint_path().is_file()
        if self.stage == 'router' and not is_resume:
            self._initialize_router()
        return super().run()
