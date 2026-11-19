"""Reuse the offline train/validation loop for the native MMPC routing baseline."""
from pathlib import Path

import numpy as np
import torch

from modular_policy.common.error_moments import file_hash, write_json
from modular_policy.model.frozen_expert_router import FrozenExpertRouter, direct_router_loss, shared_state_dimension
from modular_policy.workspace.train_error_moments import TrainErrorMomentsWorkspace, risk_report, mean_fusion_risk


class TrainFrozenRouterWorkspace(TrainErrorMomentsWorkspace):
    def dataset_roles(self):
        selection=self.cfg.get('router_data','risk_train')
        if selection=='expert_train':
            return 'expert_train','expert_val'
        if selection=='risk_train':
            return super().dataset_roles()
        raise ValueError('Unknown router data protocol')
    def training_label(self):
        return 'Native frozen MMPC router'

    def format_metrics(self, metrics):
        for before,after in [('train_moment_loss','train_router_loss'),
                             ('sampled_train_moment_loss','sampled_train_router_loss'),
                             ('val_moment_loss','val_router_loss'),('best_val_loss','best_val_router_loss')]:
            metrics[after] = metrics.pop(before)
        return metrics

    def signature_extras(self):
        return {'workspace_sha256': file_hash(__file__),
                'router_source_sha256': file_hash(Path(__file__).parents[1]/'policy/dp_unets_spec.py'),
                'router_model_sha256': file_hash(Path(__file__).parents[1]/'model/frozen_expert_router.py')}

    def verify_experts(self, identity):
        for modality in ['rgb','pcd']:
            info = identity['experts'][modality]
            if file_hash(info['path']) != info['sha256']:
                raise ValueError(f'{modality} checkpoint changed since cached expert predictions')

    def build_model(self, dataset, statistics):
        if self.cfg.risk_mode != 'shared' or self.cfg.history_contexts != 1:
            raise ValueError('Native direct router requires shared weights and history_contexts=1')
        identity = dataset.metadata['identity']
        from modular_policy.common.frozen_router_data import router_cache_roles
        roles=router_cache_roles(identity)
        if set(roles)!=set(self.dataset_roles()):
            raise ValueError('Requested router data protocol differs from cached episodes')
        self.verify_experts(identity)
        state_dim = shared_state_dimension(identity)
        model = FrozenExpertRouter(dataset.rgb_dim, dataset.pcd_dim, state_dim)
        for episode in dataset.episodes:
            model.router_input(torch.from_numpy(episode['rgb_features']), torch.from_numpy(episode['pcd_features']))
        return model

    def run(self):
        super().run()
        import json
        metadata = json.loads((Path(self.output_dir)/'cache_metadata.json').read_text())
        self.verify_experts(metadata['identity'])
        summary_path = Path(self.output_dir)/'run_summary.json'
        if summary_path.exists():
            summary = json.loads(summary_path.read_text())
            if 'best_val_moment_loss' in summary:
                summary['best_val_router_loss'] = summary.pop('best_val_moment_loss')
            summary['model_kind'] = 'native_mmpc_frozen_router'
            write_json(summary_path,summary)
        write_json(Path(self.output_dir)/'expert_freeze_audit.json',{
            'expert_files_unchanged_before_and_after_training': True,
            'expert_parameters_present_in_optimizer': False,
            'trainable_parameters': [name for name,p in self.model.named_parameters() if p.requires_grad],
            'expert_hashes': {m: metadata['identity']['experts'][m]['sha256'] for m in ['rgb','pcd']}})

    def training_loss(self, prediction, target, dataset):
        return direct_router_loss(prediction, target)

    def extra_audit(self):
        return {'model_kind': 'native_mmpc_frozen_router', 'shared_state_dim': self.model.shared_state_dim,
                'router_input_dim': self.model.input_dim, 'expert_weights_in_optimizer': False,
                'objective': 'cached exact frozen-expert mixture epsilon MSE; no moment regression',
                'primary_checkpoint_metric': 'episode-balanced validation mixture epsilon MSE',
                'router_source_sha256': file_hash(Path(__file__).parents[1]/'policy/dp_unets_spec.py'),
                'router_model_sha256': file_hash(Path(__file__).parents[1]/'model/frozen_expert_router.py')}

    @torch.no_grad()
    def validate(self, loader, dataset, statistics, device, tolerance):
        self.model.eval()
        prediction = np.empty((len(dataset),2),np.float64)
        loss = 0.
        for batch in loader:
            values = self.predict(self.model,batch,device)
            local = direct_router_loss(values,batch['target'].to(device))
            loss += float((local.cpu().double()*batch['weight']).sum())
            prediction[batch['index'].numpy()] = values.cpu().numpy()
        if not np.isfinite(loss):
            raise FloatingPointError('Nonfinite router validation risk')
        loss = mean_fusion_risk(dataset,prediction[:,0])
        return loss,prediction,prediction[:,0],np.zeros(len(dataset),dtype=bool)

    def store_validation(self,output,prefix,dataset,baselines,prediction,weights,fallback,
                         epoch,val_loss,global_weight,criterion):
        report = risk_report(dataset,{**baselines,'fixed_0.6': .6,'conditional': weights})
        report.update(epoch=epoch,val_router_loss=val_loss,val_moment_loss=None,
            checkpoint_selection='minimum validation fusion epsilon MSE' if prefix!='latest' else 'last completed epoch',
            model_kind='native_mmpc_frozen_router',rgb_weight_range=[float(weights.min()),float(weights.max())],
            scope=f'{self.dataset_roles()} development split; no calibration or final test used')
        write_json(output/f'{prefix}_validation.json',report)
        np.savez_compressed(output/f'{prefix}_validation_predictions.npz',probabilities=prediction,
            rgb_weights=weights,targets=np.concatenate([e['moments'] for e in dataset.episodes]),
            episode_ids=np.concatenate([np.full(len(e['moments']),e['episode_id']) for e in dataset.episodes]),
            anchor_frames=np.concatenate([e['anchor_frames'] for e in dataset.episodes]))
