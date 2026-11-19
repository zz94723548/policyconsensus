"""Native router tests using synthetic features/cache, never real experts or sim."""
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn
from omegaconf import OmegaConf

from modular_policy.common.episode_split import canonical_hash
from modular_policy.common.error_moments import file_hash
from modular_policy.model.frozen_expert_router import FrozenExpertRouter, direct_router_loss
from modular_policy.policy.dp_unets_spec import make_weight_predictor
from test_error_moments import cache


def test_native_architecture_and_shared_state_not_duplicated():
    torch.manual_seed(7)
    native = make_weight_predictor(4,2)
    torch.manual_seed(7)
    original = nn.Sequential(nn.Linear(4,4),nn.ReLU(),nn.Linear(4,2))
    for a,b in zip(native.state_dict().values(),original.state_dict().values()):
        torch.testing.assert_close(a,b)
    model = FrozenExpertRouter(3,2,1)
    r,p = torch.tensor([[1.,2.,9.]]),torch.tensor([[3.,9.]])
    torch.testing.assert_close(model.router_input(r,p),torch.tensor([[1.,2.,3.,9.]]))
    probs = model(r[:,None],p[:,None],torch.ones(1,dtype=torch.long))
    torch.testing.assert_close(probs.sum(-1),torch.ones(1))
    with pytest.raises(ValueError,match='robot states'):
        model.router_input(r,p+1)


def test_cached_objective_and_gradient_equal_original_epsilon_mixture_mse():
    torch.manual_seed(4)
    er,ep,noise = [torch.randn(3,16,9,dtype=torch.double) for _ in range(3)]
    logits = torch.randn(3,2,dtype=torch.double,requires_grad=True)
    probabilities = logits.softmax(-1)
    moments = torch.stack([er.square().mean((1,2)),ep.square().mean((1,2)),(er*ep).mean((1,2))],-1)
    cached = direct_router_loss(probabilities,moments).mean()
    w = probabilities[:,0,None,None]
    original = (w*(noise+er)+(1-w)*(noise+ep)-noise).square().mean()
    torch.testing.assert_close(cached,original)
    torch.testing.assert_close(torch.autograd.grad(cached,logits,retain_graph=True)[0],
                               torch.autograd.grad(original,logits)[0])


def test_router_training_freezes_files_saves_and_loads_only_router(cache,tmp_path,monkeypatch):
    from modular_policy.workspace.train_frozen_router import TrainFrozenRouterWorkspace
    from modular_policy.policy.error_moment_fusion import load_native_router_policy
    from types import SimpleNamespace
    path,metadata = cache
    identity = metadata['identity']
    identity['experts'] = {}
    for modality in ['rgb','pcd']:
        frozen = tmp_path/f'{modality}.ckpt'
        frozen.write_text('frozen expert bytes')
        identity['experts'][modality] = {'path':str(frozen),'sha256':file_hash(frozen),'weights':'ema_model',
            'policy_config':{'n_obs_steps':1,'shape_meta':{'obs':{'robot_state':{'shape':[1],'type':'robot_state'}}}}}
    identity.update(val_noise_repeats=8,sampler_sha256='sampler',dataset_sha256='dataset',policy_sha256='policy')
    for record in metadata['episodes']:
        target = path/record['file']
        with np.load(target) as data:
            values = {k:data[k] for k in data.files}
        values['pcd_features'][:,-1] = values['rgb_features'][:,-1]
        np.savez_compressed(target,**values)
        record['sha256'] = file_hash(target)
    metadata['signature'] = canonical_hash(identity)
    (path/'metadata.json').write_text(json.dumps(metadata))
    cfg = OmegaConf.create({'_target_':'modular_policy.workspace.train_frozen_router.TrainFrozenRouterWorkspace',
        'cache_dir':str(path),'risk_mode':'shared','history_contexts':1,'weight_tolerance':1e-8,'resume':False,'model':None,
        'optimizer':{'_target_':'torch.optim.AdamW','lr':3e-4,'weight_decay':1e-4},
        'training':{'device':'cpu','seed':42,'batch_size':64,'num_epochs':2,'early_stopping':False,
            'min_epochs':1,'patience':1,'snapshot_every':1,'gradient_clip_norm':1.}})
    output = tmp_path/'router_run'
    workspace = TrainFrozenRouterWorkspace(cfg,output_dir=str(output))
    workspace.run()
    report = json.loads((output/'best_risk_validation.json').read_text())
    freeze = json.loads((output/'expert_freeze_audit.json').read_text())
    assert report['model_kind'] == 'native_mmpc_frozen_router'
    assert report['val_moment_loss'] is None
    assert freeze['expert_files_unchanged_before_and_after_training']
    assert not freeze['expert_parameters_present_in_optimizer']
    assert all(name.startswith('weight_predictor.') for name in freeze['trainable_parameters'])
    assert len({id(p) for g in workspace.optimizer.param_groups for p in g['params']}) == 4
    cfg.resume = True
    workspace = TrainFrozenRouterWorkspace(cfg,output_dir=str(output))
    workspace.run()
    assert workspace.epoch == 2

    class TinyExpert(nn.Module):
        def __init__(self):
            super().__init__()
            self.parameter = nn.Parameter(torch.ones(1))
            self.n_action_steps = 8
    monkeypatch.setattr('modular_policy.policy.error_moment_fusion.load_frozen_expert',
        lambda path,modality,device:(TinyExpert(),None,identity['experts'][modality]))
    monkeypatch.setattr('modular_policy.policy.error_moment_fusion.audit_experts',lambda *args:np.array([99,0]))
    policy,provenance = load_native_router_policy('rgb.ckpt','pcd.ckpt',output/'checkpoints/best_risk.ckpt',42,torch.device('cpu'))
    assert policy.mode == 'router' and policy.history.length == 1
    assert provenance['scheme'] == 'native_mmpc_router'

    # Reuse the real offline entry with a synthetic final cache and reference report.
    from test_moment_calibration import make_heldout_cache
    from modular_policy.dataset.error_moments_dataset import ErrorMomentsDataset
    from modular_policy.workspace.train_error_moments import risk_report
    from modular_policy.workspace.calibrate_error_moments import CalibrateErrorMomentsWorkspace
    test_dir=tmp_path/'test_cache'
    test_dir.mkdir()
    _,_,test_meta=make_heldout_cache(test_dir,'test')
    test_meta['identity']['experts']=identity['experts']
    files={m+'_checkpoint':Path(identity['experts'][m]['path']) for m in ['rgb','pcd']}
    for name in ['statistics_checkpoint','calibration','source_manifest','risk_split']:
        f=tmp_path/f'frozen_{name}'
        f.write_text(name)
        files[name]=f
    protocol={'purpose':'frozen_final_offline_error_moments','risk_mode':'shared','fixed_rgb_weight':.6,
        'seed':2026,'test_noise_repeats':8,
        'files':{name:{'path':str(f),'sha256':file_hash(f)} for name,f in files.items()}}
    protocol_path=tmp_path/'final_protocol.json'
    protocol_path.write_text(json.dumps(protocol))
    test_meta['identity'].update(final_protocol=protocol,final_protocol_sha256=file_hash(protocol_path),seed=2026)
    test_meta['signature']=canonical_hash(test_meta['identity'])
    (test_dir/'metadata.json').write_text(json.dumps(test_meta))
    dataset=ErrorMomentsDataset(test_dir,'test',history_contexts=1)
    weights={'rgb':1.,'pcd':0.,'equal':.5,'fixed':.6,'point':.4,'robust':.55}
    reference=risk_report(dataset,weights)
    reference.update(test_cache_signature=test_meta['signature'],frozen_protocol_sha256=file_hash(protocol_path),fixed_rgb_weight=.6)
    ref_path=tmp_path/'reference_report.json'
    ref_path.write_text(json.dumps(reference))
    ref_npz=tmp_path/'reference_predictions.npz'
    np.savez_compressed(ref_npz,targets=np.concatenate([e['moments'] for e in dataset.episodes]),
        point_rgb_weights=np.full(len(dataset),.4),robust_rgb_weights=np.full(len(dataset),.55),
        episode_ids=np.concatenate([np.full(len(e['moments']),e['episode_id']) for e in dataset.episodes]),
        anchor_frames=np.concatenate([e['anchor_frames'] for e in dataset.episodes]))
    cfg=OmegaConf.create({'evaluation_only':True,'router_checkpoint':str(output/'checkpoints/best_risk.ckpt'),
        'final_protocol':str(protocol_path),'test_cache_dir':str(test_dir),'reference_report':str(ref_path),
        'reference_predictions':str(ref_npz),'device':'cpu','batch_size':7})
    offline=tmp_path/'offline'
    evaluator=CalibrateErrorMomentsWorkspace(cfg,output_dir=str(offline))
    evaluator.run()
    result=json.loads((offline/'offline_error_report.json').read_text())
    assert result['episode_count']==16 and set(result['mean_risk'])==set(weights)|{'router'}
    for method,value in reference['mean_risk'].items():
        assert result['mean_risk'][method]==pytest.approx(value)
    assert (offline/'offline_error_comparison.csv').exists()
    with pytest.raises(FileExistsError,match='exists'):
        evaluator.run()


def test_matched_comparison_reads_existing_results_and_checks_router_identity(tmp_path):
    from modular_policy.common.matched_eval_comparison import compare_matched
    from modular_policy.common.fixed_weight_search import TASKS
    for method in ['rgb','pcd','equal','point','robust','router']:
        folder=tmp_path/method
        folder.mkdir()
        info={'seed':42,'test_start_seed':1000,'n_parallel_envs':4,'n_test':4,
              'rgb':{'sha256':'rgb'},'pcd':{'sha256':'pcd'},'router_sha256':'router'}
        log={'fusion':info,'mean_success_rate':.5,'episode_results':{
            str(1000+i):{'seed':1000+i,'task':task,'success':i<2} for i,task in enumerate(TASKS)}}
        (folder/'eval_log.json').write_text(json.dumps(log))
    folder=tmp_path/'offline_errors/router'
    folder.mkdir(parents=True)
    report={'router':{'router_sha256':'router'},'mean_risk':{'router':.003,'point':.004}}
    (folder/'offline_error_report.json').write_text(json.dumps(report))
    summary=compare_matched(tmp_path)
    assert len(summary['rows'])==6
    assert next(r for r in summary['rows'] if r['method']=='router')['offline_epsilon_mse']==.003
    local_fixed=tmp_path/'fixed_0.6'
    local_fixed.mkdir()
    log['fusion'].update(mode='fixed',fixed_rgb_weight=.6)
    (local_fixed/'eval_log.json').write_text(json.dumps(log))
    summary=compare_matched(tmp_path)
    fixed=next(r for r in summary['rows'] if r['method']=='fixed')
    assert fixed['simulation_source']==str(local_fixed/'eval_log.json')
    report['router']['router_sha256']='different'
    (folder/'offline_error_report.json').write_text(json.dumps(report))
    with pytest.raises(ValueError,match='different checkpoints'):
        compare_matched(tmp_path)


def test_expert_data_router_uses_original_120_20_and_no_reserved_episodes(cache,tmp_path):
    from modular_policy.common.frozen_router_data import router_cache_roles
    from modular_policy.dataset.error_moments_dataset import ErrorMomentsDataset
    from modular_policy.workspace.train_frozen_router import TrainFrozenRouterWorkspace
    path,metadata=cache
    identity=metadata['identity']
    identity['collection_partition']='expert_router'
    original=identity['source_manifest']
    roles=router_cache_roles(identity)
    assert len(roles['expert_train'])==120 and len(roles['expert_val'])==20
    assert not (set(roles['expert_train']+roles['expert_val'])&set(original['splits']['risk_train']))
    identity['experts']={}
    for modality in ['rgb','pcd']:
        frozen=tmp_path/f'frozen_{modality}.ckpt'
        frozen.write_text('unchanged')
        identity['experts'][modality]={'path':str(frozen),'sha256':file_hash(frozen),
            'policy_config':{'n_obs_steps':1,'shape_meta':{'obs':{'robot_state':{'shape':[1],'type':'robot_state'}}}}}
    metadata['episodes']=[]
    for role,ids in roles.items():
        (path/role).mkdir()
        for eid in ids:
            first=original['source']['episode_ends'][eid-1] if eid else 0
            relative=f'{role}/episode_{eid:04d}.npz'
            values=np.array([[[1.,4.,.5],[3.,2.,-.5]]]*2)
            np.savez_compressed(path/relative,episode_id=eid,timesteps=[99,0],
                rgb_features=np.ones((2,3)),pcd_features=np.ones((2,2)),
                moments=values.mean(1),moments_by_step=values,anchor_frames=[first,first+8])
            metadata['episodes'].append({'episode_id':eid,'role':role,'file':relative,'sha256':file_hash(path/relative),
                'task':original['task_by_episode'][eid],'contexts':2,'rgb_dim':3,'pcd_dim':2})
    metadata['signature']=canonical_hash(identity)
    (path/'metadata.json').write_text(json.dumps(metadata))
    training=ErrorMomentsDataset(path,'expert_train',history_contexts=1)
    validation=ErrorMomentsDataset(path,'expert_val',history_contexts=1)
    assert len(training.episodes)==120 and len(validation.episodes)==20
    training.training_statistics()
    with pytest.raises(ValueError,match='training'):
        validation.training_statistics()
    cfg=OmegaConf.create({'router_data':'expert_train','risk_mode':'shared','history_contexts':1})
    workspace=TrainFrozenRouterWorkspace(cfg,output_dir=str(tmp_path/'run'))
    assert workspace.dataset_roles()==('expert_train','expert_val')
    model=workspace.build_model(training,training.training_statistics())
    assert model.input_dim==4
    with pytest.raises(ValueError,match='not present'):
        ErrorMomentsDataset(path,'risk_model_train')
