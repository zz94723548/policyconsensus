"""Frozen-model calibration and final evaluation using the same offline reports."""

import json
from pathlib import Path

import dill
import numpy as np
from omegaconf import OmegaConf
import torch
from torch.utils.data import DataLoader

from modular_policy.common.episode_split import canonical_hash
from modular_policy.common.error_moments import check_moments, file_hash, optimal_weight, write_json
from modular_policy.common.moment_calibration import (
    audit_calibration_identity, fit_shared_radius, normalized_frobenius_residual,
    robust_weight, worst_case_risk,
)
from modular_policy.dataset.error_moments_dataset import ErrorMomentsDataset
from modular_policy.model.error_moments import HistoryErrorMoments
from modular_policy.workspace.base_workspace import BaseWorkspace
from modular_policy.workspace.train_error_moments import risk_report


class CalibrateErrorMomentsWorkspace(BaseWorkspace):
    @torch.inference_mode()
    def run(self):
        cfg = self.cfg
        if cfg.get('evaluation_only', False):
            if cfg.get('router_checkpoint'):
                return self.evaluate_router_test()
            return self.evaluate_final_test()
        output = Path(self.output_dir)
        output.mkdir(parents=True, exist_ok=True)
        if (output / "calibration.json").exists():
            raise FileExistsError("Calibration already exists: freeze this artifact or select a fresh hydra.run.dir")
        if cfg.batch_size < 1:
            raise ValueError("batch_size must be positive")
        payload = torch.load(cfg.checkpoint, map_location="cpu", pickle_module=dill)
        training_cfg = payload["cfg"]
        if training_cfg.risk_mode != "shared":
            raise ValueError("This calibration supports scheme A only")
        metadata = json.loads(Path(cfg.training_cache_metadata).read_text())
        audit = json.loads(Path(cfg.training_audit).read_text())
        checkpoint_signature = dill.loads(payload["pickles"]["signature"])
        if audit["signature"] != checkpoint_signature or audit["cache_signature"] != metadata["signature"]:
            raise ValueError("Frozen checkpoint, training audit and cached labels have different provenance")
        if metadata["status"] != "complete" or canonical_hash(metadata["identity"]) != metadata["signature"]:
            raise ValueError("Invalid frozen training metadata")
        state = payload["state_dicts"]["model"]
        model = HistoryErrorMoments(state["rgb_mean"].numel(), state["pcd_mean"].numel(), **training_cfg.model)
        model.load_state_dict(state, strict=True)
        model.eval().requires_grad_(False)
        checkpoint_epoch = dill.loads(payload["pickles"]["epoch"]) - 1
        del payload
        device = torch.device(cfg.device)
        model.to(device)
        dataset = ErrorMomentsDataset(cfg.calibration_cache_dir, "risk_calibration", training_cfg.history_contexts)
        audit_calibration_identity(metadata["identity"], dataset.metadata["identity"])
        if (dataset.rgb_dim, dataset.pcd_dim) != (model.rgb_mean.numel(), model.pcd_mean.numel()):
            raise ValueError("Frozen model and calibration features have different dimensions")
        # Refuse a higher requested level before running any calibration predictions.
        from modular_policy.common.moment_calibration import episode_conformal_threshold
        for task in dataset.task_order:
            n = sum(e["task"] == task for e in dataset.episodes)
            episode_conformal_threshold(np.zeros(n), cfg.alpha)
        forbidden = set(audit["train_episode_ids"] + audit["val_episode_ids"])
        if forbidden & {e["episode_id"] for e in dataset.episodes}:
            raise ValueError("Calibration overlaps model training/selection episodes")
        loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=0,
                            generator=torch.Generator().manual_seed(2026))
        predictions = np.empty((len(dataset), 3), np.float64)
        for batch in loader:
            values = model(batch["rgb_features"].to(device), batch["pcd_features"].to(device), batch["length"])
            predictions[batch["index"].numpy()] = values.cpu().numpy()
        check_moments(predictions)
        scale = float(model.output_scale)
        targets = np.concatenate([e["moments"] for e in dataset.episodes])
        scores = normalized_frobenius_residual(predictions, targets, scale)
        episodes, offset = [], 0
        for item in dataset.episodes:
            n = len(item["moments"])
            episodes.append({"episode_id": item["episode_id"], "task": item["task"], "contexts": n,
                             "score": float(scores[offset:offset + n].max())})
            offset += n
        fitted = fit_shared_radius(episodes, dataset.task_order, cfg.alpha, scale)
        # The fallback was frozen during training; read the recorded train-only baseline.
        baseline = json.loads(Path(cfg.training_baselines).read_text())
        global_weight = float(baseline["global_rgb_weight"])
        if not 0 <= global_weight <= 1:
            raise ValueError("Invalid frozen global weight")
        expected = float(optimal_weight(baseline["global_moments"], tolerance=training_cfg.weight_tolerance)[0])
        expected_scale = max(float(np.asarray(baseline["global_moments"])[:2].mean()), 1e-8)
        if not np.isclose(expected, global_weight, rtol=1e-7, atol=1e-10) or not np.isclose(expected_scale, scale, rtol=1e-6):
            raise ValueError("Frozen train-only baseline does not match model normalization")
        point, point_fallback = optimal_weight(predictions, fallback=global_weight,
                                               tolerance=training_cfg.weight_tolerance)
        robust, robust_fallback = robust_weight(predictions, fitted["radius"], fallback=global_weight,
                                                tolerance=training_cfg.weight_tolerance)
        artifact = {
            "version": 1, "method": "task-stratified episode conformal; shared maximum radius; PSD Frobenius ball",
            "risk_mode": "shared", "checkpoint": str(Path(cfg.checkpoint).resolve()),
            "checkpoint_sha256": file_hash(cfg.checkpoint), "checkpoint_epoch": checkpoint_epoch,
            "training_signature": checkpoint_signature, "training_cache_signature": metadata["signature"],
            "calibration_cache_signature": dataset.metadata["signature"],
            "training_baselines_sha256": file_hash(cfg.training_baselines),
            "history_contexts": training_cfg.history_contexts, "global_fallback_rgb_weight": global_weight,
            "weight_tolerance": training_cfg.weight_tolerance,
            "known_tasks": dataset.task_order, "timesteps": dataset.timesteps.tolist(),
            "lambda": dataset.step_weights.tolist(), "fitted": fitted, "episode_scores": episodes,
            "uncertainty_set": "Gamma PSD and ||Gamma-Gamma_hat(h)||_F <= radius",
            "robust_rule": "clip((B+radius-C)/(A+B+2*radius-2*C),0,1)",
            "worst_case_risk": "R(w;Gamma_hat)+radius*(w*w+(1-w)*(1-w))",
            "coverage_unit": "entire new episode: all sampled causal contexts jointly",
            "coverage_target": "finite-MC demonstration moment labels, not the unknown conditional population moments",
            "assumptions": ["fixed model and protocol independent of calibration labels",
                            "exchangeable episodes within each known task",
                            "same context extraction and MC noise repeat protocol for future labeled evaluation"],
            "limitations": ["No guarantee for reverse-sampling state shift, unseen tasks or task success",
                            "Calibration-fit coverage and risk are in-sample diagnostics, not independent validation",
                            "Min-max upper bound concerns true risk only if its actual moment matrix lies in the set",
                            "No OOD detector or closed-loop deployment integration is implemented"],
            "calibrator_sha256": file_hash(__file__),
        }
        # Calibration residuals select the radius, never a new checkpoint or alpha.
        report = risk_report(dataset, {"rgb": 1., "pcd": 0., "equal": 0.5,
                                      "global": global_weight, "point": point, "robust": robust})
        report.update(scope="calibration in-sample diagnostics only; not final-test results",
                      calibration_fit_episode_coverage=float(np.mean([e["score"] <= fitted["normalized_radius"] for e in episodes])),
                      point_weight_range=[float(point.min()), float(point.max())],
                      robust_weight_range=[float(robust.min()), float(robust.max())],
                      point_fallback_fraction=float(np.dot(dataset.sample_weights, point_fallback)),
                      robust_fallback_fraction=float(np.dot(dataset.sample_weights, robust_fallback)))
        write_json(output / "calibration_diagnostics.json", report)
        np.savez_compressed(output / "calibration_predictions.npz", predictions=predictions, targets=targets,
            normalized_residuals=scores, point_rgb_weights=point, robust_rgb_weights=robust,
            predicted_worst_case_risk=worst_case_risk(predictions, robust, fitted["radius"]),
            episode_ids=np.concatenate([np.full(len(e["moments"]), e["episode_id"]) for e in dataset.episodes]),
            anchor_frames=np.concatenate([e["anchor_frames"] for e in dataset.episodes]))
        write_json(output / "calibration.json", artifact)
        OmegaConf.save(cfg, output / "calibration_config.yaml", resolve=True)
        print(f"Calibration complete: {len(episodes)} episodes; shared radius={fitted['radius']:.6g}")
        print("Coverage target is episode-level finite-MC labels under within-task exchangeability; final test untouched.")

    @torch.inference_mode()
    def evaluate_final_test(self):
        from modular_policy.common.final_error_moments import read_final_protocol
        cfg = self.cfg
        protocol = read_final_protocol(cfg.final_protocol)
        output = Path(self.output_dir)
        output.mkdir(parents=True, exist_ok=True)
        if (output / 'final_test_report.json').exists():
            raise FileExistsError('Final report already exists; preserve it and use a fresh output directory if needed')
        if cfg.batch_size < 1:
            raise ValueError('batch_size must be positive')
        artifact = json.loads(Path(protocol['files']['calibration']['path']).read_text())
        checkpoint = protocol['files']['statistics_checkpoint']
        if checkpoint['sha256'] != artifact['checkpoint_sha256']:
            raise ValueError('Frozen statistics model differs from calibrated model')
        run = Path(checkpoint['path']).resolve().parents[1]
        metadata = json.loads((run / 'cache_metadata.json').read_text())
        audit = json.loads((run / 'training_audit.json').read_text())
        if (metadata['signature'] != artifact['training_cache_signature'] or
                canonical_hash(metadata['identity']) != metadata['signature'] or
                audit['signature'] != artifact['training_signature']):
            raise ValueError('Frozen training provenance differs from calibration')
        payload = torch.load(checkpoint['path'], map_location='cpu', pickle_module=dill)
        if dill.loads(payload['pickles']['signature']) != artifact['training_signature']:
            raise ValueError('Final-test checkpoint signature differs')
        if (payload['cfg'].risk_mode != 'shared' or
                payload['cfg'].history_contexts != artifact['history_contexts']):
            raise ValueError('Final-test scheme/history differs from calibration')
        state = payload['state_dicts']['model']
        model = HistoryErrorMoments(state['rgb_mean'].numel(), state['pcd_mean'].numel(), **payload['cfg'].model)
        model.load_state_dict(state, strict=True)
        model.eval().requires_grad_(False).to(cfg.device)
        del payload
        dataset = ErrorMomentsDataset(cfg.test_cache_dir, 'test', artifact['history_contexts'])
        identity = dataset.metadata['identity']
        if (identity.get('final_protocol_sha256') != file_hash(cfg.final_protocol) or
                identity.get('final_protocol') != protocol):
            raise ValueError('Final cache was not collected under this frozen protocol')
        audit_calibration_identity(metadata['identity'], identity, partition='test')
        if (identity['seed'] != protocol['seed'] or identity['test_noise_repeats'] != protocol['test_noise_repeats']):
            raise ValueError('Final-test noise protocol differs')
        ids = {e['episode_id'] for e in dataset.episodes}
        forbidden = set(audit['train_episode_ids'] + audit['val_episode_ids'])
        forbidden.update(e['episode_id'] for e in artifact['episode_scores'])
        if ids & forbidden:
            raise ValueError('Final-test episodes overlap model fitting/selection or calibration')
        if (dataset.rgb_dim, dataset.pcd_dim) != (model.rgb_mean.numel(), model.pcd_mean.numel()):
            raise ValueError('Final-test feature dimensions differ')
        loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=0,
                            generator=torch.Generator().manual_seed(protocol['seed']))
        prediction = np.empty((len(dataset), 3), np.float64)
        for batch in loader:
            values = model(batch['rgb_features'].to(cfg.device), batch['pcd_features'].to(cfg.device), batch['length'])
            prediction[batch['index'].numpy()] = values.cpu().numpy()
        check_moments(prediction)
        target = np.concatenate([e['moments'] for e in dataset.episodes])
        fallback, tolerance = artifact['global_fallback_rgb_weight'], artifact['weight_tolerance']
        radius = artifact['fitted']['radius']
        point, pf = optimal_weight(prediction, fallback, tolerance)
        robust, rf = robust_weight(prediction, radius, fallback, tolerance)
        report = risk_report(dataset, {'rgb': 1., 'pcd': 0., 'equal': .5,
            'fixed': protocol['fixed_rgb_weight'], 'point': point, 'robust': robust})
        residuals = normalized_frobenius_residual(prediction, target, float(model.output_scale))
        episode_scores, offset = [], 0
        for item in dataset.episodes:
            n = len(item['moments'])
            score = float(residuals[offset:offset+n].max())
            episode_scores.append({'episode_id': item['episode_id'], 'task': item['task'],
                'score': score, 'covered': score <= artifact['fitted']['normalized_radius']})
            offset += n
        report.update(scope='final held-out demonstration forward-noising epsilon MSE; not task success',
            fixed_rgb_weight=protocol['fixed_rgb_weight'], frozen_radius=radius,
            frozen_protocol=protocol, frozen_protocol_sha256=file_hash(cfg.final_protocol),
            test_cache_signature=dataset.metadata['signature'],
            moment_mse=np.einsum('n,nc->c', dataset.sample_weights, (prediction-target)**2).tolist(),
            normalized_frobenius_mse=float(np.dot(dataset.sample_weights, residuals**2)),
            weight_ranges={'point': [float(point.min()), float(point.max())],
                           'robust': [float(robust.min()), float(robust.max())]},
            fallback_fractions={'point': float(np.dot(dataset.sample_weights, pf)),
                                'robust': float(np.dot(dataset.sample_weights, rf))},
            episode_coverage=float(np.mean([e['covered'] for e in episode_scores])),
            episode_scores=episode_scores,
            coverage_target='finite-MC labels at sampled contexts; not population moments or closed-loop states')
        np.savez_compressed(output / 'final_test_predictions.npz', predictions=prediction, targets=target,
            point_rgb_weights=point, robust_rgb_weights=robust, normalized_residuals=residuals,
            episode_ids=np.concatenate([np.full(len(e['moments']), e['episode_id']) for e in dataset.episodes]),
            anchor_frames=np.concatenate([e['anchor_frames'] for e in dataset.episodes]))
        write_json(output / 'final_test_report.json', report)
        OmegaConf.save(cfg, output / 'final_test_config.yaml', resolve=True)
        print(f'Final offline test complete: {len(dataset.episodes)} episodes; no parameters or radius updated.')
        print(json.dumps(report['mean_risk'], indent=2))

    @torch.inference_mode()
    def evaluate_router_test(self):
        """Add the frozen native router to the SAME final-cache comparison."""
        import csv
        from modular_policy.common.final_error_moments import read_final_protocol
        from modular_policy.policy.error_moment_fusion import load_native_router_checkpoint
        cfg = self.cfg
        output = Path(self.output_dir)
        output.mkdir(parents=True,exist_ok=True)
        if (output/'offline_error_report.json').exists():
            raise FileExistsError('Router offline report exists; preserve it or choose a new output directory')
        if cfg.batch_size < 1:
            raise ValueError('batch_size must be positive')
        protocol = read_final_protocol(cfg.final_protocol)
        model,provenance,training_metadata,audit = load_native_router_checkpoint(cfg.router_checkpoint,torch.device(cfg.device))
        dataset = ErrorMomentsDataset(cfg.test_cache_dir,'test',history_contexts=1)
        identity = dataset.metadata['identity']
        if (identity.get('final_protocol_sha256') != file_hash(cfg.final_protocol) or identity.get('final_protocol') != protocol):
            raise ValueError('Router evaluation requires the exact original final-test cache/protocol')
        audit_calibration_identity(training_metadata['identity'],identity,partition='test')
        if (identity['seed'] != protocol['seed'] or identity['test_noise_repeats'] != protocol['test_noise_repeats']):
            raise ValueError('Router final-test noise protocol differs')
        forbidden = set(audit['train_episode_ids']+audit['val_episode_ids'])
        if forbidden & {e['episode_id'] for e in dataset.episodes}:
            raise ValueError('Router test overlaps training or validation')
        if (dataset.rgb_dim,dataset.pcd_dim) != (model.rgb_dim,model.pcd_dim):
            raise ValueError('Router offline feature dimensions differ')
        reference = json.loads(Path(cfg.reference_report).read_text())
        if (reference['test_cache_signature'] != dataset.metadata['signature'] or
                reference['frozen_protocol_sha256'] != file_hash(cfg.final_protocol) or
                reference['fixed_rgb_weight'] != protocol['fixed_rgb_weight']):
            raise ValueError('Reference results use different final labels or fixed weight')
        ids = np.concatenate([np.full(len(e['moments']),e['episode_id']) for e in dataset.episodes])
        anchors = np.concatenate([e['anchor_frames'] for e in dataset.episodes])
        targets = np.concatenate([e['moments'] for e in dataset.episodes])
        with np.load(cfg.reference_predictions,allow_pickle=False) as data:
            if (not np.array_equal(ids,data['episode_ids']) or not np.array_equal(anchors,data['anchor_frames']) or
                    not np.array_equal(targets,data['targets'])):
                raise ValueError('Reference predictions are not aligned to the original final contexts')
            point = data['point_rgb_weights'].copy()
            robust = data['robust_rgb_weights'].copy()
        for weights in [point,robust]:
            if weights.shape != (len(dataset),) or not np.isfinite(weights).all() or np.any((weights<0)|(weights>1)):
                raise ValueError('Invalid frozen reference weights')
        probabilities = np.empty((len(dataset),2),np.float64)
        loader = DataLoader(dataset,batch_size=cfg.batch_size,shuffle=False,num_workers=0,
                            generator=torch.Generator().manual_seed(protocol['seed']))
        for batch in loader:
            values = model(batch['rgb_features'].to(cfg.device),batch['pcd_features'].to(cfg.device),batch['length'])
            probabilities[batch['index'].numpy()] = values.cpu().numpy()
        if not np.isfinite(probabilities).all() or not np.allclose(probabilities.sum(-1),1,atol=1e-6):
            raise ValueError('Invalid router output probabilities')
        report = risk_report(dataset,{'rgb':1.,'pcd':0.,'equal':.5,'fixed':protocol['fixed_rgb_weight'],
            'point':point,'robust':robust,'router':probabilities[:,0]})
        for method,value in reference['mean_risk'].items():
            if not np.isclose(report['mean_risk'][method],value,rtol=1e-10,atol=1e-14):
                raise ValueError(f'Recomputed reference risk differs: {method}')
        report.update(scope='Appended router comparison on the same previously inspected 16 final demonstration episodes; not task success',
            router=provenance,test_cache_signature=dataset.metadata['signature'],
            frozen_protocol_sha256=file_hash(cfg.final_protocol),fixed_rgb_weight=protocol['fixed_rgb_weight'],
            reference_report_sha256=file_hash(cfg.reference_report),
            reference_predictions_sha256=file_hash(cfg.reference_predictions),
            router_rgb_weight_range=[float(probabilities[:,0].min()),float(probabilities[:,0].max())])
        np.savez_compressed(output/'offline_error_predictions.npz',router_probabilities=probabilities,
            router_rgb_weights=probabilities[:,0],point_rgb_weights=point,robust_rgb_weights=robust,
            targets=targets,episode_ids=ids,anchor_frames=anchors)
        with (output/'offline_error_comparison.csv').open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=['method','mean_epsilon_mse',*dataset.task_order])
            writer.writeheader()
            for method,value in report['mean_risk'].items():
                writer.writerow({'method':method,'mean_epsilon_mse':value,
                    **{task:report['per_task'][task][method] for task in dataset.task_order}})
        write_json(output/'offline_error_report.json',report)
        OmegaConf.save(cfg,output/'offline_error_config.yaml',resolve=True)
        print('Router offline comparison complete: 16 episodes; no training, recollection or recalibration.')
        print(json.dumps(report['mean_risk'],indent=2))
