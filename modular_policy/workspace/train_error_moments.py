"""Train shared (A) or per-step (B) moments on the same isolated 20/8 cache."""

import json
from pathlib import Path
import random

import hydra
import numpy as np
from omegaconf import OmegaConf
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from modular_policy.common.episode_split import canonical_hash
from modular_policy.common.error_moments import bootstrap_delta, file_hash, optimal_weight, quadratic_risk, write_json
from modular_policy.common.json_logger import JsonLogger
from modular_policy.common.pytorch_util import optimizer_to
from modular_policy.dataset.error_moments_dataset import ErrorMomentsDataset
from modular_policy.model.error_moments import HistoryErrorMoments, matrix_regression_loss
from modular_policy.workspace.base_workspace import BaseWorkspace


def risk_report(dataset, weights):
    """Episode-balanced risks using the SAME stored paired errors for all methods."""
    rows, step_risks = [], []
    offset = 0
    for item in dataset.episodes:
        n = len(item["moments"])
        row = {"episode_id": item["episode_id"], "task": item["task"]}
        step = {}
        for name, weight in weights.items():
            if np.ndim(weight) == 2:
                w = np.broadcast_to(weight, (len(dataset), len(dataset.timesteps)))[offset:offset + n]
                step[name] = quadratic_risk(item['moments_by_step'], w).mean(0)
                row[name] = float(np.dot(dataset.step_weights, step[name]))
            else:
                w = np.broadcast_to(weight, (len(dataset),))[offset:offset + n]
                row[name] = float(quadratic_risk(item["moments"], w).mean())
                step[name] = quadratic_risk(item["moments_by_step"], w[:, None]).mean(0)
        rows.append(row)
        step_risks.append(step)
        offset += n
    means = {name: float(np.mean([r[name] for r in rows])) for name in weights}
    per_task = {task: {name: float(np.mean([r[name] for r in rows if r["task"] == task]))
                       for name in weights} for task in dataset.task_order}
    report = {"episode_count": len(rows), "mean_risk": means, "per_task": per_task, "per_episode": rows,
              "timesteps": dataset.timesteps.tolist(),
              "risk_by_step": {name: np.mean([r[name] for r in step_risks], axis=0).tolist() for name in weights}}
    if "conditional" in weights:
        report["conditional_minus_baseline"] = {
            name: bootstrap_delta([r["conditional"] - r[name] for r in rows], [r["task"] for r in rows])
            for name in weights if name != "conditional"}
    return report


def should_stop(training, epoch, bad_epochs):
    return (training.get("early_stopping", True)
            and epoch >= training.min_epochs and bad_epochs >= training.patience)


def mean_fusion_risk(dataset, weights):
    if np.ndim(weights) == 2:
        moments = np.concatenate([e['moments_by_step'] for e in dataset.episodes])
        risk = np.einsum('nk,k->n', quadratic_risk(moments, weights), dataset.step_weights)
        return float(np.dot(dataset.sample_weights, risk))
    moments = np.concatenate([e["moments"] for e in dataset.episodes])
    return float(np.dot(dataset.sample_weights, quadratic_risk(moments, weights)))


def per_context_moment_loss(model, prediction, target, dataset):
    loss = matrix_regression_loss(prediction, target, model.output_scale)
    if loss.ndim == 2:
        loss = (loss * torch.as_tensor(dataset.step_weights, device=loss.device, dtype=loss.dtype)).sum(-1)
    return loss


def save_validation(output, prefix, dataset, baselines, prediction, weights, fallback,
                    epoch, val_loss, global_weight, criterion):
    report = risk_report(dataset, {**baselines, "conditional": weights})
    errors, jitter, mean_step_weights, offset = [], [], [], 0
    label_key = 'moments_by_step' if prediction.ndim == 3 else 'moments'
    for item in dataset.episodes:
        n = len(item["moments"])
        errors.append(((prediction[offset:offset + n] - item[label_key]) ** 2).mean(0))
        if weights.ndim == 2:
            mean_step_weights.append(weights[offset:offset + n].mean(0))
        if n > 1:
            jitter.append(float(np.abs(np.diff(weights[offset:offset + n], axis=0)).mean()))
        offset += n
    report.update(epoch=epoch, val_moment_loss=val_loss, checkpoint_selection=criterion,
                  moment_mse=np.mean(errors, axis=0).tolist(),
                  rgb_weight_range=[float(weights.min()), float(weights.max())],
                  fallback_fraction=float(np.dot(dataset.sample_weights, fallback.mean(1) if fallback.ndim == 2 else fallback)),
                  mean_episode_weight_change=float(np.mean(jitter)) if jitter else 0.0,
                  psd_valid_fraction=1.0, global_rgb_weight=global_weight,
                  scope="development episodes used for checkpoint selection; not independent test or task success")
    report['risk_mode'] = 'per_step' if prediction.ndim == 3 else 'shared'
    if weights.ndim == 2:
        report['mean_rgb_weight_by_step'] = np.mean(mean_step_weights, axis=0).tolist()
        report['mean_diffusion_step_weight_change'] = float(np.abs(np.diff(weights, axis=1)).mean()) if weights.shape[1] > 1 else 0.
    write_json(output / f"{prefix}_validation.json", report)
    np.savez_compressed(output / f"{prefix}_validation_predictions.npz", moments=prediction,
        targets=np.concatenate([e[label_key] for e in dataset.episodes]), rgb_weights=weights, fallback=fallback,
        timesteps=dataset.timesteps,
        episode_ids=np.concatenate([np.full(len(e["moments"]), e["episode_id"]) for e in dataset.episodes]),
        anchor_frames=np.concatenate([e["anchor_frames"] for e in dataset.episodes]))


class TrainErrorMomentsWorkspace(BaseWorkspace):
    include_keys = ("epoch", "global_step", "best_val_loss", "best_val_risk", "bad_epochs", "signature", "sampler_state")

    def __init__(self, cfg, output_dir=None):
        super().__init__(cfg, output_dir)
        self.epoch, self.global_step, self.bad_epochs = 0, 0, 0
        self.best_val_loss = float("inf")
        self.best_val_risk = float("inf")
        self.signature, self.sampler_state = None, None

    @staticmethod
    def predict(model, batch, device):
        return model(batch["rgb_features"].to(device), batch["pcd_features"].to(device), batch["length"])

    def build_model(self, dataset, statistics):
        options = dict(self.cfg.model)
        if self.cfg.risk_mode == 'per_step':
            options['num_steps'] = len(dataset.timesteps)
        model = HistoryErrorMoments(dataset.rgb_dim, dataset.pcd_dim, **options)
        model.set_statistics(statistics)
        return model

    def training_loss(self, prediction, target, dataset):
        return per_context_moment_loss(self.model, prediction, target, dataset)

    def store_validation(self, *args):
        save_validation(*args)

    def extra_audit(self):
        return {}

    def signature_extras(self):
        return {}

    def training_label(self):
        return 'Scheme B' if self.cfg.risk_mode == 'per_step' else 'Scheme A'

    def dataset_roles(self):
        return 'risk_model_train','risk_model_val'

    def format_metrics(self, metrics):
        return metrics

    @torch.no_grad()
    def validate(self, loader, dataset, statistics, device, tolerance):
        self.model.eval()
        shape = (len(dataset), len(dataset.timesteps), 3) if dataset.risk_mode == 'per_step' else (len(dataset), 3)
        prediction = np.empty(shape, np.float64)
        loss = 0.0
        for batch in loader:
            pred = self.predict(self.model, batch, device)
            target = batch["target"].to(device)
            local = per_context_moment_loss(self.model, pred, target, dataset)
            loss += float((local.cpu().double() * batch["weight"]).sum())
            prediction[batch["index"].numpy()] = pred.cpu().numpy()
        if not np.isfinite(loss):
            raise FloatingPointError("Nonfinite validation moment loss")
        global_weight = float(optimal_weight(statistics["global_moments"], tolerance=tolerance)[0])
        weights, fallback = optimal_weight(prediction, fallback=global_weight, tolerance=tolerance)
        return loss, prediction, weights, fallback

    def run(self):
        cfg = self.cfg
        if cfg.risk_mode not in ('shared', 'per_step'):
            raise ValueError('risk_mode must be shared (A) or per_step (B)')
        if min(cfg.training.num_epochs, cfg.training.min_epochs, cfg.training.patience,
               cfg.training.snapshot_every, cfg.training.batch_size) < 1:
            raise ValueError("Training counts and intervals must be positive")
        if cfg.training.min_epochs > cfg.training.num_epochs or cfg.training.seed < 0:
            raise ValueError("Invalid training epoch budget or seed")
        seed = cfg.training.seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        train_role,val_role=self.dataset_roles()
        train = ErrorMomentsDataset(cfg.cache_dir, train_role, cfg.history_contexts, cfg.risk_mode)
        val = ErrorMomentsDataset(cfg.cache_dir, val_role, cfg.history_contexts, cfg.risk_mode)
        if (train.rgb_dim, train.pcd_dim) != (val.rgb_dim, val.pcd_dim):
            raise ValueError("Training and validation feature dimensions differ")
        if train.metadata["signature"] != val.metadata["signature"]:
            raise ValueError("Training and validation use different caches")
        statistics = train.training_statistics()
        self.model = self.build_model(train, statistics)
        self.optimizer = hydra.utils.instantiate(cfg.optimizer, params=self.model.parameters())
        signature_config = OmegaConf.to_container(cfg, resolve=True)
        signature_config.pop("resume")
        signature = canonical_hash({"cache_signature": train.metadata["signature"], "config": signature_config,
            "trainer_sha256": file_hash(__file__),
            "model_sha256": file_hash(Path(__file__).parents[1] / "model/error_moments.py"),
            "dataset_sha256": file_hash(Path(__file__).parents[1] / "dataset/error_moments_dataset.py"),
            "model_specific": self.signature_extras()})
        output = Path(self.output_dir)
        output.mkdir(parents=True, exist_ok=True)
        latest = self.get_checkpoint_path()
        if latest.exists() and not cfg.resume:
            raise FileExistsError("Existing run: use resume=true or a fresh hydra.run.dir")
        audit_path = output / "training_audit.json"
        if audit_path.exists() and json.loads(audit_path.read_text())["signature"] != signature:
            raise ValueError("Existing run has different cache/model/training settings")
        if cfg.resume and latest.exists():
            self.load_checkpoint(path=latest, map_location="cpu", include_keys=self.include_keys)
            if self.signature != signature:
                raise ValueError("Checkpoint provenance differs from the requested run")
        self.signature = signature
        device = torch.device(cfg.training.device)
        self.model.to(device)
        optimizer_to(self.optimizer, device)
        generator = torch.Generator().manual_seed(seed)
        if self.sampler_state is not None:
            generator.set_state(self.sampler_state)
        sampler = WeightedRandomSampler(torch.tensor(train.sample_weights, dtype=torch.double),
                                       len(train), replacement=True, generator=generator)
        train_loader = DataLoader(train, batch_size=cfg.training.batch_size, sampler=sampler, num_workers=0,
                                  generator=torch.Generator().manual_seed(seed + 1))
        val_loader = DataLoader(val, batch_size=cfg.training.batch_size, shuffle=False, num_workers=0,
                                generator=torch.Generator().manual_seed(seed + 2))
        train_eval_loader = DataLoader(train, batch_size=cfg.training.batch_size, shuffle=False, num_workers=0,
                                       generator=torch.Generator().manual_seed(seed + 3))
        tolerance = cfg.weight_tolerance
        global_weight = float(optimal_weight(statistics["global_moments"], tolerance=tolerance)[0])
        baselines = {"rgb": 1.0, "pcd": 0.0, "equal": 0.5, "global": global_weight}
        write_json(output / "baselines.json", {"global_rgb_weight": global_weight,
                   "global_moments": statistics["global_moments"].tolist(), "validation": risk_report(val, baselines)})
        write_json(output / "cache_metadata.json", train.metadata)
        write_json(audit_path, {"signature": signature, "cache_signature": train.metadata["signature"],
                   "train_episode_ids": [e["episode_id"] for e in train.episodes],
                   "val_episode_ids": [e["episode_id"] for e in val.episodes],
                   "feature_dimensions": {"rgb": train.rgb_dim, "pcd": train.pcd_dim},
                   "output_scale": self.model.output_scale.detach().cpu().tolist(), "history_contexts": cfg.history_contexts,
                   "parameter_count": sum(p.numel() for p in self.model.parameters()),
                   "primary_checkpoint_metric": "episode-balanced normalized moment Frobenius squared error",
                   "risk_checkpoint_metric": "episode-balanced validation fusion epsilon MSE",
                   "early_stopping": cfg.training.get("early_stopping", True),
                   "normalization_fit_role": train_role, "risk_mode": cfg.risk_mode,
                   "timesteps": train.timesteps.tolist(), "lambda": train.step_weights.tolist(), **self.extra_audit()})
        OmegaConf.save(cfg, output / "risk_config.yaml", resolve=True)
        if self.epoch >= cfg.training.num_epochs:
            print("Resumed run has already completed its epoch budget.")
            return
        if should_stop(cfg.training, self.epoch, self.bad_epochs):
            print("Resumed run already reached early stopping; no further updates.")
            return
        print(f"{self.training_label()}: {len(train.episodes)} train / {len(val.episodes)} val episodes; "
              f"{len(train)} / {len(val)} contexts; global RGB weight={global_weight:.6f}")
        with JsonLogger(output / "logs.json.txt") as logger:
            while self.epoch < cfg.training.num_epochs:
                epoch = self.epoch
                self.model.train()
                total, count = 0.0, 0
                for batch in train_loader:
                    self.optimizer.zero_grad(set_to_none=True)
                    prediction = self.predict(self.model, batch, device)
                    loss = self.training_loss(prediction, batch["target"].to(device), train).mean()
                    if not torch.isfinite(loss):
                        raise FloatingPointError("Nonfinite training loss")
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.training.gradient_clip_norm, error_if_nonfinite=True)
                    self.optimizer.step()
                    n = len(batch["target"])
                    total += loss.item() * n
                    count += n
                    self.global_step += 1
                train_loss, _, train_weights, _ = self.validate(train_eval_loader, train, statistics, device, tolerance)
                val_loss, pred, weights, fallback = self.validate(val_loader, val, statistics, device, tolerance)
                val_risk = mean_fusion_risk(val, weights)
                improved = val_loss < self.best_val_loss
                risk_improved = val_risk < self.best_val_risk
                if risk_improved:
                    self.best_val_risk = val_risk
                if improved:
                    self.best_val_loss, self.bad_epochs = val_loss, 0
                else:
                    self.bad_epochs += 1
                self.epoch += 1
                self.sampler_state = generator.get_state()
                if improved:
                    self.save_checkpoint(path=output / "checkpoints/best.ckpt", use_thread=False)
                    self.store_validation(output, "best", val, baselines, pred, weights, fallback,
                                    epoch, val_loss, global_weight, "minimum validation moment loss")
                if risk_improved:
                    self.save_checkpoint(path=output / "checkpoints/best_risk.ckpt", use_thread=False)
                    self.store_validation(output, "best_risk", val, baselines, pred, weights, fallback,
                                    epoch, val_loss, global_weight, "minimum validation fusion risk")
                self.save_checkpoint(use_thread=False)
                if self.epoch % cfg.training.snapshot_every == 0:
                    self.save_checkpoint(path=output / f"checkpoints/epoch_{self.epoch:04d}.ckpt", use_thread=False)
                metrics = {"epoch": epoch, "global_step": self.global_step, "train_moment_loss": train_loss,
                           "sampled_train_moment_loss": total / count,
                           "train_fusion_risk": mean_fusion_risk(train, train_weights),
                           "val_fusion_risk": val_risk, "best_val_fusion_risk": self.best_val_risk,
                           "val_moment_loss": val_loss, "best_val_loss": self.best_val_loss,
                           "bad_epochs": self.bad_epochs}
                metrics = self.format_metrics(metrics)
                logger.log(metrics)
                print(json.dumps(metrics), flush=True)
                if should_stop(cfg.training, self.epoch, self.bad_epochs):
                    print("Early stopping: validation moment loss has not improved.")
                    break
        self.store_validation(output, "latest", val, baselines, pred, weights, fallback,
                        self.epoch - 1, val_loss, global_weight, "last completed epoch")
        write_json(output / "run_summary.json", {"completed_epochs": self.epoch,
                   "budget_epochs": cfg.training.num_epochs,
                   "early_stopping": cfg.training.get("early_stopping", True),
                   "best_val_moment_loss": self.best_val_loss, "best_val_fusion_risk": self.best_val_risk,
                   "risk_mode": cfg.risk_mode, "num_steps": len(train.timesteps)})
