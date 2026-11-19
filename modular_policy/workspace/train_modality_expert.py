"""Single-modality denoising training with explicit, isolated episode splits."""

import json
import math
from pathlib import Path

import hydra
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb

from modular_policy.common.episode_split import canonical_hash
from modular_policy.common.json_logger import JsonLogger
from modular_policy.common.lr_scheduler import get_scheduler
from modular_policy.common.pytorch_util import dict_apply, optimizer_to
from modular_policy.workspace.train_policy import TrainPolicyWorkspace


class TrainModalityExpertWorkspace(TrainPolicyWorkspace):
    include_keys = tuple(TrainPolicyWorkspace.include_keys) + ("best_val_loss", "training_signature")

    def __init__(self, cfg, output_dir=None):
        super().__init__(cfg, output_dir=output_dir)
        self.best_val_loss = float("inf")
        self.training_signature = None

    def run(self):
        # Evaluation constructs this workspace only to load its model weights.
        # Importing the dataset at module scope also loads Numba's LLVM, which
        # can collide with the different LLVM used by Mesa/CoppeliaSim rendering.
        from modular_policy.dataset.episode_split_rlbench_dataset import EpisodeSplitRlbenchDataset

        cfg = self.cfg
        output = Path(self.output_dir)
        output.mkdir(parents=True, exist_ok=True)
        dataset = hydra.utils.instantiate(cfg.task.dataset)
        if not isinstance(dataset, EpisodeSplitRlbenchDataset):
            raise TypeError("Expert training requires EpisodeSplitRlbenchDataset")
        if dataset.split != "expert_train" or dataset.validation_split != "expert_val":
            raise ValueError("Experts must train on expert_train and validate on expert_val")
        expected_modality = {"rgb": "rgb", "pcd": "pointcloud"}[cfg.expert_modality]
        modalities = set(self.model.get_observation_modalities()) - {"robot_state"}
        if modalities != {expected_modality}:
            raise ValueError(f"Expected a single {expected_modality} expert; got {modalities}")
        if set(self.model.get_observation_ports()) != set(dataset.obs_keys):
            raise ValueError("Dataset observation keys must exactly match the expert's inputs")

        # Keep a portable snapshot beside each run and inside the checkpoint config.
        manifest_hash = canonical_hash(dataset.manifest)
        training_config = OmegaConf.to_container(cfg.training, resolve=True)
        training_config.pop("resume")
        signature = canonical_hash({
            "manifest_sha256": manifest_hash,
            "policy": OmegaConf.to_container(cfg.policy, resolve=True),
            "optimizer": OmegaConf.to_container(cfg.optimizer, resolve=True),
            "training": training_config,
            "dataloader": OmegaConf.to_container(cfg.dataloader, resolve=True),
            "val_dataloader": OmegaConf.to_container(cfg.val_dataloader, resolve=True),
            "ema": OmegaConf.to_container(cfg.ema, resolve=True),
            "dataset_options": {k: v for k, v in OmegaConf.to_container(cfg.task.dataset, resolve=True).items()
                                if k not in ("zarr_path", "split_manifest")},
            "normalization": "expert_train_limits_rgb_fixed_range_v1",
        })
        latest = self.get_checkpoint_path()
        if latest.exists() and not cfg.training.resume:
            raise FileExistsError(f"Existing run: {latest}; set training.resume=true or choose a new output")
        audit_path = output / "training_audit.json"
        if audit_path.exists():
            old_audit = json.loads(audit_path.read_text())
            if old_audit["training_signature"] != signature:
                raise ValueError("Training settings/split changed; use a new output directory")

        snapshot = output.resolve() / "split_manifest.json"
        snapshot.write_text(json.dumps(dataset.manifest, indent=2) + "\n")
        cfg.data_split_manifest = str(snapshot)
        cfg.task.dataset.split_manifest = str(snapshot)
        cfg.task.dataset.zarr_path = str(Path(dataset.zarr_path).resolve())
        OmegaConf.update(cfg, "split_manifest_sha256", manifest_hash, force_add=True)

        normalizer = dataset.get_normalizer()
        self.model.set_normalizer(normalizer)
        if self.ema_model is not None:
            self.ema_model.set_normalizer(normalizer)
        validation = dataset.get_validation_dataset()
        train_loader = DataLoader(
            dataset, **cfg.dataloader,
            generator=torch.Generator().manual_seed(cfg.training.seed),
        )
        val_loader = DataLoader(
            validation, **cfg.val_dataloader,
            generator=torch.Generator().manual_seed(cfg.training.validation_seed),
        )
        train_batches = len(train_loader)
        if cfg.training.max_train_steps is not None:
            train_batches = min(train_batches, cfg.training.max_train_steps)
        if train_batches <= 0:
            raise ValueError("Training loader is empty")
        accumulate = cfg.training.gradient_accumulate_every
        if accumulate < 1 or cfg.training.val_every < 1 or cfg.training.checkpoint_every < 1:
            raise ValueError("Training/validation intervals must be positive")
        self.lr_scheduler = get_scheduler(
            cfg.training.lr_scheduler, optimizer=self.optimizer,
            num_warmup_steps=cfg.training.lr_warmup_steps,
            num_training_steps=math.ceil(train_batches / accumulate) * cfg.training.num_epochs,
        )
        if cfg.training.resume and latest.exists():
            self.load_checkpoint(path=latest, map_location="cpu")
            if self.training_signature != signature:
                raise ValueError("Checkpoint training signature differs; start a new run")
        self.training_signature = signature
        device = torch.device(cfg.training.device)
        self.model.to(device)
        if self.ema_model is not None:
            self.ema_model.to(device)
        optimizer_to(self.optimizer, device)
        ema = hydra.utils.instantiate(cfg.ema, model=self.ema_model) if self.ema_model is not None else None
        if ema is not None:
            ema.optimization_step = self.global_step

        action_params = normalizer["action"].params_dict
        audit = {
            "expert_modality": cfg.expert_modality,
            "training_signature": signature,
            "signature_config": OmegaConf.to_container(cfg.training, resolve=True),
            "manifest_sha256": manifest_hash,
            "normalizer_fit_split": "expert_train",
            "train_episode_ids": dataset.episode_ids.tolist(),
            "validation_episode_ids": validation.episode_ids.tolist(),
            "reserved_episode_ids": {name: dataset.manifest["splits"][name]
                                     for name in ("risk_train", "risk_calibration", "test")},
            "train_sequences": len(dataset), "validation_sequences": len(validation),
            "action_scale": action_params["scale"].tolist(),
            "action_offset": action_params["offset"].tolist(),
            "checkpoint_metric": "EMA denoising MSE" if cfg.training.use_ema else "denoising MSE",
        }
        audit_path.write_text(json.dumps(audit, indent=2) + "\n")
        OmegaConf.save(cfg, output / "expert_config.yaml", resolve=True)
        run = wandb.init(dir=str(output), config=OmegaConf.to_container(cfg, resolve=True), **cfg.logging)
        print(f"{cfg.expert_modality}: {len(dataset.episode_ids)} train / "
              f"{len(validation.episode_ids)} val episodes; reserved splits are excluded")
        try:
            with JsonLogger(output / "logs.json.txt") as logger:
                while self.epoch < cfg.training.num_epochs:
                    epoch = self.epoch
                    self.model.train()
                    self.optimizer.zero_grad(set_to_none=True)
                    total, examples = 0.0, 0
                    for i, batch in enumerate(tqdm(train_loader, total=train_batches, desc=f"Epoch {epoch}")):
                        batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                        loss = self.model.compute_loss(batch)
                        if not torch.isfinite(loss):
                            raise FloatingPointError("Nonfinite denoising training loss")
                        group_start = (i // accumulate) * accumulate
                        group_size = min(accumulate, train_batches - group_start)
                        (loss / group_size).backward()
                        batch_size = len(batch["action"])
                        total += loss.item() * batch_size
                        examples += batch_size
                        if (i + 1) % accumulate == 0 or i + 1 == train_batches:
                            self.optimizer.step()
                            self.optimizer.zero_grad(set_to_none=True)
                            self.lr_scheduler.step()
                            self.global_step += 1
                            if ema is not None:
                                ema.step(self.model)
                        if i + 1 == train_batches:
                            break
                    metrics = {"epoch": epoch, "global_step": self.global_step,
                               "train_loss": total / examples,
                               "lr": self.lr_scheduler.get_last_lr()[0]}
                    improved = False
                    if epoch % cfg.training.val_every == 0 or epoch + 1 == cfg.training.num_epochs:
                        policy = self.ema_model if self.ema_model is not None else self.model
                        metrics["val_loss"] = self._validate(policy, val_loader, device, cfg)
                        improved = metrics["val_loss"] < self.best_val_loss
                        if improved:
                            self.best_val_loss = metrics["val_loss"]
                    self.epoch += 1  # Checkpoints resume at the next epoch.
                    metrics["best_val_loss"] = self.best_val_loss
                    if improved:
                        self.save_checkpoint(path=output / "checkpoints/best.ckpt", use_thread=False)
                    if self.epoch % cfg.training.checkpoint_every == 0 or self.epoch == cfg.training.num_epochs:
                        self.save_checkpoint(use_thread=False)
                    run.log(metrics, step=self.global_step)
                    logger.log(metrics)
                    print(json.dumps(metrics))
        finally:
            run.finish()

    @staticmethod
    @torch.no_grad()
    def _validate(policy, loader, device, cfg):
        policy.eval()
        total, examples = 0.0, 0
        cuda_devices = list(range(torch.cuda.device_count())) if device.type == "cuda" else []
        # Fixed draws make checkpoint comparisons less noisy; restore training RNG afterwards.
        with torch.random.fork_rng(devices=cuda_devices):
            torch.manual_seed(cfg.training.validation_seed)
            for i, batch in enumerate(loader):
                batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                loss = policy.compute_loss(batch)
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite validation loss")
                size = len(batch["action"])
                total += loss.item() * size
                examples += size
                if cfg.training.max_val_steps is not None and i + 1 >= cfg.training.max_val_steps:
                    break
        if examples == 0:
            raise ValueError("Validation loader is empty")
        return total / examples
