"""The training loop.

Designed around the assumption that the process can die at any moment (Colab
disconnects, idle timeouts, preemption).

Two checkpoints are written:

* ``best.pt`` — weights with the best validation score so far (this is the
  model you download and export). Saved *before* ``last.pt`` so a disconnect
  cannot leave you with only the latest epoch.
* ``last.pt`` — full trainer state so ``--resume`` can continue.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from torch.utils.data import DataLoader

from rallyvision.common.checkpoint import (
    TrainingState,
    find_resume_checkpoint,
    prune_epoch_checkpoints,
    restore_training,
    save_checkpoint,
)
from rallyvision.common.device import DeviceInfo, autocast_context, memory_summary
from rallyvision.common.logging_utils import get_logger, log_banner
from rallyvision.ball_tracking.evaluation.metrics import format_metrics
from rallyvision.common.paths import RunPaths
from rallyvision.common.utils import CSVLogger, count_parameters, flatten_metrics, format_duration, write_json
from rallyvision.ball_tracking.evaluation.evaluator import evaluate
from rallyvision.ball_tracking.training.engine.optim import build_optimizer, build_scaler, build_scheduler, current_lr, is_plateau

logger = get_logger(__name__)


class Trainer:
    """Owns the model, the loop, and everything written to the run directory."""

    def __init__(
        self,
        cfg,
        model: torch.nn.Module,
        loss_fn: torch.nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader],
        device_info: DeviceInfo,
        paths: RunPaths,
    ):
        self.cfg = cfg
        self.device_info = device_info
        self.device = device_info.device
        self.model = model.to(self.device)
        self.loss_fn = loss_fn
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.paths = paths.create()

        self.optimizer = build_optimizer(cfg, self.model)
        self.scheduler = build_scheduler(cfg, self.optimizer, steps_per_epoch=len(train_loader))
        self.scaler = build_scaler(device_info)

        train_cfg = cfg.train
        self.epochs = int(train_cfg.epochs)
        self.grad_clip = float(cfg.optim.get("grad_clip_norm", 0.0) or 0.0)
        self.log_every = max(1, int(train_cfg.get("log_every", 20)))
        self.val_every = max(1, int(train_cfg.get("val_every", 1)))
        self.checkpoint_every = max(1, int(train_cfg.get("checkpoint_every", 1)))
        self.keep_last_n = int(train_cfg.get("keep_last_n", 0))
        self.max_train_batches = train_cfg.get("max_train_batches")
        self.max_val_batches = train_cfg.get("max_val_batches")
        self.early_stopping_patience = int(train_cfg.get("early_stopping_patience", 0))

        self.state = TrainingState(
            monitor=str(train_cfg.get("monitor", "val/accuracy_at_5px")),
            monitor_mode=str(train_cfg.get("monitor_mode", "max")),
        )
        self.csv_logger = CSVLogger(self.paths.metrics_csv)
        self._epochs_without_improvement = 0

    # -- resume -----------------------------------------------------------
    def maybe_resume(self, resume: Any = None) -> Optional[Path]:
        """Restore state from a checkpoint.

        ``resume`` may be True/"auto" (find `last.pt` in this run) or a path.
        Returns the checkpoint used, or None when starting fresh.
        """
        if not resume:
            return None
        if resume is True or str(resume) in ("auto", "latest", "last"):
            path = find_resume_checkpoint(self.paths.checkpoints)
            if path is None:
                logger.info("no checkpoint found in %s; starting from scratch", self.paths.checkpoints)
                return None
        else:
            path = Path(resume)
            if not path.is_file():
                raise FileNotFoundError(f"--resume checkpoint not found: {path}")

        self.state = restore_training(
            path,
            self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            scaler=self.scaler,
            map_location=self.device,
        )
        self.state.monitor = str(self.cfg.train.get("monitor", self.state.monitor))
        self.state.monitor_mode = str(self.cfg.train.get("monitor_mode", self.state.monitor_mode))
        logger.info(
            "resumed from %s at epoch %d (best %s = %s)",
            path,
            self.state.epoch,
            self.state.monitor,
            _fmt(self.state.best_metric),
        )
        return path

    # -- loop -------------------------------------------------------------
    def train(self) -> TrainingState:
        self._log_header()
        if self.state.epoch >= self.epochs:
            logger.info(
                "nothing to do: %d epoch(s) already completed and train.epochs=%d",
                self.state.epoch,
                self.epochs,
            )
            return self.state

        start_wall = time.perf_counter()
        for epoch in range(self.state.epoch, self.epochs):
            epoch_start = time.perf_counter()
            train_metrics = self.train_one_epoch(epoch)

            val_metrics: Dict[str, Any] = {}
            should_validate = self.val_loader is not None and (
                (epoch + 1) % self.val_every == 0 or (epoch + 1) == self.epochs
            )
            if should_validate:
                val_metrics, _ = evaluate(
                    self.model,
                    self.val_loader,
                    self.device_info,
                    self.cfg,
                    loss_fn=self.loss_fn,
                    max_batches=self.max_val_batches,
                    keep_samples=False,
                    progress=True,
                    desc=f"val {epoch + 1}/{self.epochs}",
                )

            self.state.epoch = epoch + 1
            lr = current_lr(self.optimizer)
            self._step_scheduler(val_metrics)

            record = {
                "epoch": self.state.epoch,
                "lr": lr,
                "seconds": time.perf_counter() - epoch_start,
                **flatten_metrics(train_metrics, prefix="train/"),
                **flatten_metrics(val_metrics, prefix="val/"),
            }
            record.update({f"mem/{k}": v for k, v in memory_summary(self.device_info).items()})
            self.state.history.append(record)
            self.csv_logger.log(record)
            self._log_epoch(record, val_metrics)

            monitored = record.get(self.state.monitor)
            if should_validate and monitored is None:
                raise RuntimeError(
                    f"validation ran but {self.state.monitor!r} is missing; "
                    "cannot select best.pt. Check eval metrics and train.monitor."
                )
            improved = monitored is not None and self.state.is_improvement(float(monitored))
            if improved:
                self.state.best_metric = float(monitored)
                self.state.best_epoch = self.state.epoch
                self._epochs_without_improvement = 0
            elif monitored is not None:
                self._epochs_without_improvement += 1
            elif self.val_loader is None:
                logger.warning(
                    "no val loader; cannot pick a best model. Only last.pt will be written."
                )

            self._save(epoch_index=self.state.epoch, is_best=improved)
            write_json(self.paths.history_file, {"history": self.state.history})

            if (
                self.early_stopping_patience
                and self._epochs_without_improvement >= self.early_stopping_patience
            ):
                logger.info(
                    "early stopping: %s has not improved for %d epoch(s)",
                    self.state.monitor,
                    self._epochs_without_improvement,
                )
                break

        total = time.perf_counter() - start_wall
        log_banner(logger, "training finished")
        logger.info("elapsed          : %s", format_duration(total))
        logger.info(
            "best %s: %s (epoch %s)",
            self.state.monitor,
            _fmt(self.state.best_metric),
            self.state.best_epoch,
        )
        logger.info("checkpoints      : %s", self.paths.checkpoints)
        if self.val_loader is not None and not self.paths.best_checkpoint.is_file():
            raise RuntimeError(
                f"training finished without {self.paths.best_checkpoint.name}; "
                "best.pt is the val-selected model and must be written whenever validation runs."
            )
        return self.state

    def train_one_epoch(self, epoch: int) -> Dict[str, float]:
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        limit = self.max_train_batches
        total_batches = len(self.train_loader) if limit is None else min(len(self.train_loader), limit)
        iterator = _maybe_tqdm(
            self.train_loader, desc=f"epoch {epoch + 1}/{self.epochs}", total=total_batches
        )

        for i, batch in enumerate(iterator):
            if limit is not None and i >= limit:
                break
            frames = batch["frames"].to(self.device, non_blocking=True)
            target = batch["heatmap"].to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)
            with autocast_context(self.device_info):
                logits = self.model.forward_logits(frames)
                loss = self.loss_fn(logits, target)

            if self.scaler.is_enabled():
                self.scaler.scale(loss).backward()
                if self.grad_clip > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                if self.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.optimizer.step()

            loss_value = float(loss.detach())
            total_loss += loss_value
            num_batches += 1
            self.state.global_step += 1

            if hasattr(iterator, "set_postfix"):
                iterator.set_postfix(loss=f"{loss_value:.4f}", lr=f"{current_lr(self.optimizer):.2e}")
            elif (i + 1) % self.log_every == 0:
                logger.info(
                    "epoch %d | batch %d/%s | loss %.4f | lr %.2e",
                    epoch + 1,
                    i + 1,
                    total_batches,
                    loss_value,
                    current_lr(self.optimizer),
                )

        return {"loss": total_loss / max(1, num_batches), "batches": num_batches}

    # -- helpers ----------------------------------------------------------
    def _step_scheduler(self, val_metrics: Dict[str, Any]) -> None:
        if self.scheduler is None:
            return
        if is_plateau(self.scheduler):
            key = self.state.monitor.split("/", 1)[-1]
            value = val_metrics.get(key)
            if value is not None:
                self.scheduler.step(float(value))
            return
        self.scheduler.step()

    def _save(self, epoch_index: int, is_best: bool) -> None:
        config_dict = self.cfg.to_dict() if hasattr(self.cfg, "to_dict") else dict(self.cfg)
        extra = {
            "model": self.cfg.model.to_dict() if hasattr(self.cfg.model, "to_dict") else None,
            "input_height": int(self.cfg.data.input_height),
            "input_width": int(self.cfg.data.input_width),
            "num_frames": int(self.cfg.data.num_frames),
        }
        payload = dict(
            model=self.model,
            state=self.state,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            scaler=self.scaler,
            config=config_dict,
            extra=extra,
        )
        # Write best.pt first. last.pt is only for resume; if Colab dies mid-save
        # the file we care about (the best val snapshot) should already be on disk.
        if is_best:
            save_checkpoint(self.paths.best_checkpoint, **payload)
            logger.info(
                "saved BEST model %s = %s (epoch %s) -> %s",
                self.state.monitor,
                _fmt(self.state.best_metric),
                self.state.best_epoch,
                self.paths.best_checkpoint,
            )
        save_checkpoint(self.paths.last_checkpoint, **payload)
        if self.keep_last_n > 0 and epoch_index % self.checkpoint_every == 0:
            save_checkpoint(
                self.paths.checkpoints / f"epoch_{epoch_index:04d}.pt",
                self.model,
                self.state,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                scaler=self.scaler,
                config=config_dict,
                extra=extra,
                include_rng=False,
            )
            prune_epoch_checkpoints(self.paths.checkpoints, self.keep_last_n)

    def _log_header(self) -> None:
        params = count_parameters(self.model)
        log_banner(logger, "training")
        for line in self.device_info.summary().splitlines():
            logger.info(line)
        logger.info("model            : %s (%.2fM params)", type(self.model).__name__, params["total"] / 1e6)
        logger.info("input            : %d x 3 x %d x %d",
                    int(self.cfg.data.num_frames), int(self.cfg.data.input_height), int(self.cfg.data.input_width))
        logger.info("train batches    : %d (batch size %d)", len(self.train_loader), self.train_loader.batch_size)
        if self.val_loader is not None:
            logger.info("val batches      : %d", len(self.val_loader))
        logger.info("loss / optimizer : %s / %s", type(self.loss_fn).__name__, type(self.optimizer).__name__)
        logger.info("scheduler        : %s", type(self.scheduler).__name__ if self.scheduler else "none")
        logger.info("epochs           : %d (starting at %d)", self.epochs, self.state.epoch)
        logger.info("monitor          : %s (%s)", self.state.monitor, self.state.monitor_mode)
        logger.info("run directory    : %s", self.paths.root)

    def _log_epoch(self, record: Dict[str, Any], val_metrics: Dict[str, Any]) -> None:
        msg = (
            f"epoch {record['epoch']}/{self.epochs} | "
            f"train loss {record.get('train/loss', float('nan')):.4f} | "
            f"lr {record['lr']:.2e} | {format_duration(record['seconds'])}"
        )
        if "val/loss" in record:
            msg += f" | val loss {record['val/loss']:.4f}"
        logger.info(msg)
        if val_metrics:
            for line in format_metrics(val_metrics).splitlines():
                logger.info("  %s", line)


def _fmt(value: Optional[float]) -> str:
    return "n/a" if value is None else f"{value:.5f}"


def _maybe_tqdm(iterable, desc: str, total: Optional[int]):
    try:
        from tqdm.auto import tqdm
    except ImportError:
        return iterable
    return tqdm(iterable, desc=desc, total=total, leave=False)
