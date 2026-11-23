"""Checkpoint save/restore.

A Colab runtime can disconnect at any moment, so a checkpoint has to carry
everything needed to continue as if nothing happened: model, optimizer,
scheduler, AMP scaler, epoch counter, best metric, history and RNG state.

Checkpoints are written atomically (temp file + rename) so a disconnect in the
middle of `torch.save` cannot leave a truncated `last.pt` behind.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

CHECKPOINT_FORMAT_VERSION = 1


@dataclass
class TrainingState:
    """Everything that defines "where training got to"."""

    epoch: int = 0  # number of *completed* epochs
    global_step: int = 0
    best_metric: Optional[float] = None
    best_epoch: Optional[int] = None
    monitor: str = "val/accuracy_at_5px"
    monitor_mode: str = "max"
    history: List[Dict[str, Any]] = field(default_factory=list)

    def is_improvement(self, value: float) -> bool:
        if value is None:
            return False
        if self.best_metric is None:
            return True
        if self.monitor_mode == "max":
            return value > self.best_metric
        return value < self.best_metric

    def to_dict(self) -> Dict[str, Any]:
        return {
            "epoch": self.epoch,
            "global_step": self.global_step,
            "best_metric": self.best_metric,
            "best_epoch": self.best_epoch,
            "monitor": self.monitor,
            "monitor_mode": self.monitor_mode,
            "history": self.history,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TrainingState":
        return cls(
            epoch=int(data.get("epoch", 0)),
            global_step=int(data.get("global_step", 0)),
            best_metric=data.get("best_metric"),
            best_epoch=data.get("best_epoch"),
            monitor=data.get("monitor", "val/accuracy_at_5px"),
            monitor_mode=data.get("monitor_mode", "max"),
            history=list(data.get("history", [])),
        )


def _rng_state() -> Dict[str, Any]:
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng(state: Optional[Dict[str, Any]]) -> None:
    if not state:
        return
    try:
        random.setstate(state["python"])
        np.random.set_state(state["numpy"])
        torch.set_rng_state(_as_byte_tensor(state["torch"]))
        if torch.cuda.is_available() and "cuda" in state:
            torch.cuda.set_rng_state_all([_as_byte_tensor(s) for s in state["cuda"]])
    except Exception:  # noqa: BLE001 - RNG restore is best effort
        pass


def _as_byte_tensor(value: Any) -> torch.Tensor:
    tensor = value if isinstance(value, torch.Tensor) else torch.as_tensor(value)
    return tensor.cpu().to(torch.uint8)


def save_checkpoint(
    path: Path | str,
    model: torch.nn.Module,
    state: TrainingState,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    scaler: Optional[Any] = None,
    config: Optional[Dict[str, Any]] = None,
    extra: Optional[Dict[str, Any]] = None,
    include_rng: bool = True,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    module = model.module if hasattr(model, "module") else model
    payload: Dict[str, Any] = {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "model": {k: v.detach().cpu() for k, v in module.state_dict().items()},
        "model_class": type(module).__name__,
        "state": state.to_dict(),
        "config": config,
        "extra": extra or {},
    }
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    if scheduler is not None:
        payload["scheduler"] = scheduler.state_dict()
    if scaler is not None and getattr(scaler, "is_enabled", lambda: False)():
        payload["scaler"] = scaler.state_dict()
    if include_rng:
        payload["rng"] = _rng_state()

    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)
    return path


def load_checkpoint(path: Path | str, map_location: Any = "cpu") -> Dict[str, Any]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {path}")
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:  # torch < 2.0 has no weights_only kwarg
        return torch.load(path, map_location=map_location)


def restore_training(
    path: Path | str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    scaler: Optional[Any] = None,
    map_location: Any = "cpu",
    strict: bool = True,
    restore_rng: bool = True,
) -> TrainingState:
    """Restore a full training state in place and return it."""
    ckpt = load_checkpoint(path, map_location=map_location)
    module = model.module if hasattr(model, "module") else model
    module.load_state_dict(ckpt["model"], strict=strict)

    if optimizer is not None and ckpt.get("optimizer"):
        optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler is not None and ckpt.get("scheduler"):
        scheduler.load_state_dict(ckpt["scheduler"])
    if scaler is not None and ckpt.get("scaler"):
        scaler.load_state_dict(ckpt["scaler"])
    if restore_rng:
        _restore_rng(ckpt.get("rng"))

    return TrainingState.from_dict(ckpt.get("state", {}))


def load_model_weights(
    path: Path | str,
    model: torch.nn.Module,
    map_location: Any = "cpu",
    strict: bool = True,
) -> Dict[str, Any]:
    """Load only weights (for evaluation, inference and Core ML export).

    Accepts a full RallyVision checkpoint or a bare `state_dict`.
    """
    ckpt = load_checkpoint(path, map_location=map_location)
    if isinstance(ckpt, dict) and "model" in ckpt:
        state_dict = ckpt["model"]
        meta = {k: v for k, v in ckpt.items() if k != "model"}
    else:
        state_dict = ckpt
        meta = {}
    module = model.module if hasattr(model, "module") else model
    module.load_state_dict(state_dict, strict=strict)
    return meta


def find_resume_checkpoint(checkpoint_dir: Path | str, prefer: str = "last") -> Optional[Path]:
    """Return the checkpoint to resume from, or None if there is nothing yet."""
    checkpoint_dir = Path(checkpoint_dir)
    if not checkpoint_dir.is_dir():
        return None
    candidates = [checkpoint_dir / f"{prefer}.pt", checkpoint_dir / "last.pt", checkpoint_dir / "best.pt"]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    epochs = sorted(checkpoint_dir.glob("epoch_*.pt"))
    return epochs[-1] if epochs else None


def prune_epoch_checkpoints(checkpoint_dir: Path | str, keep_last_n: int) -> None:
    if keep_last_n <= 0:
        return
    files = sorted(Path(checkpoint_dir).glob("epoch_*.pt"))
    for stale in files[:-keep_last_n]:
        stale.unlink(missing_ok=True)
