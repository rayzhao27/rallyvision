"""Centralized device selection.

Every script goes through :func:`resolve_device` so the CUDA/MPS/CPU decision
lives in exactly one place:

    CUDA  -- the only supported target for real training (Colab).
    MPS   -- local Apple Silicon, for debugging and tiny smoke tests only.
    CPU   -- fallback, and the target for Core ML comparison runs.
"""

from __future__ import annotations

import os
import platform
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import torch

VALID_DEVICES = ("auto", "cuda", "mps", "cpu")


class DeviceError(RuntimeError):
    """Raised when the requested device is unavailable or disallowed."""


@dataclass
class DeviceInfo:
    """Resolved device plus everything worth printing in a training header."""

    device: torch.device
    kind: str  # "cuda" | "mps" | "cpu"
    name: str
    amp_enabled: bool
    amp_dtype: Optional[torch.dtype]
    details: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_cuda(self) -> bool:
        return self.kind == "cuda"

    @property
    def is_mps(self) -> bool:
        return self.kind == "mps"

    def summary(self) -> str:
        lines = [
            f"device      : {self.device} ({self.name})",
            f"torch       : {torch.__version__}",
        ]
        if self.is_cuda:
            lines.append(f"cuda        : {torch.version.cuda}")
            total = self.details.get("total_memory_gb")
            if total:
                lines.append(f"gpu memory  : {total:.1f} GB")
            cc = self.details.get("capability")
            if cc:
                lines.append(f"capability  : {cc}")
        lines.append(
            "amp         : "
            + (f"enabled ({self.amp_dtype})" if self.amp_enabled else "disabled")
        )
        return "\n".join(lines)


def cuda_available() -> bool:
    return torch.cuda.is_available()


def mps_available() -> bool:
    backend = getattr(torch.backends, "mps", None)
    return bool(backend is not None and backend.is_available() and backend.is_built())


def best_device() -> str:
    """"cuda", else "mps" (Apple Silicon), else "cpu"."""
    return _pick_kind("auto")


def _pick_kind(preference: str) -> str:
    preference = (preference or "auto").lower()
    if preference not in VALID_DEVICES:
        raise DeviceError(f"device must be one of {VALID_DEVICES}, got {preference!r}")

    if preference == "auto":
        if cuda_available():
            return "cuda"
        if mps_available():
            return "mps"
        return "cpu"

    if preference == "cuda" and not cuda_available():
        raise DeviceError(
            "device='cuda' was requested but torch.cuda.is_available() is False.\n"
            "On Colab: Runtime > Change runtime type > Hardware accelerator > GPU."
        )
    if preference == "mps" and not mps_available():
        raise DeviceError(
            "device='mps' was requested but the MPS backend is unavailable. "
            "MPS needs macOS 12.3+ and an arm64 PyTorch build."
        )
    return preference


def resolve_device(
    preference: str = "auto",
    *,
    amp: bool = False,
    cudnn_benchmark: bool = True,
    deterministic: bool = False,
) -> DeviceInfo:
    """Resolve `preference` into a concrete device and configure backends.

    AMP is enabled only on CUDA. MPS autocast is not reliable across PyTorch
    versions and CPU autocast buys nothing here, so both are forced off.
    """
    kind = _pick_kind(preference)
    device = torch.device(kind if kind != "cuda" else "cuda:0")
    details: Dict[str, Any] = {}

    if kind == "cuda":
        idx = device.index or 0
        props = torch.cuda.get_device_properties(idx)
        name = props.name
        details["total_memory_gb"] = props.total_memory / (1024**3)
        details["capability"] = f"{props.major}.{props.minor}"
        details["device_count"] = torch.cuda.device_count()
        torch.backends.cudnn.benchmark = bool(cudnn_benchmark) and not deterministic
        torch.backends.cudnn.deterministic = bool(deterministic)
    elif kind == "mps":
        name = f"Apple {platform.machine()} (MPS)"
        # Let unsupported ops silently fall back to CPU rather than crash a
        # debugging session.
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    else:
        name = platform.processor() or platform.machine() or "cpu"
        details["threads"] = torch.get_num_threads()

    amp_enabled = bool(amp) and kind == "cuda"
    amp_dtype = None
    if amp_enabled:
        major = torch.cuda.get_device_properties(device.index or 0).major
        amp_dtype = torch.bfloat16 if major >= 8 else torch.float16
        details["amp_dtype"] = str(amp_dtype)

    return DeviceInfo(
        device=device,
        kind=kind,
        name=name,
        amp_enabled=amp_enabled,
        amp_dtype=amp_dtype,
        details=details,
    )


def resolve_device_from_config(cfg, override: Optional[str] = None) -> DeviceInfo:
    """Build a :class:`DeviceInfo` from a loaded :class:`~rallyvision.config.Config`."""
    runtime = cfg.get("runtime", {})
    return resolve_device(
        override or runtime.get("device", "auto"),
        amp=bool(runtime.get("amp", False)),
        cudnn_benchmark=bool(runtime.get("cudnn_benchmark", True)),
        deterministic=bool(runtime.get("deterministic", False)),
    )


def guard_training_device(info: DeviceInfo, cfg, allow_local: bool = False) -> None:
    """Refuse to start a real training run on a non-CUDA device.

    This is a deliberate guardrail: full training belongs on Colab. Smoke tests
    bypass it by passing ``allow_local=True``.
    """
    if allow_local or info.is_cuda:
        return
    if not bool(cfg.get("runtime", {}).get("require_cuda_for_training", True)):
        return
    raise DeviceError(
        f"Refusing to start full training on '{info.kind}'.\n"
        "Full training is a Colab/CUDA workload. Options:\n"
        "  * run this in Colab with a GPU runtime, or\n"
        "  * pass --smoke for a tiny pipeline check, or\n"
        "  * pass --allow-local-training if you really mean it, or\n"
        "  * set runtime.require_cuda_for_training: false in config.yaml."
    )


def autocast_context(info: DeviceInfo):
    """Return an autocast context manager (a no-op unless AMP is active)."""
    if not info.amp_enabled:
        return _NullContext()
    return torch.autocast(device_type="cuda", dtype=info.amp_dtype)


class _NullContext:
    def __enter__(self):
        return None

    def __exit__(self, *exc):
        return False


def empty_cache(info: DeviceInfo) -> None:
    if info.is_cuda:
        torch.cuda.empty_cache()
    elif info.is_mps and hasattr(torch, "mps"):
        torch.mps.empty_cache()


def memory_summary(info: DeviceInfo) -> Dict[str, float]:
    """Peak memory in GB, when the backend can report it."""
    if info.is_cuda:
        idx = info.device.index or 0
        return {
            "allocated_gb": torch.cuda.memory_allocated(idx) / (1024**3),
            "max_allocated_gb": torch.cuda.max_memory_allocated(idx) / (1024**3),
            "reserved_gb": torch.cuda.memory_reserved(idx) / (1024**3),
        }
    return {}
