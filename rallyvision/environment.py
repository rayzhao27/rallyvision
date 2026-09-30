"""Runtime environment reporting and Colab/Drive path helpers.

The notebook (and `scripts/check_environment.py`) call this so every training
session starts with a single, unambiguous picture of:

    * which device will be used
    * whether CUDA is actually present
    * where persistent storage lives (Google Drive vs local `artifacts/`)

Real training is a Colab/CUDA job. This module never starts it; it only
describes the machine and, when asked, refuses to continue without a GPU.
"""

from __future__ import annotations

import os
import platform
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import torch

from .config import Config, REPO_ROOT, resolve_path
from .device import DeviceError, DeviceInfo, cuda_available, resolve_device_from_config

COLAB_DRIVE_MOUNT = Path("/content/drive")
COLAB_MYDRIVE = COLAB_DRIVE_MOUNT / "MyDrive"
DEFAULT_COLAB_PROJECT_DIR = "rallyvision"


class EnvironmentError(RuntimeError):
    """Raised when the runtime cannot satisfy a hard requirement (usually CUDA)."""


@dataclass
class EnvironmentReport:
    """Everything the notebook prints in the Environment Setup cell."""

    platform: str
    machine: str
    python: str
    torch_version: str
    cuda_compiled: Optional[str]
    cuda_available: bool
    in_colab: bool
    drive_mounted: bool
    device: DeviceInfo
    details: Dict[str, Any] = field(default_factory=dict)

    def format(self) -> str:
        lines = [
            f"platform     : {self.platform} ({self.machine})",
            f"python       : {self.python}",
            f"torch        : {self.torch_version}",
            f"cuda build   : {self.cuda_compiled or 'cpu-only build'}",
            f"cuda avail   : {'yes' if self.cuda_available else 'no'}",
            f"colab        : {'yes' if self.in_colab else 'no'}",
            f"drive mounted: {'yes' if self.drive_mounted else 'no'}",
        ]
        lines.extend(self.device.summary().splitlines())
        gpu = self.details.get("gpus")
        if gpu:
            lines.append(f"gpu(s)       : {gpu}")
        free = self.details.get("disk_free")
        if free:
            lines.append(f"disk free    : {free}")
        return "\n".join(lines)


def in_colab() -> bool:
    """True inside a Google Colab runtime (including Colab Pro)."""
    if os.environ.get("COLAB_RELEASE_TAG") or os.environ.get("COLAB_GPU"):
        return True
    try:
        import google.colab  # noqa: F401
    except ImportError:
        return False
    return True


def drive_is_mounted() -> bool:
    return COLAB_MYDRIVE.is_dir()


def default_drive_root(project_dir: str = DEFAULT_COLAB_PROJECT_DIR) -> Path:
    """`/content/drive/MyDrive/rallyvision` when Drive is mounted."""
    return COLAB_MYDRIVE / project_dir


def report_environment(
    cfg: Optional[Config] = None,
    *,
    device_override: Optional[str] = None,
    require_cuda: bool = False,
) -> EnvironmentReport:
    """Inspect the runtime and optionally refuse to continue without CUDA.

    ``require_cuda=True`` is what the Colab notebook uses before training.
    Local unit tests and smoke checks leave it False.
    """
    if cfg is None:
        from .config import load_config

        cfg = load_config()

    try:
        device = resolve_device_from_config(cfg, override=device_override)
    except DeviceError as exc:
        raise EnvironmentError(str(exc)) from exc

    gpus = []
    if cuda_available():
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            gpus.append(f"{props.name} ({props.total_memory / (1024**3):.1f} GB)")

    disk = shutil.disk_usage(os.getcwd())
    from .utils import format_bytes

    report = EnvironmentReport(
        platform=f"{platform.system()} {platform.release()}",
        machine=platform.machine(),
        python=sys.version.split()[0],
        torch_version=torch.__version__,
        cuda_compiled=torch.version.cuda,
        cuda_available=cuda_available(),
        in_colab=in_colab(),
        drive_mounted=drive_is_mounted(),
        device=device,
        details={
            "gpus": ", ".join(gpus) if gpus else None,
            "disk_free": format_bytes(disk.free),
            "repo_root": str(REPO_ROOT),
        },
    )

    if require_cuda and not device.is_cuda:
        raise EnvironmentError(
            "CUDA is required for RallyVision training but is not available.\n"
            f"Resolved device: {device.device} ({device.name})\n"
            f"torch={torch.__version__}, cuda compiled={torch.version.cuda!r}, "
            f"cuda.is_available()={cuda_available()}.\n"
            "On Colab: Runtime > Change runtime type > Hardware accelerator > GPU, "
            "then restart the runtime."
        )
    return report


def apply_persistent_storage(cfg: Config, root: Path | str) -> Config:
    """Point data + artifacts at a persistent directory (typically Google Drive).

    Layout under ``root``:

        data/raw          raw downloads
        data/prepared     manifest + optional cached frames
        artifacts         runs, checkpoints, logs, metrics, videos

    Absolute paths, so they survive a working-directory change.
    """
    root = Path(root).expanduser().resolve()
    colab_cfg = cfg.get("colab", {})
    raw_sub = colab_cfg.get("raw_subdir", "data/raw") if colab_cfg is not None else "data/raw"
    prep_sub = colab_cfg.get("prepared_subdir", "data/prepared") if colab_cfg is not None else "data/prepared"
    out_sub = colab_cfg.get("output_subdir", "artifacts") if colab_cfg is not None else "artifacts"

    cfg["data.raw_dir"] = str(root / raw_sub)
    cfg["data.prepared_dir"] = str(root / prep_sub)
    cfg["data.local_path"] = str(root / raw_sub)
    cfg["project.output_root"] = str(root / out_sub)
    for path in (cfg["data.raw_dir"], cfg["data.prepared_dir"], cfg["project.output_root"]):
        Path(path).mkdir(parents=True, exist_ok=True)
    return cfg


def resolve_storage_root(
    cfg: Config,
    *,
    drive_root: Optional[Path | str] = None,
    prefer_drive: bool = True,
) -> Path:
    """Pick the persistent root: explicit Drive path, mounted Drive, or local artifacts."""
    if drive_root:
        return Path(drive_root).expanduser().resolve()
    colab_cfg = cfg.get("colab", {}) or {}
    configured = colab_cfg.get("drive_root")
    if prefer_drive and drive_is_mounted():
        if configured:
            return Path(configured)
        return default_drive_root()
    return resolve_path(cfg, cfg.project.output_root).parent
