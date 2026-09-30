"""Collect a finished training run into a single, downloadable folder.

The Colab notebook calls this at the end of a session so the Mac can pick up:

    * the best (and last) checkpoint
    * the exact config the run used
    * evaluation metrics (JSON + CSV)
    * training history
    * prediction grids
    * annotated demo video, if one was produced

Core ML conversion is deliberately *not* included: that happens on the Mac.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from .paths import RunPaths, resolve_run
from .utils import dir_size_bytes, format_bytes, write_json


# (source relative to the run root, destination name inside the bundle)
BUNDLE_FILES: Sequence[tuple[str, str]] = (
    ("checkpoints/best.pt", "best.pt"),
    ("checkpoints/last.pt", "last.pt"),
    ("config.yaml", "config.yaml"),
    ("run_info.json", "run_info.json"),
    ("metrics/history.json", "history.json"),
    ("logs/metrics.csv", "metrics.csv"),
    ("logs/train.log", "train.log"),
    ("metrics/val_metrics.json", "val_metrics.json"),
    ("metrics/val_metrics.csv", "val_metrics.csv"),
    ("metrics/val_per_sample.csv", "val_per_sample.csv"),
)


def collect_run_artifacts(
    run: RunPaths,
    dest: Path | str,
    *,
    zip_bundle: bool = False,
    extra: Optional[Sequence[Path]] = None,
) -> Dict[str, Any]:
    """Copy every durable artifact of ``run`` into ``dest``.

    Missing optional files (no video yet, evaluation not run) are skipped,
    never treated as errors. Returns a manifest of what was copied.
    """
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)

    copied: List[Dict[str, Any]] = []
    missing: List[str] = []

    for relative, name in BUNDLE_FILES:
        src = run.root / relative
        if src.is_file():
            target = dest / name
            shutil.copy2(src, target)
            copied.append({"name": name, "source": str(src), "bytes": src.stat().st_size})
        else:
            missing.append(relative)

    pred_dir = dest / "predictions"
    if run.predictions.is_dir():
        for png in sorted(run.predictions.glob("*.png")):
            pred_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(png, pred_dir / png.name)
            copied.append({"name": f"predictions/{png.name}", "source": str(png), "bytes": png.stat().st_size})

    video_dir = dest / "videos"
    if run.videos.is_dir():
        for video in sorted(run.videos.glob("*")):
            if not video.is_file():
                continue
            video_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(video, video_dir / video.name)
            copied.append({"name": f"videos/{video.name}", "source": str(video), "bytes": video.stat().st_size})

    for path in extra or []:
        path = Path(path)
        if path.is_file():
            target = dest / path.name
            shutil.copy2(path, target)
            copied.append({"name": path.name, "source": str(path), "bytes": path.stat().st_size})

    zip_path = None
    if zip_bundle:
        archive = shutil.make_archive(str(dest), "zip", root_dir=dest)
        zip_path = str(Path(archive))

    manifest = {
        "run_dir": str(run.root),
        "bundle_dir": str(dest),
        "zip": zip_path,
        "copied": copied,
        "missing": missing,
        "total_bytes": sum(item["bytes"] for item in copied),
        "total_size": format_bytes(sum(item["bytes"] for item in copied)),
    }
    write_json(dest / "bundle_manifest.json", manifest)
    return manifest


def collect_from_config(cfg, dest: Path | str, run: Optional[str] = None, zip_bundle: bool = False) -> Dict[str, Any]:
    """Resolve a run via config (`latest` if ``run`` is None) and collect it."""
    return collect_run_artifacts(resolve_run(cfg, run), dest, zip_bundle=zip_bundle)


def format_bundle(manifest: Dict[str, Any]) -> str:
    lines = [
        f"run            : {manifest['run_dir']}",
        f"bundle         : {manifest['bundle_dir']}",
        f"size           : {manifest['total_size']}",
    ]
    if manifest.get("zip"):
        lines.append(f"zip            : {manifest['zip']}")
    lines.append("copied:")
    for item in manifest["copied"]:
        lines.append(f"  {item['name']:<28} {format_bytes(item['bytes'])}")
    if manifest["missing"]:
        lines.append("not present (ok if that step hasn't run):")
        for name in manifest["missing"]:
            lines.append(f"  {name}")
    return "\n".join(lines)


def run_disk_usage(run: RunPaths) -> str:
    return format_bytes(dir_size_bytes(run.root))
