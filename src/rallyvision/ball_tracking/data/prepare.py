"""Turn a raw tennis clip dataset into a prepared, manifest-backed dataset."""

from __future__ import annotations

import datetime as _dt
import fnmatch
import hashlib
import random
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import cv2

from rallyvision.common.config import Config, resolve_path
from rallyvision.common.logging_utils import get_logger
from rallyvision.ball_tracking.data.labels import parse_label_csv
from rallyvision.ball_tracking.data.manifest import ClipRecord, DatasetManifest

logger = get_logger(__name__)


class PrepareError(RuntimeError):
    """Raised when the raw dataset cannot be interpreted."""


def preview_dir(path: Path, limit: int = 20) -> str:
    """Short listing used in error messages so an empty/wrong folder is obvious."""
    path = Path(path)
    if not path.exists():
        return f"{path} does not exist"
    if not path.is_dir():
        return f"{path} is a file, not a directory"
    entries = sorted(path.iterdir(), key=lambda p: p.name)
    if not entries:
        return f"{path} is empty"
    lines = [f"{path} contains {len(entries)} item(s):"]
    for item in entries[:limit]:
        kind = "dir" if item.is_dir() else "file"
        lines.append(f"  {item.name}  ({kind})")
    if len(entries) > limit:
        lines.append(f"  ... {len(entries) - limit} more")
    n_labels = sum(1 for _ in path.rglob("Label.csv"))
    lines.append(f"Label.csv files found recursively: {n_labels}")
    return "\n".join(lines)


def _natural_key(name: str):
    """Sort `9.jpg` before `10.jpg`."""
    parts: List[object] = []
    digits = ""
    for ch in name:
        if ch.isdigit():
            digits += ch
        else:
            if digits:
                parts.append(int(digits))
                digits = ""
            parts.append(ch)
    if digits:
        parts.append(int(digits))
    return parts


def discover_clip_dirs(raw_root: Path, clip_glob: str, label_filename: str) -> List[Path]:
    """Find clip directories, falling back to a recursive search for label files."""
    raw_root = Path(raw_root)
    if not raw_root.is_dir():
        raise PrepareError(
            f"raw dataset directory does not exist: {raw_root}\n"
            "Run scripts/download_dataset.py, or point data.local_path at your copy."
        )

    matches = sorted(p for p in raw_root.glob(clip_glob) if p.is_dir())
    matches = [p for p in matches if (p / label_filename).is_file()]
    if matches:
        return matches

    # The archive may nest the dataset one level deeper, or use a different
    # naming scheme. Fall back to "any directory containing a label file".
    logger.warning(
        "no clips matched glob %r under %s; falling back to a recursive search for %s",
        clip_glob,
        raw_root,
        label_filename,
    )
    found = sorted({p.parent for p in raw_root.rglob(label_filename)})
    if not found:
        raise PrepareError(
            f"no clips found under {raw_root}.\n"
            f"Expected directories matching {clip_glob!r} that each contain {label_filename!r}.\n"
            "DATA_SOURCE=local means the TrackNet frames are already extracted there.\n"
            "Put game*/Clip*/Label.csv (and the jpg frames) on Drive, or switch DATA_SOURCE "
            "to gdrive/url/kaggle and re-run download.\n\n"
            f"{preview_dir(raw_root)}"
        )
    return found


def _list_frames(clip_dir: Path, extensions: Sequence[str]) -> List[str]:
    exts = {e.lower() for e in extensions}
    names = [p.name for p in clip_dir.iterdir() if p.is_file() and p.suffix.lower() in exts]
    return sorted(names, key=_natural_key)


def game_from_clip_path(rel: str) -> str:
    """Game id is the clip's parent folder, not the dataset root.

    TrackNet layout is ``gameN/ClipM``. gdown often nests that under one extra
    folder (``Dataset/game1/Clip1``). Using parts[0] then tags every clip as
    the same game, the by-game split puts 100% in train, and val is empty.
    """
    parts = [p for p in Path(rel).parts if p not in ("", ".")]
    if len(parts) >= 2:
        return parts[-2]
    return "game"


def _clip_identity(clip_dir: Path, raw_root: Path) -> tuple[str, str, str]:
    rel = clip_dir.relative_to(raw_root)
    clip_id = str(rel).replace("\\", "/")
    parts = rel.parts
    name = parts[-1] if parts else clip_dir.name
    return clip_id, game_from_clip_path(clip_id), name


def build_clip_record(
    clip_dir: Path,
    raw_root: Path,
    label_filename: str,
    frame_extensions: Sequence[str],
) -> ClipRecord:
    """Read one clip's frames + labels into a :class:`ClipRecord`."""
    clip_id, game, name = _clip_identity(clip_dir, raw_root)
    frame_files = _list_frames(clip_dir, frame_extensions)
    if not frame_files:
        raise PrepareError(f"{clip_dir}: contains no frames with extensions {frame_extensions}")

    records = parse_label_csv(clip_dir / label_filename)
    by_file = {str(r["file"]).strip(): r for r in records}

    # Labels are matched by filename; when they disagree (some mirrors label
    # `0000.jpg` but store `0000.png`) fall back to positional matching.
    matched = sum(1 for f in frame_files if f in by_file)
    positional = matched < 0.5 * len(frame_files)
    if positional:
        if len(records) < len(frame_files):
            logger.warning(
                "%s: %d labels for %d frames; truncating to the labelled prefix",
                clip_id,
                len(records),
                len(frame_files),
            )
            frame_files = frame_files[: len(records)]
        logger.warning("%s: label filenames do not match frames; matching by position", clip_id)

    first = cv2.imread(str(clip_dir / frame_files[0]))
    if first is None:
        raise PrepareError(f"{clip_dir / frame_files[0]}: could not be read by OpenCV")
    height, width = first.shape[:2]

    files: List[str] = []
    visible: List[int] = []
    xs: List[Optional[float]] = []
    ys: List[Optional[float]] = []
    for i, fname in enumerate(frame_files):
        rec = records[i] if positional else by_file.get(fname)
        if rec is None:
            files.append(fname)
            visible.append(0)
            xs.append(None)
            ys.append(None)
            continue
        is_visible = bool(rec["visible"])
        x, y = rec["x"], rec["y"]
        if is_visible and (x is None or y is None or not (0 <= x < width and 0 <= y < height)):
            is_visible = False
            x = y = None
        files.append(fname)
        visible.append(1 if is_visible else 0)
        xs.append(float(x) if is_visible else None)
        ys.append(float(y) if is_visible else None)

    return ClipRecord(
        id=clip_id,
        game=game,
        name=name,
        source_dir=clip_id,
        files=files,
        visible=visible,
        x=xs,
        y=ys,
        width=width,
        height=height,
    )


def assign_splits(
    clips: List[ClipRecord],
    strategy: str = "by_game",
    val_fraction: float = 0.2,
    val_games: Optional[Sequence[str]] = None,
    seed: int = 1337,
) -> List[ClipRecord]:
    """Assign each clip to train or val, in place.

    ``by_game`` is the default and the only split that avoids leakage between
    train and val: frames from one rally are highly correlated, so splitting at
    frame or even clip level would inflate validation accuracy.
    """
    if not clips:
        return clips

    if val_games:
        wanted = {str(g) for g in val_games}
        for clip in clips:
            match = clip.game in wanted or any(fnmatch.fnmatch(clip.id, pat) for pat in wanted)
            clip.split = "val" if match else "train"
        if not any(c.split == "val" for c in clips):
            raise PrepareError(
                f"data.split.val_games={sorted(wanted)} matched no clips. "
                f"Available games: {sorted({c.game for c in clips})}"
            )
        return clips

    if strategy == "random":
        rng = random.Random(seed)
        shuffled = list(clips)
        rng.shuffle(shuffled)
        n_val = max(1, int(round(val_fraction * len(shuffled)))) if val_fraction > 0 else 0
        val_ids = {c.id for c in shuffled[:n_val]}
        for clip in clips:
            clip.split = "val" if clip.id in val_ids else "train"
        return clips

    key = (lambda c: c.game) if strategy == "by_game" else (lambda c: c.id)
    groups: Dict[str, List[ClipRecord]] = {}
    for clip in clips:
        groups.setdefault(key(clip), []).append(clip)

    # One game cannot make a by-game hold-out. Split clips instead so val exists.
    if strategy == "by_game" and len(groups) < 2 and len(clips) > 1 and val_fraction > 0:
        only = next(iter(groups), "unknown")
        logger.warning("only one game (%s); splitting by clip so the val split is not empty", only)
        return assign_splits(clips, strategy="by_clip", val_fraction=val_fraction, seed=seed)

    # Deterministic ordering that does not depend on directory iteration order.
    ordered = sorted(groups, key=lambda g: hashlib.md5(f"{seed}:{g}".encode()).hexdigest())
    total_frames = sum(len(c) for c in clips)
    target = val_fraction * total_frames

    val_groups: set = set()
    accumulated = 0
    for group in ordered:
        # Keep at least one group in train when there are 2+ groups.
        if accumulated >= target or (len(ordered) > 1 and len(val_groups) >= len(ordered) - 1):
            break
        val_groups.add(group)
        accumulated += sum(len(c) for c in groups[group])

    if val_fraction > 0 and not val_groups and len(ordered) > 1:
        val_groups.add(ordered[0])

    for clip in clips:
        clip.split = "val" if key(clip) in val_groups else "train"

    n_val = sum(1 for c in clips if c.split == "val")
    n_train = sum(1 for c in clips if c.split == "train")
    if val_fraction > 0 and n_val == 0:
        raise PrepareError(
            "validation split is empty. The by-game split needs more than one game "
            "(or more than one clip). Re-run prepare after pulling the split fix, "
            "or pass data.split.strategy=by_clip."
        )
    if n_train == 0:
        raise PrepareError(
            "training split is empty (everything was assigned to val). "
            "Lower data.split.val_fraction or add more clips."
        )
    return clips


def reassign_manifest_splits(manifest: DatasetManifest, cfg) -> DatasetManifest:
    """Rewrite train/val on an existing manifest without re-caching frames.

    Use this when prepare accidentally put every clip in train (nested gdown
    folder, or a single inferred game).
    """
    if not manifest.clips:
        raise PrepareError("manifest has no clips")
    for clip in manifest.clips:
        clip.game = game_from_clip_path(clip.source_dir or clip.id)
    split_cfg = cfg.data.split
    assign_splits(
        manifest.clips,
        strategy=split_cfg.get("strategy", "by_game"),
        val_fraction=float(split_cfg.get("val_fraction", 0.2)),
        val_games=split_cfg.get("val_games"),
        seed=int(cfg.project.seed),
    )
    manifest.save(Path(manifest.prepared_root) / "manifest.json")
    logger.info(
        "reassigned splits: %s",
        {split: sum(1 for c in manifest.clips if c.split == split) for split in ("train", "val")},
    )
    return manifest


def cache_clip_frames(
    clip: ClipRecord,
    raw_root: Path,
    prepared_root: Path,
    height: int,
    width: int,
    jpeg_quality: int = 92,
    overwrite: bool = False,
) -> ClipRecord:
    """Write resized copies of a clip's frames under ``prepared_root/frames``.

    Caching shrinks the dataset by roughly an order of magnitude and removes
    per-epoch resizing, which matters a lot when the data lives on Drive.
    Labels are rescaled to match.
    """
    rel_dir = Path("frames") / clip.id
    out_dir = prepared_root / rel_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    sx = width / clip.width
    sy = height / clip.height
    new_files: List[str] = []
    size_checked = False
    for i, fname in enumerate(clip.files):
        src = raw_root / clip.source_dir / fname
        dst = out_dir / (Path(fname).stem + ".jpg")
        new_files.append(dst.name)
        # Rescale before the skip: labels come from the raw CSV every time, even
        # when the resized frame is already on disk.
        if clip.visible[i]:
            clip.x[i] = float(clip.x[i]) * sx
            clip.y[i] = float(clip.y[i]) * sy
        if dst.is_file() and not overwrite:
            if not size_checked:
                existing = cv2.imread(str(dst))
                if existing is None or existing.shape[:2] != (height, width):
                    raise PrepareError(
                        f"cached frame {dst} is not {width}x{height}; "
                        "re-run prepare with --recache-frames"
                    )
                size_checked = True
            continue
        img = cv2.imread(str(src))
        if img is None:
            raise PrepareError(f"could not read frame {src}")
        resized = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)
        if not cv2.imwrite(str(dst), resized, [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)]):
            raise PrepareError(f"could not write cached frame {dst}")

    clip.files = new_files
    clip.cached_dir = str(rel_dir).replace("\\", "/")
    clip.width = width
    clip.height = height
    return clip


def prepare_dataset(
    cfg: Config,
    raw_root: Optional[Path] = None,
    prepared_root: Optional[Path] = None,
    cache_frames: Optional[bool] = None,
    limit_clips: Optional[int] = None,
    overwrite: bool = False,
    recache_frames: bool = False,
) -> DatasetManifest:
    """Build (or reuse) the prepared dataset and return its manifest.

    ``overwrite`` rebuilds manifest.json (clip index, game ids, split) from the
    raw labels. Already-cached frames are reused unless ``recache_frames``;
    re-writing ~20k frames through Google Drive takes hours.
    """
    data_cfg = cfg.data
    raw_root = Path(raw_root) if raw_root else resolve_path(cfg, data_cfg.raw_dir)
    prepared_root = Path(prepared_root) if prepared_root else resolve_path(cfg, data_cfg.prepared_dir)
    if cache_frames is None:
        cache_frames = bool(data_cfg.get("cache_frames", False))

    manifest_path = prepared_root / "manifest.json"
    if manifest_path.is_file() and not overwrite:
        logger.info("reusing prepared dataset at %s (pass --overwrite to rebuild)", prepared_root)
        return DatasetManifest.load(manifest_path)

    source_root = raw_root
    local_path = data_cfg.get("local_path")
    if local_path:
        candidate = resolve_path(cfg, local_path)
        if candidate.is_dir():
            source_root = candidate

    clip_dirs = discover_clip_dirs(source_root, data_cfg.clip_glob, data_cfg.label_filename)
    if limit_clips:
        clip_dirs = clip_dirs[:limit_clips]
    logger.info("found %d clip(s) under %s", len(clip_dirs), source_root)

    clips: List[ClipRecord] = []
    for clip_dir in clip_dirs:
        try:
            clips.append(
                build_clip_record(
                    clip_dir, source_root, data_cfg.label_filename, list(data_cfg.frame_extensions)
                )
            )
        except Exception as exc:  # noqa: BLE001 - skip unusable clips, keep going
            logger.warning("skipping %s: %s", clip_dir, exc)
    if not clips:
        raise PrepareError(f"no usable clips found under {source_root}")

    split_cfg = data_cfg.split
    assign_splits(
        clips,
        strategy=split_cfg.get("strategy", "by_game"),
        val_fraction=float(split_cfg.get("val_fraction", 0.2)),
        val_games=split_cfg.get("val_games"),
        seed=int(cfg.project.seed),
    )

    height = int(data_cfg.input_height)
    width = int(data_cfg.input_width)
    if cache_frames:
        logger.info("caching resized frames to %s (%dx%d)", prepared_root / "frames", width, height)
        for i, clip in enumerate(clips, start=1):
            cache_clip_frames(
                clip,
                source_root,
                prepared_root,
                height=height,
                width=width,
                jpeg_quality=int(data_cfg.get("cache_jpeg_quality", 92)),
                overwrite=recache_frames,
            )
            if i % 10 == 0 or i == len(clips):
                logger.info("  cached %d/%d clips", i, len(clips))

    manifest = DatasetManifest(
        raw_root=str(source_root),
        prepared_root=str(prepared_root),
        clips=clips,
        input_height=height,
        input_width=width,
        cached=bool(cache_frames),
        created=_dt.datetime.now().isoformat(timespec="seconds"),
        source=str(data_cfg.get("source", "local")),
    )
    manifest.save(manifest_path)
    logger.info("wrote %s", manifest_path)
    return manifest
