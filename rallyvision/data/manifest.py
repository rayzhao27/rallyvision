"""The prepared-dataset manifest.

Preparing a dataset means: walk the raw clips, parse their label CSVs, decide
the train/val split, optionally cache resized frames, and write a single
`manifest.json`. Training then never touches the raw layout again, so the same
prepared directory can be cached on Google Drive and reused across Colab
sessions without re-downloading or re-preprocessing anything.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

MANIFEST_VERSION = 1
MANIFEST_FILENAME = "manifest.json"


@dataclass
class ClipRecord:
    """One rally clip: an ordered list of frames with per-frame ball labels."""

    id: str
    game: str
    name: str
    source_dir: str  # relative to the raw dataset root
    files: List[str] = field(default_factory=list)
    visible: List[int] = field(default_factory=list)
    x: List[Optional[float]] = field(default_factory=list)
    y: List[Optional[float]] = field(default_factory=list)
    width: int = 0
    height: int = 0
    split: str = "train"
    cached_dir: Optional[str] = None  # relative to the prepared root

    def __len__(self) -> int:
        return len(self.files)

    @property
    def num_visible(self) -> int:
        return int(sum(self.visible))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "game": self.game,
            "name": self.name,
            "source_dir": self.source_dir,
            "files": self.files,
            "visible": self.visible,
            "x": self.x,
            "y": self.y,
            "width": self.width,
            "height": self.height,
            "split": self.split,
            "cached_dir": self.cached_dir,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ClipRecord":
        return cls(
            id=data["id"],
            game=data.get("game", ""),
            name=data.get("name", ""),
            source_dir=data["source_dir"],
            files=list(data.get("files", [])),
            visible=[int(v) for v in data.get("visible", [])],
            x=list(data.get("x", [])),
            y=list(data.get("y", [])),
            width=int(data.get("width", 0)),
            height=int(data.get("height", 0)),
            split=data.get("split", "train"),
            cached_dir=data.get("cached_dir"),
        )


@dataclass
class DatasetManifest:
    raw_root: str
    prepared_root: str
    clips: List[ClipRecord] = field(default_factory=list)
    input_height: int = 360
    input_width: int = 640
    cached: bool = False
    created: str = ""
    source: str = "local"
    version: int = MANIFEST_VERSION

    # -- io ---------------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "created": self.created,
            "source": self.source,
            "raw_root": self.raw_root,
            "prepared_root": self.prepared_root,
            "input_height": self.input_height,
            "input_width": self.input_width,
            "cached": self.cached,
            "clips": [c.to_dict() for c in self.clips],
        }

    def save(self, path: Path | str) -> Path:
        path = Path(path)
        if path.is_dir():
            path = path / MANIFEST_FILENAME
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".json.tmp")
        with tmp.open("w") as fh:
            json.dump(self.to_dict(), fh)
        tmp.replace(path)
        return path

    @classmethod
    def load(cls, path: Path | str) -> "DatasetManifest":
        path = Path(path)
        if path.is_dir():
            path = path / MANIFEST_FILENAME
        if not path.is_file():
            raise FileNotFoundError(
                f"prepared dataset manifest not found: {path}\n"
                "Run scripts/prepare_dataset.py first."
            )
        with path.open() as fh:
            data = json.load(fh)
        # Always bind prepared_root to the directory we actually loaded from.
        # The folder is frequently copied onto Google Drive or remounted, and
        # the path stored at prepare-time would then be stale.
        return cls(
            raw_root=data["raw_root"],
            prepared_root=str(path.parent.resolve()),
            clips=[ClipRecord.from_dict(c) for c in data.get("clips", [])],
            input_height=int(data.get("input_height", 360)),
            input_width=int(data.get("input_width", 640)),
            cached=bool(data.get("cached", False)),
            created=data.get("created", ""),
            source=data.get("source", "local"),
            version=int(data.get("version", MANIFEST_VERSION)),
        )

    # -- queries ----------------------------------------------------------
    def split_clips(self, split: Optional[str]) -> List[ClipRecord]:
        if split is None or split == "all":
            return list(self.clips)
        return [c for c in self.clips if c.split == split]

    def games(self) -> List[str]:
        seen: Dict[str, None] = {}
        for clip in self.clips:
            seen.setdefault(clip.game, None)
        return list(seen)

    def split_problems(self, num_frames: int, frame_stride: int) -> List[str]:
        """Reasons the train/val split cannot give an honest held-out score.

        An empty list means: 2+ games, both splits have usable samples, and no
        game appears in both train and val.
        """
        train_games = {c.game for c in self.split_clips("train")}
        val_games = {c.game for c in self.split_clips("val")}
        problems: List[str] = []
        if len(self.games()) < 2:
            problems.append(f"only {len(self.games())} game id(s) {self.games()}; a by-game split needs 2+")
        if self.usable_samples("train", num_frames, frame_stride) == 0:
            problems.append("train split has no usable samples")
        if self.usable_samples("val", num_frames, frame_stride) == 0:
            problems.append("val split has no usable samples")
        overlap = sorted(train_games & val_games)
        if overlap:
            problems.append(f"games in both train and val (leakage): {overlap}")
        return problems

    def usable_samples(self, split: Optional[str], num_frames: int, frame_stride: int) -> int:
        """Frames that can act as the target of a full temporal stack."""
        span = (num_frames - 1) * frame_stride
        return sum(max(0, len(c) - span) for c in self.split_clips(split))

    def stats(self, num_frames: int = 3, frame_stride: int = 1) -> Dict[str, Any]:
        """Aggregate counts, computed from the manifest - never hard-coded."""
        resolutions: Dict[str, int] = {}
        for clip in self.clips:
            key = f"{clip.width}x{clip.height}"
            resolutions[key] = resolutions.get(key, 0) + len(clip)

        per_split: Dict[str, Any] = {}
        for split in ("train", "val"):
            clips = self.split_clips(split)
            frames = sum(len(c) for c in clips)
            visible = sum(c.num_visible for c in clips)
            per_split[split] = {
                "clips": len(clips),
                "games": sorted({c.game for c in clips}),
                "frames": frames,
                "visible_frames": visible,
                "invisible_frames": frames - visible,
                "usable_samples": self.usable_samples(split, num_frames, frame_stride),
            }

        total_frames = sum(len(c) for c in self.clips)
        total_visible = sum(c.num_visible for c in self.clips)
        return {
            "clips": len(self.clips),
            "games": len(self.games()),
            "frames": total_frames,
            "visible_frames": total_visible,
            "invisible_frames": total_frames - total_visible,
            "visible_fraction": (total_visible / total_frames) if total_frames else 0.0,
            "source_resolutions": resolutions,
            "network_input": {
                "height": self.input_height,
                "width": self.input_width,
                "num_frames": num_frames,
                "frame_stride": frame_stride,
            },
            "frames_cached": self.cached,
            "usable_samples": self.usable_samples(None, num_frames, frame_stride),
            "splits": per_split,
        }

    def format_stats(self, num_frames: int = 3, frame_stride: int = 1) -> str:
        s = self.stats(num_frames, frame_stride)
        train, val = s["splits"]["train"], s["splits"]["val"]
        total_usable = train["usable_samples"] + val["usable_samples"]
        lines = [
            f"clips              : {s['clips']} across {s['games']} game(s)",
            f"frames             : {s['frames']:,} "
            f"({s['visible_frames']:,} with a visible ball, "
            f"{100 * s['visible_fraction']:.1f}%)",
            f"source resolution  : {_format_resolutions(s['source_resolutions'])}",
            f"network input      : {num_frames} x 3 x {s['network_input']['height']}"
            f" x {s['network_input']['width']} (stride {frame_stride})",
            f"frames cached      : {'yes' if s['frames_cached'] else 'no (reads raw frames)'}",
            f"train split        : {train['clips']} clips, {train['frames']:,} frames, "
            f"{train['usable_samples']:,} usable samples",
            f"val split          : {val['clips']} clips, {val['frames']:,} frames, "
            f"{val['usable_samples']:,} usable samples",
            f"usable samples     : {total_usable:,} "
            f"({_pct(train['usable_samples'], total_usable)} train / "
            f"{_pct(val['usable_samples'], total_usable)} val)",
        ]
        if train["games"] and val["games"]:
            lines.append(f"train games        : {', '.join(train['games'])}")
            lines.append(f"val games          : {', '.join(val['games'])}")
        return "\n".join(lines)


def _format_resolutions(resolutions: Dict[str, int]) -> str:
    if not resolutions:
        return "unknown"
    items = sorted(resolutions.items(), key=lambda kv: -kv[1])
    return ", ".join(f"{res} ({count:,} frames)" for res, count in items)


def _pct(part: int, whole: int) -> str:
    return f"{(100.0 * part / whole):.0f}%" if whole else "0%"


def resolve_clip_frame_path(
    clip: ClipRecord,
    filename: str,
    raw_root: Path,
    prepared_root: Path,
) -> Path:
    """Path to a frame, preferring the resized cache when one exists."""
    if clip.cached_dir:
        return prepared_root / clip.cached_dir / filename
    return raw_root / clip.source_dir / filename


def summarize_split_assignment(clips: Sequence[ClipRecord]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for clip in clips:
        counts[clip.split] = counts.get(clip.split, 0) + 1
    return counts
