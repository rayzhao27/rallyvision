"""E2E-Spot Tennis labels (github.com/jhong93/spot, data/tennis): clips, events, per-frame targets.

Each clip is a frame range of a YouTube broadcast match, named ``<source video>_<start>_<end>``;
event frames are relative to the clip start, at the source frame rate (US Open 29.97, Wimbledon 25).
Classes: {near,far}_court_{bounce,swing,serve}. "far/near" is the half of the court (bounces) or the
player (swings, serves). Timing in the labels (serve -> bounce 0.44 s, swing -> bounce 0.92 s,
bounce -> swing 0.40 s, train+val medians) matches racket-ball contact, so swing and serve map to "hit".
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

OTHER, BOUNCE, HIT = 0, 1, 2
CLASS_NAMES = ("other", "bounce", "hit")


def parse_clip_name(name: str) -> Tuple[str, int, int]:
    """'usopen_2019_..._14849_15383' -> ('usopen_2019_...', 14849, 15383)."""
    m = re.fullmatch(r"(.+)_(\d+)_(\d+)", name)
    if not m:
        raise ValueError(f"not a clip name: {name}")
    return m.group(1), int(m.group(2)), int(m.group(3))


def load_split(path: Path) -> List[Dict]:
    """Clips of one split file, each with source, start, end added."""
    clips = json.loads(Path(path).read_text())
    for c in clips:
        c["source"], c["start"], c["end"] = parse_clip_name(c["video"])
        if c["end"] - c["start"] != c["num_frames"]:
            raise ValueError(f"{c['video']}: frame range {c['end'] - c['start']} != num_frames {c['num_frames']}")
    return clips


def load_all(labels_dir: Path) -> Dict[str, List[Dict]]:
    return {s: load_split(Path(labels_dir) / f"{s}.json") for s in ("train", "val", "test")}


def load_videos(labels_dir: Path) -> Dict[str, Dict]:
    """videos.csv: name -> {resolution, fps, yt_id}."""
    lines = (Path(labels_dir) / "videos.csv").read_text().strip().splitlines()[1:]
    out = {}
    for line in lines:
        name, res, fps, yt = line.split(",")
        out[name] = {"resolution": res, "fps": float(fps), "yt_id": yt}
    return out


def event_kind(label: str) -> Tuple[int, str]:
    """'far_court_bounce' -> (BOUNCE, 'far'); swings and serves are hits."""
    side, _, what = label.split("_")
    return (BOUNCE if what == "bounce" else HIT), side


def frame_targets(clip: Dict) -> Tuple[np.ndarray, np.ndarray]:
    """Per-frame class (OTHER/BOUNCE/HIT) and side (0 none, 1 near, 2 far) at the labelled frames."""
    n = clip["num_frames"]
    cls, side = np.zeros(n, dtype=np.int64), np.zeros(n, dtype=np.int64)
    for e in clip["events"]:
        k, s = event_kind(e["label"])
        if 0 <= e["frame"] < n:
            cls[e["frame"]], side[e["frame"]] = k, 1 if s == "near" else 2
    return cls, side
