"""Small shared helpers: seeding, timing, JSON/CSV IO, human-readable sizes."""

from __future__ import annotations

import csv
import json
import os
import random
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np
import torch


def set_seed(seed: int, deterministic: bool = False) -> int:
    """Seed python, numpy and torch. Returns the seed for logging."""
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except TypeError:  # older torch
            torch.use_deterministic_algorithms(True)
    return seed


def worker_init_fn(worker_id: int) -> None:
    """Give every DataLoader worker a distinct, reproducible seed."""
    base = torch.initial_seed() % (2**32)
    seed = (base + worker_id) % (2**32)
    np.random.seed(seed)
    random.seed(seed)


def seed_generator(seed: int) -> torch.Generator:
    g = torch.Generator()
    g.manual_seed(seed)
    return g


@contextmanager
def timer(label: str = "", printer=None):
    """Time a block; yields a one-element list that receives the elapsed seconds."""
    holder: List[float] = [0.0]
    start = time.perf_counter()
    try:
        yield holder
    finally:
        holder[0] = time.perf_counter() - start
        if printer is not None:
            printer(f"{label} took {format_duration(holder[0])}")


def format_duration(seconds: float) -> str:
    if seconds < 1:
        return f"{seconds * 1000:.0f}ms"
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, secs = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h{minutes:02d}m{secs:02d}s"
    return f"{minutes}m{secs:02d}s"


def format_bytes(num_bytes: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(num_bytes) < 1024 or unit == "TB":
            return f"{num_bytes:.1f} {unit}"
        num_bytes /= 1024
    return f"{num_bytes:.1f} TB"


def dir_size_bytes(path: os.PathLike | str) -> int:
    total = 0
    for root, _dirs, files in os.walk(path):
        for name in files:
            fp = Path(root) / name
            if fp.is_file():
                total += fp.stat().st_size
    return total


def count_parameters(model: torch.nn.Module) -> Dict[str, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable}


def write_json(path: os.PathLike | str, payload: Any, indent: int = 2) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as fh:
        json.dump(payload, fh, indent=indent, default=_json_default)
    tmp.replace(path)
    return path


def read_json(path: os.PathLike | str) -> Any:
    with Path(path).open() as fh:
        return json.load(fh)


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    return str(obj)


def write_csv(
    path: os.PathLike | str,
    rows: Sequence[Mapping[str, Any]],
    fieldnames: Optional[Iterable[str]] = None,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return path
    if fieldnames is None:
        seen: Dict[str, None] = {}
        for row in rows:
            for key in row:
                seen.setdefault(key, None)
        fieldnames = list(seen)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return path


class CSVLogger:
    """Append-only CSV writer that tolerates new columns appearing later."""

    def __init__(self, path: os.PathLike | str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fieldnames: List[str] = []
        self.rows: List[Dict[str, Any]] = []
        if self.path.is_file():
            with self.path.open() as fh:
                reader = csv.DictReader(fh)
                self.fieldnames = list(reader.fieldnames or [])
                self.rows = [dict(r) for r in reader]

    def log(self, row: Mapping[str, Any]) -> None:
        new_keys = [k for k in row if k not in self.fieldnames]
        self.fieldnames.extend(new_keys)
        self.rows.append({k: _scalar(v) for k, v in row.items()})
        self.flush()

    def flush(self) -> None:
        write_csv(self.path, self.rows, self.fieldnames)


def _scalar(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.item() if value.numel() == 1 else value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    return value


def flatten_metrics(metrics: Mapping[str, Any], prefix: str = "") -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key, value in metrics.items():
        name = f"{prefix}{key}"
        if isinstance(value, Mapping):
            out.update(flatten_metrics(value, prefix=f"{name}/"))
        else:
            out[name] = _scalar(value)
    return out
