"""Run directory layout.

Every artifact of a run lives under one directory so a Colab session can point
`project.output_root` at Google Drive and survive a disconnect:

    <output_root>/runs/<run_name>/
        config.yaml          exact config the run started with
        checkpoints/         last.pt, best.pt, epoch_*.pt
        logs/                train.log, metrics.csv
        metrics/             history.json, eval_*.json, eval_*.csv
        predictions/         sample visualizations
        videos/              annotated inference output
"""

from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from rallyvision.common.config import Config, output_root


@dataclass
class RunPaths:
    root: Path

    @property
    def checkpoints(self) -> Path:
        return self.root / "checkpoints"

    @property
    def logs(self) -> Path:
        return self.root / "logs"

    @property
    def metrics(self) -> Path:
        return self.root / "metrics"

    @property
    def predictions(self) -> Path:
        return self.root / "predictions"

    @property
    def videos(self) -> Path:
        return self.root / "videos"

    @property
    def config_file(self) -> Path:
        return self.root / "config.yaml"

    @property
    def last_checkpoint(self) -> Path:
        return self.checkpoints / "last.pt"

    @property
    def best_checkpoint(self) -> Path:
        return self.checkpoints / "best.pt"

    @property
    def history_file(self) -> Path:
        return self.metrics / "history.json"

    @property
    def metrics_csv(self) -> Path:
        return self.logs / "metrics.csv"

    @property
    def train_log(self) -> Path:
        return self.logs / "train.log"

    def create(self) -> "RunPaths":
        for path in (self.root, self.checkpoints, self.logs, self.metrics, self.predictions, self.videos):
            path.mkdir(parents=True, exist_ok=True)
        return self

    def exists(self) -> bool:
        return self.root.is_dir()


def default_run_name(prefix: str = "ball") -> str:
    return f"{prefix}_{_dt.datetime.now().strftime('%Y%m%d_%H%M%S')}"


def get_run_paths(cfg: Config, run_name: Optional[str] = None, create: bool = False) -> RunPaths:
    """Resolve the run directory, generating a timestamped name if needed."""
    name = run_name or cfg.get("project.run_name") or default_run_name()
    paths = RunPaths(output_root(cfg) / "runs" / str(name))
    return paths.create() if create else paths


def latest_run(cfg: Config) -> Optional[RunPaths]:
    """Most recently modified run directory, or None."""
    runs_dir = output_root(cfg) / "runs"
    if not runs_dir.is_dir():
        return None
    candidates = [p for p in runs_dir.iterdir() if p.is_dir()]
    if not candidates:
        return None
    return RunPaths(max(candidates, key=lambda p: p.stat().st_mtime))


def resolve_run(cfg: Config, run: Optional[str]) -> RunPaths:
    """Resolve a run argument that may be a name, a path, or 'latest'."""
    if run in (None, "", "latest"):
        found = latest_run(cfg)
        if found is None:
            raise FileNotFoundError(
                f"no runs found under {output_root(cfg) / 'runs'}; train a model first"
            )
        return found
    candidate = Path(run)
    if candidate.is_dir():
        return RunPaths(candidate)
    return RunPaths(output_root(cfg) / "runs" / str(run))
