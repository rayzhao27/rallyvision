"""Model and data locations from configs/paths.yaml, resolved against the repo root."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from rallyvision.common.config import REPO_ROOT

PATHS_FILE = REPO_ROOT / "configs" / "paths.yaml"
PIPELINE_FILE = REPO_ROOT / "configs" / "pipeline.yaml"


def _resolve(value: Any):
    if isinstance(value, list):
        return [_resolve(v) for v in value]
    p = Path(value).expanduser()
    return p if p.is_absolute() else REPO_ROOT / p


def artifact(module: str, key: str):
    """Path (or list of paths) of ``key`` in section ``module`` of configs/paths.yaml."""
    table = yaml.safe_load(PATHS_FILE.read_text())
    value = table[module][key]
    if value is None:
        raise FileNotFoundError(f"configs/paths.yaml: {module}.{key} is not set (external dependency)")
    return _resolve(value)


def pipeline_config() -> dict:
    return yaml.safe_load(PIPELINE_FILE.read_text())
