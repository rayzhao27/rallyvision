"""Configuration loading.

`config.yaml` at the repo root is the single source of truth for defaults.
Everything else (CLI flags, the Colab notebook) layers overrides on top of it,
so there is exactly one place where a default lives.

    cfg = load_config()                       # repo-root config.yaml
    cfg = load_config(overrides={"train.epochs": 1})
    cfg.train.epochs                          # attribute access
    cfg["train.epochs"]                       # dotted access
"""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterator, Mapping, MutableMapping, Optional, Sequence

import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]          # src/rallyvision/common/config.py
DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "ball_tracking.yaml"

_MISSING = object()


class ConfigError(RuntimeError):
    """Raised when a configuration file or override is invalid."""


class Config(MutableMapping):
    """A nested dict with attribute access and dotted-key lookup."""

    def __init__(self, data: Optional[Mapping[str, Any]] = None):
        object.__setattr__(self, "_data", {})
        for key, value in (data or {}).items():
            self._data[key] = _wrap(value)

    # -- mapping protocol -------------------------------------------------
    def __getitem__(self, key: str) -> Any:
        if "." in key:
            node: Any = self
            for part in key.split("."):
                if not isinstance(node, Config) or part not in node._data:
                    raise KeyError(key)
                node = node._data[part]
            return node
        return self._data[key]

    def __setitem__(self, key: str, value: Any) -> None:
        if "." in key:
            head, rest = key.split(".", 1)
            node = self._data.get(head)
            if not isinstance(node, Config):
                node = Config()
                self._data[head] = node
            node[rest] = value
            return
        self._data[key] = _wrap(value)

    def __delitem__(self, key: str) -> None:
        del self._data[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    # -- attribute access -------------------------------------------------
    def __getattr__(self, key: str) -> Any:
        try:
            return object.__getattribute__(self, "_data")[key]
        except KeyError as exc:
            raise AttributeError(
                f"config has no key {key!r}; available: {sorted(self._data)}"
            ) from exc

    def __setattr__(self, key: str, value: Any) -> None:
        self._data[key] = _wrap(value)

    def __repr__(self) -> str:
        return f"Config({json.dumps(self.to_dict(), indent=2, default=str)})"

    # -- helpers ----------------------------------------------------------
    def get(self, key: str, default: Any = None) -> Any:
        try:
            return self[key]
        except KeyError:
            return default

    def require(self, key: str) -> Any:
        """Fetch a key, raising a descriptive error when it is unset."""
        value = self.get(key, _MISSING)
        if value is _MISSING or value is None:
            raise ConfigError(f"required config key {key!r} is not set")
        return value

    def to_dict(self) -> Dict[str, Any]:
        return {k: (v.to_dict() if isinstance(v, Config) else v) for k, v in self._data.items()}

    def copy(self) -> "Config":
        return Config(copy.deepcopy(self.to_dict()))

    def merge(self, other: Mapping[str, Any]) -> "Config":
        """Return a new Config with `other` deep-merged on top of this one."""
        merged = _deep_merge(self.to_dict(), dict(other))
        return Config(merged)

    def apply_overrides(self, overrides: Mapping[str, Any]) -> "Config":
        """Return a new Config with dotted-key overrides applied."""
        out = self.copy()
        for key, value in overrides.items():
            if value is None:
                continue
            out[key] = value
        return out

    def save(self, path: os.PathLike | str) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as fh:
            yaml.safe_dump(self.to_dict(), fh, sort_keys=False, default_flow_style=False)
        return path


def _wrap(value: Any) -> Any:
    if isinstance(value, Config):
        return value
    if isinstance(value, Mapping):
        return Config(value)
    if isinstance(value, list):
        return [_wrap(v) for v in value]
    return value


def _deep_merge(base: Dict[str, Any], override: Mapping[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value.to_dict() if isinstance(value, Config) else value
    return out


def parse_override(text: str) -> tuple[str, Any]:
    """Parse a ``key.path=value`` string; the value is decoded as YAML."""
    if "=" not in text:
        raise ConfigError(f"override {text!r} must be of the form key.path=value")
    key, raw = text.split("=", 1)
    key = key.strip()
    if not key:
        raise ConfigError(f"override {text!r} has an empty key")
    try:
        value = yaml.safe_load(raw)
    except yaml.YAMLError as exc:
        raise ConfigError(f"could not parse value in override {text!r}: {exc}") from exc
    # YAML 1.1 does not recognize unpunctuated scientific notation, so `1e-4`
    # would arrive as the string "1e-4" and silently break a learning rate.
    if isinstance(value, str):
        try:
            value = float(value)
        except ValueError:
            pass
    return key, value


def load_config(
    path: Optional[os.PathLike | str] = None,
    overrides: Optional[Mapping[str, Any]] = None,
    set_args: Optional[Sequence[str]] = None,
) -> Config:
    """Load `config.yaml` and apply overrides.

    Args:
        path: config file; defaults to the repo-root `config.yaml`.
        overrides: mapping of dotted keys to values. ``None`` values are
            ignored, which lets argparse defaults pass straight through.
        set_args: raw ``key=value`` strings, e.g. from ``--set``.
    """
    path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    if not path.is_file():
        raise ConfigError(f"config file not found: {path}")
    with path.open() as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ConfigError(f"config file {path} must contain a YAML mapping")

    cfg = Config(data)
    cfg["_meta"] = {"config_path": str(path)}
    if overrides:
        cfg = cfg.apply_overrides(overrides)
    for item in set_args or []:
        key, value = parse_override(item)
        cfg[key] = value
    return cfg


def resolve_path(cfg: Config, value: os.PathLike | str) -> Path:
    """Resolve a possibly relative path against the repo root.

    Absolute paths are returned unchanged, so a Colab notebook can point
    `project.output_root` at `/content/drive/MyDrive/...` without surprises.
    """
    p = Path(value).expanduser()
    return p if p.is_absolute() else (REPO_ROOT / p)


def output_root(cfg: Config) -> Path:
    return resolve_path(cfg, cfg.project.output_root)
