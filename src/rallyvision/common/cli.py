"""Shared command-line plumbing for the training, evaluation and inference entry points.

The repo root (``REPO_ROOT``) comes from rallyvision.common.config; entry points run as
``python -m rallyvision.<module>...`` or through the forwarders in scripts/.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, Optional

from rallyvision.common.config import REPO_ROOT, Config, load_config  # noqa: E402,F401
from rallyvision.common.logging_utils import set_level, setup_logging  # noqa: E402


def add_config_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="path to a config file (default: configs/ball_tracking.yaml)",
    )
    parser.add_argument(
        "--set",
        dest="set_args",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="override any config key, e.g. --set train.epochs=5 (repeatable)",
    )
    parser.add_argument("--quiet", action="store_true", help="only log warnings and errors")
    return parser


def add_device_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument(
        "--device",
        choices=["auto", "cuda", "mps", "cpu"],
        default=None,
        help="override runtime.device (default: config value, usually auto)",
    )
    return parser


def build_config(args: argparse.Namespace, overrides: Optional[Dict[str, Any]] = None) -> Config:
    """Load config.yaml and layer CLI overrides on top."""
    merged: Dict[str, Any] = dict(overrides or {})
    if getattr(args, "device", None):
        merged["runtime.device"] = args.device
    if getattr(args, "seed", None) is not None:
        merged["project.seed"] = args.seed
    return load_config(args.config, overrides=merged, set_args=getattr(args, "set_args", []))


def init_logging(args: argparse.Namespace, log_file: Optional[Path] = None):
    import logging

    level = logging.WARNING if getattr(args, "quiet", False) else logging.INFO
    logger = setup_logging(log_file=log_file, level=level)
    set_level(level)
    return logger


def fail(message: str, code: int = 1) -> None:
    """Print a clear error and exit."""
    print(f"\nERROR: {message}\n", file=sys.stderr)
    raise SystemExit(code)
