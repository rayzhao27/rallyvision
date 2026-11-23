from __future__ import annotations

from pathlib import Path

import pytest

from rallyvision.common.environment import (
    EnvironmentError,
    apply_persistent_storage,
    in_colab,
    report_environment,
    resolve_storage_root,
)


def test_environment_report_always_includes_torch(base_config):
    report = report_environment(base_config)
    text = report.format()
    assert "torch" in text
    assert "python" in text
    assert report.device.kind in {"cuda", "mps", "cpu"}
    assert report.in_colab is False or isinstance(report.in_colab, bool)


def test_require_cuda_fails_on_cpu(base_config):
    base_config["runtime.device"] = "cpu"
    with pytest.raises(EnvironmentError, match="CUDA is required"):
        report_environment(base_config, require_cuda=True)


def test_apply_persistent_storage_rewrites_paths(base_config, tmp_path):
    apply_persistent_storage(base_config, tmp_path / "drive" / "rallyvision")
    root = tmp_path / "drive" / "rallyvision"
    assert Path(base_config.data.raw_dir) == root / "data" / "raw"
    assert Path(base_config.data.prepared_dir) == root / "data" / "prepared"
    assert Path(base_config.project.output_root) == root / "artifacts"
    assert (root / "data" / "raw").is_dir()
    assert (root / "artifacts").is_dir()
    assert Path(base_config.data.raw_dir).is_absolute()


def test_resolve_storage_root_uses_explicit_drive(base_config, tmp_path):
    chosen = resolve_storage_root(base_config, drive_root=tmp_path / "drv")
    assert chosen == (tmp_path / "drv").resolve()


def test_in_colab_is_false_on_this_machine():
    # This Mac is not Colab. The helper must not false-positive.
    assert in_colab() is False


def test_config_exposes_colab_defaults(repo_root):
    from rallyvision.common.config import load_config

    cfg = load_config(repo_root / "configs" / "ball_tracking.yaml")
    assert cfg.colab.require_cuda is True
    assert "MyDrive" in str(cfg.colab.drive_root)
