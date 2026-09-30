from __future__ import annotations

import pytest

from rallyvision.config import Config, ConfigError, load_config, parse_override, resolve_path


def test_loads_repo_config(repo_root):
    cfg = load_config(repo_root / "config.yaml")
    assert cfg.project.name == "rallyvision"
    assert cfg.data.num_frames >= 1
    assert cfg.model.name in {"tracknet"}


def test_dotted_and_attribute_access_agree(base_config):
    assert base_config["data.input_height"] == base_config.data.input_height


def test_override_creates_missing_nodes():
    cfg = Config({"a": {"b": 1}})
    cfg["a.c.d"] = 5
    assert cfg.a.c.d == 5
    assert cfg.a.b == 1


def test_apply_overrides_ignores_none():
    cfg = Config({"train": {"epochs": 30}})
    out = cfg.apply_overrides({"train.epochs": None})
    assert out.train.epochs == 30


def test_parse_override_types():
    assert parse_override("train.epochs=5") == ("train.epochs", 5)
    assert parse_override("runtime.amp=false") == ("runtime.amp", False)
    assert parse_override("optim.lr=1e-4") == ("optim.lr", 1e-4)
    assert parse_override("eval.tolerances_px=[1, 2]") == ("eval.tolerances_px", [1, 2])


def test_parse_override_rejects_bad_input():
    with pytest.raises(ConfigError):
        parse_override("no_equals_sign")


def test_require_raises_for_missing_key(base_config):
    with pytest.raises(ConfigError):
        base_config.require("data.kaggle_dataset")


def test_save_and_reload_roundtrip(base_config, tmp_path):
    path = base_config.save(tmp_path / "cfg.yaml")
    reloaded = load_config(path)
    assert reloaded.data.input_height == base_config.data.input_height
    assert reloaded.model.width == base_config.model.width


def test_resolve_path_keeps_absolute(base_config, tmp_path):
    assert resolve_path(base_config, tmp_path) == tmp_path
    assert resolve_path(base_config, "data/raw").is_absolute()


def test_missing_config_file_is_a_clear_error(tmp_path):
    with pytest.raises(ConfigError, match="not found"):
        load_config(tmp_path / "nope.yaml")
