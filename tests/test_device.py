from __future__ import annotations

import pytest
import torch

from rallyvision.device import (
    DeviceError,
    autocast_context,
    guard_training_device,
    resolve_device,
    resolve_device_from_config,
)


def test_cpu_is_always_resolvable():
    info = resolve_device("cpu")
    assert info.kind == "cpu"
    assert info.device == torch.device("cpu")
    assert info.amp_enabled is False


def test_auto_never_raises():
    info = resolve_device("auto")
    assert info.kind in {"cuda", "mps", "cpu"}


def test_amp_is_cuda_only():
    info = resolve_device("cpu", amp=True)
    assert info.amp_enabled is False
    assert info.amp_dtype is None


def test_unknown_device_rejected():
    with pytest.raises(DeviceError):
        resolve_device("tpu")


def test_autocast_is_a_noop_without_amp():
    info = resolve_device("cpu")
    with autocast_context(info):
        assert torch.zeros(1).dtype == torch.float32


def test_summary_mentions_device_and_torch():
    text = resolve_device("cpu").summary()
    assert "device" in text and "torch" in text


def test_training_guard_blocks_non_cuda(base_config):
    info = resolve_device("cpu")
    with pytest.raises(DeviceError, match="Refusing to start full training"):
        guard_training_device(info, base_config)


def test_training_guard_allows_explicit_local_override(base_config):
    info = resolve_device("cpu")
    guard_training_device(info, base_config, allow_local=True)  # must not raise


def test_training_guard_respects_config_opt_out(base_config):
    base_config["runtime.require_cuda_for_training"] = False
    guard_training_device(resolve_device("cpu"), base_config)  # must not raise


def test_resolve_from_config_uses_runtime_section(base_config):
    info = resolve_device_from_config(base_config)
    assert info.kind == "cpu"
