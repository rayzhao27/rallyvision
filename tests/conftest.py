"""Shared fixtures. Everything here is tiny and CPU-only by construction."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rallyvision.common.config import load_config  # noqa: E402
from rallyvision.ball_tracking.data.manifest import DatasetManifest  # noqa: E402
from rallyvision.ball_tracking.data.prepare import prepare_dataset  # noqa: E402
from rallyvision.ball_tracking.data.synthetic import generate_dataset  # noqa: E402

# Small enough to stay fast, still divisible by the model's stride of 8.
TEST_HEIGHT = 72
TEST_WIDTH = 128
TEST_FRAMES = 3


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return REPO_ROOT


@pytest.fixture
def base_config():
    """The real config.yaml, shrunk to test scale."""
    cfg = load_config(REPO_ROOT / "configs" / "ball_tracking.yaml")
    cfg["data.input_height"] = TEST_HEIGHT
    cfg["data.input_width"] = TEST_WIDTH
    cfg["data.num_frames"] = TEST_FRAMES
    cfg["data.loader.batch_size"] = 2
    cfg["data.loader.num_workers"] = 0
    cfg["data.loader.pin_memory"] = False
    cfg["data.loader.persistent_workers"] = False
    cfg["data.augment.enabled"] = False
    cfg["model.width"] = 0.125
    cfg["runtime.device"] = "cpu"
    cfg["runtime.amp"] = False
    cfg["train.epochs"] = 1
    cfg["train.max_train_batches"] = 2
    cfg["train.max_val_batches"] = 1
    return cfg


@pytest.fixture(scope="session")
def synthetic_raw(tmp_path_factory) -> Path:
    """A 4-clip synthetic dataset in the raw on-disk layout."""
    root = tmp_path_factory.mktemp("raw")
    return generate_dataset(
        root / "synthetic",
        num_games=2,
        clips_per_game=2,
        frames_per_clip=10,
        width=TEST_WIDTH * 2,
        height=TEST_HEIGHT * 2,
        seed=11,
    )


@pytest.fixture
def prepared_dataset(base_config, synthetic_raw, tmp_path) -> DatasetManifest:
    """A prepared manifest built from the synthetic clips."""
    base_config["data.source"] = "local"
    base_config["data.local_path"] = str(synthetic_raw)
    base_config["data.raw_dir"] = str(synthetic_raw)
    base_config["data.prepared_dir"] = str(tmp_path / "prepared")
    base_config["data.split.strategy"] = "by_game"
    base_config["data.split.val_fraction"] = 0.5
    return prepare_dataset(
        base_config,
        raw_root=synthetic_raw,
        prepared_root=tmp_path / "prepared",
        cache_frames=False,
        overwrite=True,
    )
