from __future__ import annotations

import torch

from rallyvision.checkpoint import (
    TrainingState,
    find_resume_checkpoint,
    load_model_weights,
    prune_epoch_checkpoints,
    restore_training,
    save_checkpoint,
)
from rallyvision.models.tracknet import TrackNet


def _fixture():
    model = TrackNet(num_frames=3, width=0.125)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)
    return model, optimizer, scheduler


def test_round_trip_restores_model_optimizer_scheduler_and_epoch(tmp_path):
    model, optimizer, scheduler = _fixture()
    # Take a real step so the optimizer accumulates state worth restoring.
    model.forward_logits(torch.rand(1, 9, 72, 128)).mean().backward()
    optimizer.step()
    scheduler.step()

    state = TrainingState(epoch=7, global_step=123, best_metric=0.42, best_epoch=5)
    path = save_checkpoint(tmp_path / "last.pt", model, state, optimizer, scheduler)
    assert path.is_file()

    fresh_model, fresh_optimizer, fresh_scheduler = _fixture()
    restored = restore_training(path, fresh_model, fresh_optimizer, fresh_scheduler)

    assert restored.epoch == 7
    assert restored.global_step == 123
    assert restored.best_metric == 0.42
    assert restored.best_epoch == 5
    assert fresh_scheduler.last_epoch == scheduler.last_epoch
    assert fresh_optimizer.param_groups[0]["lr"] == optimizer.param_groups[0]["lr"]
    for a, b in zip(model.state_dict().values(), fresh_model.state_dict().values()):
        assert torch.allclose(a.float(), b.float())


def test_restored_model_produces_identical_output(tmp_path):
    model, optimizer, _ = _fixture()
    model.eval()
    x = torch.rand(1, 9, 72, 128)
    with torch.no_grad():
        expected = model(x)

    save_checkpoint(tmp_path / "last.pt", model, TrainingState(epoch=1), optimizer)
    fresh = TrackNet(num_frames=3, width=0.125).eval()
    load_model_weights(tmp_path / "last.pt", fresh)
    with torch.no_grad():
        assert torch.allclose(expected, fresh(x), atol=1e-6)


def test_best_metric_direction():
    maximize = TrainingState(monitor_mode="max", best_metric=0.5)
    assert maximize.is_improvement(0.6)
    assert not maximize.is_improvement(0.4)

    minimize = TrainingState(monitor_mode="min", best_metric=0.5)
    assert minimize.is_improvement(0.4)
    assert not minimize.is_improvement(0.6)

    fresh = TrainingState()
    assert fresh.is_improvement(-100.0)


def test_history_survives_the_round_trip(tmp_path):
    model, optimizer, _ = _fixture()
    state = TrainingState(epoch=2, history=[{"epoch": 1, "train/loss": 1.0}, {"epoch": 2, "train/loss": 0.5}])
    save_checkpoint(tmp_path / "last.pt", model, state, optimizer)
    restored = restore_training(tmp_path / "last.pt", TrackNet(num_frames=3, width=0.125))
    assert len(restored.history) == 2
    assert restored.history[-1]["train/loss"] == 0.5


def test_find_resume_prefers_last_then_best(tmp_path):
    ckpt_dir = tmp_path / "checkpoints"
    ckpt_dir.mkdir()
    assert find_resume_checkpoint(ckpt_dir) is None

    (ckpt_dir / "best.pt").write_bytes(b"x")
    assert find_resume_checkpoint(ckpt_dir).name == "best.pt"

    (ckpt_dir / "last.pt").write_bytes(b"x")
    assert find_resume_checkpoint(ckpt_dir).name == "last.pt"


def test_prune_keeps_only_the_newest_epoch_files(tmp_path):
    for i in range(5):
        (tmp_path / f"epoch_{i:04d}.pt").write_bytes(b"x")
    prune_epoch_checkpoints(tmp_path, keep_last_n=2)
    remaining = sorted(p.name for p in tmp_path.glob("epoch_*.pt"))
    assert remaining == ["epoch_0003.pt", "epoch_0004.pt"]


def test_save_is_atomic_and_leaves_no_temp_file(tmp_path):
    model, optimizer, _ = _fixture()
    save_checkpoint(tmp_path / "last.pt", model, TrainingState(), optimizer)
    assert list(tmp_path.glob("*.tmp")) == []
