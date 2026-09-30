"""End-to-end pipeline checks.

These are deliberately minuscule: a handful of tiny CPU batches over synthetic
data. They verify that the wiring works (data -> loader -> forward -> loss ->
backward -> step -> checkpoint -> resume -> evaluate), not that the model
learns anything. Real training happens on Colab.
"""

from __future__ import annotations

import torch

from rallyvision.data.dataset import build_dataloader, build_dataset
from rallyvision.device import resolve_device
from rallyvision.engine.evaluator import evaluate
from rallyvision.engine.optim import build_optimizer, build_scheduler
from rallyvision.engine.trainer import Trainer
from rallyvision.losses import build_loss
from rallyvision.models import build_model
from rallyvision.paths import RunPaths


def _trainer(cfg, manifest, run_dir):
    device_info = resolve_device("cpu")
    train_dataset = build_dataset(cfg, manifest, split="train")
    val_dataset = build_dataset(cfg, manifest, split="val")
    return Trainer(
        cfg=cfg,
        model=build_model(cfg),
        loss_fn=build_loss(cfg),
        train_loader=build_dataloader(cfg, train_dataset, split="train", device_kind="cpu"),
        val_loader=build_dataloader(cfg, val_dataset, split="val", shuffle=False, device_kind="cpu"),
        device_info=device_info,
        paths=RunPaths(run_dir),
    )


def test_one_tiny_epoch_completes_and_saves_a_checkpoint(base_config, prepared_dataset, tmp_path):
    trainer = _trainer(base_config, prepared_dataset, tmp_path / "run")
    state = trainer.train()

    assert state.epoch == 1
    assert state.global_step > 0
    assert trainer.paths.last_checkpoint.is_file()
    assert trainer.paths.best_checkpoint.is_file()
    assert trainer.paths.history_file.is_file()
    assert trainer.paths.metrics_csv.is_file()
    assert len(state.history) == 1
    assert torch.isfinite(torch.tensor(state.history[0]["train/loss"]))


def test_weights_actually_change(base_config, prepared_dataset, tmp_path):
    trainer = _trainer(base_config, prepared_dataset, tmp_path / "run")
    before = trainer.model.head.weight.detach().clone()
    trainer.train()
    after = trainer.model.head.weight.detach()
    assert not torch.allclose(before, after)


def test_resume_continues_instead_of_restarting(base_config, prepared_dataset, tmp_path):
    run_dir = tmp_path / "run"
    first = _trainer(base_config, prepared_dataset, run_dir)
    first.train()
    assert first.state.epoch == 1

    base_config["train.epochs"] = 2
    second = _trainer(base_config, prepared_dataset, run_dir)
    used = second.maybe_resume("auto")
    assert used is not None
    assert second.state.epoch == 1  # picks up where the first left off

    second.train()
    assert second.state.epoch == 2
    assert len(second.state.history) == 2


def test_resume_restores_optimizer_state(base_config, prepared_dataset, tmp_path):
    run_dir = tmp_path / "run"
    first = _trainer(base_config, prepared_dataset, run_dir)
    first.train()
    expected = first.optimizer.state_dict()["state"]

    second = _trainer(base_config, prepared_dataset, run_dir)
    second.maybe_resume("auto")
    restored = second.optimizer.state_dict()["state"]
    assert set(expected) == set(restored)
    if expected:
        key = next(iter(expected))
        assert torch.allclose(expected[key]["exp_avg"], restored[key]["exp_avg"])


def test_resume_with_no_checkpoint_starts_from_scratch(base_config, prepared_dataset, tmp_path):
    trainer = _trainer(base_config, prepared_dataset, tmp_path / "empty_run")
    assert trainer.maybe_resume("auto") is None
    assert trainer.state.epoch == 0


def test_best_checkpoint_is_written_when_the_monitor_improves(base_config, prepared_dataset, tmp_path):
    trainer = _trainer(base_config, prepared_dataset, tmp_path / "run")
    trainer.train()
    # The first epoch is always an improvement over "no metric yet".
    assert trainer.paths.best_checkpoint.is_file()
    assert trainer.state.best_epoch == 1
    assert "val/accuracy_at_5px" in trainer.state.history[0]


def test_best_checkpoint_is_not_replaced_when_val_does_not_improve(base_config, prepared_dataset, tmp_path):
    trainer = _trainer(base_config, prepared_dataset, tmp_path / "run")
    trainer.train()
    best_bytes = trainer.paths.best_checkpoint.read_bytes()
    best_epoch = trainer.state.best_epoch
    trainer.state.best_metric = 1e9  # later val Acc@5px cannot beat this
    trainer.epochs = 2
    trainer.train()
    assert trainer.state.epoch == 2
    assert trainer.state.best_epoch == best_epoch
    assert trainer.paths.best_checkpoint.read_bytes() == best_bytes
    assert trainer.paths.last_checkpoint.is_file()
    assert trainer.paths.last_checkpoint.stat().st_size > 0


def test_evaluation_produces_the_expected_metric_keys(base_config, prepared_dataset, tmp_path):
    trainer = _trainer(base_config, prepared_dataset, tmp_path / "run")
    metrics, accumulator = evaluate(
        trainer.model,
        trainer.val_loader,
        resolve_device("cpu"),
        base_config,
        loss_fn=trainer.loss_fn,
        keep_samples=True,
    )
    for key in (
        "mean_error_px",
        "median_error_px",
        "accuracy_at_5px",
        "accuracy_at_10px",
        "accuracy_at_20px",
        "precision",
        "recall",
        "f1",
        "loss",
    ):
        assert key in metrics
    assert metrics["num_samples"] > 0
    assert len(accumulator.samples) == metrics["num_samples"]


def test_evaluate_and_save_writes_json_and_csv(base_config, prepared_dataset, tmp_path):
    from rallyvision.engine.evaluator import evaluate_and_save

    trainer = _trainer(base_config, prepared_dataset, tmp_path / "run")
    out = tmp_path / "metrics"
    evaluate_and_save(
        trainer.model,
        trainer.val_loader,
        resolve_device("cpu"),
        base_config,
        output_dir=out,
        prefix="val",
        progress=False,
    )
    assert (out / "val_metrics.json").is_file()
    assert (out / "val_per_sample.csv").is_file()


def test_max_batches_caps_the_epoch(base_config, prepared_dataset, tmp_path):
    base_config["train.max_train_batches"] = 1
    trainer = _trainer(base_config, prepared_dataset, tmp_path / "run")
    trainer.train()
    assert trainer.state.history[0]["train/batches"] == 1


def test_gradients_are_finite_after_a_step(base_config, prepared_dataset, tmp_path):
    trainer = _trainer(base_config, prepared_dataset, tmp_path / "run")
    batch = next(iter(trainer.train_loader))
    logits = trainer.model.forward_logits(batch["frames"])
    trainer.loss_fn(logits, batch["heatmap"]).backward()
    for param in trainer.model.parameters():
        if param.grad is not None:
            assert torch.isfinite(param.grad).all()


def test_cosine_scheduler_decays_the_learning_rate(base_config):
    base_config["train.epochs"] = 10
    base_config["scheduler.name"] = "cosine"
    base_config["scheduler.warmup_epochs"] = 0
    model = build_model(base_config)
    optimizer = build_optimizer(base_config, model)
    scheduler = build_scheduler(base_config, optimizer)
    start = optimizer.param_groups[0]["lr"]
    for _ in range(9):
        scheduler.step()
    assert optimizer.param_groups[0]["lr"] < start


def test_warmup_raises_the_learning_rate_first(base_config):
    base_config["train.epochs"] = 10
    base_config["scheduler.name"] = "cosine"
    base_config["scheduler.warmup_epochs"] = 2
    model = build_model(base_config)
    optimizer = build_optimizer(base_config, model)
    scheduler = build_scheduler(base_config, optimizer)
    first = optimizer.param_groups[0]["lr"]
    scheduler.step()
    assert optimizer.param_groups[0]["lr"] > first


def test_optimizer_excludes_norm_and_bias_from_weight_decay(base_config):
    model = build_model(base_config)
    optimizer = build_optimizer(base_config, model)
    assert len(optimizer.param_groups) == 2
    assert optimizer.param_groups[1]["weight_decay"] == 0.0
