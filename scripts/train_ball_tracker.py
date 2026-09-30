#!/usr/bin/env python3
"""Train the ball tracker.

    COLAB GPU JOB. Full training will refuse to start on MPS/CPU.

Typical use:

    # Colab, full run
    python scripts/train_ball_tracker.py --run-name v1

    # Colab, continue after a disconnect
    python scripts/train_ball_tracker.py --run-name v1 --resume

    # anywhere, tiny pipeline check (a handful of batches, seconds)
    python scripts/train_ball_tracker.py --smoke

The `--smoke` path exists so the pipeline can be verified without training:
data -> DataLoader -> device -> forward -> loss -> backward -> optimizer step
-> checkpoint save. It is explicitly allowed on CPU/MPS. Everything else is
not.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

from _common import add_config_args, add_device_args, build_config, fail, init_logging

from rallyvision.checkpoint import load_model_weights
from rallyvision.config import resolve_path
from rallyvision.data.dataset import build_dataloader, build_dataset
from rallyvision.data.manifest import DatasetManifest
from rallyvision.device import DeviceError, guard_training_device, resolve_device_from_config
from rallyvision.engine.trainer import Trainer
from rallyvision.logging_utils import banner
from rallyvision.losses import build_loss
from rallyvision.models import build_model
from rallyvision.paths import RunPaths, default_run_name, get_run_paths
from rallyvision.utils import count_parameters, set_seed, write_json


SMOKE_OVERRIDES: Dict[str, Any] = {
    "train.epochs": 1,
    "train.max_val_batches": 2,
    "train.val_every": 1,
    "train.checkpoint_every": 1,
    "data.loader.batch_size": 2,
    "data.loader.num_workers": 0,
    "data.loader.persistent_workers": False,
    "data.augment.enabled": False,
    "model.width": 0.25,
    "runtime.amp": False,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_config_args(parser)
    add_device_args(parser)
    parser.add_argument("--run-name", default=None, help="run directory name (default: timestamped)")
    parser.add_argument("--epochs", type=int, default=None, help="override train.epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="override data.loader.batch_size")
    parser.add_argument("--lr", type=float, default=None, help="override optim.lr")
    parser.add_argument("--num-workers", type=int, default=None, help="override data.loader.num_workers")
    parser.add_argument("--seed", type=int, default=None, help="override project.seed")
    parser.add_argument(
        "--resume",
        nargs="?",
        const="auto",
        default=None,
        metavar="CHECKPOINT",
        help="resume training; bare --resume picks up last.pt from the run directory",
    )
    parser.add_argument(
        "--init-weights",
        type=Path,
        default=None,
        help="start from these weights but reset the epoch/optimizer state",
    )
    parser.add_argument("--prepared-dir", type=Path, default=None, help="override data.prepared_dir")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="tiny end-to-end pipeline check; safe to run locally",
    )
    parser.add_argument(
        "--smoke-batches", type=int, default=3, help="training batches to run in --smoke mode"
    )
    parser.add_argument(
        "--allow-local-training",
        action="store_true",
        help="bypass the CUDA requirement (do not use for real training)",
    )
    parser.add_argument(
        "--no-eval",
        action="store_true",
        help="skip the final evaluation pass on the best checkpoint",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    overrides: Dict[str, Any] = {}
    if args.epochs is not None:
        overrides["train.epochs"] = args.epochs
    if args.batch_size is not None:
        overrides["data.loader.batch_size"] = args.batch_size
    if args.lr is not None:
        overrides["optim.lr"] = args.lr
    if args.num_workers is not None:
        overrides["data.loader.num_workers"] = args.num_workers
    if args.smoke:
        overrides.update(SMOKE_OVERRIDES)
        overrides["train.max_train_batches"] = max(1, args.smoke_batches)
        if args.batch_size is not None:
            overrides["data.loader.batch_size"] = args.batch_size

    cfg = build_config(args, overrides)

    run_name = args.run_name or ("smoke_" + default_run_name("run") if args.smoke else default_run_name())
    paths: RunPaths = get_run_paths(cfg, run_name, create=True)
    logger = init_logging(args, log_file=paths.train_log)

    set_seed(int(cfg.project.seed), deterministic=bool(cfg.runtime.get("deterministic", False)))

    # -- device ----------------------------------------------------------
    try:
        device_info = resolve_device_from_config(cfg)
        guard_training_device(device_info, cfg, allow_local=args.smoke or args.allow_local_training)
    except DeviceError as exc:
        fail(str(exc))
        return 1

    if args.smoke:
        logger.info(banner("SMOKE TEST - not real training"))

    # -- data ------------------------------------------------------------
    prepared_dir = Path(args.prepared_dir) if args.prepared_dir else resolve_path(cfg, cfg.data.prepared_dir)
    try:
        manifest = DatasetManifest.load(prepared_dir)
    except FileNotFoundError as exc:
        fail(f"{exc}\nRun scripts/download_dataset.py then scripts/prepare_dataset.py first.")
        return 1

    train_dataset = build_dataset(cfg, manifest, split="train")
    val_dataset = build_dataset(cfg, manifest, split="val")
    if len(train_dataset) == 0:
        fail("the training split is empty; check data.split in config.yaml")
        return 1

    logger.info(banner("dataset"))
    for line in manifest.format_stats(int(cfg.data.num_frames), int(cfg.data.frame_stride)).splitlines():
        logger.info(line)
    logger.info("train samples    : %d", len(train_dataset))
    logger.info("val samples      : %d", len(val_dataset))
    logger.info("train games      : %s", sorted({c.game for c in manifest.split_clips("train")}))
    logger.info("val games        : %s", sorted({c.game for c in manifest.split_clips("val")}))

    problems = manifest.split_problems(int(cfg.data.num_frames), int(cfg.data.frame_stride))
    if problems and not args.smoke:
        fail(
            "refusing to train: the train/val split cannot give a held-out score.\n  - "
            + "\n  - ".join(problems)
            + "\nRebuild it: python scripts/prepare_dataset.py --overwrite --cache-frames"
        )
        return 1

    train_loader = build_dataloader(cfg, train_dataset, split="train", device_kind=device_info.kind)
    val_loader = (
        build_dataloader(cfg, val_dataset, split="val", shuffle=False, device_kind=device_info.kind)
        if len(val_dataset)
        else None
    )
    if val_loader is None:
        if args.smoke:
            logger.warning("no validation samples; smoke test will write last.pt only")
        else:
            fail(
                "the validation split is empty, so training cannot save best.pt "
                "(best.pt is the highest val/accuracy_at_5px snapshot, not last.pt). "
                "Check data.split in config.yaml."
            )
            return 1

    # -- model -----------------------------------------------------------
    model = build_model(cfg)
    if args.init_weights:
        load_model_weights(args.init_weights, model)
        logger.info("initialized weights from %s", args.init_weights)
    loss_fn = build_loss(cfg)

    cfg["project.run_name"] = run_name
    cfg.save(paths.config_file)
    write_json(
        paths.root / "run_info.json",
        {
            "run_name": run_name,
            "device": {"kind": device_info.kind, "name": device_info.name, **device_info.details},
            "parameters": count_parameters(model),
            "train_samples": len(train_dataset),
            "val_samples": len(val_dataset),
            "smoke": bool(args.smoke),
        },
    )

    # -- train -----------------------------------------------------------
    trainer = Trainer(
        cfg=cfg,
        model=model,
        loss_fn=loss_fn,
        train_loader=train_loader,
        val_loader=val_loader,
        device_info=device_info,
        paths=paths,
    )
    trainer.maybe_resume(args.resume)
    state = trainer.train()

    # -- final evaluation on the best checkpoint --------------------------
    # Smoke runs skip it: a full pass over the real val split takes many minutes
    # and says nothing about whether the pipeline is wired correctly.
    if args.smoke:
        logger.info("smoke test: skipping the final full-val evaluation")
    elif not args.no_eval and val_loader is not None and paths.best_checkpoint.is_file():
        from rallyvision.engine.evaluator import evaluate_and_save
        from rallyvision.metrics import format_metrics

        logger.info(banner("final evaluation (best checkpoint)"))
        load_model_weights(paths.best_checkpoint, trainer.model, map_location=device_info.device)
        metrics, _ = evaluate_and_save(
            trainer.model,
            val_loader,
            device_info,
            cfg,
            output_dir=paths.metrics,
            prefix="val",
            save_per_sample=bool(cfg.eval.get("save_per_sample", True)),
        )
        for line in format_metrics(metrics).splitlines():
            logger.info(line)

    print()
    print(banner("done"))
    print(f"run directory : {paths.root}")
    print(f"best checkpoint: {paths.best_checkpoint if paths.best_checkpoint.is_file() else 'n/a'}")
    print(f"last checkpoint : {paths.last_checkpoint if paths.last_checkpoint.is_file() else 'n/a'}")
    print("download best.pt (val-selected), not last.pt")
    print(f"epochs completed: {state.epoch}")
    if state.best_epoch is not None:
        print(f"best epoch      : {state.best_epoch} ({state.monitor}={state.best_metric})")
    if args.smoke:
        print("\nThis was a SMOKE TEST. Real training must run on Colab with a GPU.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
