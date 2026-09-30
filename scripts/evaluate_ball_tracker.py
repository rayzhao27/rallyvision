#!/usr/bin/env python3
"""Evaluate a trained checkpoint and write metrics as JSON + CSV.

Colab GPU job for the real dataset.

    python scripts/evaluate_ball_tracker.py --run v1
    python scripts/evaluate_ball_tracker.py --checkpoint path/to/best.pt --split val
"""

from __future__ import annotations

import argparse
from pathlib import Path

from _common import add_config_args, add_device_args, build_config, fail, init_logging

from rallyvision.checkpoint import load_model_weights
from rallyvision.config import resolve_path
from rallyvision.data.dataset import build_dataloader, build_dataset
from rallyvision.data.manifest import DatasetManifest
from rallyvision.device import DeviceError, resolve_device_from_config
from rallyvision.engine.evaluator import evaluate_and_save
from rallyvision.logging_utils import banner
from rallyvision.losses import build_loss
from rallyvision.metrics import format_metrics
from rallyvision.models import build_model
from rallyvision.paths import resolve_run
from rallyvision.utils import write_csv


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_config_args(parser)
    add_device_args(parser)
    parser.add_argument("--run", default=None, help="run name, run directory, or 'latest'")
    parser.add_argument("--checkpoint", type=Path, default=None, help="explicit checkpoint path")
    parser.add_argument("--split", default="val", choices=["val", "train", "all"], help="split to evaluate")
    parser.add_argument("--batch-size", type=int, default=None, help="override eval.batch_size")
    parser.add_argument("--threshold", type=float, default=None, help="override eval.detection_threshold")
    parser.add_argument("--prepared-dir", type=Path, default=None, help="override data.prepared_dir")
    parser.add_argument("--output-dir", type=Path, default=None, help="where to write metrics")
    parser.add_argument("--prefix", default=None, help="output file prefix (default: the split name)")
    parser.add_argument(
        "--no-per-sample", action="store_true", help="skip the per-sample CSV"
    )
    parser.add_argument(
        "--use-run-config",
        action="store_true",
        help="load the run's saved config.yaml instead of the repo-root one",
    )
    args = parser.parse_args()

    logger = init_logging(args)
    overrides = {}
    if args.threshold is not None:
        overrides["eval.detection_threshold"] = args.threshold
    cfg = build_config(args, overrides)

    # -- locate the checkpoint -------------------------------------------
    run_paths = None
    checkpoint = args.checkpoint
    if checkpoint is None or args.run is not None:
        try:
            run_paths = resolve_run(cfg, args.run)
        except FileNotFoundError as exc:
            fail(str(exc))
            return 1
    if checkpoint is None:
        checkpoint = run_paths.best_checkpoint
        if not checkpoint.is_file():
            checkpoint = run_paths.last_checkpoint
            logger.warning("no best.pt in %s; evaluating last.pt instead", run_paths.checkpoints)
    if not Path(checkpoint).is_file():
        fail(f"checkpoint not found: {checkpoint}")
        return 1

    if args.use_run_config and run_paths is not None and run_paths.config_file.is_file():
        args.config = run_paths.config_file
        cfg = build_config(args, overrides)
        logger.info("using the run's saved config: %s", run_paths.config_file)

    if args.batch_size is not None:
        cfg["data.loader.batch_size"] = args.batch_size
    else:
        cfg["data.loader.batch_size"] = int(cfg.eval.get("batch_size", cfg.data.loader.batch_size))

    try:
        device_info = resolve_device_from_config(cfg)
    except DeviceError as exc:
        fail(str(exc))
        return 1

    # -- data + model ----------------------------------------------------
    prepared_dir = Path(args.prepared_dir) if args.prepared_dir else resolve_path(cfg, cfg.data.prepared_dir)
    try:
        manifest = DatasetManifest.load(prepared_dir)
    except FileNotFoundError as exc:
        fail(str(exc))
        return 1

    dataset = build_dataset(cfg, manifest, split=args.split, augment=False)
    if len(dataset) == 0:
        stats = manifest.stats(int(cfg.data.num_frames), int(cfg.data.frame_stride))
        train_n = stats["splits"]["train"]["usable_samples"]
        val_n = stats["splits"]["val"]["usable_samples"]
        fail(
            f"split {args.split!r} has no samples "
            f"(train usable={train_n}, val usable={val_n}, clips={len(manifest.clips)}). "
            "The train/val split was empty at prepare time — usually every clip was tagged "
            "as one game. git pull, then:\n"
            "  python scripts/prepare_dataset.py --reassign-splits\n"
            "Then evaluate last.pt. That does not retrain."
        )
        return 1
    loader = build_dataloader(cfg, dataset, split="val", shuffle=False, device_kind=device_info.kind)

    model = build_model(cfg)
    load_model_weights(checkpoint, model)
    model.to(device_info.device)

    output_dir = Path(args.output_dir) if args.output_dir else (
        run_paths.metrics if run_paths is not None else Path(checkpoint).parent
    )
    prefix = args.prefix or args.split

    logger.info(banner("evaluation"))
    logger.info("checkpoint       : %s", checkpoint)
    logger.info("split            : %s (%d samples)", args.split, len(dataset))
    logger.info("device           : %s (%s)", device_info.device, device_info.name)
    logger.info("threshold        : %.3f", float(cfg.eval.get("detection_threshold", 0.5)))

    metrics, accumulator = evaluate_and_save(
        model,
        loader,
        device_info,
        cfg,
        output_dir=output_dir,
        prefix=prefix,
        loss_fn=build_loss(cfg),
        save_per_sample=not args.no_per_sample,
    )

    # A flat one-row CSV, so the README can pick numbers up without parsing JSON.
    write_csv(output_dir / f"{prefix}_metrics.csv", [metrics])

    print()
    print(banner(f"{args.split} metrics"))
    print(format_metrics(metrics))
    print()
    print(f"json: {output_dir / f'{prefix}_metrics.json'}")
    print(f"csv : {output_dir / f'{prefix}_metrics.csv'}")
    if not args.no_per_sample and accumulator.samples:
        print(f"per-sample: {output_dir / f'{prefix}_per_sample.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
