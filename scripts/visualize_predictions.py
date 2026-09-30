#!/usr/bin/env python3
"""Render validation predictions as PNG grids: good cases and failure cases.

    python scripts/visualize_predictions.py --run v1 --num 6
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
from rallyvision.engine.evaluator import evaluate
from rallyvision.logging_utils import banner
from rallyvision.models import build_model
from rallyvision.paths import resolve_run
from rallyvision.viz import plot_predictions, predict_samples


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_config_args(parser)
    add_device_args(parser)
    parser.add_argument("--run", default=None, help="run name, run directory, or 'latest'")
    parser.add_argument("--checkpoint", type=Path, default=None, help="explicit checkpoint path")
    parser.add_argument("--num", type=int, default=6, help="examples per grid")
    parser.add_argument("--split", default="val", choices=["val", "train"], help="split to sample from")
    parser.add_argument("--output-dir", type=Path, default=None, help="where to write the PNGs")
    parser.add_argument("--max-batches", type=int, default=None, help="cap the scoring pass")
    args = parser.parse_args()

    logger = init_logging(args)
    cfg = build_config(args)

    run_paths = None
    checkpoint = args.checkpoint
    if checkpoint is None:
        try:
            run_paths = resolve_run(cfg, args.run)
        except FileNotFoundError as exc:
            fail(str(exc))
            return 1
        checkpoint = run_paths.best_checkpoint
        if not checkpoint.is_file():
            checkpoint = run_paths.last_checkpoint
    if not Path(checkpoint).is_file():
        fail(f"checkpoint not found: {checkpoint}")
        return 1

    try:
        device_info = resolve_device_from_config(cfg)
    except DeviceError as exc:
        fail(str(exc))
        return 1

    try:
        manifest = DatasetManifest.load(resolve_path(cfg, cfg.data.prepared_dir))
    except FileNotFoundError as exc:
        fail(str(exc))
        return 1

    dataset = build_dataset(cfg, manifest, split=args.split, augment=False)
    if len(dataset) == 0:
        fail(f"split {args.split!r} is empty")
        return 1

    model = build_model(cfg)
    load_model_weights(checkpoint, model)
    model.to(device_info.device)

    logger.info(banner("scoring the split to rank examples"))
    loader = build_dataloader(cfg, dataset, split="val", shuffle=False, device_kind=device_info.kind)
    _, accumulator = evaluate(
        model, loader, device_info, cfg, max_batches=args.max_batches, keep_samples=True, progress=True
    )

    output_dir = Path(args.output_dir) if args.output_dir else (
        run_paths.predictions if run_paths is not None else Path("artifacts/predictions")
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    best = [s.index for s in accumulator.best_samples(args.num)]
    worst = [s.index for s in accumulator.worst_samples(args.num)]
    # An untrained or badly-performing model localizes nothing, which would
    # otherwise produce no output at all. Show the first samples instead.
    if not best:
        logger.warning("no localized predictions to rank; showing the first samples instead")
        best = [s.index for s in accumulator.samples[: args.num]]

    written = []
    for label, indices, title in (
        ("good", best, "Best predictions (smallest localization error)"),
        ("failures", worst, "Failure cases (largest error or missed detection)"),
    ):
        if not indices:
            continue
        results = predict_samples(
            model,
            dataset,
            indices,
            device_info.device,
            detection_threshold=float(cfg.eval.get("detection_threshold", 0.5)),
            min_blob_area=int(cfg.eval.get("min_blob_area", 0)),
        )
        path = output_dir / f"{args.split}_{label}.png"
        plot_predictions(results, save_path=path, title=title)
        written.append(path)
        logger.info("wrote %s", path)

    print()
    print(banner("prediction grids"))
    for path in written:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
