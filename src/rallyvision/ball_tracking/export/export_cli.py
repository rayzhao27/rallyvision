#!/usr/bin/env python3
"""Convert a trained checkpoint to Core ML.

    LOCAL APPLE SILICON MAC JOB. Do not run this in Colab.

    python scripts/export_coreml.py --run v1 --precision fp16
    python scripts/export_coreml.py --checkpoint best.pt --precision both --validate
"""

from __future__ import annotations

import argparse
from pathlib import Path

from rallyvision.common.cli import add_config_args, build_config, fail, init_logging

from rallyvision.common.checkpoint import load_model_weights
from rallyvision.common.config import output_root, resolve_path
from rallyvision.ball_tracking.export.coreml_export import CoreMLUnavailable, check_macos, export_coreml
from rallyvision.common.logging_utils import banner
from rallyvision.ball_tracking.model import build_model
from rallyvision.common.paths import resolve_run
from rallyvision.common.utils import format_bytes


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_config_args(parser)
    parser.add_argument("--run", default=None, help="run name, run directory, or 'latest'")
    parser.add_argument("--checkpoint", type=Path, default=None, help="explicit checkpoint path")
    parser.add_argument(
        "--precision",
        choices=["fp16", "fp32", "both"],
        default=None,
        help="override coreml.precision; 'both' exports two models for comparison",
    )
    parser.add_argument("--output-dir", type=Path, default=None, help="override coreml.output_dir")
    parser.add_argument("--name", default="rallyvision_ball", help="base name for the exported model")
    parser.add_argument("--batch-size", type=int, default=1, help="fixed batch dimension of the export")
    parser.add_argument(
        "--deployment-target", default=None, help="override coreml.minimum_deployment_target, e.g. iOS16"
    )
    parser.add_argument("--compute-units", default=None, help="override coreml.compute_units")
    parser.add_argument("--validate", action="store_true", help="run numerical validation after export")
    parser.add_argument(
        "--validate-on-dataset",
        action="store_true",
        help="validate on real prepared frames instead of random tensors",
    )
    args = parser.parse_args()

    logger = init_logging(args)
    cfg = build_config(args)

    try:
        check_macos()
    except CoreMLUnavailable as exc:
        fail(str(exc))
        return 1

    # -- checkpoint ------------------------------------------------------
    run_paths = None
    checkpoint = args.checkpoint
    if checkpoint is None:
        try:
            run_paths = resolve_run(cfg, args.run)
        except FileNotFoundError as exc:
            fail(f"{exc}\nTrain on Colab first, then copy the checkpoint here.")
            return 1
        checkpoint = run_paths.best_checkpoint
        if not checkpoint.is_file():
            checkpoint = run_paths.last_checkpoint
    if not Path(checkpoint).is_file():
        fail(f"checkpoint not found: {checkpoint}")
        return 1

    model = build_model(cfg)
    meta = load_model_weights(checkpoint, model)
    model.eval()

    height = int(cfg.data.input_height)
    width = int(cfg.data.input_width)
    channels = int(cfg.data.num_frames) * 3
    input_shape = (int(args.batch_size), channels, height, width)

    precision = args.precision or str(cfg.coreml.get("precision", "fp16"))
    precisions = ["fp32", "fp16"] if precision == "both" else [precision]
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else resolve_path(cfg, cfg.coreml.get("output_dir", output_root(cfg) / "coreml"))
    )
    target = args.deployment_target or str(cfg.coreml.get("minimum_deployment_target", "iOS17"))
    compute_units = args.compute_units or str(cfg.coreml.get("compute_units", "ALL"))

    logger.info(banner("core ml export"))
    logger.info("checkpoint       : %s", checkpoint)
    logger.info("input shape      : %s", input_shape)
    logger.info("precision        : %s", ", ".join(precisions))
    logger.info("deployment target: %s", target)

    trained_epochs = None
    if isinstance(meta.get("state"), dict):
        trained_epochs = meta["state"].get("epoch")

    exported = {}
    for prec in precisions:
        path = export_coreml(
            model,
            output_dir / f"{args.name}_{prec}.mlpackage",
            input_shape=input_shape,
            precision=prec,
            minimum_deployment_target=target,
            compute_units=compute_units,
            metadata={
                "checkpoint": str(checkpoint),
                "epochs_trained": trained_epochs,
                "num_frames": int(cfg.data.num_frames),
                "input_height": height,
                "input_width": width,
            },
        )
        exported[prec] = path

    # -- optional validation ---------------------------------------------
    if args.validate:
        from rallyvision.ball_tracking.export.coreml_validate import format_validation_report, validate_coreml

        samples = None
        if args.validate_on_dataset:
            from rallyvision.ball_tracking.data.dataset import build_dataset
            from rallyvision.ball_tracking.data.manifest import DatasetManifest
            from rallyvision.ball_tracking.export.coreml_validate import inputs_from_dataset

            try:
                manifest = DatasetManifest.load(resolve_path(cfg, cfg.data.prepared_dir))
                dataset = build_dataset(cfg, manifest, split="val", augment=False)
                samples = inputs_from_dataset(
                    dataset, int(cfg.coreml.validate.get("num_samples", 8)), seed=int(cfg.project.seed)
                )
            except (FileNotFoundError, ValueError) as exc:
                logger.warning("falling back to random validation inputs: %s", exc)

        for prec, path in exported.items():
            print()
            print(banner(f"validation: {prec}"))
            report = validate_coreml(
                model,
                path,
                input_shape=input_shape,
                samples=samples,
                num_samples=int(cfg.coreml.validate.get("num_samples", 8)),
                max_abs_diff=float(cfg.coreml.validate.get("max_abs_diff", 0.02)),
                max_peak_distance_px=float(cfg.coreml.validate.get("max_peak_distance_px", 2.0)),
                detection_threshold=float(cfg.eval.get("detection_threshold", 0.5)),
                seed=int(cfg.project.seed),
                output_json=output_dir / f"{args.name}_{prec}_validation.json",
            )
            print(format_validation_report(report))

    print()
    print(banner("exported"))
    for prec, path in exported.items():
        size = sum(f.stat().st_size for f in Path(path).rglob("*") if f.is_file())
        print(f"{prec:<5}: {path}  ({format_bytes(size)})")
    print(f"\nnext: python scripts/benchmark_coreml.py --model-dir {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
