#!/usr/bin/env python3
"""Check that an exported Core ML model matches its PyTorch source numerically.

    LOCAL APPLE SILICON MAC JOB.

    python scripts/validate_coreml.py --model artifacts/coreml/rallyvision_ball_fp16.mlpackage --run v1
"""

from __future__ import annotations

import argparse
from pathlib import Path

from rallyvision.common.cli import add_config_args, build_config, fail, init_logging

from rallyvision.common.checkpoint import load_model_weights
from rallyvision.common.config import resolve_path
from rallyvision.ball_tracking.export.coreml_export import CoreMLUnavailable, check_macos
from rallyvision.ball_tracking.export.coreml_validate import format_validation_report, validate_coreml
from rallyvision.common.logging_utils import banner
from rallyvision.ball_tracking.model import build_model
from rallyvision.common.paths import resolve_run


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_config_args(parser)
    parser.add_argument("--model", type=Path, required=True, help="path to the .mlpackage")
    parser.add_argument("--run", default=None, help="run name, run directory, or 'latest'")
    parser.add_argument("--checkpoint", type=Path, default=None, help="explicit checkpoint path")
    parser.add_argument("--num-samples", type=int, default=None, help="override coreml.validate.num_samples")
    parser.add_argument("--batch-size", type=int, default=1, help="batch dimension used at export time")
    parser.add_argument("--compute-units", default="ALL", help="ALL | CPU_AND_GPU | CPU_AND_NE | CPU_ONLY")
    parser.add_argument("--random-inputs", action="store_true", help="use random tensors, not real frames")
    parser.add_argument("--output", type=Path, default=None, help="where to write the JSON report")
    args = parser.parse_args()

    logger = init_logging(args)
    cfg = build_config(args)

    try:
        check_macos()
    except CoreMLUnavailable as exc:
        fail(str(exc))
        return 1

    if not args.model.exists():
        fail(f"Core ML model not found: {args.model}")
        return 1

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

    model = build_model(cfg)
    load_model_weights(checkpoint, model)
    model.eval()

    input_shape = (
        int(args.batch_size),
        int(cfg.data.num_frames) * 3,
        int(cfg.data.input_height),
        int(cfg.data.input_width),
    )
    num_samples = int(args.num_samples or cfg.coreml.validate.get("num_samples", 8))

    samples = None
    if not args.random_inputs:
        from rallyvision.ball_tracking.data.dataset import build_dataset
        from rallyvision.ball_tracking.data.manifest import DatasetManifest
        from rallyvision.ball_tracking.export.coreml_validate import inputs_from_dataset

        try:
            manifest = DatasetManifest.load(resolve_path(cfg, cfg.data.prepared_dir))
            dataset = build_dataset(cfg, manifest, split="val", augment=False)
            samples = inputs_from_dataset(dataset, num_samples, seed=int(cfg.project.seed))
            logger.info("validating on %d real frames from the val split", len(samples))
        except (FileNotFoundError, ValueError) as exc:
            logger.warning("no prepared dataset available (%s); using random inputs", exc)

    print(banner("core ml validation"))
    report = validate_coreml(
        model,
        args.model,
        input_shape=input_shape,
        samples=samples,
        num_samples=num_samples,
        max_abs_diff=float(cfg.coreml.validate.get("max_abs_diff", 0.02)),
        max_peak_distance_px=float(cfg.coreml.validate.get("max_peak_distance_px", 2.0)),
        detection_threshold=float(cfg.eval.get("detection_threshold", 0.5)),
        compute_units=args.compute_units,
        seed=int(cfg.project.seed),
        output_json=args.output or args.model.parent / f"{args.model.stem}_validation.json",
    )
    print(format_validation_report(report))
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
