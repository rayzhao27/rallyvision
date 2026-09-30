#!/usr/bin/env python3
"""Measure Core ML latency, FPS and model size on this Mac.

    LOCAL APPLE SILICON MAC JOB. These numbers describe deployment
    performance on Apple hardware, which is exactly why they cannot be
    measured in Colab.

    python scripts/benchmark_coreml.py --model-dir artifacts/coreml
    python scripts/benchmark_coreml.py --model m.mlpackage --compute-units ALL CPU_AND_NE
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

from _common import add_config_args, build_config, fail, init_logging

from rallyvision.config import output_root, resolve_path
from rallyvision.export.benchmark import format_benchmark_table, run_benchmark_suite, system_info
from rallyvision.export.coreml_export import CoreMLUnavailable, check_macos
from rallyvision.logging_utils import banner
from rallyvision.utils import write_csv


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_config_args(parser)
    parser.add_argument("--model", type=Path, action="append", default=[], help="a .mlpackage (repeatable)")
    parser.add_argument("--model-dir", type=Path, default=None, help="benchmark every .mlpackage in this dir")
    parser.add_argument(
        "--compute-units",
        nargs="+",
        default=["ALL", "CPU_AND_GPU", "CPU_AND_NE", "CPU_ONLY"],
        help="Core ML compute unit configurations to compare",
    )
    parser.add_argument("--iters", type=int, default=None, help="override coreml.benchmark.iters")
    parser.add_argument("--warmup", type=int, default=None, help="override coreml.benchmark.warmup_iters")
    parser.add_argument("--batch-size", type=int, default=1, help="batch dimension used at export time")
    parser.add_argument(
        "--torch-baseline",
        nargs="*",
        default=None,
        metavar="DEVICE",
        help="also benchmark the PyTorch model, e.g. --torch-baseline cpu mps (needs --checkpoint)",
    )
    parser.add_argument("--checkpoint", type=Path, default=None, help="checkpoint for the torch baseline")
    parser.add_argument("--output", type=Path, default=None, help="where to write the JSON report")
    args = parser.parse_args()

    logger = init_logging(args)
    cfg = build_config(args)

    try:
        check_macos()
    except CoreMLUnavailable as exc:
        fail(str(exc))
        return 1

    # -- collect models --------------------------------------------------
    models: Dict[str, Path] = {}
    for path in args.model:
        models[Path(path).stem] = Path(path)
    model_dir = args.model_dir
    if model_dir is None and not models:
        model_dir = resolve_path(cfg, cfg.coreml.get("output_dir", output_root(cfg) / "coreml"))
    if model_dir is not None:
        for path in sorted(Path(model_dir).glob("*.mlpackage")):
            models[path.stem] = path
    if not models:
        fail(
            f"no .mlpackage found (looked in {model_dir}).\n"
            "Run scripts/export_coreml.py first."
        )
        return 1

    input_shape = (
        int(args.batch_size),
        int(cfg.data.num_frames) * 3,
        int(cfg.data.input_height),
        int(cfg.data.input_width),
    )
    iters = int(args.iters or cfg.coreml.benchmark.get("iters", 100))
    warmup = int(args.warmup if args.warmup is not None else cfg.coreml.benchmark.get("warmup_iters", 10))

    torch_model = None
    torch_devices = tuple(args.torch_baseline or ())
    if torch_devices:
        if args.checkpoint is None or not args.checkpoint.is_file():
            fail("--torch-baseline needs --checkpoint pointing at a trained .pt")
            return 1
        from rallyvision.checkpoint import load_model_weights
        from rallyvision.models import build_model

        torch_model = build_model(cfg)
        load_model_weights(args.checkpoint, torch_model)

    print(banner("benchmark environment"))
    for key, value in system_info().items():
        print(f"{key:<14}: {value}")
    print()
    logger.info("input shape %s, %d iterations (%d warmup)", input_shape, iters, warmup)

    output = args.output or (Path(model_dir) if model_dir else Path(".")) / "benchmark.json"
    report = run_benchmark_suite(
        models,
        input_shape=input_shape,
        compute_units=args.compute_units,
        warmup_iters=warmup,
        iters=iters,
        torch_model=torch_model,
        torch_devices=torch_devices,
        output_json=output,
    )

    rows = [
        {
            "model": r.get("label"),
            "compute_units": r.get("compute_units", r.get("backend")),
            "median_ms": round(r["median_ms"], 3),
            "mean_ms": round(r["mean_ms"], 3),
            "p95_ms": round(r["p95_ms"], 3),
            "fps": round(r["fps"], 2),
            "model_size_bytes": r.get("model_size_bytes"),
        }
        for r in report["results"]
        if "error" not in r
    ]
    csv_path = Path(output).with_suffix(".csv")
    write_csv(csv_path, rows)

    print()
    print(banner("results"))
    print(format_benchmark_table(report))
    print()
    print(f"json: {output}")
    print(f"csv : {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
