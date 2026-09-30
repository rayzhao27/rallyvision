"""Latency / FPS / size benchmarking on Apple Silicon.

This measures deployment performance, which is why it belongs on the Mac and
never in Colab: the number that matters is what the Neural Engine and GPU in
this specific machine can do.
"""

from __future__ import annotations

import platform
import statistics
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from ..logging_utils import get_logger
from ..utils import format_bytes, write_json
from .coreml_export import coreml_predict, load_coreml_model, model_size_bytes

logger = get_logger(__name__)

DEFAULT_COMPUTE_UNITS = ("ALL", "CPU_AND_GPU", "CPU_AND_NE", "CPU_ONLY")


def _latency_stats(samples_ms: Sequence[float]) -> Dict[str, float]:
    ordered = sorted(samples_ms)
    n = len(ordered)
    return {
        "iterations": n,
        "mean_ms": float(statistics.fmean(ordered)),
        "median_ms": float(statistics.median(ordered)),
        "min_ms": float(ordered[0]),
        "max_ms": float(ordered[-1]),
        "stdev_ms": float(statistics.pstdev(ordered)) if n > 1 else 0.0,
        "p95_ms": float(ordered[min(n - 1, int(round(0.95 * (n - 1))))]),
        "p99_ms": float(ordered[min(n - 1, int(round(0.99 * (n - 1))))]),
        "fps": float(1000.0 / statistics.fmean(ordered)) if statistics.fmean(ordered) > 0 else 0.0,
    }


def benchmark_coreml(
    coreml_path: Path | str,
    input_shape: Tuple[int, int, int, int],
    compute_units: str = "ALL",
    warmup_iters: int = 10,
    iters: int = 100,
    seed: int = 0,
) -> Dict[str, Any]:
    """Time single-batch Core ML predictions."""
    mlmodel = load_coreml_model(coreml_path, compute_units=compute_units)
    rng = np.random.default_rng(seed)
    batch = rng.random(input_shape, dtype=np.float32)

    for _ in range(max(0, warmup_iters)):
        coreml_predict(mlmodel, batch)

    timings: List[float] = []
    for _ in range(max(1, iters)):
        start = time.perf_counter()
        coreml_predict(mlmodel, batch)
        timings.append((time.perf_counter() - start) * 1000.0)

    stats = _latency_stats(timings)
    size = model_size_bytes(coreml_path)
    return {
        "backend": "coreml",
        "model_path": str(coreml_path),
        "compute_units": compute_units,
        "input_shape": list(input_shape),
        "model_size_bytes": size,
        "model_size_human": format_bytes(size),
        "warmup_iters": warmup_iters,
        **stats,
    }


def benchmark_torch(
    model: torch.nn.Module,
    input_shape: Tuple[int, int, int, int],
    device: str = "cpu",
    warmup_iters: int = 5,
    iters: int = 30,
    seed: int = 0,
) -> Dict[str, Any]:
    """Time the PyTorch model, as a baseline to compare Core ML against."""
    torch_device = torch.device(device)
    model = model.eval().to(torch_device)
    generator = torch.Generator().manual_seed(seed)
    batch = torch.rand(*input_shape, generator=generator).to(torch_device)

    def _sync() -> None:
        if torch_device.type == "mps" and hasattr(torch, "mps"):
            torch.mps.synchronize()
        elif torch_device.type == "cuda":
            torch.cuda.synchronize()

    with torch.no_grad():
        for _ in range(max(0, warmup_iters)):
            model.forward_logits(batch)
        _sync()

        timings: List[float] = []
        for _ in range(max(1, iters)):
            start = time.perf_counter()
            model.forward_logits(batch)
            _sync()
            timings.append((time.perf_counter() - start) * 1000.0)

    params = sum(p.numel() for p in model.parameters())
    return {
        "backend": f"torch:{device}",
        "input_shape": list(input_shape),
        "parameters": params,
        "model_size_bytes": params * 4,
        "model_size_human": format_bytes(params * 4),
        "warmup_iters": warmup_iters,
        **_latency_stats(timings),
    }


def system_info() -> Dict[str, Any]:
    info = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python": platform.python_version(),
        "torch": torch.__version__,
    }
    try:
        import subprocess

        brand = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True, timeout=5
        )
        if brand.returncode == 0:
            info["cpu"] = brand.stdout.strip()
    except Exception:  # noqa: BLE001 - informational only
        pass
    try:
        import coremltools as ct

        info["coremltools"] = ct.__version__
    except ImportError:
        pass
    return info


def run_benchmark_suite(
    coreml_models: Dict[str, Path],
    input_shape: Tuple[int, int, int, int],
    compute_units: Sequence[str] = ("ALL",),
    warmup_iters: int = 10,
    iters: int = 100,
    torch_model: Optional[torch.nn.Module] = None,
    torch_devices: Sequence[str] = (),
    output_json: Optional[Path | str] = None,
) -> Dict[str, Any]:
    """Benchmark every (model, compute unit) pair plus optional torch baselines."""
    results: List[Dict[str, Any]] = []

    for label, path in coreml_models.items():
        for units in compute_units:
            logger.info("benchmarking %s on %s ...", label, units)
            try:
                record = benchmark_coreml(
                    path,
                    input_shape,
                    compute_units=units,
                    warmup_iters=warmup_iters,
                    iters=iters,
                )
            except Exception as exc:  # noqa: BLE001 - keep going through the matrix
                logger.warning("  failed: %s", exc)
                results.append({"backend": "coreml", "label": label, "compute_units": units, "error": str(exc)})
                continue
            record["label"] = label
            results.append(record)
            logger.info(
                "  %s / %s: %.2f ms median, %.1f FPS, %s",
                label,
                units,
                record["median_ms"],
                record["fps"],
                record["model_size_human"],
            )

    if torch_model is not None:
        for device in torch_devices:
            logger.info("benchmarking torch on %s ...", device)
            try:
                record = benchmark_torch(
                    torch_model, input_shape, device=device, warmup_iters=5, iters=max(10, iters // 5)
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("  failed: %s", exc)
                continue
            record["label"] = f"torch-{device}"
            results.append(record)
            logger.info("  torch/%s: %.2f ms median, %.1f FPS", device, record["median_ms"], record["fps"])

    report = {"system": system_info(), "input_shape": list(input_shape), "results": results}
    if output_json:
        write_json(output_json, report)
    return report


def format_benchmark_table(report: Dict[str, Any]) -> str:
    """Fixed-width table of the benchmark matrix."""
    rows = [r for r in report.get("results", []) if "error" not in r]
    if not rows:
        return "no successful benchmark runs"

    width = max(14, *(len(str(r.get("label", r.get("backend", "?")))) for r in rows))
    header = f"{'model':<{width}} {'compute':<13} {'median ms':>10} {'p95 ms':>9} {'FPS':>8} {'size':>10}"
    lines = [header, "-" * len(header)]
    for r in rows:
        lines.append(
            f"{str(r.get('label', r.get('backend', '?'))):<{width}} "
            f"{str(r.get('compute_units', r.get('backend', '-')))[:13]:<13} "
            f"{r['median_ms']:>10.2f} {r['p95_ms']:>9.2f} {r['fps']:>8.1f} "
            f"{r.get('model_size_human', '-'):>10}"
        )
    return "\n".join(lines)
