"""Numerical validation of a Core ML model against its PyTorch source.

Raw tensor diff is necessary but not sufficient: fp16 conversion can shift
individual heatmap values noticeably while leaving the decoded ball position
untouched, and it is the decoded position that the app actually uses. So both
are measured, and the peak distance is the metric that matters.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from rallyvision.common.heatmap import decode_peak
from rallyvision.common.logging_utils import get_logger
from rallyvision.common.utils import write_json
from rallyvision.ball_tracking.export.coreml_export import INPUT_NAME, coreml_predict, load_coreml_model, model_size_bytes

logger = get_logger(__name__)


def _random_inputs(n: int, shape: Tuple[int, int, int, int], seed: int = 0) -> List[np.ndarray]:
    rng = np.random.default_rng(seed)
    return [rng.random(shape, dtype=np.float32) for _ in range(n)]


def inputs_from_dataset(dataset, num_samples: int, seed: int = 0) -> List[np.ndarray]:
    """Sample real frame stacks; synthetic noise is a much weaker test."""
    if len(dataset) == 0:
        raise ValueError("dataset is empty")
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(dataset), size=min(num_samples, len(dataset)), replace=False)
    return [dataset[int(i)]["frames"].unsqueeze(0).numpy().astype(np.float32) for i in indices]


def validate_coreml(
    torch_model: torch.nn.Module,
    coreml_path: Path | str,
    input_shape: Tuple[int, int, int, int],
    samples: Optional[Sequence[np.ndarray]] = None,
    num_samples: int = 8,
    max_abs_diff: float = 0.02,
    max_peak_distance_px: float = 2.0,
    detection_threshold: float = 0.5,
    compute_units: str = "ALL",
    seed: int = 0,
    output_json: Optional[Path | str] = None,
) -> Dict[str, Any]:
    """Compare PyTorch and Core ML outputs; returns a report with `passed`."""
    mlmodel = load_coreml_model(coreml_path, compute_units=compute_units)
    torch_model = torch_model.eval().to("cpu")

    if samples is None:
        samples = _random_inputs(num_samples, input_shape, seed=seed)
        source = "random"
    else:
        samples = list(samples)[:num_samples]
        source = "dataset"
    if not samples:
        raise ValueError("no validation samples")

    records: List[Dict[str, Any]] = []
    for i, batch in enumerate(samples):
        with torch.no_grad():
            torch_out = torch.sigmoid(
                torch_model.forward_logits(torch.from_numpy(batch))
            ).numpy()
        coreml_out = coreml_predict(mlmodel, batch)
        if coreml_out.shape != torch_out.shape:
            coreml_out = coreml_out.reshape(torch_out.shape)

        diff = np.abs(torch_out - coreml_out)
        t_peak, t_score = decode_peak(torch_out[0], threshold=detection_threshold)
        c_peak, c_score = decode_peak(coreml_out[0], threshold=detection_threshold)

        if t_peak is not None and c_peak is not None:
            peak_distance = float(np.hypot(t_peak[0] - c_peak[0], t_peak[1] - c_peak[1]))
        elif t_peak is None and c_peak is None:
            peak_distance = 0.0
        else:
            peak_distance = float("inf")  # one detected, the other did not

        records.append(
            {
                "sample": i,
                "max_abs_diff": float(diff.max()),
                "mean_abs_diff": float(diff.mean()),
                "torch_peak": list(t_peak) if t_peak else None,
                "coreml_peak": list(c_peak) if c_peak else None,
                "torch_score": float(t_score),
                "coreml_score": float(c_score),
                "peak_distance_px": peak_distance,
            }
        )

    max_diff = max(r["max_abs_diff"] for r in records)
    mean_diff = float(np.mean([r["mean_abs_diff"] for r in records]))
    finite_distances = [r["peak_distance_px"] for r in records if np.isfinite(r["peak_distance_px"])]
    worst_distance = max((r["peak_distance_px"] for r in records), default=0.0)
    detection_mismatches = sum(1 for r in records if not np.isfinite(r["peak_distance_px"]))

    passed = (
        max_diff <= max_abs_diff
        and detection_mismatches == 0
        and worst_distance <= max_peak_distance_px
    )
    report = {
        "passed": bool(passed),
        "coreml_path": str(coreml_path),
        "model_size_bytes": model_size_bytes(coreml_path),
        "input_name": INPUT_NAME,
        "input_shape": list(input_shape),
        "num_samples": len(records),
        "sample_source": source,
        "compute_units": compute_units,
        "max_abs_diff": max_diff,
        "mean_abs_diff": mean_diff,
        "worst_peak_distance_px": worst_distance,
        "mean_peak_distance_px": float(np.mean(finite_distances)) if finite_distances else 0.0,
        "detection_mismatches": detection_mismatches,
        "thresholds": {
            "max_abs_diff": max_abs_diff,
            "max_peak_distance_px": max_peak_distance_px,
        },
        "samples": records,
    }

    logger.info(
        "core ml validation %s: max|diff| %.5f (limit %.5f), worst peak shift %.3f px (limit %.3f), "
        "%d detection mismatch(es)",
        "PASSED" if passed else "FAILED",
        max_diff,
        max_abs_diff,
        worst_distance,
        max_peak_distance_px,
        detection_mismatches,
    )
    if output_json:
        write_json(output_json, report)
    return report


def format_validation_report(report: Dict[str, Any]) -> str:
    return "\n".join(
        [
            f"status              : {'PASS' if report['passed'] else 'FAIL'}",
            f"samples             : {report['num_samples']} ({report['sample_source']})",
            f"max |torch-coreml|  : {report['max_abs_diff']:.6f} "
            f"(limit {report['thresholds']['max_abs_diff']})",
            f"mean |torch-coreml| : {report['mean_abs_diff']:.6f}",
            f"worst peak shift    : {report['worst_peak_distance_px']:.3f} px "
            f"(limit {report['thresholds']['max_peak_distance_px']})",
            f"detection mismatches: {report['detection_mismatches']}",
        ]
    )
