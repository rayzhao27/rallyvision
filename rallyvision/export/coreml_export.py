"""PyTorch -> Core ML conversion. Runs on the local Apple Silicon Mac.

`coremltools` is imported lazily so the rest of the package stays importable in
Colab, where it is neither installed nor wanted.
"""

from __future__ import annotations

import platform
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch

from ..logging_utils import get_logger
from ..utils import count_parameters, dir_size_bytes, format_bytes, write_json

logger = get_logger(__name__)

INPUT_NAME = "frames"
OUTPUT_NAME = "heatmap"


class CoreMLUnavailable(RuntimeError):
    """Raised when coremltools is missing or the platform is not macOS."""


def require_coremltools():
    try:
        import coremltools as ct
    except ImportError as exc:
        raise CoreMLUnavailable(
            "coremltools is not installed. On your Mac: pip install -r requirements-mac.txt"
        ) from exc
    return ct


def check_macos() -> None:
    if platform.system() != "Darwin":
        raise CoreMLUnavailable(
            f"Core ML work must run on macOS; this is {platform.system()}. "
            "Export and benchmark on the Mac, not in Colab."
        )


class _ExportWrapper(torch.nn.Module):
    """Always emits sigmoid probabilities, so the deployed model is self-contained."""

    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.model.forward_logits(frames))


def trace_model(
    model: torch.nn.Module,
    input_shape: Tuple[int, int, int, int],
) -> torch.jit.ScriptModule:
    """TorchScript-trace the model on CPU in eval mode."""
    wrapper = _ExportWrapper(model).eval().to("cpu")
    example = torch.rand(*input_shape, dtype=torch.float32)
    with torch.no_grad():
        traced = torch.jit.trace(wrapper, example, strict=False)
        traced = torch.jit.freeze(traced)
    return traced


def _precision(ct, name: str):
    name = str(name).lower()
    if name in ("fp16", "float16", "half"):
        return ct.precision.FLOAT16
    if name in ("fp32", "float32", "full"):
        return ct.precision.FLOAT32
    raise ValueError(f"unknown coreml precision {name!r}; expected fp16 or fp32")


def _deployment_target(ct, name: str):
    # ct.target members are spelled `iOS17` / `macOS14`, so the lookup has to be
    # case-insensitive rather than a straight getattr.
    wanted = str(name).replace(".", "").replace("_", "").lower()
    available = [t for t in dir(ct.target) if not t.startswith("_")]
    for candidate in available:
        if candidate.lower() == wanted:
            return getattr(ct.target, candidate)
    raise ValueError(f"unknown deployment target {name!r}; available: {sorted(available)}")


def _compute_units(ct, name: str):
    name = str(name).upper()
    if not hasattr(ct.ComputeUnit, name):
        raise ValueError(
            f"unknown compute_units {name!r}; expected one of "
            f"{[u.name for u in ct.ComputeUnit]}"
        )
    return getattr(ct.ComputeUnit, name)


def export_coreml(
    model: torch.nn.Module,
    output_path: Path | str,
    input_shape: Tuple[int, int, int, int],
    precision: str = "fp16",
    minimum_deployment_target: str = "iOS17",
    compute_units: str = "ALL",
    metadata: Optional[Dict[str, Any]] = None,
) -> Path:
    """Convert to an `.mlpackage` and return its path."""
    ct = require_coremltools()
    output_path = Path(output_path)
    if output_path.suffix != ".mlpackage":
        output_path = output_path.with_suffix(".mlpackage")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("tracing model with input shape %s", tuple(input_shape))
    traced = trace_model(model, input_shape)

    logger.info("converting to Core ML (%s, %s)", precision, minimum_deployment_target)
    mlmodel = ct.convert(
        traced,
        convert_to="mlprogram",
        inputs=[ct.TensorType(name=INPUT_NAME, shape=tuple(input_shape), dtype=np.float32)],
        outputs=[ct.TensorType(name=OUTPUT_NAME, dtype=np.float32)],
        compute_precision=_precision(ct, precision),
        minimum_deployment_target=_deployment_target(ct, minimum_deployment_target),
        compute_units=_compute_units(ct, compute_units),
    )

    n, c, h, w = input_shape
    mlmodel.author = "RallyVision"
    mlmodel.short_description = (
        "Tennis ball heatmap detector. Input: a stack of consecutive RGB frames "
        f"({c // 3} frames x 3 channels) at {w}x{h}, values in [0, 1], NCHW. "
        "Output: a single-channel probability heatmap at the same resolution; "
        "the argmax is the ball position in the most recent frame."
    )
    mlmodel.input_description[INPUT_NAME] = f"float32 [{n}, {c}, {h}, {w}], RGB in [0, 1]"
    mlmodel.output_description[OUTPUT_NAME] = f"float32 [{n}, 1, {h}, {w}] ball probability"

    user_metadata = {
        "rallyvision.precision": str(precision),
        "rallyvision.input_shape": str(list(input_shape)),
        "rallyvision.num_frames": str(c // 3),
        "rallyvision.parameters": str(count_parameters(model)["total"]),
    }
    for key, value in (metadata or {}).items():
        user_metadata[f"rallyvision.{key}"] = str(value)
    mlmodel.user_defined_metadata.update(user_metadata)

    mlmodel.save(str(output_path))
    size = dir_size_bytes(output_path) if output_path.is_dir() else output_path.stat().st_size
    logger.info("saved %s (%s)", output_path, format_bytes(size))

    write_json(
        output_path.parent / f"{output_path.stem}_export.json",
        {
            "model_path": str(output_path),
            "size_bytes": size,
            "size_human": format_bytes(size),
            "input_name": INPUT_NAME,
            "output_name": OUTPUT_NAME,
            "input_shape": list(input_shape),
            "precision": precision,
            "minimum_deployment_target": minimum_deployment_target,
            "compute_units": compute_units,
            "torch_version": torch.__version__,
            "coremltools_version": ct.__version__,
            "parameters": count_parameters(model),
            "metadata": metadata or {},
        },
    )
    return output_path


def load_coreml_model(path: Path | str, compute_units: str = "ALL"):
    """Load an `.mlpackage` for validation or benchmarking."""
    ct = require_coremltools()
    check_macos()
    return ct.models.MLModel(str(path), compute_units=_compute_units(ct, compute_units))


def coreml_predict(mlmodel, batch: np.ndarray) -> np.ndarray:
    """Run a single forward pass; returns the heatmap as a numpy array."""
    out = mlmodel.predict({INPUT_NAME: batch.astype(np.float32)})
    return np.asarray(out[OUTPUT_NAME])


def model_size_bytes(path: Path | str) -> int:
    path = Path(path)
    return dir_size_bytes(path) if path.is_dir() else path.stat().st_size
