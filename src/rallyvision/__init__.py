"""RallyVision: tennis video analysis from three independent ML subsystems and one integration pipeline.

    rallyvision.ball_tracking     ball detection (TrackNet) and tracking
    rallyvision.court_detection   court keypoints and homography
    rallyvision.bounce_detection  bounce events (temporal Transformer v3 on a frozen E2E-Spot backbone)
    rallyvision.pipeline          runs the three and fuses their outputs
    rallyvision.common            configuration, paths, devices, shared TrackNet and heatmap code

Module names from before the 2026-11-22 module-first refactor (e.g. ``rallyvision.court``,
``rallyvision.tracker``) still import, so archived experiments and Colab bundles keep working;
new code should use the module paths above.
"""

import importlib
import importlib.abc
import importlib.util
import sys

__version__ = "0.1.0"

__all__ = ["__version__"]

_LEGACY = {
    "rallyvision.config": "rallyvision.common.config",
    "rallyvision.paths": "rallyvision.common.paths",
    "rallyvision.device": "rallyvision.common.device",
    "rallyvision.checkpoint": "rallyvision.common.checkpoint",
    "rallyvision.logging_utils": "rallyvision.common.logging_utils",
    "rallyvision.utils": "rallyvision.common.utils",
    "rallyvision.environment": "rallyvision.common.environment",
    "rallyvision.artifacts": "rallyvision.common.artifacts",
    "rallyvision.models": "rallyvision.ball_tracking.model",
    "rallyvision.models.tracknet": "rallyvision.common.tracknet",
    "rallyvision.losses": "rallyvision.ball_tracking.model.losses",
    "rallyvision.data": "rallyvision.ball_tracking.data",
    "rallyvision.data.heatmap": "rallyvision.common.heatmap",
    "rallyvision.data.dataset": "rallyvision.ball_tracking.data.dataset",
    "rallyvision.data.labels": "rallyvision.ball_tracking.data.labels",
    "rallyvision.data.manifest": "rallyvision.ball_tracking.data.manifest",
    "rallyvision.data.prepare": "rallyvision.ball_tracking.data.prepare",
    "rallyvision.data.sources": "rallyvision.ball_tracking.data.sources",
    "rallyvision.data.synthetic": "rallyvision.ball_tracking.data.synthetic",
    "rallyvision.data.transforms": "rallyvision.ball_tracking.data.transforms",
    "rallyvision.engine": "rallyvision.ball_tracking.training.engine",
    "rallyvision.engine.trainer": "rallyvision.ball_tracking.training.engine.trainer",
    "rallyvision.engine.optim": "rallyvision.ball_tracking.training.engine.optim",
    "rallyvision.engine.evaluator": "rallyvision.ball_tracking.evaluation.evaluator",
    "rallyvision.metrics": "rallyvision.ball_tracking.evaluation.metrics",
    "rallyvision.tracker": "rallyvision.ball_tracking.inference.tracker",
    "rallyvision.ball_select": "rallyvision.ball_tracking.inference.ball_select",
    "rallyvision.inference": "rallyvision.ball_tracking.inference",
    "rallyvision.inference.video": "rallyvision.ball_tracking.inference.video",
    "rallyvision.viz": "rallyvision.ball_tracking.viz",
    "rallyvision.export": "rallyvision.ball_tracking.export",
    "rallyvision.export.benchmark": "rallyvision.ball_tracking.export.benchmark",
    "rallyvision.export.coreml_export": "rallyvision.ball_tracking.export.coreml_export",
    "rallyvision.export.coreml_validate": "rallyvision.ball_tracking.export.coreml_validate",
    "rallyvision.court": "rallyvision.court_detection",
    "rallyvision.bounce": "rallyvision.bounce_detection.experiments.gbdt.bounce",
    "rallyvision.events": "rallyvision.bounce_detection.experiments.event_sequence.events",
    "rallyvision.event_tf": "rallyvision.bounce_detection.experiments.previous_transformers.event_tf",
    "rallyvision.e2e_tennis": "rallyvision.bounce_detection.experiments.previous_transformers.e2e_tennis",
}


class _LegacyNames(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    """Resolve a pre-refactor module name to the module it moved to (the same module object)."""

    def find_spec(self, fullname, path, target=None):
        if fullname in _LEGACY:
            return importlib.util.spec_from_loader(fullname, self)
        return None

    def create_module(self, spec):
        return importlib.import_module(_LEGACY[spec.name])

    def exec_module(self, module):
        pass


if not any(isinstance(f, _LegacyNames) for f in sys.meta_path):
    sys.meta_path.insert(0, _LegacyNames())
