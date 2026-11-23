"""Training and evaluation loops."""

from rallyvision.ball_tracking.evaluation.evaluator import evaluate, evaluate_and_save
from rallyvision.ball_tracking.training.engine.optim import build_optimizer, build_scaler, build_scheduler
from rallyvision.ball_tracking.training.engine.trainer import Trainer

__all__ = [
    "Trainer",
    "build_optimizer",
    "build_scaler",
    "build_scheduler",
    "evaluate",
    "evaluate_and_save",
]
