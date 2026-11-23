"""The training/validation dataset.

One sample is a stack of ``num_frames`` consecutive frames; the label is the
ball position in the **last** frame of the stack. That matches how the tracker
runs on video (predict the current frame from the current plus previous frames)
and keeps training and inference geometry identical.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from rallyvision.common.config import Config, resolve_path
from rallyvision.common.utils import seed_generator, worker_init_fn
from rallyvision.common.heatmap import render_gaussian
from rallyvision.ball_tracking.data.manifest import ClipRecord, DatasetManifest
from rallyvision.ball_tracking.data.transforms import (
    AugmentConfig,
    StackAugmentor,
    bgr_to_rgb,
    frames_to_tensor_array,
    resize_frame,
)


class BallDataset(Dataset):
    """Temporal frame stacks with gaussian heatmap targets."""

    def __init__(
        self,
        manifest: DatasetManifest,
        split: str = "train",
        num_frames: int = 3,
        frame_stride: int = 1,
        input_height: int = 360,
        input_width: int = 640,
        heatmap_downsample: int = 1,
        heatmap_sigma: float = 2.5,
        heatmap_radius: int = 8,
        drop_invisible: bool = False,
        invisible_keep_fraction: float = 1.0,
        augment: Optional[AugmentConfig] = None,
        seed: int = 1337,
        raw_root: Optional[Path] = None,
        prepared_root: Optional[Path] = None,
    ):
        if num_frames < 1:
            raise ValueError("num_frames must be >= 1")
        if input_height % heatmap_downsample or input_width % heatmap_downsample:
            raise ValueError(
                f"input size {input_width}x{input_height} is not divisible by "
                f"heatmap downsample {heatmap_downsample}"
            )

        self.manifest = manifest
        self.split = split
        self.num_frames = int(num_frames)
        self.frame_stride = int(frame_stride)
        self.input_height = int(input_height)
        self.input_width = int(input_width)
        self.heatmap_downsample = int(heatmap_downsample)
        self.heatmap_height = self.input_height // self.heatmap_downsample
        self.heatmap_width = self.input_width // self.heatmap_downsample
        self.heatmap_sigma = float(heatmap_sigma)
        self.heatmap_radius = int(heatmap_radius)
        self.seed = int(seed)

        self.raw_root = Path(raw_root) if raw_root else Path(manifest.raw_root)
        self.prepared_root = Path(prepared_root) if prepared_root else Path(manifest.prepared_root)

        self.clips: List[ClipRecord] = manifest.split_clips(split)
        self.samples: List[Tuple[int, int]] = self._build_index(
            drop_invisible=drop_invisible,
            invisible_keep_fraction=float(invisible_keep_fraction),
        )

        aug = augment if augment is not None else AugmentConfig(enabled=False)
        self.augmentor = StackAugmentor(aug, rng=random.Random(self.seed))
        self.augment_enabled = aug.enabled

    # -- construction -----------------------------------------------------
    def _build_index(
        self, drop_invisible: bool, invisible_keep_fraction: float
    ) -> List[Tuple[int, int]]:
        span = (self.num_frames - 1) * self.frame_stride
        rng = random.Random(self.seed)
        samples: List[Tuple[int, int]] = []
        for ci, clip in enumerate(self.clips):
            for fi in range(span, len(clip)):
                if clip.visible[fi]:
                    samples.append((ci, fi))
                elif drop_invisible:
                    continue
                elif invisible_keep_fraction >= 1.0 or rng.random() < invisible_keep_fraction:
                    samples.append((ci, fi))
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    # -- item loading -----------------------------------------------------
    def _frame_path(self, clip: ClipRecord, index: int) -> Path:
        filename = clip.files[index]
        if clip.cached_dir:
            return self.prepared_root / clip.cached_dir / filename
        return self.raw_root / clip.source_dir / filename

    def _read_frame(self, clip: ClipRecord, index: int) -> np.ndarray:
        path = self._frame_path(clip, index)
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"could not read frame {path}")
        return resize_frame(bgr_to_rgb(image), self.input_width, self.input_height)

    def describe_sample(self, index: int) -> Dict[str, Any]:
        clip_idx, frame_idx = self.samples[index]
        clip = self.clips[clip_idx]
        return {
            "index": index,
            "clip_id": clip.id,
            "game": clip.game,
            "frame_index": frame_idx,
            "frame_file": clip.files[frame_idx],
            "path": str(self._frame_path(clip, frame_idx)),
            "visible": bool(clip.visible[frame_idx]),
        }

    def __getitem__(self, index: int) -> Dict[str, Any]:
        clip_idx, frame_idx = self.samples[index]
        clip = self.clips[clip_idx]

        indices = [
            max(0, frame_idx - offset * self.frame_stride)
            for offset in reversed(range(self.num_frames))
        ]
        frames = [self._read_frame(clip, i) for i in indices]

        # Source -> network-input scale, so predictions can be reported back in
        # original frame coordinates.
        sx = self.input_width / float(clip.width)
        sy = self.input_height / float(clip.height)

        point: Optional[Tuple[float, float]] = None
        if clip.visible[frame_idx]:
            point = (float(clip.x[frame_idx]) * sx, float(clip.y[frame_idx]) * sy)

        if self.augment_enabled:
            frames, point = self.augmentor(frames, point)

        stack = frames_to_tensor_array(frames)
        visible = point is not None

        hs = 1.0 / self.heatmap_downsample
        if visible:
            heatmap = render_gaussian(
                self.heatmap_height,
                self.heatmap_width,
                x=point[0] * hs,
                y=point[1] * hs,
                sigma=self.heatmap_sigma,
                radius=self.heatmap_radius,
            )
        else:
            heatmap = np.zeros((self.heatmap_height, self.heatmap_width), dtype=np.float32)

        target_xy = np.array(point if visible else (-1.0, -1.0), dtype=np.float32)
        return {
            "frames": torch.from_numpy(stack),
            "heatmap": torch.from_numpy(heatmap).unsqueeze(0),
            "visible": torch.tensor(1.0 if visible else 0.0, dtype=torch.float32),
            # Ball position in network-input pixels.
            "target_xy": torch.from_numpy(target_xy),
            # Multiply input-space coords by this to get original-frame coords.
            "to_original": torch.tensor([1.0 / sx, 1.0 / sy], dtype=torch.float32),
            "index": torch.tensor(index, dtype=torch.long),
            "clip_id": clip.id,
            "frame_file": clip.files[frame_idx],
        }


def build_dataset(
    cfg: Config,
    manifest: DatasetManifest,
    split: str = "train",
    augment: Optional[bool] = None,
) -> BallDataset:
    """Create a :class:`BallDataset` from config. Validation is never augmented."""
    data = cfg.data
    use_augment = (split == "train") if augment is None else bool(augment)
    aug_cfg = AugmentConfig.from_config(data.get("augment")) if use_augment else AugmentConfig(enabled=False)
    if not use_augment:
        aug_cfg.enabled = False

    return BallDataset(
        manifest=manifest,
        split=split,
        num_frames=int(data.num_frames),
        frame_stride=int(data.frame_stride),
        input_height=int(data.input_height),
        input_width=int(data.input_width),
        heatmap_downsample=int(data.heatmap.downsample),
        heatmap_sigma=float(data.heatmap.sigma),
        heatmap_radius=int(data.heatmap.radius),
        drop_invisible=bool(data.get("drop_invisible", False)),
        invisible_keep_fraction=float(data.get("invisible_keep_fraction", 1.0)),
        augment=aug_cfg,
        seed=int(cfg.project.seed),
        raw_root=resolve_path(cfg, manifest.raw_root),
        prepared_root=resolve_path(cfg, manifest.prepared_root),
    )


def collate_batch(batch: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Collate tensors, keeping string metadata as plain lists."""
    out: Dict[str, Any] = {}
    for key in batch[0]:
        values = [item[key] for item in batch]
        out[key] = torch.stack(values) if isinstance(values[0], torch.Tensor) else values
    return out


def build_dataloader(
    cfg: Config,
    dataset: BallDataset,
    split: str = "train",
    batch_size: Optional[int] = None,
    shuffle: Optional[bool] = None,
    device_kind: str = "cpu",
) -> DataLoader:
    """Create a DataLoader with settings that are safe on both Colab and macOS."""
    loader_cfg = cfg.data.loader
    batch_size = int(batch_size or loader_cfg.batch_size)
    shuffle = (split == "train") if shuffle is None else bool(shuffle)

    num_workers = int(loader_cfg.get("num_workers", 4))
    # Pinned memory only helps for CUDA transfers, and on MPS it is a pessimization.
    pin_memory = bool(loader_cfg.get("pin_memory", True)) and device_kind == "cuda"
    persistent = bool(loader_cfg.get("persistent_workers", True)) and num_workers > 0

    kwargs: Dict[str, Any] = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "drop_last": shuffle and len(dataset) > batch_size,
        "collate_fn": collate_batch,
        "worker_init_fn": worker_init_fn,
        "generator": seed_generator(int(cfg.project.seed)),
    }
    if num_workers > 0:
        kwargs["persistent_workers"] = persistent
        kwargs["prefetch_factor"] = int(loader_cfg.get("prefetch_factor", 2))
    return DataLoader(dataset, **kwargs)
