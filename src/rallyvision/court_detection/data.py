"""Court keypoint dataset: Roboflow COCO export, video-level split, preprocessing, targets and augmentation."""

from __future__ import annotations

import json
import re
import zlib
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from rallyvision.court_detection.geometry import FLIP_INDEX, INPUT_H, INPUT_W, NUM_KEYPOINTS

# ----------------------------------------------------------------------------- data

def load_coco_split(split_dir: Path) -> List[Dict]:
    """Read one Roboflow COCO split into [{"file", "kps" (14,2), "labeled" (14,)}].

    The export stores keypoints in the order of `categories[].keypoints` (names
    "0", "4", "6", ...); they are reordered here so index k is keypoint "k".
    """
    split_dir = Path(split_dir)
    coco = json.loads((split_dir / "_annotations.coco.json").read_text())

    cats = [c for c in coco["categories"] if c.get("keypoints")]
    if len(cats) != 1:
        raise ValueError(f"expected one keypoint category, found {len(cats)}")
    names = [int(n) for n in cats[0]["keypoints"]]
    if sorted(names) != list(range(NUM_KEYPOINTS)):
        raise ValueError(f"unexpected keypoint names: {names}")
    cat_id = cats[0]["id"]

    anns_by_image: Dict[int, List[Dict]] = {}
    for a in coco["annotations"]:
        if a["category_id"] == cat_id and a.get("keypoints"):
            anns_by_image.setdefault(a["image_id"], []).append(a)

    samples = []
    for img in coco["images"]:
        anns = anns_by_image.get(img["id"], [])
        if len(anns) != 1:          # no court, or ambiguous: skip
            continue
        raw = np.asarray(anns[0]["keypoints"], dtype=np.float64).reshape(-1, 3)
        if raw.shape[0] != NUM_KEYPOINTS:
            raise ValueError(f"{img['file_name']}: {raw.shape[0]} keypoints")
        kps = np.zeros((NUM_KEYPOINTS, 2))
        labeled = np.zeros(NUM_KEYPOINTS, dtype=bool)
        for pos, name in enumerate(names):
            kps[name] = raw[pos, :2]
            labeled[name] = raw[pos, 2] > 0
        # Express coordinates as fractions of the image so any resize is a multiply.
        kps /= [img["width"], img["height"]]
        samples.append({"file": str(split_dir / img["file_name"]), "kps": kps, "labeled": labeled})
    return samples


def video_id(file_name: str) -> str:
    """Group key: the source video of a Roboflow frame.

    "IMG_9400_MOV-0014_jpg.rf.ab12.jpg"     -> "IMG_9400_MOV"
    "1-UMg_jEcdA_3350_png_jpg.rf.ab12.jpg"  -> "1-UMg_jEcdA"   (YouTube id + frame, converted twice)
    """
    name = Path(file_name).name
    name = re.sub(r"\.rf\.[0-9a-zA-Z]+\.\w+$", "", name)
    name = re.sub(r"(_(jpe?g|png))+$", "", name, flags=re.IGNORECASE)
    return re.sub(r"[-_]\d+$", "", name)


def split_by_video(samples: List[Dict], val_frac: float = 0.1, test_frac: float = 0.1) -> Dict[str, List[Dict]]:
    """Deterministic video-level split, so no video has frames in two splits."""
    out = {"train": [], "val": [], "test": []}
    for s in samples:
        u = zlib.crc32(video_id(s["file"]).encode()) % 1000 / 1000.0
        key = "test" if u < test_frac else "val" if u < test_frac + val_frac else "train"
        out[key].append(s)
    return out


def preprocess(image_bgr: np.ndarray) -> torch.Tensor:
    """BGR uint8 (any size) -> RGB float tensor [3, 360, 640] in [0, 1]."""
    img = cv2.resize(image_bgr, (INPUT_W, INPUT_H), interpolation=cv2.INTER_AREA)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return torch.from_numpy(img.astype(np.float32) / 255.0).permute(2, 0, 1)


class CourtDataset(Dataset):
    """Images resized to 640x360, with keypoints [14, 2] in pixels and a labeled mask [14].

    Heatmap targets are rendered on the GPU by `render_targets`: 14 full-size
    heatmaps per image would be ~13 MB to pass through DataLoader workers.
    """

    def __init__(self, samples: List[Dict], augment: bool = False, seed: int = 0):
        self.samples = samples
        self.augment = augment
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i: int):
        s = self.samples[i]
        img = cv2.imread(s["file"])
        if img is None:
            raise FileNotFoundError(s["file"])
        img = cv2.resize(img, (INPUT_W, INPUT_H), interpolation=cv2.INTER_AREA)
        kps = s["kps"] * [INPUT_W, INPUT_H]
        labeled = s["labeled"].copy()
        if self.augment:
            img, kps, labeled = augment(img, kps, labeled, self.rng)
        x = torch.from_numpy(cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0).permute(2, 0, 1)
        return x, torch.from_numpy(kps.astype(np.float32)), torch.from_numpy(labeled)

    def worker_init(self, worker_id: int) -> None:
        """DataLoader worker_init_fn: give each worker its own random stream."""
        self.rng = np.random.default_rng(torch.initial_seed() % 2**32 + worker_id)


def render_targets(kps: torch.Tensor, labeled: torch.Tensor, height: int = INPUT_H, width: int = INPUT_W,
                   sigma: float = 2.5) -> torch.Tensor:
    """[B, 14, 2] keypoints + [B, 14] mask -> [B, 14, H, W] gaussian heatmaps (peak 1).

    A keypoint gets an all-zero heatmap when it is unlabeled or outside the
    image, which teaches the model to stay silent there. Values beyond 3 sigma
    are zeroed, as in `render_gaussian`.
    """
    ys = torch.arange(height, device=kps.device, dtype=torch.float32).view(1, 1, -1, 1)
    xs = torch.arange(width, device=kps.device, dtype=torch.float32).view(1, 1, 1, -1)
    x, y = kps[..., 0, None, None], kps[..., 1, None, None]
    d2 = (xs - x) ** 2 + (ys - y) ** 2
    hm = torch.exp(-d2 / (2 * sigma ** 2)) * (d2 <= (3 * sigma) ** 2)
    inside = labeled & (kps[..., 0] >= 0) & (kps[..., 0] <= width - 1) & (kps[..., 1] >= 0) & (kps[..., 1] <= height - 1)
    return hm * inside[..., None, None]


# ----------------------------------------------------------------------------- augmentation

def augment(img: np.ndarray, kps: np.ndarray, labeled: np.ndarray, rng: np.random.Generator):
    """Geometry (flip, scale/shift/rotate) then appearance (colour, noise, blur, quality, occlusion)."""
    img, kps, labeled = augment_geometry(img, kps, labeled, rng)
    return augment_appearance(img, kps, labeled, rng), kps, labeled


def augment_geometry(img: np.ndarray, kps: np.ndarray, labeled: np.ndarray, rng: np.random.Generator):
    """Random flip and zoom/pan/roll; keypoints move with the image."""
    h, w = img.shape[:2]
    kps = kps.copy()

    if rng.random() < 0.5:                                    # horizontal flip, swap left/right names
        img = img[:, ::-1].copy()
        kps[:, 0] = (w - 1) - kps[:, 0]
        kps, labeled = kps[FLIP_INDEX], labeled[FLIP_INDEX]

    if rng.random() < 0.7:                                    # zoom in/out, pan, slight roll
        scale = rng.uniform(0.75, 1.4)                         # >1 zooms in: some points leave the frame
        angle = rng.uniform(-5, 5)
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, scale)
        M[:, 2] += rng.uniform(-0.12, 0.12, size=2) * [w, h]
        img = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        kps = kps @ M[:, :2].T + M[:, 2]
    # Points pushed outside the image get a zero heatmap (render_targets), no change needed here.
    return img, kps, labeled


def augment_appearance(img: np.ndarray, kps: np.ndarray, labeled: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Colour, noise, blur, low resolution, compression and occluding boxes; keypoints unchanged."""
    h, w = img.shape[:2]
    img = img.astype(np.float32)
    if rng.random() < 0.8:                                    # brightness / contrast / colour cast (night, shade, surfaces)
        img = img * rng.uniform(0.6, 1.4) + rng.uniform(-40, 40)
        img = img * rng.uniform(0.85, 1.15, size=3)
    if rng.random() < 0.3:                                    # sensor noise
        img = img + rng.normal(0, rng.uniform(3, 12), size=img.shape)
    img = np.clip(img, 0, 255).astype(np.uint8)

    if rng.random() < 0.3:                                    # defocus / motion softness
        k = int(rng.choice([3, 5, 7]))
        img = cv2.GaussianBlur(img, (k, k), 0)
    if rng.random() < 0.3:                                    # low-resolution source upscaled
        f = rng.uniform(0.3, 0.7)
        small = cv2.resize(img, (int(w * f), int(h * f)), interpolation=cv2.INTER_AREA)
        img = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
    if rng.random() < 0.3:                                    # heavy compression
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, int(rng.integers(20, 70))])
        if ok:
            img = cv2.imdecode(buf, cv2.IMREAD_COLOR)

    if rng.random() < 0.5:                                    # players / objects covering some points
        for _ in range(int(rng.integers(1, 4))):
            bw, bh = int(w * rng.uniform(0.04, 0.15)), int(h * rng.uniform(0.08, 0.3))
            vis = np.flatnonzero(labeled)
            if len(vis) and rng.random() < 0.7:               # usually centred on a keypoint
                cx, cy = kps[rng.choice(vis)]
            else:
                cx, cy = rng.uniform(0, w), rng.uniform(0, h)
            x0, y0 = int(cx - bw / 2), int(cy - bh / 2)
            img[max(0, y0):max(0, y0 + bh), max(0, x0):max(0, x0 + bw)] = rng.integers(0, 256, size=3)
        # Covered points keep their labels: the model should infer them from the lines around.
    return img
