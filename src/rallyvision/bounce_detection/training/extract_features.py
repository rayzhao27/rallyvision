"""Extract and cache frozen E2E-Spot backbone features (and the original GRU probabilities) for every clip.

Data: the E2E-Spot tennis clips and labels (Hugging Face package OpenSportsLab/soccernetpro-localization-tennis).
Run through ``python -m rallyvision.bounce_detection.training.train extract ...`` (see train.py).
"""

from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np

from rallyvision.bounce_detection.model.event_transformer.backbone import backbone, decode, load_spot
from rallyvision.bounce_detection.model.event_transformer.model import CLASSES

SPLITS = {"train": "train", "valid": "val", "test": "test"}


def extract(args):
    device = "cuda"
    impl, clip_len = load_spot(args.spot, args.spot_model, device)
    items = []
    for hf_split, split in SPLITS.items():
        d = json.loads((args.data / f"annotations-localization-{hf_split}.json").read_text())
        rows = d["data"][:args.max_clips] if args.max_clips else d["data"]
        for it in rows:
            fps = float(it["inputs"][0]["fps"])
            ev = [(int(round(int(e["position_ms"]) / 1000 * fps)), CLASSES.index(e["label"]) + 1) for e in it["events"]]
            items.append((split, it["id"], args.data / it["inputs"][0]["path"], fps, ev))
    todo = [x for x in items if not (args.cache / x[0] / f"{x[1]}.npz").exists()]
    print(f"{len(items)} clips, {len(todo)} to extract", flush=True)
    t0, nframes = time.time(), 0
    with ThreadPoolExecutor(args.threads) as pool:
        for c0 in range(0, len(todo), args.threads):
            chunk = todo[c0:c0 + args.threads]
            for (split, cid, path, fps, ev), frames in zip(chunk, pool.map(lambda x: decode(x[2]), chunk)):
                feat, prob = backbone(impl, frames, clip_len, device)
                ev = [(f, c) for f, c in ev if f < len(frames)]
                out = args.cache / split / f"{cid}.npz"
                out.parent.mkdir(parents=True, exist_ok=True)
                np.savez(out, feat=feat.astype(np.float16), spot=prob.astype(np.float16), fps=fps,
                         ev_frame=np.array([f for f, _ in ev], int), ev_cls=np.array([c for _, c in ev], int))
                nframes += len(frames)
            el = time.time() - t0
            print(f"  {c0 + len(chunk)}/{len(todo)} clips, {nframes} frames, {el:.0f}s ({nframes / max(el, 1e-6):.0f} frames/s)", flush=True)
    if args.sinner:
        cap = cv2.VideoCapture(str(args.sinner))
        fps, n = cap.get(cv2.CAP_PROP_FPS), int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        keep = np.unique(np.round(np.arange(0, n, fps / 30.0)).astype(int).clip(0, n - 1))     # 30 fps like the dataset
        frames = decode(args.sinner, keep)
        keep = keep[:len(frames)]
        feat, prob = backbone(impl, frames, clip_len, device)
        np.savez(args.cache / "sinner.npz", feat=feat.astype(np.float16), spot=prob.astype(np.float16),
                 video_frame=keep, src_fps=fps)
        print(f"Sinner: {len(frames)} frames at 30 fps (source {fps:.2f} fps)")
    (args.cache / "extract_info.json").write_text(json.dumps(
        {"frames": nframes, "seconds": time.time() - t0, "backbone": str(args.spot_model.name)}))
