#!/usr/bin/env python
"""Write the top heatmap peaks of every frame of a split, for evaluating ball selection offline.

    python scripts/dump_ball_peaks.py --checkpoint best.pt --split train --out train_peaks.csv \
        --prepared-dir /content/prepared_local --set data.loader.num_workers=2

One row per (frame, peak), up to 3 peaks scoring >= 0.05; a frame without peaks has
one row with empty x, y. Coordinates are in the same pixels as *_per_sample.csv
(model input scaled by `to_original`), so scripts/evaluate_ball_select.py can
compare against the labels and feed scripts/evaluate_bounce.py.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch

from rallyvision.common.cli import add_config_args, add_device_args, build_config
from rallyvision.ball_tracking.inference.ball_select import top_peaks
from rallyvision.common.checkpoint import load_model_weights
from rallyvision.common.config import resolve_path
from rallyvision.ball_tracking.data.dataset import build_dataloader, build_dataset
from rallyvision.ball_tracking.data.manifest import DatasetManifest
from rallyvision.common.device import resolve_device_from_config
from rallyvision.ball_tracking.model import build_model


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_config_args(ap)
    add_device_args(ap)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--split", choices=["train", "val"], required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--prepared-dir", type=Path, default=None)
    args = ap.parse_args()

    cfg = build_config(args)
    device = resolve_device_from_config(cfg)
    manifest = DatasetManifest.load(args.prepared_dir or resolve_path(cfg, cfg.data.prepared_dir))
    dataset = build_dataset(cfg, manifest, split=args.split, augment=False)
    if len(dataset) == 0:
        raise ValueError(f"split {args.split} is empty")
    loader = build_dataloader(cfg, dataset, split="val", shuffle=False, device_kind=device.kind)
    model = build_model(cfg)
    load_model_weights(args.checkpoint, model)
    model.to(device.device).eval()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["clip_id", "frame_file", "visible", "gt_x", "gt_y", "rank", "x", "y", "score"])
        for batch in loader:
            with torch.no_grad():
                probs = torch.sigmoid(model.forward_logits(batch["frames"].to(device.device)).float())[:, 0].cpu().numpy()
            sx_sy = batch["to_original"].numpy().reshape(-1, 2)
            txy = batch["target_xy"].numpy().reshape(-1, 2)
            vis = batch["visible"].numpy().reshape(-1) > 0.5
            for i, hm in enumerate(probs):
                sx, sy = sx_sy[i]
                head = [batch["clip_id"][i], batch["frame_file"][i], bool(vis[i]),
                        f"{txy[i, 0] * sx:.2f}" if vis[i] else "", f"{txy[i, 1] * sy:.2f}" if vis[i] else ""]
                peaks = top_peaks(hm)
                if len(peaks) == 0:
                    w.writerow(head + [0, "", "", ""])
                for r, (x, y, s) in enumerate(peaks):
                    w.writerow(head + [r, f"{x * sx:.2f}", f"{y * sy:.2f}", f"{s:.4f}"])
    print("wrote", args.out)


if __name__ == "__main__":
    main()
