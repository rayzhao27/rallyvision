#!/usr/bin/env python
"""Train the court keypoint model (single-frame TrackNet, 14 heatmaps).

    python scripts/train_court.py --data /content/court_all --out runs/court_v1

`--data` is the unzipped Roboflow COCO export (train/ valid/ test/ folders).
The three Roboflow splits are merged and re-split by source video, so no video
has frames in two splits. Selection uses the val split only; test is left for
scripts/evaluate_court.py, run once.

Every epoch writes last.pt (for --resume after a Colab disconnect) and keeps
best.pt by highest val PCK@5px at threshold 0.5 (share of labeled in-frame
points found within 5 px; ties go to the later epoch), the same rule the ball
detector uses. The threshold itself is tuned later on val predictions.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader


from rallyvision.court_detection import (INPUT_H, INPUT_W, NUM_KEYPOINTS, CourtDataset, decode_keypoints,  # noqa: E402
                               load_coco_split, render_targets, split_by_video, video_id)
from rallyvision.ball_tracking.model.losses import FocalHeatmapLoss  # noqa: E402
from rallyvision.common.tracknet import TrackNet  # noqa: E402


def load_splits(data_dir: Path):
    samples = []
    for name in ("train", "valid", "test"):
        if (data_dir / name / "_annotations.coco.json").is_file():
            samples += load_coco_split(data_dir / name)
    if not samples:
        raise FileNotFoundError(f"no */_annotations.coco.json under {data_dir}")
    splits = split_by_video(samples)
    for k, v in splits.items():
        print(f"{k:5s}: {len(v):5d} images from {len({video_id(s['file']) for s in v}):3d} videos")
    if not splits["train"] or not splits["val"]:
        raise ValueError("empty train or val split")
    return splits


def pick_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@torch.no_grad()
def validate(model, loader, loss_fn, device, sigma, threshold=0.5, tol_px=5.0):
    """Mean val loss and PCK@tol (share of in-frame labeled points found within tol px)."""
    model.eval()
    total, n_batches, hits, n_gt = 0.0, 0, 0, 0
    for x, kps, labeled in loader:
        logits = model.forward_logits(x.to(device))
        target = render_targets(kps.to(device), labeled.to(device), sigma=sigma)
        total += loss_fn(logits.float(), target).item()
        n_batches += 1
        probs = torch.sigmoid(logits).float().cpu().numpy()
        for hm, gt, lab in zip(probs, kps.numpy(), labeled.numpy()):
            pts, _ = decode_keypoints(hm, threshold)
            inside = lab & (gt[:, 0] >= 0) & (gt[:, 0] < INPUT_W) & (gt[:, 1] >= 0) & (gt[:, 1] < INPUT_H)
            err = np.linalg.norm(pts - gt, axis=1)          # NaN where not detected
            hits += int(np.sum(inside & (err <= tol_px)))
            n_gt += int(inside.sum())
    return total / max(n_batches, 1), hits / max(n_gt, 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--width", type=float, default=1.0, help="channel multiplier (0.25 for smoke tests)")
    ap.add_argument("--sigma", type=float, default=2.5)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", action="store_true", help="continue from OUT/last.pt")
    ap.add_argument("--max-batches", type=int, default=0, help="cap batches per epoch (smoke tests)")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = pick_device(args.device)
    args.out.mkdir(parents=True, exist_ok=True)
    splits = load_splits(args.data)
    with open(args.out / "split.json", "w") as f:       # record which file went where
        json.dump({k: [Path(s["file"]).name for s in v] for k, v in splits.items()}, f)

    train_ds = CourtDataset(splits["train"], augment=True, seed=args.seed)
    val_ds = CourtDataset(splits["val"], augment=False)
    train_dl = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=args.workers,
                          drop_last=True, worker_init_fn=train_ds.worker_init, pin_memory=device.type == "cuda",
                          persistent_workers=args.workers > 0)
    val_dl = DataLoader(val_ds, batch_size=args.batch, shuffle=False, num_workers=args.workers)

    model = TrackNet(num_frames=1, out_channels=NUM_KEYPOINTS, width=args.width).to(device)
    loss_fn = FocalHeatmapLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=1e-5)
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    start_epoch, best_pck = 0, -1.0
    if args.resume and (args.out / "last.pt").is_file():
        ck = torch.load(args.out / "last.pt", map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["optimizer"])
        sched.load_state_dict(ck["scheduler"])
        scaler.load_state_dict(ck["scaler"])
        start_epoch, best_pck = ck["epoch"] + 1, ck["best_val_pck5"]
        print(f"resumed at epoch {start_epoch}, best val PCK@5px {best_pck:.4f}")

    log_path = args.out / "history.csv"
    if start_epoch == 0:
        log_path.write_text("epoch,train_loss,val_loss,val_pck5,lr,seconds\n")

    for epoch in range(start_epoch, args.epochs):
        model.train()
        t0, running, n = time.time(), 0.0, 0
        for i, (x, kps, labeled) in enumerate(train_dl):
            if args.max_batches and i >= args.max_batches:
                break
            x = x.to(device, non_blocking=True)
            target = render_targets(kps.to(device), labeled.to(device), sigma=args.sigma)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                logits = model.forward_logits(x)
            loss = loss_fn(logits.float(), target)       # loss in fp32 for stable log terms
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss at epoch {epoch} batch {i}")
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(opt)
            scaler.update()
            running += loss.item()
            n += 1
        sched.step()

        val_loss, pck = validate(model, val_dl, loss_fn, device, args.sigma)
        lr = opt.param_groups[0]["lr"]
        secs = time.time() - t0
        is_best = pck >= best_pck
        best_pck = max(best_pck, pck)
        print(f"epoch {epoch + 1:3d}/{args.epochs}  train {running / max(n, 1):.4f}  val {val_loss:.4f}  "
              f"PCK@5px {pck:.3f}  lr {lr:.2e}  {secs:.0f}s" + ("  *best" if is_best else ""))
        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow([epoch + 1, f"{running / max(n, 1):.5f}", f"{val_loss:.5f}", f"{pck:.4f}", f"{lr:.2e}", f"{secs:.1f}"])

        ck = {"model": model.state_dict(), "width": args.width, "sigma": args.sigma, "epoch": epoch,
              "best_val_pck5": best_pck, "val_pck5": pck, "val_loss": val_loss, "input_wh": (INPUT_W, INPUT_H)}
        if is_best:
            torch.save(ck, args.out / "best.pt")
        ck.update(optimizer=opt.state_dict(), scheduler=sched.state_dict(), scaler=scaler.state_dict())
        torch.save(ck, args.out / "last.pt")

    print(f"done. best val PCK@5px {best_pck:.4f} -> {args.out / 'best.pt'}")


if __name__ == "__main__":
    main()
