#!/usr/bin/env python
"""Train / evaluate the temporal event detector on preprocessed E2E-Spot Tennis clips (run on Colab).

    python scripts/train_event_tf.py --data /content/e2e --out /content/runs/A --mode motion
    python scripts/train_event_tf.py --data /content/e2e --out /content/runs/B --mode vision --cache /content/emb

  --mode motion   experiment A: ball motion + court geometry per frame
  --mode vision   experiment B: A + frozen ResNet18 (ImageNet) features of the ball crop and the whole frame

Split as in the dataset: train to fit, val to choose the epoch and the decoding thresholds, test scored
once at the end. The GBDT baseline (scripts/evaluate_bounce.py, fitted on the TrackNet train games) is
run on the same tracks and scored the same way. Clips above 40 fps (e.g. the Sinner video) are
evaluated on a 30 fps resampling of their frames.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from rallyvision.common.config import REPO_ROOT  # noqa: E402
REPO = REPO_ROOT
from rallyvision.bounce_detection.experiments.previous_transformers.e2e_tennis import BOUNCE, HIT  # noqa: E402
from rallyvision.bounce_detection.experiments.previous_transformers.event_tf import (EventTransformer, WindowDataset, decode_events, load_clip,  # noqa: E402
                                  predict_clip, score_bounces, score_class)

THR_GRID = np.round(np.arange(0.1, 0.95, 0.05), 2)


# --------------------------------------------------------------------------- vision features
def vision_features(paths, cache: Path, device) -> dict:
    """Frozen ResNet18 features per frame: [crop 512 | whole frame 512] float16, cached per clip."""
    import torchvision
    cache.mkdir(parents=True, exist_ok=True)
    net = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
    net.fc = nn.Identity()
    net = net.to(device).eval()
    mean = torch.tensor([0.485, 0.456, 0.406], device=device)[:, None, None]
    std = torch.tensor([0.229, 0.224, 0.225], device=device)[:, None, None]

    @torch.no_grad()
    def embed(imgs, size):
        out = []
        for i in range(0, len(imgs), 512):
            x = torch.from_numpy(imgs[i:i + 512]).to(device).permute(0, 3, 1, 2).contiguous().float() / 255
            if size:
                x = nn.functional.interpolate(x, size=size, mode="bilinear", align_corners=False)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.startswith("cuda")):
                out.append(net((x - mean) / std).float().cpu().numpy())
        return np.concatenate(out)

    feats = {}
    for p in paths:
        f = cache / f"{p.stem}.npy"
        if not f.exists():
            z = np.load(p)
            np.save(f, np.concatenate([embed(z["crops"], (112, 112)), embed(z["thumbs"], None)], 1).astype(np.float16))
        feats[p.stem] = np.load(f).astype(np.float32)
    return feats


# --------------------------------------------------------------------------- helpers
def resampled(c, vision, target_fps=30.0):
    """Frame indices used for prediction: every frame, or a 30 fps subsampling for high frame rates."""
    if c["fps"] <= 40:
        return np.arange(len(c["cls"]))
    return np.unique(np.round(np.arange(0, len(c["cls"]), c["fps"] / target_fps)).astype(int).clip(0, len(c["cls"]) - 1))


def clip_probs(model, clips, vision, window, device):
    out = {}
    model.eval()
    for c in clips:
        idx = resampled(c, vision)
        f = c["motion"][idx]
        if vision is not None:
            f = np.concatenate([f, vision[c["name"]][idx]], 1)
        p = predict_clip(model, f.astype(np.float32), window, max(window // 4, 1), device)
        out[c["name"]] = (p, idx)
    return out


def decode_all(probs, cls, thr, min_sep_s, fps_of):
    preds = {}
    for name, (p, idx) in probs.items():
        sep = max(int(round(min_sep_s * min(fps_of[name], 30.0))), 1)
        preds[name] = [int(idx[t]) for t in decode_events(p, cls, thr, sep)]
    return preds


def best_threshold(clips, probs, cls, tol, fps_of):
    best = (-1, None)
    for t in THR_GRID:
        preds = decode_all(probs, cls, t, 0.2, fps_of)
        f1 = score_bounces(clips, preds, tol)["all"][5] if cls == BOUNCE else score_class(clips, preds, cls, tol)[5]
        if f1 > best[0]:
            best = (f1, float(t))
    return best[1], best[0]


# --------------------------------------------------------------------------- GBDT baseline
def gbdt_baseline(clips, data_root, tracknet_csv, tracknet_labels, tol):
    """First bounces from the deployed GBDT (fitted on the TrackNet train games) on the same ball tracks."""
    from rallyvision.common.cli import load_config
    from rallyvision.bounce_detection.experiments.gbdt.evaluate_bounce import VAL_GAMES, fit_gbdt, load_clips, track_data
    from rallyvision.bounce_detection.experiments.gbdt.bounce import first_bounces_by_gap
    from rallyvision.ball_tracking.inference.tracker import BallTracker
    cfg = load_config()
    detect = fit_gbdt([c for c in load_clips(tracknet_csv, tracknet_labels, cfg) if c["game"] not in VAL_GAMES], "pred")
    preds = {}
    for c in clips:
        z = np.load(next(Path(data_root).glob(f"*/{c['name']}.npz")))
        fps, balls = c["fps"], z["ball"]
        tr = BallTracker(max_jump_px=float(cfg.tracker.max_jump_px),
                         max_gap_frames=int(round(cfg.tracker.max_gap_frames * fps / 30)))
        for f, p in enumerate(balls):
            tr.update(f, tuple(p) if np.isfinite(p[0]) else None)
        t = tr.finalize()
        x = np.array([d.x if d.has_position else np.nan for d in t])
        y = np.array([d.y if d.has_position else np.nan for d in t])
        m = np.array([d.source != "detected" for d in t])
        ts = np.arange(len(t)) / fps
        t30 = np.arange(0, ts[-1], 1 / 30)
        lo = np.clip(np.searchsorted(ts, t30, side="right") - 1, 0, len(t) - 2)
        w = (t30 - ts[lo]) * fps
        x30, y30, m30 = (1 - w) * x[lo] + w * x[lo + 1], (1 - w) * y[lo] + w * y[lo + 1], m[lo] | m[lo + 1]
        f30 = np.arange(len(t30))
        b = first_bounces_by_gap(detect({"frames": f30, "pred": track_data(x30, y30, m30, f30, [])}),
                                 detect.params["same_shot_gap"])
        preds[c["name"]] = [min(int(round(f / 30 * fps)), len(t) - 1) for f in b]
    return {"bounce": score_bounces(clips, preds, tol), "params": detect.params}, preds


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, required=True, help="folder of <match>/<clip>.npz")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--mode", choices=("motion", "vision"), default="motion")
    ap.add_argument("--cache", type=Path, default=Path("/content/emb"))
    ap.add_argument("--extra", type=Path, nargs="*", default=[], help="extra clips to report (e.g. the Sinner video)")
    ap.add_argument("--window", type=int, default=32)
    ap.add_argument("--stride", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--class-weight", type=float, default=5.0, help="loss weight of bounce and hit frames")
    ap.add_argument("--tol", type=int, default=2, help="frames; matching tolerance")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tracknet-csv", type=Path, nargs="*", default=[])
    ap.add_argument("--tracknet-labels", type=Path)
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    args.out.mkdir(parents=True, exist_ok=True)

    paths = sorted(p for p in args.data.glob("*/*.npz"))
    t0 = time.time()
    clips = [load_clip(p) for p in paths]
    extra = [dict(load_clip(p), split="extra") for p in args.extra]
    vision = vision_features(paths + list(args.extra), args.cache, device) if args.mode == "vision" else None
    by = {s: [c for c in clips if c["split"] == s] for s in ("train", "val", "test")}
    fps_of = {c["name"]: c["fps"] for c in clips + extra}
    print(f"loaded {len(clips)} clips in {time.time() - t0:.0f}s: " +
          ", ".join(f"{s} {len(v)} clips / {sum((c['cls'] == BOUNCE).sum() for c in v)} bounces" for s, v in by.items()))

    train_ds = WindowDataset(by["train"], args.window, args.stride, vision, augment=True)
    in_dim = train_ds[0][0].shape[1]
    model = EventTransformer(in_dim, window=args.window).to(device)
    print(f"mode {args.mode}: input {in_dim} per frame, {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M params, "
          f"{len(train_ds)} training windows")
    loader = torch.utils.data.DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=2, drop_last=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, args.lr, total_steps=args.epochs * len(loader), pct_start=0.1)
    loss_fn = nn.CrossEntropyLoss(weight=torch.tensor([1.0, args.class_weight, args.class_weight], device=device))

    best, history = (-1, None, None), []
    for ep in range(1, args.epochs + 1):
        model.train()
        tot = 0.0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            loss = loss_fn(model(x).reshape(-1, 3), y.reshape(-1))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            tot += loss.item()
        probs = clip_probs(model, by["val"], vision, args.window, device)
        thr, f1 = best_threshold(by["val"], probs, BOUNCE, args.tol, fps_of)
        history.append({"epoch": ep, "loss": tot / len(loader), "val_bounce_f1": f1, "thr": thr})
        print(f"epoch {ep:2d}  loss {tot / len(loader):.4f}  val bounce F1 {f1:.3f} (thr {thr})")
        if f1 > best[0]:
            best = (f1, ep, {k: v.detach().clone() for k, v in model.state_dict().items()})
    model.load_state_dict(best[2])
    torch.save({"model": best[2], "in_dim": in_dim, "window": args.window, "mode": args.mode}, args.out / "model.pt")

    # thresholds on val, then test once
    vprobs = clip_probs(model, by["val"], vision, args.window, device)
    t_b, _ = best_threshold(by["val"], vprobs, BOUNCE, args.tol, fps_of)
    t_h, _ = best_threshold(by["val"], vprobs, HIT, args.tol, fps_of)
    result = {"mode": args.mode, "best_epoch": best[1], "thresholds": {"bounce": t_b, "hit": t_h},
              "history": history, "args": {k: str(v) for k, v in vars(args).items()}}
    for split in ("val", "test"):
        if not by[split]:
            continue
        pr = clip_probs(model, by[split], vision, args.window, device)
        pb, ph = decode_all(pr, BOUNCE, t_b, 0.2, fps_of), decode_all(pr, HIT, t_h, 0.2, fps_of)
        result[split] = {"bounce": score_bounces(by[split], pb, args.tol), "hit": score_class(by[split], ph, HIT, args.tol)}
        (args.out / f"pred_{split}.json").write_text(json.dumps({"bounce": pb, "hit": ph}))
    if extra:
        pr = clip_probs(model, extra, vision, args.window, device)
        result["extra"] = {}
        for c in extra:
            p, idx = pr[c["name"]]
            pb = decode_all({c["name"]: (p, idx)}, BOUNCE, t_b, 0.2, fps_of)[c["name"]]
            gt = np.flatnonzero(c["cls"] == BOUNCE).tolist()
            tol = int(np.ceil(args.tol * c["fps"] / 30))
            # probability of bounce near each labelled bounce (max within ±tol frames)
            near = {int(g): float(p[(np.abs(idx - g) <= tol), 1].max()) if (np.abs(idx - g) <= tol).any() else 0.0
                    for g in gt}
            result["extra"][c["name"]] = {"pred_bounces": pb, "gt_bounces": gt, "p_bounce_at_gt": near,
                                          "score": score_bounces([c], {c["name"]: pb}, tol)}
    if args.tracknet_csv:
        for split in ("val", "test"):
            if by[split]:
                result.setdefault("gbdt", {})[split] = gbdt_baseline(by[split], args.data, args.tracknet_csv,
                                                                     args.tracknet_labels, args.tol)[0]
    (args.out / "result.json").write_text(json.dumps(result, indent=1, default=float))

    def line(name, s):
        return (f"  {name:10s} all P {s['all'][3]:.3f} R {s['all'][4]:.3f} F1 {s['all'][5]:.3f} | "
                f"near F1 {s['near'][5]:.3f} (R {s['near'][4]:.3f}) | far F1 {s['far'][5]:.3f} (R {s['far'][4]:.3f})")
    print(f"\nbest epoch {best[1]}, thresholds bounce {t_b} hit {t_h}")
    for split in ("val", "test"):
        if split in result:
            print(f"{split}:")
            print(line("bounce", result[split]["bounce"]))
            h = result[split]["hit"]
            print(f"  hit        P {h[3]:.3f} R {h[4]:.3f} F1 {h[5]:.3f}")
            if "gbdt" in result and split in result["gbdt"]:
                print(line("GBDT", result["gbdt"][split]["bounce"]))
    for name, r in result.get("extra", {}).items():
        print(f"{name}: predicted bounces {r['pred_bounces']}")
        print(f"  p(bounce) at labelled bounces: {r['p_bounce_at_gt']}")


if __name__ == "__main__":
    main()
