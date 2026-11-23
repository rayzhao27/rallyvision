#!/usr/bin/env python
"""Small temporal Transformer for bounce / hit, trained on ONE preprocessed E2E-Spot match (run on Colab).

    python scripts/train_event_small.py --clips DIR_OF_NPZ --sinner SINNER.npz --out OUT

Per frame: ball motion + court geometry (rallyvision.event_tf.motion_features, 15 values) and a frozen
ResNet18 (ImageNet) embedding of the saved 64x64 ball crop (512 values). 2-layer Transformer, d_model 64,
4 heads, 48-frame windows, 4 classes per frame: other / near bounce / far bounce / hit (swing or serve).

Split inside the one match: clips sorted by time, the last ``--val-frac`` of them (a contiguous block,
no clip overlapping the training block) is the validation block; it picks the epoch and the thresholds.
This is a single-match preliminary experiment, not a generalisation result. The Sinner video is then
scored independently against the corrected hand labels (S3 = frame 248, S11 = frame 864).
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
from rallyvision.bounce_detection.experiments.previous_transformers.event_tf import load_clip  # noqa: E402

CLASSES = ("other", "near_bounce", "far_bounce", "hit")
SINNER_GT = {"S1": (57, "far"), "S2": (163, "near"), "S3": (248, "far"), "S4": (308, "near"), "S5": (389, "far"),
             "S6": (456, "near"), "S7": (534, "far"), "S8": (605, "near"), "S9": (676, "far"), "S10": (764, "near"),
             "S11": (864, "far"), "S12": (950, "far")}


# --------------------------------------------------------------------------- data
def targets(c, radius=1):
    """Per-frame class 0..3; each event also labels its ±radius neighbours (the label frame is ±1 uncertain)."""
    y = np.zeros(len(c["cls"]), np.int64)
    for t in np.flatnonzero(c["cls"] != 0):
        k = 3 if c["cls"][t] == HIT else (1 if c["side"][t] == 1 else 2)
        lo, hi = max(t - radius, 0), min(t + radius + 1, len(y))
        y[lo:hi][y[lo:hi] == 0] = k
    return y


def crop_embeddings(paths, cache: Path, device):
    """Frozen ResNet18 (ImageNet) on the saved 64x64 ball crops (upsampled to 112): [n, 512] float32 per clip."""
    import torchvision
    cache.mkdir(parents=True, exist_ok=True)
    net = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
    net.fc = nn.Identity()
    net = net.to(device).eval()
    mean = torch.tensor([0.485, 0.456, 0.406], device=device)[:, None, None]
    std = torch.tensor([0.229, 0.224, 0.225], device=device)[:, None, None]
    out = {}
    for p in paths:
        f = cache / f"{Path(p).stem}.npy"
        if not f.exists():
            crops = np.load(p)["crops"]
            feats = []
            with torch.no_grad():
                for i in range(0, len(crops), 512):
                    x = torch.from_numpy(crops[i:i + 512]).to(device).permute(0, 3, 1, 2).float() / 255
                    x = nn.functional.interpolate(x, size=(112, 112), mode="bilinear", align_corners=False)
                    feats.append(net((x - mean) / std).float().cpu().numpy())
            np.save(f, np.concatenate(feats).astype(np.float16))
        out[Path(p).stem] = np.load(f).astype(np.float32)
    return out


class Windows(torch.utils.data.Dataset):
    def __init__(self, clips, window, stride):
        self.clips, self.window = clips, window
        self.idx = [(k, s) for k, c in enumerate(clips) for s in range(0, len(c["x"]) - window + 1, stride)]

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        k, s = self.idx[i]
        c = self.clips[k]
        return torch.from_numpy(c["x"][s:s + self.window]), torch.from_numpy(c["y"][s:s + self.window])


# --------------------------------------------------------------------------- model
class SmallEventTransformer(nn.Module):
    def __init__(self, motion_dim, vision_dim, d_model=64, layers=2, heads=4, window=48, n_cls=4, dropout=0.1):
        super().__init__()
        self.motion_dim = motion_dim
        self.motion = nn.Sequential(nn.Linear(motion_dim, d_model), nn.GELU(), nn.Linear(d_model, d_model))
        self.vision = nn.Sequential(nn.LayerNorm(vision_dim), nn.Linear(vision_dim, d_model))
        self.pos = nn.Parameter(torch.zeros(1, window, d_model))
        layer = nn.TransformerEncoderLayer(d_model, heads, 2 * d_model, dropout, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, layers)
        self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, n_cls))

    def forward(self, x):
        h = self.motion(x[..., :self.motion_dim]) + self.vision(x[..., self.motion_dim:])
        return self.head(self.encoder(h + self.pos[:, :x.shape[1]]))


@torch.no_grad()
def predict(model, x, window, device):
    """Per-frame class probabilities [n, 4], overlapping windows (stride window/4) averaged."""
    n = len(x)
    if n < window:
        x = np.concatenate([x, np.repeat(x[-1:], window - n, 0)])
    starts = list(range(0, len(x) - window + 1, window // 4))
    if starts[-1] != len(x) - window:
        starts.append(len(x) - window)
    p = torch.softmax(model(torch.from_numpy(np.stack([x[s:s + window] for s in starts])).to(device)), -1).cpu().numpy()
    acc, cnt = np.zeros((len(x), p.shape[-1])), np.zeros(len(x))
    for s, q in zip(starts, p):
        acc[s:s + window] += q
        cnt[s:s + window] += 1
    return (acc / cnt[:, None])[:n]


# --------------------------------------------------------------------------- decoding and scoring
def peaks(score, thr, sep):
    keep = []
    for t in np.argsort(-score):
        if score[t] < thr:
            break
        if all(abs(t - k) >= sep for k in keep):
            keep.append(int(t))
    return sorted(keep)


def match(pred, gt, tol):
    pairs = sorted((abs(a - b), i, j) for i, a in enumerate(pred) for j, b in enumerate(gt) if abs(a - b) <= tol)
    up, ug, out = set(), set(), []
    for _, i, j in pairs:
        if i not in up and j not in ug:
            up.add(i)
            ug.add(j)
            out.append((i, j))
    return out


def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return [tp, fp, fn, p, r, 2 * p * r / (p + r) if p + r else 0.0]


def decode(prob, thr_bounce, thr_hit, fps):
    """Bounce frames with side (from near vs far probability) and hit frames."""
    sep = max(int(round(0.25 * fps)), 1)
    b = peaks(prob[:, 1] + prob[:, 2], thr_bounce, sep)
    side = ["near" if prob[t, 1] >= prob[t, 2] else "far" for t in b]
    return b, side, peaks(prob[:, 3], thr_hit, sep)


def score_clips(clips, probs, thr_b, thr_h, tol=2):
    tot = {k: [0, 0, 0] for k in ("bounce", "near_bounce", "far_bounce", "hit")}
    for c, p in zip(clips, probs):
        b, side, h = decode(p, thr_b, thr_h, c["fps"])
        gt_b = np.flatnonzero(c["cls"] == BOUNCE)
        gt_side = ["near" if c["side"][t] == 1 else "far" for t in gt_b]
        for key, pr, gt in (("bounce", b, gt_b.tolist()), ("hit", h, np.flatnonzero(c["cls"] == HIT).tolist())):
            m = len(match(pr, gt, tol))
            tot[key] = [tot[key][0] + m, tot[key][1] + len(pr) - m, tot[key][2] + len(gt) - m]
        for s in ("near", "far"):
            pr = [t for t, sd in zip(b, side) if sd == s]
            gt = [t for t, sd in zip(gt_b, gt_side) if sd == s]
            m = len(match(pr, gt, tol))
            k = f"{s}_bounce"
            tot[k] = [tot[k][0] + m, tot[k][1] + len(pr) - m, tot[k][2] + len(gt) - m]
    return {k: prf(*v) for k, v in tot.items()}


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clips", type=Path, required=True, help="folder with the match's <clip>.npz files")
    ap.add_argument("--sinner", type=Path, help="preprocessed Sinner video .npz")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--cache", type=Path, default=Path("/content/crop_emb"))
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--window", type=int, default=48)
    ap.add_argument("--d-model", type=int, default=64)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--weights", type=float, nargs=4, default=[1.0, 10.0, 10.0, 5.0], help="loss weight per class")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    args.out.mkdir(parents=True, exist_ok=True)

    # data: clips in time order, contiguous train / val blocks with no overlap
    paths = sorted(args.clips.glob("*.npz"))
    starts = {p: int(np.load(p)["start"]) for p in paths}
    paths.sort(key=lambda p: starts[p])
    clips = [dict(load_clip(p), start=starts[p], path=p) for p in paths]
    n_val = max(1, int(round(args.val_frac * len(clips))))
    train, val = clips[:-n_val], clips[-n_val:]
    train_end = max(c["start"] + len(c["cls"]) for c in train)
    val = [c for c in val if c["start"] >= train_end]                 # drop any clip overlapping the train block
    emb = crop_embeddings([c["path"] for c in clips] + ([args.sinner] if args.sinner else []), args.cache, device)
    for c in clips:
        c["x"] = np.concatenate([c["motion"], emb[c["name"]]], 1).astype(np.float32)
        c["y"] = targets(c)

    def count(cs):
        allc = np.concatenate([c["cls"] for c in cs]); alls = np.concatenate([c["side"] for c in cs])
        return {"clips": len(cs), "frames": int(len(allc)), "near_bounce": int(((allc == BOUNCE) & (alls == 1)).sum()),
                "far_bounce": int(((allc == BOUNCE) & (alls == 2)).sum()), "hit": int((allc == HIT).sum())}
    info = {"train": count(train), "val": count(val), "train_frames_end": train_end, "val_first_start": val[0]["start"]}
    print("data:", json.dumps(info))

    motion_dim, vision_dim = clips[0]["motion"].shape[1], emb[clips[0]["name"]].shape[1]
    model = SmallEventTransformer(motion_dim, vision_dim, args.d_model, args.layers, args.heads, args.window).to(device)
    print(f"model: {sum(p.numel() for p in model.parameters()) / 1e3:.0f}k parameters")
    loader = torch.utils.data.DataLoader(Windows(train, args.window, 8), batch_size=64, shuffle=True, drop_last=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, args.lr, total_steps=args.epochs * len(loader), pct_start=0.1)
    loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(args.weights, device=device))
    grid = np.round(np.arange(0.1, 0.95, 0.05), 2)

    best, history = (-1.0, 0, None, None), []
    for ep in range(1, args.epochs + 1):
        model.train()
        tot = 0.0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            loss = loss_fn(model(x).reshape(-1, 4), y.reshape(-1))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            tot += loss.item()
        model.eval()
        vp = [predict(model, c["x"], args.window, device) for c in val]
        f1, thr = max((score_clips(val, vp, t, 0.5)["bounce"][5], float(t)) for t in grid)
        history.append({"epoch": ep, "loss": tot / len(loader), "val_bounce_f1": f1, "thr": thr})
        print(f"epoch {ep:2d}  loss {tot / len(loader):.4f}  val bounce F1 {f1:.3f} (thr {thr})", flush=True)
        if f1 > best[0]:
            best = (f1, ep, {k: v.detach().clone() for k, v in model.state_dict().items()}, thr)
    model.load_state_dict(best[2])

    vp = [predict(model, c["x"], args.window, device) for c in val]
    thr_b = best[3]
    thr_h = max((score_clips(val, vp, thr_b, t)["hit"][5], float(t)) for t in grid)[1]
    val_scores = score_clips(val, vp, thr_b, thr_h)
    result = {"data": info, "best_epoch": best[1], "thresholds": {"bounce": thr_b, "hit": thr_h},
              "val": val_scores, "history": history, "args": {k: str(v) for k, v in vars(args).items()}}
    torch.save({"model": best[2], "motion_dim": motion_dim, "vision_dim": vision_dim, "classes": CLASSES,
                "config": {k: getattr(args, k) for k in ("window", "d_model", "layers", "heads")},
                "thresholds": result["thresholds"]}, args.out / "small_event_tf.pt")
    print(f"\nbest epoch {best[1]}; thresholds bounce {thr_b} hit {thr_h}")
    print("VALIDATION (contiguous block of the same match; preliminary, not a generalisation result):")
    for k, v in val_scores.items():
        print(f"  {k:12s} P {v[3]:.3f} R {v[4]:.3f} F1 {v[5]:.3f}  (tp {v[0]} fp {v[1]} fn {v[2]})")

    if args.sinner:
        c = load_clip(args.sinner)
        x = np.concatenate([c["motion"], emb[c["name"]]], 1).astype(np.float32)
        fps = c["fps"]
        idx = np.unique(np.round(np.arange(0, len(x), fps / 30.0)).astype(int).clip(0, len(x) - 1))  # 30 fps like training
        p = predict(model, x[idx], args.window, device)
        b, side, h = decode(p, thr_b, thr_h, 30.0)
        pred = [(int(idx[t]), s, round(float(p[t, 1] + p[t, 2]), 3)) for t, s in zip(b, side)]
        gt = [(name, f, s) for name, (f, s) in SINNER_GT.items()]
        tol = 4                                                          # video frames at 54.6 fps (= ±2 at 30 fps)
        pairs = match([q[0] for q in pred], [g[1] for g in gt], tol)
        tp = len(pairs)
        s = prf(tp, len(pred) - tp, len(gt) - tp)
        matched = {j: i for i, j in pairs}
        rows = {}
        print("\nSINNER VIDEO (independent; corrected labels S3 = 248, S11 = 864)")
        print("  predicted bounces (video frame, side, prob):", pred)
        for j, (name, f, sd) in enumerate(gt):
            near_t = np.abs(idx - f) <= tol
            pmax = float((p[near_t, 1] + p[near_t, 2]).max()) if near_t.any() else 0.0
            got = pred[matched[j]] if j in matched else None
            rows[name] = {"frame": f, "side": sd, "max_bounce_prob": pmax, "found": got is not None,
                          "pred_side": got[1] if got else None}
            print(f"  {name:4s} frame {f:4d} ({sd}): max bounce prob {pmax:.3f}  "
                  + (f"FOUND at {got[0]} (side {got[1]})" if got else "missed"))
        print(f"  bounce P {s[3]:.3f} R {s[4]:.3f} F1 {s[5]:.3f}  (tp {s[0]} fp {s[1]} fn {s[2]}); "
              f"false: {[pred[i][0] for i in range(len(pred)) if i not in {i for i, _ in pairs}]}")
        result["sinner"] = {"pred_bounces": pred, "per_bounce": rows, "bounce_prf": s}
    (args.out / "result.json").write_text(json.dumps(result, indent=1, default=float))


if __name__ == "__main__":
    main()
