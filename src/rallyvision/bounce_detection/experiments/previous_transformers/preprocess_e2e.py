#!/usr/bin/env python
"""One E2E-Spot Tennis source video -> per-clip ball tracks, court homographies and image crops (run on Colab).

    python scripts/preprocess_e2e.py --labels LABELS_DIR --video match.mp4 --source usopen_2020_... \
        --out OUT_DIR --ball-checkpoint hr1280.pt --court-checkpoint court_v2.pt

Same components as the deployed pipeline, unchanged: 1280x720 TrackNet (hr1280_v1, time-based input
frames), top-3 heatmap peaks, the new select_ball (parameters chosen on the TrackNet train games) +
drop_static, BallTracker gap filling, court_v2 keypoints + homography at 640x360.

Per clip it writes <clip>.npz (all arrays per frame of the clip, coordinates in 640x360 pixels):
  peaks [n,3,3] (x, y, score; NaN padded)   ball [n,2] selected peak (NaN = none)
  track [n,2], track_src [n] (0 none, 1 detected, 2 interpolated)
  H [n,3,3] court->image, H_valid [n]
  crops [n,64,64,3] uint8 RGB around the ball at 1280x720 (centre: track, else last known position)
  crop_xy [n,2] (640x360 coordinates of the crop centre), thumbs [n,90,160,3] uint8 RGB whole frame
Clips that overlap in the source video are read once (frames are processed per merged segment).
Existing outputs are skipped, so a disconnected run resumes.
"""

from __future__ import annotations

import argparse
import json
import queue
import threading
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from rallyvision.common.config import REPO_ROOT  # noqa: E402
REPO = REPO_ROOT
from rallyvision.common.cli import load_config  # noqa: E402
from rallyvision.ball_tracking.inference.ball_select import drop_static, input_offsets, select_ball  # noqa: E402
from rallyvision.bounce_detection.experiments.gbdt.rally_gbdt import pad_16x9  # noqa: E402
from rallyvision.common.checkpoint import load_model_weights  # noqa: E402
from rallyvision.court_detection import fit_homography, load_court_model  # noqa: E402
from rallyvision.bounce_detection.experiments.previous_transformers.e2e_tennis import load_all, load_videos  # noqa: E402
from rallyvision.ball_tracking.model import build_model  # noqa: E402
from rallyvision.ball_tracking.inference.tracker import BallTracker  # noqa: E402

BALL_W, BALL_H, COURT_W, COURT_H = 1280, 720, 640, 360
BALL_THRESHOLD = 0.16                                   # chosen on the TrackNet train games (hr1280_v1)
SELECT = {"follow": 0.1, "base_px": 10.0, "speed_px_s": 900.0}   # local_archive/articrafts/ball_select_v2/tune_new.py
CROP, THUMB = 64, (160, 90)


def gpu_top_peaks(hm: torch.Tensor, k: int = 3, min_score: float = 0.05, size: int = 17) -> np.ndarray:
    """Batched rallyvision.ball_select.top_peaks: [B,H,W] -> [B,k,3] (x, y, score; NaN padded).

    Local maxima in a size x size window, positions refined by the centroid of the 5x5 neighbourhood
    above half the peak value.
    """
    b, h, w = hm.shape
    is_max = (hm == F.max_pool2d(hm[:, None], size, 1, size // 2)[:, 0]) & (hm >= min_score)
    score, idx = torch.where(is_max, hm, torch.zeros_like(hm)).flatten(1).topk(k, dim=1)
    ys, xs = idx // w, idx % w
    off = torch.arange(-2, 3, device=hm.device)
    yy = (ys[..., None, None] + off[:, None]).clamp(0, h - 1).expand(b, k, 5, 5)
    xx = (xs[..., None, None] + off[None, :]).clamp(0, w - 1).expand(b, k, 5, 5)
    patch = hm[torch.arange(b, device=hm.device)[:, None, None, None], yy, xx]
    wgt = (patch - 0.5 * score[..., None, None]).clamp(min=0)
    norm = wgt.sum((-1, -2)).clamp(min=1e-12)
    out = torch.stack([(wgt * xx).sum((-1, -2)) / norm, (wgt * yy).sum((-1, -2)) / norm, score], -1)
    out[score <= 0] = float("nan")
    return out.float().cpu().numpy()


def merged_segments(clips, back):
    """Merge [start - back, end) ranges of clips into disjoint segments: [(lo, hi, [clips])]."""
    segs = []
    for c in sorted(clips, key=lambda c: c["start"]):
        lo, hi = max(c["start"] - back, 0), c["end"]
        if segs and lo <= segs[-1][1]:
            segs[-1][1] = max(segs[-1][1], hi)
            segs[-1][2].append(c)
        else:
            segs.append([lo, hi, [c]])
    return segs


def gpu_keypoints(hm: torch.Tensor, threshold: float = 0.2) -> np.ndarray:
    """Batched rallyvision.data.heatmap.decode_peak per keypoint channel: [B,14,H,W] -> [B,14,2] (NaN = missing).

    Argmax per channel, refined by the centroid of its 5x5 neighbourhood weighted by (value - threshold/2).
    """
    b, k, h, w = hm.shape
    flat = hm.flatten(2)
    score, idx = flat.max(dim=2)
    ys, xs = idx // w, idx % w
    off = torch.arange(-2, 3, device=hm.device)
    yy = ys[..., None, None] + off[:, None]
    xx = xs[..., None, None] + off[None, :]
    inside = (yy >= 0) & (yy < h) & (xx >= 0) & (xx < w)
    yc, xc = yy.clamp(0, h - 1).expand(b, k, 5, 5), xx.clamp(0, w - 1).expand(b, k, 5, 5)
    bi = torch.arange(b, device=hm.device)[:, None, None, None]
    ki = torch.arange(k, device=hm.device)[None, :, None, None]
    patch = hm[bi, ki, yc, xc]
    wgt = ((patch - 0.5 * threshold).clamp(min=0) * inside).float()
    tot = wgt.sum((-1, -2))
    cx = torch.where(tot > 1e-8, (wgt * xc).sum((-1, -2)) / tot.clamp(min=1e-12), xs.float())
    cy = torch.where(tot > 1e-8, (wgt * yc).sum((-1, -2)) / tot.clamp(min=1e-12), ys.float())
    out = torch.stack([cx, cy], -1)
    out[score < threshold] = float("nan")
    return out.cpu().numpy().astype(np.float64)


class FrameReader:
    """Frames by exact index. Reading runs in a background thread so decoding overlaps the GPU work.

    Long forward jumps use a seek, but only if a seek was checked to land on exactly the same frame
    as sequential decoding for this file; otherwise frames are skipped one by one (always exact).
    """
    SEEK_MIN = 300

    def __init__(self, path):
        self.path = str(path)
        self.cap = cv2.VideoCapture(self.path)
        if not self.cap.isOpened():
            raise FileNotFoundError(path)
        self.pos = 0
        self.seek_ok = self._seek_is_exact()

    def _seek_is_exact(self, k: int = 450) -> bool:
        seq = cv2.VideoCapture(self.path)
        for _ in range(k + 1):
            ok, ref = seq.read()
            if not ok:
                return False
        jump = cv2.VideoCapture(self.path)
        jump.set(cv2.CAP_PROP_POS_FRAMES, k)
        ok, got = jump.read()
        return bool(ok and np.array_equal(ref, got))

    def read_range(self, lo, hi):
        if lo < self.pos:
            raise ValueError(f"frame {lo} is behind the reader ({self.pos})")
        if self.seek_ok and lo - self.pos > self.SEEK_MIN:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, lo)
            self.pos = lo
        q = queue.Queue(maxsize=64)

        def work():
            try:
                while self.pos < lo:
                    self.cap.grab()
                    self.pos += 1
                for i in range(lo, hi):
                    ok, frame = self.cap.read()
                    self.pos += 1
                    if not ok:
                        q.put(EOFError(f"frame {i} could not be read"))
                        return
                    q.put(frame)
            finally:
                q.put(None)

        threading.Thread(target=work, daemon=True).start()
        while (item := q.get()) is not None:
            if isinstance(item, Exception):
                raise item
            yield item


def process_segment(reader, lo, hi, fps, ball_model, court_model, device, batch, court_every):
    """Per frame of [lo, hi): peaks [n,3,3], H, H_valid, thumbs, and the frames kept on the GPU (uint8 RGB)."""
    far, near = input_offsets(fps)
    n = hi - lo
    peaks = np.full((n, 3, 3), np.nan)
    Hs, valid, done = np.full((n, 3, 3), np.nan), np.zeros(n, bool), np.zeros(n, bool)
    thumbs = np.zeros((n, THUMB[1], THUMB[0], 3), np.uint8)
    gpu, pending = [], []
    amp = torch.bfloat16 if device.startswith("cuda") and torch.cuda.is_bf16_supported() else torch.float16

    def flush():
        idx = list(pending)
        pending.clear()
        x = torch.stack([torch.cat([gpu[max(i - far, 0)], gpu[max(i - near, 0)], gpu[i]]) for i in idx]).float().div_(255)
        cur = x[:, -3:]
        thumbs[idx] = (F.interpolate(cur, size=(THUMB[1], THUMB[0]), mode="area") * 255).round().byte() \
            .permute(0, 2, 3, 1).cpu().numpy()
        cj = [j for j, i in enumerate(idx) if i % court_every == 0 or i == n - 1]
        with torch.no_grad(), torch.autocast("cuda", dtype=amp, enabled=device.startswith("cuda")):
            hm = torch.sigmoid(ball_model.forward_logits(x).float())[:, 0]
            chm = court_model.predict_heatmap(F.interpolate(cur[cj], size=(COURT_H, COURT_W), mode="area")).float() \
                if cj else None
        p = gpu_top_peaks(hm)
        p[..., :2] /= 2                      # 1280x720 -> 640x360, the pixels of the tracker parameters
        peaks[idx] = p
        for j, kp in zip(cj, gpu_keypoints(chm, 0.2) if cj else []):
            c = fit_homography(kp)
            if c["H"] is not None:
                Hs[idx[j]] = c["H"]
            valid[idx[j]], done[idx[j]] = c["valid"], True

    with torch.no_grad():
        for i, bgr in enumerate(reader.read_range(lo, hi)):
            if bgr.shape[:2] != (BALL_H, BALL_W):
                bgr = cv2.resize(pad_16x9(bgr)[0], (BALL_W, BALL_H), interpolation=cv2.INTER_AREA)
            gpu.append(torch.from_numpy(bgr).to(device).permute(2, 0, 1).flip(0).contiguous())   # BGR -> RGB
            pending.append(i)
            if len(pending) == batch:
                flush()
        if pending:
            flush()
    # the court changes slowly: frames between court evaluations take the nearest evaluated one
    ev = np.flatnonzero(done)
    near_ev = ev[np.clip(np.searchsorted(ev, np.arange(n)), 0, len(ev) - 1)]
    prev_ev = ev[np.clip(np.searchsorted(ev, np.arange(n)) - 1, 0, len(ev) - 1)]
    nearest = np.where(np.abs(near_ev - np.arange(n)) <= np.abs(prev_ev - np.arange(n)), near_ev, prev_ev)
    return peaks, Hs[nearest], valid[nearest], thumbs, gpu


def clip_outputs(c, lo, peaks, Hs, valid, thumbs, gpu, fps):
    a, b = c["start"] - lo, c["end"] - lo
    pk = peaks[a:b]
    plist = [r[np.isfinite(r[:, 2])] for r in pk]
    balls = drop_static(select_ball(plist, fps, acquire=BALL_THRESHOLD, **SELECT), fps)
    tracker = BallTracker(max_jump_px=100.0, max_gap_frames=int(round(4 * fps / 30)))
    for f, p in enumerate(balls):
        tracker.update(f, p)
    trk = tracker.finalize()
    track = np.array([(d.x, d.y) if d.has_position else (np.nan, np.nan) for d in trk], dtype=float)
    src = np.array([0 if not d.has_position else (1 if d.source == "detected" else 2) for d in trk], dtype=np.int8)

    # 64x64 crops centred on the ball (last known position when missing), zero padded outside the frame
    crop_xy = np.full((len(pk), 2), np.nan)
    crops = torch.zeros((len(pk), 3, CROP, CROP), dtype=torch.uint8, device=gpu[0].device)
    last = np.array([COURT_W / 2, COURT_H / 2])
    for f in range(len(pk)):
        if src[f]:
            last = track[f]
        x0, y0 = int(round(last[0] * 2)) - CROP // 2, int(round(last[1] * 2)) - CROP // 2
        sx0, sy0 = max(x0, 0), max(y0, 0)
        sx1, sy1 = min(x0 + CROP, BALL_W), min(y0 + CROP, BALL_H)
        if sx1 > sx0 and sy1 > sy0:
            crops[f, :, sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = gpu[a + f][:, sy0:sy1, sx0:sx1]
        crop_xy[f] = last
    crops = crops.permute(0, 2, 3, 1).cpu().numpy()
    return dict(peaks=pk, ball=np.array([p if p is not None else (np.nan, np.nan) for p in balls], float),
                track=track, track_src=src, H=Hs[a:b], H_valid=valid[a:b], crops=crops, crop_xy=crop_xy,
                thumbs=thumbs[a:b], fps=fps, start=c["start"], events=json.dumps(c["events"]))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labels", type=Path, required=True)
    ap.add_argument("--video", type=Path, required=True)
    ap.add_argument("--source", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--ball-checkpoint", type=Path, required=True)
    ap.add_argument("--court-checkpoint", type=Path, required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--court-every", type=int, default=1, help="court model on every k-th frame (1 = every frame)")
    ap.add_argument("--max-clips", type=int, default=0, help="only the first N clips (smoke test)")
    args = ap.parse_args()

    meta = load_videos(args.labels)[args.source]
    clips = [dict(c, split=s) for s, cs in load_all(args.labels).items() for c in cs if c["source"] == args.source]
    clips.sort(key=lambda c: c["start"])
    if args.max_clips:
        clips = clips[:args.max_clips]
    out = args.out / args.source
    out.mkdir(parents=True, exist_ok=True)
    todo = [c for c in clips if not (out / f"{c['video']}.npz").exists()]
    print(f"{args.source}: {len(clips)} clips, {len(todo)} to do", flush=True)
    if not todo:
        return

    reader = FrameReader(args.video)
    fps = reader.cap.get(cv2.CAP_PROP_FPS)
    print(f"  {int(reader.cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(reader.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}, "
          f"{fps:.3f} fps, exact seek: {reader.seek_ok}", flush=True)
    total = int(reader.cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if abs(fps - meta["fps"]) > 0.05:
        raise SystemExit(f"video fps {fps:.3f} != labelled {meta['fps']:.3f}: wrong download format")
    if total < max(c["end"] for c in clips):
        raise SystemExit(f"video has {total} frames, labels need {max(c['end'] for c in clips)}")
    fps = meta["fps"]

    cfg = load_config()
    ball_model = build_model(cfg)
    load_model_weights(args.ball_checkpoint, ball_model)
    ball_model.to(args.device).eval()
    court_model, _ = load_court_model(args.court_checkpoint, args.device)

    far, _ = input_offsets(fps)
    for lo, hi, seg_clips in merged_segments(todo, far):
        peaks, Hs, valid, thumbs, gpu = process_segment(reader, lo, hi, fps, ball_model, court_model, args.device,
                                                        args.batch, args.court_every)
        for c in seg_clips:
            np.savez(out / f"{c['video']}.npz", split=c["split"],
                     **clip_outputs(c, lo, peaks, Hs, valid, thumbs, gpu, fps))
        del gpu
        print(f"  frames {lo}-{hi}: {len(seg_clips)} clip(s) written", flush=True)
    (out / "_done.json").write_text(json.dumps({"clips": len(clips), "fps": fps, "frames": total}))


if __name__ == "__main__":
    main()
