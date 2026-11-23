"""RallyVision temporal event detector on E2E-Spot Tennis: frozen E2E-Spot visual backbone + our Transformer.

Run on Colab, two stages:

  extract  frames of every packaged clip (OpenSportsLab/soccernetpro-localization-tennis, the E2E-Spot
           tennis clips + labels) go through the pretrained E2E-Spot RegNetY-008 + GSM backbone exactly as
           in its own inference (398x224, centre crop 224, ImageNet normalisation, 100-frame clips with
           50-frame overlap). Saved per clip: backbone features [n, 768] and, for the baseline, the
           original E2E-Spot GRU's per-frame class probabilities [n, 7]. Optional: the Sinner video.
  train    our Transformer (3 layers, d_model 128, 4 heads, 100-frame window) on the cached features of
           the train split; epoch and per-class thresholds chosen on val; test split scored once, and the
           E2E-Spot GRU scored with the same decoding, matching and val-chosen thresholds.

    python -m rallyvision.bounce_detection.training.train extract --data /content/osl_tennis --spot /content/spot \
        --spot-model /content/e2e-spot-models/tennis_rny008gsm_gru_rgb --cache /content/feat [--sinner video.mov]
    python -m rallyvision.bounce_detection.training.train train --cache /content/feat --out /content/run

Classes (E2E-Spot class.txt order, 0 = background): far_court_bounce, far_court_swing, far_court_serve,
near_court_bounce, near_court_swing, near_court_serve.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from rallyvision.bounce_detection.evaluation.metrics import (BASE_SEP, GROUPS, SEP_GRID, THR, choose_decoding,  # noqa: F401
                                                            close_pairs, evaluate, group_peaks, match, nms_diff, prf,
                                                            score_group)
from rallyvision.bounce_detection.inference.decode import peaks
from rallyvision.bounce_detection.inference.predict import predict
from rallyvision.bounce_detection.model.event_transformer.backbone import (CROP, H_IN, W_IN, backbone, decode,  # noqa: F401
                                                                          load_spot, pad169)
from rallyvision.bounce_detection.model.event_transformer.model import CLASSES, EventTransformer
from rallyvision.bounce_detection.training.data import GROUP_OF, dilate, gpu_windows, load_split, take  # noqa: F401
from rallyvision.bounce_detection.training.extract_features import SPLITS, extract  # noqa: F401

SINNER_GT = {"S1": (57, "far"), "S2": (163, "near"), "S3": (248, "far"), "S4": (308, "near"), "S5": (389, "far"),
             "S6": (456, "near"), "S7": (534, "far"), "S8": (605, "near"), "S9": (676, "far"), "S10": (764, "near"),
             "S11": (864, "far"), "S12": (950, "far")}
UNCERTAIN = {"S12": "drop volley; label not re-checked on frame sheets yet"}


def train_one(args, data, gpu, seed, device):
    torch.manual_seed(seed)
    np.random.seed(seed)
    (tf, ty, ts), (vf, vy, vs) = gpu["train"], gpu["val"]
    model = EventTransformer(tf.shape[1], args.d_model, args.layers, args.heads, args.ff, args.window,
                             dropout=args.dropout, feat_drop=args.feat_drop).to(device)
    steps = len(ts) // args.batch
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, args.lr, total_steps=max(args.epochs * steps, 20), pct_start=0.1)
    weight = torch.tensor([1.0] + [args.fg_weight] * len(CLASSES), device=device)   # E2E-Spot uses fg weight 5
    loss_fn = nn.CrossEntropyLoss(weight=weight)
    amp = dict(device_type="cuda", dtype=torch.bfloat16, enabled=device == "cuda")
    history, best, t0 = [], (-1.0, 0, None), time.time()
    for ep in range(1, args.epochs + 1):
        model.train()
        perm, tl = torch.randperm(len(ts), device=device), 0.0
        for k in range(steps):
            x, y = take(tf, ty, ts[perm[k * args.batch:(k + 1) * args.batch]], args.window)
            with torch.autocast(**amp):
                loss = loss_fn(model(x).float().reshape(-1, len(CLASSES) + 1), y.reshape(-1))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            tl += loss.item()
        model.eval()
        with torch.no_grad(), torch.autocast(**amp):
            vl = [loss_fn(model(x).float().reshape(-1, len(CLASSES) + 1), y.reshape(-1)).item()
                  for x, y in (take(vf, vy, vs[i:i + 1024], args.window) for i in range(0, len(vs), 1024))]
        vprobs = [predict(model, c["feat"], args.window, device) for c in data["val"]]
        f1s = {g: max(score_group(data["val"], pk, g, t, args.select_tol)["F1"] for t in THR)
               for g, pk in ((g, group_peaks(data["val"], vprobs, g)) for g in GROUP_OF.values())}
        macro = float(np.mean(list(f1s.values())))
        history.append({"seed": seed, "epoch": ep, "train_loss": tl / steps, "val_loss": float(np.mean(vl)),
                        "val_macro_f1": macro, **{f"val_f1_{k}": v for k, v in f1s.items()}, "seconds": time.time() - t0})
        print(f"seed {seed} epoch {ep:2d}  train loss {tl / steps:.4f}  val loss {np.mean(vl):.4f}  val macro F1 {macro:.3f}  "
              f"far bounce {f1s['far_bounce']:.3f}  near bounce {f1s['near_bounce']:.3f}  ({time.time() - t0:.0f}s)", flush=True)
        if macro > best[0]:
            best = (macro, ep, {k: v.detach().clone() for k, v in model.state_dict().items()})
        elif ep - best[1] >= args.patience:
            print(f"seed {seed}: early stop (no val improvement for {args.patience} epochs)", flush=True)
            break
    model.load_state_dict(best[2])
    return model, best, history, time.time() - t0


def train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    args.out.mkdir(parents=True, exist_ok=True)
    data = {s: load_split(args.cache, s) for s in ("train", "val", "test")}
    for c in data["train"]:
        c["y_train"] = dilate(c["y"], args.dilate)
    info = {s: {"clips": len(v), "frames": int(sum(len(c["y"]) for c in v)),
                **{n: int(sum((c["y"] == i + 1).sum() for c in v)) for i, n in enumerate(CLASSES)}} for s, v in data.items()}
    print("data:", json.dumps(info), flush=True)
    gpu = {"train": gpu_windows(data["train"], "y_train", args.window, args.stride, device),
           "val": gpu_windows(data["val"], "y", args.window, args.window, device)}
    print(f"train windows on GPU: {len(gpu['train'][2])}, batch {args.batch}", flush=True)

    runs = [train_one(args, data, gpu, seed, device) for seed in args.seeds]
    seed_results = []
    for (m, best, hist, secs), seed in zip(runs, args.seeds):
        pv = [predict(m, c["feat"], args.window, device) for c in data["val"]]
        pt = [predict(m, c["feat"], args.window, device) for c in data["test"]]
        d0 = choose_decoding(data["val"], pv, args.select_tol, (BASE_SEP,))
        d1 = choose_decoding(data["val"], pv, args.select_tol)
        seed_results.append({"seed": seed, "best_epoch": best[1], "epochs_run": max(h["epoch"] for h in hist),
                             "val_macro_f1": best[0], "train_seconds": secs, "decoding_base": d0, "decoding_nms": d1,
                             "test_base": evaluate(data["test"], pt, d0), "test": evaluate(data["test"], pt, d1)})
    k = int(np.argmax([r["val_macro_f1"] for r in seed_results]))          # model kept: best on val
    model, best, _, train_seconds = runs[k]
    history = [h for _, _, hs, _ in runs for h in hs]
    n_params = sum(p.numel() for p in model.parameters())
    print(f"kept seed {args.seeds[k]} (best val macro F1 {best[0]:.3f}, epoch {best[1]}); {n_params / 1e3:.0f}k parameters")

    # thresholds on val (both models), test scored once
    probs = {s: {"ours": [predict(model, c["feat"], args.window, device) for c in data[s]],
                 "e2e_spot_gru": [c["spot"] for c in data[s]]} for s in ("val", "test")}
    dec_base = {m: choose_decoding(data["val"], probs["val"][m], args.select_tol, (BASE_SEP,)) for m in probs["val"]}
    thr = {m: choose_decoding(data["val"], probs["val"][m], args.select_tol) for m in probs["val"]}   # with NMS window
    results = {m: {s: evaluate(data[s], probs[s][m], thr[m]) for s in ("val", "test")} for m in thr}
    results_base = {m: {s: evaluate(data[s], probs[s][m], dec_base[m]) for s in ("val", "test")} for m in thr}
    audit = {g: {"sep": thr["ours"][g]["sep"],
                 "close_label_pairs_val": close_pairs(data["val"], g, thr["ours"][g]["sep"]),
                 "close_label_pairs_test": close_pairs(data["test"], g, thr["ours"][g]["sep"]),
                 "test_vs_base_window": nms_diff(data["test"], probs["test"]["ours"], g, dec_base["ours"][g], thr["ours"][g])}
             for g in ("bounce", "far_bounce", "near_bounce")}

    # latency / throughput of our temporal head (backbone throughput is in extract_info.json)
    x1 = torch.randn(1, args.window, data["train"][0]["feat"].shape[1], device=device)
    xb = torch.randn(64, args.window, x1.shape[-1], device=device)
    with torch.no_grad():
        for _ in range(5):
            model(x1); model(xb)
        if device == "cuda":
            torch.cuda.synchronize()
        t = time.time()
        for _ in range(50):
            model(x1)
        if device == "cuda":
            torch.cuda.synchronize()
        lat = (time.time() - t) / 50
        t = time.time()
        for _ in range(20):
            model(xb)
        if device == "cuda":
            torch.cuda.synchronize()
        thr_fps = 20 * 64 * args.window / (time.time() - t)

    summary = {key: {g: {t: {"F1_mean": float(np.mean([r[key][g][t]["F1"] for r in seed_results])),
                             "F1_std": float(np.std([r[key][g][t]["F1"] for r in seed_results]))}
                         for t in ("tol4", "tol2", "tol1")} for g in GROUPS} for key in ("test_base", "test")}
    result = {"data": info, "params": n_params, "best_epoch": best[1], "train_seconds": train_seconds,
              "seeds": seed_results, "seed_summary": summary,
              "decoding": thr, "decoding_base": dec_base, "results": results, "results_base": results_base,
              "nms_audit": audit, "history": history,
              "latency_ms_per_100_frame_window": lat * 1000, "head_throughput_frames_per_s": thr_fps,
              "config": {k: getattr(args, k) for k in ("window", "d_model", "layers", "heads", "ff", "fg_weight", "lr",
                                                       "epochs", "stride", "select_tol", "seeds", "batch", "dropout",
                                                       "feat_drop", "dilate", "weight_decay", "patience")}}
    extract_info = args.cache / "extract_info.json"
    if extract_info.exists():
        result["extract"] = json.loads(extract_info.read_text())

    # Sinner video: independent demo test (never used for training or thresholds)
    sinner = args.cache / "sinner.npz"
    if sinner.exists():
        z = np.load(sinner)
        vf = z["video_frame"]
        result["sinner"] = {}
        gt = [(n, f, sd) for n, (f, sd) in SINNER_GT.items()]
        gt30 = [int(np.argmin(np.abs(vf - f))) for _, f, _ in gt]
        p_ours = predict(model, z["feat"], args.window, device)
        for name, p, d in (("ours_base_window", p_ours, dec_base["ours"]["bounce"]), ("ours", p_ours, thr["ours"]["bounce"]),
                           ("e2e_spot_gru", z["spot"].astype(np.float32), thr["e2e_spot_gru"]["bounce"])):
            b = p[:, GROUPS["bounce"]].sum(1)
            pred = [t for t, sc in peaks(b, d["sep"]) if sc >= d["thr"]]
            side = ["far" if p[t, 1] >= p[t, 4] else "near" for t in pred]
            pairs = match(pred, gt30, 2)                                  # ±2 frames at 30 fps (≈ ±4 video frames)
            matched_pred = {i for i, _ in pairs}
            per = {}
            for j, (n, f, sd) in enumerate(gt):
                near = slice(max(gt30[j] - 2, 0), gt30[j] + 3)
                hit = [i for i, jj in pairs if jj == j]
                per[n] = {"frame": f, "side": sd, "max_bounce_prob": float(b[near].max()), "found": bool(hit),
                          "pred_frame": int(vf[pred[hit[0]]]) if hit else None,
                          "pred_side": side[hit[0]] if hit else None, "note": UNCERTAIN.get(n, "")}
            tp = len(pairs)
            result["sinner"][name] = {"decoding": d, "pred_bounces": [(int(vf[t]), sd, round(float(b[t]), 3)) for t, sd in zip(pred, side)],
                                      "false": [int(vf[pred[i]]) for i in range(len(pred)) if i not in matched_pred],
                                      "per_bounce": per, "P_R_F1": prf(tp, len(pred) - tp, len(gt) - tp),
                                      "tp": tp, "fp": len(pred) - tp, "fn": len(gt) - tp}

    torch.save({"model": best[2], "config": result["config"], "in_dim": int(x1.shape[-1]), "classes": CLASSES,
                "decoding": thr["ours"]}, args.out / "event_transformer.pt")
    (args.out / "result.json").write_text(json.dumps(result, indent=1, default=float))
    with open(args.out / "history.csv", "w") as f:
        keys = list(history[0])
        f.write(",".join(keys) + "\n")
        for h in history:
            f.write(",".join(f"{h[k]:.6g}" if isinstance(h[k], float) else str(h[k]) for k in keys) + "\n")
    report(result)


def report(r):
    print(f"\nbest epoch {r['best_epoch']}, {r['params'] / 1e3:.0f}k params, trained in {r['train_seconds']:.0f}s; "
          f"head latency {r['latency_ms_per_100_frame_window']:.2f} ms / 100-frame window, "
          f"{r['head_throughput_frames_per_s']:.0f} frames/s")
    if "extract" in r:
        e = r["extract"]
        print(f"backbone extraction: {e['frames']} frames in {e['seconds']:.0f}s ({e['frames'] / max(e['seconds'], 1):.0f} frames/s incl. decoding)")
    print("\nNMS windows chosen on val (frames):", {g: r["decoding"]["ours"][g]["sep"] for g in ("bounce", "far_bounce", "near_bounce", "swing", "serve")},
          "| GRU:", {g: r["decoding"]["e2e_spot_gru"][g]["sep"] for g in ("bounce", "far_bounce", "near_bounce")})
    for tol in ("tol4", "tol2", "tol1"):
        print(f"\nTEST ({tol[3:]} frames tolerance; thresholds and NMS windows chosen on val)")
        print(f"  {'event':12s} {'ours, 5-frame window':22s} {'ours, val-chosen NMS':22s} {'E2E-Spot GRU, val NMS':22s}  timing err ours/GRU")
        for g in ("bounce", "far_bounce", "near_bounce", "swing", "serve"):
            a0, a, b = (r["results_base"]["ours"]["test"][g][tol], r["results"]["ours"]["test"][g][tol],
                        r["results"]["e2e_spot_gru"]["test"][g][tol])
            f = lambda x: f"{x['P']:.3f}/{x['R']:.3f}/{x['F1']:.3f}"
            print(f"  {g:12s} {f(a0):22s} {f(a):22s} {f(b):22s}  {a['timing_err_mean']:.2f} / {b['timing_err_mean']:.2f}")
    print("\nNMS audit (our model, test, ±4 frames, chosen window vs 5-frame window):")
    for g, v in r["nms_audit"].items():
        print(f"  {g:12s} window {v['sep']:2d} | labelled pairs closer than the window: val {v['close_label_pairs_val'][0]}/{v['close_label_pairs_val'][1]}, "
              f"test {v['close_label_pairs_test'][0]}/{v['close_label_pairs_test'][1]} | events lost {v['test_vs_base_window']['lost']}, "
              f"gained {v['test_vs_base_window']['gained']}, false detections removed {v['test_vs_base_window']['fp_removed']}, "
              f"added {v['test_vs_base_window']['fp_added']}")
    print("\nour model over seeds (test F1 mean ± std, ±4 / ±2 frames):")
    for x in r["seeds"]:
        print(f"  seed {x['seed']}: best epoch {x['best_epoch']} of {x['epochs_run']} run, val macro F1 {x['val_macro_f1']:.3f}, "
              f"test bounce F1 {x['test_base']['bounce']['tol4']['F1']:.3f} -> {x['test']['bounce']['tol4']['F1']:.3f} with NMS")
    for key, label in (("test_base", "5-frame window"), ("test", "val-chosen NMS")):
        print(f"  {label}:")
        for g in ("bounce", "far_bounce", "near_bounce", "swing", "serve"):
            ss = r["seed_summary"][key][g]
            print(f"    {g:12s} 4f {ss['tol4']['F1_mean']:.3f} ± {ss['tol4']['F1_std']:.3f}   2f {ss['tol2']['F1_mean']:.3f} ± {ss['tol2']['F1_std']:.3f}")
    for m, sres in r.get("sinner", {}).items():
        print(f"\nSINNER ({m}, window {sres['decoding']['sep']}, thr {sres['decoding']['thr']}): TP {sres['tp']} FP {sres['fp']} FN {sres['fn']} | "
              f"P {sres['P_R_F1'][0]:.2f} R {sres['P_R_F1'][1]:.2f} F1 {sres['P_R_F1'][2]:.2f}")
        print("  false detections (video frames):", sres["false"])
        print("  per event:", " ".join(f"{n}:{'ok' if v['found'] else 'MISS'}" for n, v in sres["per_bounce"].items()),
              "| S12 label not re-checked")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="stage", required=True)
    e = sub.add_parser("extract")
    e.add_argument("--data", type=Path, required=True)
    e.add_argument("--spot", type=Path, required=True)
    e.add_argument("--spot-model", type=Path, required=True)
    e.add_argument("--cache", type=Path, required=True)
    e.add_argument("--sinner", type=Path)
    e.add_argument("--threads", type=int, default=8)
    e.add_argument("--max-clips", type=int, default=0, help="per split; 0 = all")
    t = sub.add_parser("train")
    t.add_argument("--cache", type=Path, required=True)
    t.add_argument("--out", type=Path, required=True)
    t.add_argument("--window", type=int, default=100)
    t.add_argument("--d-model", type=int, default=128)
    t.add_argument("--layers", type=int, default=3)
    t.add_argument("--heads", type=int, default=4)
    t.add_argument("--ff", type=int, default=256)
    t.add_argument("--epochs", type=int, default=40)
    t.add_argument("--patience", type=int, default=8, help="early stop after this many epochs without val improvement")
    t.add_argument("--batch", type=int, default=512)
    t.add_argument("--stride", type=int, default=10)
    t.add_argument("--lr", type=float, default=5e-4)
    t.add_argument("--weight-decay", type=float, default=0.05)
    t.add_argument("--dropout", type=float, default=0.3)
    t.add_argument("--feat-drop", type=float, default=0.2, help="dropout on the input backbone features")
    t.add_argument("--dilate", type=int, default=1, help="frames; training labels widened to ±dilate")
    t.add_argument("--fg-weight", type=float, default=5.0)
    t.add_argument("--select-tol", type=int, default=2, help="frames; tolerance for epoch and threshold selection")
    t.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    args = ap.parse_args()
    extract(args) if args.stage == "extract" else train(args)


if __name__ == "__main__":
    main()
