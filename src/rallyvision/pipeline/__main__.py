"""RallyVision: tennis video -> ball track, court homography, bounce events, bounces on a top-down court.

    rallyvision --video match.mp4 --out results/match [--layout side|pip] [--crop TOP BOTTOM]
    (same as: python -m rallyvision.pipeline ...)

Formal models only, loaded from configs/paths.yaml; nothing is trained at inference. Bounce detection needs
the external E2E-Spot code and pretrained model (configs/paths.yaml or --spot-code / --spot-model); every
input is checked before any model runs. Writes:
    bounces.json    court-space bounces (frame, time, confidence, side, court x/y in metres)
    analysis.npz    per-frame ball track, gated peaks, homographies, bounce probabilities
    bounce_map.png  all bounces on the top-down court
    rally.mp4       the untouched video (with its audio) and the top-down court (side panel or top-right inset)
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import cv2
import numpy as np

from rallyvision.common.device import best_device
from rallyvision.common.locations import artifact, pipeline_config
from rallyvision.pipeline.analyzer import analyze
from rallyvision.pipeline.render import bounce_map, render_video

SPOT_HELP = """E2E-Spot (external, needed by bounce detection) is not set up: {problem}
Get it once and pass the two folders (or set e2e_spot_code / e2e_spot_model in configs/paths.yaml):
    git clone https://github.com/jhong93/spot        # on numpy >= 1.24 replace np.int/np.float/np.bool by int/float/bool
    git clone https://github.com/jhong93/e2e-spot-models   # git lfs; folder tennis_rny008gsm_gru_rgb
    rallyvision ... --spot-code spot --spot-model e2e-spot-models/tennis_rny008gsm_gru_rgb"""


def spot_paths(code, model):
    """The two E2E-Spot folders, or a clear error naming what is missing."""
    try:
        code = Path(code or artifact("bounce_detection", "e2e_spot_code"))
        model = Path(model or artifact("bounce_detection", "e2e_spot_model"))
    except FileNotFoundError as e:
        raise SystemExit(SPOT_HELP.format(problem=e))
    for need in (code / "train_e2e.py", code / "data/tennis/class.txt", model / "config.json"):
        if not need.exists():
            raise SystemExit(SPOT_HELP.format(problem=f"{need} not found"))
    if not list(model.glob("checkpoint_*.pt")):
        raise SystemExit(SPOT_HELP.format(problem=f"no checkpoint_*.pt in {model} (git lfs pull?)"))
    return code, model


def check_inputs(args):
    """Fail before any model runs: video, model weights, E2E-Spot, ffmpeg."""
    cap = cv2.VideoCapture(str(args.video))
    ok = cap.isOpened() and cap.read()[0]
    cap.release()
    if not ok:
        raise SystemExit(f"cannot read video: {args.video}")
    for module, key in (("ball_tracking", "model"), ("court_detection", "model"), ("bounce_detection", "transformer_v3")):
        if not artifact(module, key).exists():
            raise SystemExit(f"model weights missing: {artifact(module, key)} (configs/paths.yaml {module}.{key})")
    if not args.no_video and shutil.which("ffmpeg") is None:
        raise SystemExit("ffmpeg not found on PATH (needed to write rally.mp4; or use --no-video)")
    return spot_paths(args.spot_code, args.spot_model)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--layout", choices=("side", "pip"), default=None, help="default: configs/pipeline.yaml")
    ap.add_argument("--crop", type=int, nargs=2, default=None, metavar=("TOP", "BOTTOM"),
                    help="source rows kept in the rendered video (to remove black bars)")
    ap.add_argument("--no-video", action="store_true")
    ap.add_argument("--spot-code", type=Path, default=None, help="E2E-Spot checkout (default: configs/paths.yaml)")
    ap.add_argument("--spot-model", type=Path, default=None, help="E2E-Spot tennis model dir (default: configs/paths.yaml)")
    ap.add_argument("--device", default=best_device())
    args = ap.parse_args()
    spot_code, spot_model = check_inputs(args)
    cfg = pipeline_config()
    args.out.mkdir(parents=True, exist_ok=True)
    r = analyze(args.video, args.device, cfg, spot_code, spot_model)
    (args.out / "bounces.json").write_text(json.dumps(
        {"video": args.video.name, "fps": r["fps"], "detector": "RallyVision Event Transformer v3",
         "decoding": r["bounce_aux"]["decoding"], "court_axes": "metres; origin net centre, x right, y towards the far baseline",
         "bounces": r["bounces"]}, indent=1))
    np.savez(args.out / "analysis.npz", fps=r["fps"], ball=r["ball"], H=r["H"], H_valid=r["H_valid"],
             peaks=np.array([(f, *row) for f, p in enumerate(r["peaks"]) for row in p], dtype=float).reshape(-1, 4),
             bounce_prob=r["bounce_aux"]["prob"].astype(np.float16), bounce_frame=r["bounce_aux"]["video_frame"])
    cv2.imwrite(str(args.out / "bounce_map.png"), bounce_map(r["bounces"]))
    print(f"{len(r['bounces'])} bounces, court valid in {r['H_valid'].mean():.1%} of frames -> {args.out}")
    if not args.no_video:
        crop = args.crop or (None, None)
        print(render_video(args.video, r["bounces"], args.out / "rally.mp4", args.layout or cfg["render"]["layout"], *crop))


if __name__ == "__main__":
    main()
