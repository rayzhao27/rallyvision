#!/usr/bin/env python3
"""Run the tracker over a video and write an annotated MP4.

Full-length videos are a Colab job; a few seconds of footage is fine locally.

    python scripts/infer_video.py --video rally.mp4 --run v1
    python scripts/infer_video.py --video rally.mp4 --checkpoint best.pt --max-frames 120
"""

from __future__ import annotations

import argparse
from pathlib import Path

from rallyvision.common.cli import add_config_args, add_device_args, build_config, fail, init_logging

from rallyvision.common.checkpoint import load_model_weights
from rallyvision.common.device import DeviceError, resolve_device_from_config
from rallyvision.ball_tracking.inference.video import run_video_inference
from rallyvision.common.logging_utils import banner
from rallyvision.ball_tracking.model import build_model
from rallyvision.common.paths import resolve_run
from rallyvision.common.utils import write_json


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_config_args(parser)
    add_device_args(parser)
    parser.add_argument("--video", type=Path, required=True, help="input video file")
    parser.add_argument("--run", default=None, help="run name, run directory, or 'latest'")
    parser.add_argument("--checkpoint", type=Path, default=None, help="explicit checkpoint path")
    parser.add_argument("--output", type=Path, default=None, help="output MP4 path")
    parser.add_argument("--max-frames", type=int, default=None, help="stop after N frames")
    parser.add_argument("--threshold", type=float, default=None, help="override detection threshold")
    parser.add_argument("--draw-heatmap", action="store_true", help="overlay the predicted heatmap")
    parser.add_argument("--no-video", action="store_true", help="track only; write JSON, no MP4")
    parser.add_argument("--save-track", type=Path, default=None, help="where to write the track JSON")
    args = parser.parse_args()

    logger = init_logging(args)
    overrides = {}
    if args.threshold is not None:
        overrides["inference.detection_threshold"] = args.threshold
    cfg = build_config(args, overrides)

    if not args.video.is_file():
        fail(f"video not found: {args.video}")
        return 1

    run_paths = None
    checkpoint = args.checkpoint
    if checkpoint is None:
        try:
            run_paths = resolve_run(cfg, args.run)
        except FileNotFoundError as exc:
            fail(str(exc))
            return 1
        checkpoint = run_paths.best_checkpoint
        if not checkpoint.is_file():
            checkpoint = run_paths.last_checkpoint
    if not Path(checkpoint).is_file():
        fail(f"checkpoint not found: {checkpoint}")
        return 1

    try:
        device_info = resolve_device_from_config(cfg)
    except DeviceError as exc:
        fail(str(exc))
        return 1

    model = build_model(cfg)
    load_model_weights(checkpoint, model)

    output_path = None
    if not args.no_video:
        if args.output is not None:
            output_path = args.output
        elif run_paths is not None:
            output_path = run_paths.videos / f"{args.video.stem}_annotated.mp4"
        else:
            output_path = args.video.with_name(f"{args.video.stem}_annotated.mp4")

    logger.info(banner("video inference"))
    logger.info("checkpoint       : %s", checkpoint)
    logger.info("device           : %s (%s)", device_info.device, device_info.name)

    result = run_video_inference(
        model,
        args.video,
        output_path,
        device_info,
        cfg,
        max_frames=args.max_frames,
        draw_heatmap=args.draw_heatmap or None,
    )

    track_path = args.save_track
    if track_path is None:
        base = run_paths.videos if run_paths is not None else args.video.parent
        track_path = Path(base) / f"{args.video.stem}_track.json"
    write_json(track_path, result)

    stats = result["stats"]
    print()
    print(banner("tracking summary"))
    print(f"frames processed : {result['video']['frames_processed']}")
    print(f"detected         : {stats['detected']}")
    print(f"interpolated     : {stats['interpolated']}")
    print(f"rejected (jump)  : {stats['rejected']}")
    print(f"coverage         : {100 * stats['coverage']:.1f}%")
    print(f"mean step        : {stats['mean_step_px']:.1f} px/frame")
    print(f"track json       : {track_path}")
    if result["output_video"]:
        print(f"annotated video  : {result['output_video']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
