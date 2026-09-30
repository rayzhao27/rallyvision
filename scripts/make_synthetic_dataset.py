#!/usr/bin/env python3
"""Generate a tiny synthetic dataset (and optionally a video) for pipeline tests.

This is a fixture, not training data. Metrics measured on it are meaningless;
its only job is to let the full pipeline run locally in seconds.

    python scripts/make_synthetic_dataset.py --out data/raw/synthetic
    python scripts/make_synthetic_dataset.py --video data/raw/demo.mp4
"""

from __future__ import annotations

import argparse
from pathlib import Path

from _common import init_logging

from rallyvision.data.synthetic import generate_dataset, generate_video
from rallyvision.utils import dir_size_bytes, format_bytes


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=Path("data/raw/synthetic"), help="output directory")
    parser.add_argument("--games", type=int, default=2)
    parser.add_argument("--clips-per-game", type=int, default=2)
    parser.add_argument("--frames-per-clip", type=int, default=24)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=360)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--video", type=Path, default=None, help="also write a short synthetic video here")
    parser.add_argument("--video-frames", type=int, default=60)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    logger = init_logging(args)

    root = generate_dataset(
        args.out,
        num_games=args.games,
        clips_per_game=args.clips_per_game,
        frames_per_clip=args.frames_per_clip,
        width=args.width,
        height=args.height,
        seed=args.seed,
    )
    total_frames = args.games * args.clips_per_game * args.frames_per_clip
    logger.info(
        "wrote %d clips / %d frames to %s (%s)",
        args.games * args.clips_per_game,
        total_frames,
        root,
        format_bytes(dir_size_bytes(root)),
    )

    if args.video:
        path = generate_video(
            args.video,
            num_frames=args.video_frames,
            width=args.width,
            height=args.height,
            seed=args.seed,
        )
        logger.info("wrote synthetic video %s", path)

    print(f"\nnext: python scripts/prepare_dataset.py --set data.source=synthetic "
          f"--set data.local_path={root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
