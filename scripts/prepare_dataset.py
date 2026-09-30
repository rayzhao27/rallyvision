#!/usr/bin/env python3
"""Index the raw clips, assign the train/val split, and write manifest.json.

Colab job for the real dataset. Re-running is cheap and idempotent: an existing
manifest is reused unless `--overwrite` is passed, which is what makes a
Drive-cached prepared dataset survive across sessions.

    python scripts/prepare_dataset.py
    python scripts/prepare_dataset.py --cache-frames      # resize once, train fast
    python scripts/prepare_dataset.py --stats-only        # just print the summary
    python scripts/prepare_dataset.py --reassign-splits   # rewrite train/val, keep cached frames
"""

from __future__ import annotations

import argparse
from pathlib import Path

from _common import add_config_args, build_config, fail, init_logging

from rallyvision.config import resolve_path
from rallyvision.data.manifest import DatasetManifest
from rallyvision.data.prepare import PrepareError, prepare_dataset, reassign_manifest_splits
from rallyvision.logging_utils import banner
from rallyvision.utils import dir_size_bytes, format_bytes, write_json


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_config_args(parser)
    parser.add_argument("--raw-dir", type=Path, default=None, help="override data.raw_dir")
    parser.add_argument("--prepared-dir", type=Path, default=None, help="override data.prepared_dir")
    parser.add_argument(
        "--cache-frames",
        dest="cache_frames",
        action="store_true",
        default=None,
        help="write resized copies of every frame (much faster training, smaller on Drive)",
    )
    parser.add_argument(
        "--no-cache-frames", dest="cache_frames", action="store_false", help="read raw frames at train time"
    )
    parser.add_argument("--limit-clips", type=int, default=None, help="only index the first N clips")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="rebuild manifest.json even if one exists (reuses already-cached frames)",
    )
    parser.add_argument(
        "--recache-frames",
        action="store_true",
        help="also re-write every cached frame (slow on Drive; only after changing input size)",
    )
    parser.add_argument(
        "--reassign-splits",
        action="store_true",
        help="keep cached frames; only rewrite train/val on the existing manifest",
    )
    parser.add_argument("--stats-only", action="store_true", help="print stats for an existing manifest")
    args = parser.parse_args()

    logger = init_logging(args)
    cfg = build_config(args)

    prepared_dir = Path(args.prepared_dir) if args.prepared_dir else resolve_path(cfg, cfg.data.prepared_dir)

    if args.stats_only:
        try:
            manifest = DatasetManifest.load(prepared_dir)
        except FileNotFoundError as exc:
            fail(str(exc))
            return 1
    elif args.reassign_splits:
        try:
            manifest = DatasetManifest.load(prepared_dir)
            manifest = reassign_manifest_splits(manifest, cfg)
        except (FileNotFoundError, PrepareError) as exc:
            fail(str(exc))
            return 1
    else:
        try:
            manifest = prepare_dataset(
                cfg,
                raw_root=args.raw_dir,
                prepared_root=prepared_dir,
                cache_frames=args.cache_frames,
                limit_clips=args.limit_clips,
                overwrite=args.overwrite or args.recache_frames,
                recache_frames=args.recache_frames,
            )
        except PrepareError as exc:
            fail(str(exc))
            return 1

    num_frames = int(cfg.data.num_frames)
    stride = int(cfg.data.frame_stride)

    print()
    print(banner("prepared dataset"))
    print(manifest.format_stats(num_frames, stride))
    prepared_root = Path(manifest.prepared_root)
    if prepared_root.is_dir():
        print(f"{'prepared size':<19}: {format_bytes(dir_size_bytes(prepared_root))}")
    print(f"{'manifest':<19}: {prepared_root / 'manifest.json'}")
    print()

    stats_path = write_json(prepared_root / "dataset_stats.json", manifest.stats(num_frames, stride))
    logger.info("wrote %s", stats_path)

    if manifest.usable_samples("val", num_frames, stride) == 0:
        fail(
            "the validation split has no usable samples. "
            "git pull, then: python scripts/prepare_dataset.py --reassign-splits"
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
