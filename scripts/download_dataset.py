#!/usr/bin/env python3
"""Download the tennis dataset.

Colab job. The dataset is large; running this locally is a waste of bandwidth
and disk.

    python scripts/download_dataset.py --source gdrive --set data.gdrive_id=<id>
    python scripts/download_dataset.py --source kaggle --set data.kaggle_dataset=<slug>
    python scripts/download_dataset.py --source url --set data.url=https://...
    python scripts/download_dataset.py --source synthetic   # tiny fake data, tests only
"""

from __future__ import annotations

import argparse
from pathlib import Path

from _common import add_config_args, build_config, fail, init_logging

from rallyvision.data.sources import DownloadError, download_dataset
from rallyvision.utils import dir_size_bytes, format_bytes


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_config_args(parser)
    parser.add_argument(
        "--source",
        choices=["local", "gdrive", "url", "kaggle", "synthetic"],
        default=None,
        help="override data.source",
    )
    parser.add_argument("--dest", type=Path, default=None, help="override data.raw_dir")
    parser.add_argument("--force", action="store_true", help="re-download even if data already exists")
    args = parser.parse_args()

    logger = init_logging(args)
    cfg = build_config(args)

    try:
        root = download_dataset(cfg, source=args.source, dest=args.dest, force=args.force)
    except DownloadError as exc:
        fail(str(exc))
        return 1

    size = dir_size_bytes(root)
    logger.info("dataset ready at %s (%s on disk)", root, format_bytes(size))
    logger.info("next: python scripts/prepare_dataset.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
