#!/usr/bin/env python3
"""Copy a run's durable artifacts into one folder (and optionally a zip).

Colab job, at the end of a session:

    python scripts/collect_artifacts.py --run v1
    python scripts/collect_artifacts.py --run v1 --zip

The bundle is what you download onto the Mac before Core ML export.
Core ML files are never produced here.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from _common import add_config_args, build_config, fail, init_logging

from rallyvision.artifacts import collect_from_config, format_bundle
from rallyvision.config import output_root
from rallyvision.logging_utils import banner


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_config_args(parser)
    parser.add_argument("--run", default="latest", help="run name, run directory, or 'latest'")
    parser.add_argument(
        "--dest",
        type=Path,
        default=None,
        help="bundle directory (default: <output_root>/exports/<run_name>)",
    )
    parser.add_argument("--zip", dest="zip_bundle", action="store_true", help="also write a .zip next to the bundle")
    args = parser.parse_args()

    init_logging(args)
    cfg = build_config(args)

    dest = args.dest
    if dest is None:
        name = "latest" if args.run in (None, "", "latest") else Path(args.run).name
        dest = output_root(cfg) / "exports" / name

    try:
        manifest = collect_from_config(cfg, dest, run=args.run, zip_bundle=args.zip_bundle)
    except FileNotFoundError as exc:
        fail(str(exc))
        return 1

    print()
    print(banner("artifact bundle"))
    print(format_bundle(manifest))
    print()
    print("Download this folder (or the zip) to your Mac, then:")
    print("  python scripts/export_coreml.py --checkpoint <bundle>/best.pt --precision both --validate")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
