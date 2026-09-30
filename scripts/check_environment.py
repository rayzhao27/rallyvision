#!/usr/bin/env python3
"""Print the runtime environment and, optionally, refuse to continue without CUDA.

    python scripts/check_environment.py
    python scripts/check_environment.py --require-cuda   # Colab training cells

Does not train, download data, or touch a GPU beyond querying its name.
"""

from __future__ import annotations

import argparse

from _common import add_config_args, add_device_args, build_config, fail, init_logging

from rallyvision.environment import EnvironmentError, report_environment
from rallyvision.logging_utils import banner


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_config_args(parser)
    add_device_args(parser)
    parser.add_argument(
        "--require-cuda",
        action="store_true",
        help="exit non-zero if the resolved device is not CUDA (Colab training)",
    )
    args = parser.parse_args()
    init_logging(args)
    cfg = build_config(args)

    try:
        report = report_environment(cfg, device_override=args.device, require_cuda=args.require_cuda)
    except EnvironmentError as exc:
        fail(str(exc))
        return 1

    print()
    print(banner("environment"))
    print(report.format())
    print()
    if args.require_cuda:
        print("CUDA is available. Safe to start training.")
    elif not report.device.is_cuda:
        print("No CUDA GPU. Full training will refuse to start; --smoke is allowed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
