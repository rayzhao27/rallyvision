"""Moved to `rallyvision.common.check_environment` (module-first refactor, 2026-11-22).

This forwarder keeps `python scripts/check_environment.py` and imports of `check_environment` from archived experiments working.
"""
import importlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
_m = importlib.import_module("rallyvision.common.check_environment")
globals().update({k: v for k, v in vars(_m).items() if not k.startswith("__")})

if __name__ == "__main__":
    _m.main()
