"""Moved to `rallyvision.ball_tracking.export.export_cli` (module-first refactor, 2026-11-22).

This forwarder keeps `python scripts/export_coreml.py` and imports of `export_coreml` from archived experiments working.
"""
import importlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
_m = importlib.import_module("rallyvision.ball_tracking.export.export_cli")
globals().update({k: v for k, v in vars(_m).items() if not k.startswith("__")})

if __name__ == "__main__":
    _m.main()
