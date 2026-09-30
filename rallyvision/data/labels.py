"""Parsing of TrackNet-style ``Label.csv`` files.

The reference tennis dataset ships one CSV per clip with columns

    file name, visibility, x-coordinate, y-coordinate, status

Header spelling varies between mirrors and forks, so column names are
normalized and matched against a list of aliases instead of being hard-coded.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List, Optional

FILE_ALIASES = ("file_name", "filename", "file", "frame", "frame_name", "image", "img", "name")
VISIBILITY_ALIASES = ("visibility", "visible", "vis", "visibility_class")
X_ALIASES = ("x_coordinate", "x", "x_coord", "ball_x", "xcoordinate", "cx")
Y_ALIASES = ("y_coordinate", "y", "y_coord", "ball_y", "ycoordinate", "cy")
STATUS_ALIASES = ("status", "state", "trajectory_pattern")


class LabelError(RuntimeError):
    """Raised when a label CSV cannot be interpreted."""


def _normalize(name: str) -> str:
    return name.strip().lower().replace("-", "_").replace(" ", "_").lstrip("\ufeff")


def _pick(headers: Dict[str, str], aliases) -> Optional[str]:
    for alias in aliases:
        if alias in headers:
            return headers[alias]
    return None


def _to_float(value: Optional[str]) -> Optional[float]:
    if value is None:
        return None
    value = value.strip()
    if value == "" or value.lower() in {"nan", "none", "null", "-"}:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def parse_label_csv(path: Path | str) -> List[Dict[str, object]]:
    """Return one record per row: ``{file, visible, x, y, status}``.

    ``visible`` is a bool derived from the visibility column (0 = not visible);
    when a row has a visibility of 0 or missing coordinates, ``x``/``y`` are
    ``None``.
    """
    path = Path(path)
    if not path.is_file():
        raise LabelError(f"label file not found: {path}")

    with path.open(newline="") as fh:
        reader = csv.reader(fh)
        try:
            raw_header = next(reader)
        except StopIteration as exc:
            raise LabelError(f"label file is empty: {path}") from exc

        headers = {_normalize(h): h for h in raw_header}
        index = {_normalize(h): i for i, h in enumerate(raw_header)}

        file_key = _pick(headers, FILE_ALIASES)
        x_key = _pick(headers, X_ALIASES)
        y_key = _pick(headers, Y_ALIASES)
        vis_key = _pick(headers, VISIBILITY_ALIASES)
        status_key = _pick(headers, STATUS_ALIASES)

        if file_key is None or x_key is None or y_key is None:
            raise LabelError(
                f"{path}: could not find file/x/y columns in header {raw_header!r}. "
                f"Expected something like 'file name, visibility, x-coordinate, y-coordinate'."
            )

        fi = index[_normalize(file_key)]
        xi = index[_normalize(x_key)]
        yi = index[_normalize(y_key)]
        vi = index[_normalize(vis_key)] if vis_key else None
        si = index[_normalize(status_key)] if status_key else None

        records: List[Dict[str, object]] = []
        for row_no, row in enumerate(reader, start=2):
            if not row or all(not c.strip() for c in row):
                continue
            if len(row) <= max(fi, xi, yi):
                raise LabelError(f"{path}:{row_no}: row has {len(row)} columns, expected more")

            name = row[fi].strip()
            if not name:
                continue
            x = _to_float(row[xi])
            y = _to_float(row[yi])
            visibility = _to_float(row[vi]) if vi is not None and len(row) > vi else None
            status = _to_float(row[si]) if si is not None and len(row) > si else None

            visible = True if visibility is None else visibility > 0
            if x is None or y is None:
                visible = False
                x = y = None
            # A few mirrors encode "not visible" as (0, 0) with visibility=1.
            if visible and x == 0.0 and y == 0.0 and visibility == 0:
                visible = False
                x = y = None

            records.append(
                {
                    "file": name,
                    "visible": bool(visible),
                    "x": x,
                    "y": y,
                    "status": int(status) if status is not None else None,
                }
            )

    if not records:
        raise LabelError(f"{path}: no usable rows")
    return records
