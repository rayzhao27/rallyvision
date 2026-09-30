"""Dataset acquisition.

Downloading is a Colab job: it is bandwidth- and disk-heavy, and the prepared
result is what gets cached on Drive. Five sources are supported so you can
point the pipeline at whatever copy of the TrackNet tennis dataset you have:

    local     already extracted somewhere on disk
    gdrive    a Google Drive file id, fetched with gdown
    url       a direct archive URL
    kaggle    `kaggle datasets download -d <slug>` (needs ~/.kaggle/kaggle.json)
    synthetic procedurally generated clips, for pipeline tests only

No download URL is hard-coded: the reference dataset has no stable public
mirror, and a broken built-in link is worse than an explicit error.
"""

from __future__ import annotations

import shutil
import subprocess
import tarfile
import zipfile
from pathlib import Path
from typing import Optional

from ..config import Config, resolve_path
from ..logging_utils import get_logger

logger = get_logger(__name__)

ARCHIVE_SUFFIXES = (".zip", ".tar", ".tar.gz", ".tgz", ".tar.bz2", ".7z")


class DownloadError(RuntimeError):
    """Raised when a dataset cannot be fetched or unpacked."""


def _is_archive(path: Path) -> bool:
    name = path.name.lower()
    return any(name.endswith(suffix) for suffix in ARCHIVE_SUFFIXES)


def extract_archive(archive: Path, dest: Path) -> Path:
    """Extract a zip/tar archive into ``dest``."""
    dest.mkdir(parents=True, exist_ok=True)
    name = archive.name.lower()
    logger.info("extracting %s -> %s", archive.name, dest)
    if name.endswith(".zip"):
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(dest)
    elif name.endswith((".tar", ".tar.gz", ".tgz", ".tar.bz2")):
        with tarfile.open(archive) as tf:
            tf.extractall(dest)
    else:
        raise DownloadError(f"unsupported archive format: {archive.name}")
    return dest


def _flatten_single_dir(path: Path) -> Path:
    """If extraction produced exactly one wrapper directory, descend into it."""
    entries = [p for p in path.iterdir() if not p.name.startswith(".")]
    if len(entries) == 1 and entries[0].is_dir():
        return entries[0]
    return path


def download_gdrive(file_id: str, dest_dir: Path, filename: Optional[str] = None, folder: bool = False) -> Path:
    try:
        import gdown
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise DownloadError("gdrive source needs `pip install gdown`") from exc

    dest_dir.mkdir(parents=True, exist_ok=True)
    if folder:
        output = dest_dir / "tennis"
        output.mkdir(parents=True, exist_ok=True)
        logger.info("downloading Google Drive folder id=%s -> %s", file_id, output)
        result = gdown.download_folder(id=file_id, output=str(output), quiet=False, remaining_ok=True)
        if result is None:
            raise DownloadError(
                f"gdown could not download folder id={file_id}. "
                "Confirm the folder is shared with 'anyone with the link'."
            )
        _extract_archives_in_tree(output)
        return output

    output = str(dest_dir / filename) if filename else None
    logger.info("downloading Google Drive file id=%s", file_id)
    result = gdown.download(id=file_id, output=output, quiet=False, fuzzy=True)
    if result is None:
        raise DownloadError(
            f"gdown could not download id={file_id}. Large Drive files often hit the "
            "virus-scan interstitial; try `gdown --fuzzy <share-url>` manually."
        )
    return Path(result)


def _extract_archives_in_tree(root: Path) -> None:
    """Unzip any archives that came along with a Drive-folder download."""
    archives = [p for p in root.rglob("*") if p.is_file() and _is_archive(p)]
    for archive in archives:
        logger.info("extracting nested archive %s", archive.name)
        extract_archive(archive, archive.parent)


def find_labelled_dataset(root: Path, label_filename: str = "Label.csv") -> Optional[Path]:
    """Return ``root`` if it already contains labelled TrackNet clips."""
    root = Path(root)
    if not root.is_dir():
        return None
    if next(root.rglob(label_filename), None) is None:
        return None
    return root


def download_url(url: str, dest_dir: Path, filename: Optional[str] = None) -> Path:
    import urllib.request

    dest_dir.mkdir(parents=True, exist_ok=True)
    target = dest_dir / (filename or Path(url.split("?")[0]).name or "dataset.zip")
    if target.is_file() and target.stat().st_size > 0:
        logger.info("reusing already-downloaded %s", target)
        return target
    logger.info("downloading %s -> %s", url, target)
    tmp = target.with_suffix(target.suffix + ".part")
    with urllib.request.urlopen(url) as response, tmp.open("wb") as fh:  # noqa: S310
        shutil.copyfileobj(response, fh)
    tmp.replace(target)
    return target


def download_kaggle(slug: str, dest_dir: Path) -> Path:
    dest_dir.mkdir(parents=True, exist_ok=True)
    if shutil.which("kaggle") is None:
        raise DownloadError(
            "kaggle source needs the Kaggle CLI: `pip install kaggle` plus credentials "
            "in ~/.kaggle/kaggle.json (in Colab, upload the token first)."
        )
    logger.info("kaggle datasets download -d %s", slug)
    proc = subprocess.run(
        ["kaggle", "datasets", "download", "-d", slug, "-p", str(dest_dir), "--force"],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise DownloadError(f"kaggle download failed:\n{proc.stdout}\n{proc.stderr}")
    archives = sorted(dest_dir.glob("*.zip"), key=lambda p: p.stat().st_mtime)
    if not archives:
        raise DownloadError(f"kaggle download produced no zip in {dest_dir}")
    return archives[-1]


def download_dataset(
    cfg: Config,
    source: Optional[str] = None,
    dest: Optional[Path] = None,
    force: bool = False,
) -> Path:
    """Fetch the dataset and return the directory holding the extracted clips."""
    data = cfg.data
    source = (source or data.get("source", "local")).lower()
    raw_dir = Path(dest) if dest else resolve_path(cfg, data.raw_dir)
    label_filename = str(data.get("label_filename", "Label.csv"))
    if source != "local":
        raw_dir.mkdir(parents=True, exist_ok=True)

    if not force:
        for candidate in (raw_dir, resolve_path(cfg, data.get("local_path") or raw_dir)):
            found = find_labelled_dataset(candidate, label_filename)
            if found is not None:
                n = sum(1 for _ in found.rglob(label_filename))
                logger.info("reusing labelled dataset at %s (%d Label.csv files)", found, n)
                return found

    if source == "local":
        local = resolve_path(cfg, data.require("local_path"))
        if not local.is_dir():
            raise DownloadError(
                f"data.source is 'local' but {local} does not exist.\n"
                "Either place the dataset there, or set data.source to "
                "gdrive/url/kaggle and provide the corresponding id/url/slug."
            )
        from .prepare import preview_dir

        raise DownloadError(
            f"data.source is 'local' but no {label_filename} files were found under {local}.\n"
            "Extract the TrackNet dataset here, or set data.source to gdrive (the Colab default).\n\n"
            f"{preview_dir(local)}"
        )

    if source == "synthetic":
        from .synthetic import generate_dataset

        target = raw_dir / "synthetic"
        if target.is_dir() and not force:
            logger.info("reusing synthetic dataset at %s", target)
            return target
        logger.warning("generating a SYNTHETIC dataset - for pipeline tests only")
        return generate_dataset(target, seed=int(cfg.project.seed))

    extract_dir = raw_dir / "tennis"

    if source == "gdrive":
        archive = download_gdrive(
            str(data.require("gdrive_id")),
            raw_dir,
            folder=bool(data.get("gdrive_folder", False)),
        )
    elif source == "url":
        archive = download_url(str(data.require("url")), raw_dir)
    elif source == "kaggle":
        archive = download_kaggle(str(data.require("kaggle_dataset")), raw_dir)
    else:
        raise DownloadError(
            f"unknown data.source {source!r}; expected one of "
            "local, gdrive, url, kaggle, synthetic"
        )

    if archive.is_dir():
        return archive
    if not _is_archive(archive):
        raise DownloadError(f"downloaded file is not a recognised archive: {archive}")

    extract_archive(archive, extract_dir)
    return _flatten_single_dir(extract_dir)
