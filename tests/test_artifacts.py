from __future__ import annotations

from pathlib import Path

from rallyvision.artifacts import collect_run_artifacts, format_bundle
from rallyvision.paths import RunPaths
from rallyvision.utils import write_json


def _fake_run(root: Path) -> RunPaths:
    paths = RunPaths(root).create()
    (paths.checkpoints / "best.pt").write_bytes(b"best-weights")
    (paths.checkpoints / "last.pt").write_bytes(b"last-weights")
    paths.config_file.write_text("project:\n  name: rallyvision\n")
    write_json(paths.history_file, {"history": [{"epoch": 1, "train/loss": 0.5}]})
    write_json(paths.metrics / "val_metrics.json", {"metrics": {"accuracy_at_5px": 0.1}})
    (paths.metrics / "val_metrics.csv").write_text("accuracy_at_5px\n0.1\n")
    (paths.predictions / "val_good.png").write_bytes(b"png")
    (paths.videos / "demo_annotated.mp4").write_bytes(b"mp4")
    return paths


def test_collect_copies_checkpoint_config_metrics_and_media(tmp_path):
    run = _fake_run(tmp_path / "run")
    dest = tmp_path / "bundle"
    manifest = collect_run_artifacts(run, dest)

    assert (dest / "best.pt").read_bytes() == b"best-weights"
    assert (dest / "config.yaml").is_file()
    assert (dest / "history.json").is_file()
    assert (dest / "val_metrics.json").is_file()
    assert (dest / "val_metrics.csv").is_file()
    assert (dest / "predictions" / "val_good.png").is_file()
    assert (dest / "videos" / "demo_annotated.mp4").is_file()
    assert (dest / "bundle_manifest.json").is_file()
    names = {item["name"] for item in manifest["copied"]}
    assert "best.pt" in names
    assert "history.json" in names
    text = format_bundle(manifest)
    assert "best.pt" in text


def test_collect_skips_missing_optional_files(tmp_path):
    paths = RunPaths(tmp_path / "empty").create()
    (paths.checkpoints / "last.pt").write_bytes(b"only-last")
    dest = tmp_path / "bundle"
    manifest = collect_run_artifacts(paths, dest)
    assert (dest / "last.pt").is_file()
    assert not (dest / "best.pt").exists()
    assert "checkpoints/best.pt" in manifest["missing"]


def test_collect_zip_writes_an_archive(tmp_path):
    run = _fake_run(tmp_path / "run")
    dest = tmp_path / "bundle"
    manifest = collect_run_artifacts(run, dest, zip_bundle=True)
    assert manifest["zip"]
    assert Path(manifest["zip"]).is_file()
