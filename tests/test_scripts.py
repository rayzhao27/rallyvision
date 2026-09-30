"""CLI-level checks.

The integration test walks the exact commands a user runs, at toy resolution on
synthetic data, so a broken argument or import surfaces here rather than after
ten minutes of Colab setup.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = REPO_ROOT / "scripts"

ALL_SCRIPTS = [
    "check_environment.py",
    "download_dataset.py",
    "prepare_dataset.py",
    "train_ball_tracker.py",
    "evaluate_ball_tracker.py",
    "infer_video.py",
    "visualize_predictions.py",
    "collect_artifacts.py",
    "make_synthetic_dataset.py",
    "export_coreml.py",
    "validate_coreml.py",
    "benchmark_coreml.py",
]

# Toy geometry: divisible by the model stride of 8, small enough to be instant.
TINY = [
    "--set", "data.input_height=48",
    "--set", "data.input_width=64",
    "--set", "data.num_frames=3",
    "--set", "model.width=0.125",
    "--set", "data.loader.num_workers=0",
    "--set", "data.loader.pin_memory=false",
    "--set", "runtime.device=cpu",
]


def run(args, timeout=300):
    env = dict(os.environ, PYTHONPATH=str(REPO_ROOT))
    return subprocess.run(
        [sys.executable, *args],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=timeout,
        env=env,
    )


@pytest.mark.parametrize("script", ALL_SCRIPTS)
def test_every_script_has_working_help(script):
    result = run([str(SCRIPTS / script), "--help"], timeout=120)
    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout.lower()


def test_training_refuses_to_run_full_training_on_cpu(tmp_path):
    """The guardrail that keeps real training off this machine."""
    result = run(
        [
            str(SCRIPTS / "train_ball_tracker.py"),
            *TINY,
            "--set", f"project.output_root={tmp_path / 'artifacts'}",
            "--set", f"data.prepared_dir={tmp_path / 'prepared'}",
            "--epochs", "1",
        ],
        timeout=180,
    )
    assert result.returncode != 0
    assert "Refusing to start full training" in (result.stderr + result.stdout)


@pytest.mark.timeout(900)
def test_full_cli_pipeline_on_synthetic_data(tmp_path):
    raw = tmp_path / "raw"
    prepared = tmp_path / "prepared"
    artifacts = tmp_path / "artifacts"
    video = tmp_path / "demo.mp4"

    common = [
        *TINY,
        "--set", f"project.output_root={artifacts}",
        "--set", f"data.prepared_dir={prepared}",
        "--set", f"data.local_path={raw}",
        "--set", f"data.raw_dir={raw}",
        "--set", "data.split.val_fraction=0.5",
    ]

    # 1. synthetic data
    result = run(
        [
            str(SCRIPTS / "make_synthetic_dataset.py"),
            "--out", str(raw),
            "--games", "2",
            "--clips-per-game", "1",
            "--frames-per-clip", "8",
            "--width", "128",
            "--height", "96",
            "--video", str(video),
            "--video-frames", "10",
        ]
    )
    assert result.returncode == 0, result.stderr
    assert video.is_file()

    # 2. prepare
    result = run([str(SCRIPTS / "prepare_dataset.py"), *common, "--cache-frames", "--overwrite"])
    assert result.returncode == 0, result.stderr
    assert (prepared / "manifest.json").is_file()
    assert (prepared / "dataset_stats.json").is_file()
    assert "usable samples" in result.stdout

    # 3. smoke training (explicitly allowed on CPU)
    result = run(
        [str(SCRIPTS / "train_ball_tracker.py"), *common, "--smoke", "--smoke-batches", "2",
         "--run-name", "citest"],
        timeout=600,
    )
    assert result.returncode == 0, result.stderr
    run_dir = artifacts / "runs" / "citest"
    assert (run_dir / "checkpoints" / "last.pt").is_file()
    assert (run_dir / "config.yaml").is_file()
    assert (run_dir / "metrics" / "history.json").is_file()
    assert "SMOKE TEST" in result.stdout + result.stderr
    assert "final evaluation" not in result.stdout + result.stderr

    # 4. resume must not restart from zero
    result = run(
        [str(SCRIPTS / "train_ball_tracker.py"), *common, "--smoke", "--smoke-batches", "2",
         "--run-name", "citest", "--resume", "--set", "train.epochs=2"],
        timeout=600,
    )
    assert result.returncode == 0, result.stderr
    assert "resumed from" in (result.stdout + result.stderr)

    # 5. evaluation writes machine-readable metrics
    result = run(
        [str(SCRIPTS / "evaluate_ball_tracker.py"), *common, "--run", "citest", "--split", "val"],
        timeout=600,
    )
    assert result.returncode == 0, result.stderr
    assert (run_dir / "metrics" / "val_metrics.json").is_file()
    assert (run_dir / "metrics" / "val_metrics.csv").is_file()
    assert "accuracy@5px" in result.stdout

    # 6. video inference writes an annotated mp4
    result = run(
        [str(SCRIPTS / "infer_video.py"), *common, "--video", str(video), "--run", "citest",
         "--max-frames", "6"],
        timeout=600,
    )
    assert result.returncode == 0, result.stderr
    assert (run_dir / "videos" / "demo_annotated.mp4").is_file()

    # 7. prediction grids
    result = run(
        [str(SCRIPTS / "visualize_predictions.py"), *common, "--run", "citest", "--num", "2"],
        timeout=600,
    )
    assert result.returncode == 0, result.stderr
    assert (run_dir / "predictions" / "val_good.png").is_file()

    # 8. artifact bundle for the Mac-side Core ML step
    bundle = tmp_path / "export"
    result = run(
        [str(SCRIPTS / "collect_artifacts.py"), *common, "--run", "citest", "--dest", str(bundle)],
        timeout=120,
    )
    assert result.returncode == 0, result.stderr
    assert (bundle / "last.pt").is_file()
    assert (bundle / "config.yaml").is_file()
    assert (bundle / "bundle_manifest.json").is_file()


def test_training_refuses_a_leaky_split(tmp_path):
    """A game in both train and val must stop full training before it starts."""
    import json

    raw = tmp_path / "raw"
    prepared = tmp_path / "prepared"
    common = [
        *TINY,
        "--set", f"project.output_root={tmp_path / 'artifacts'}",
        "--set", f"data.prepared_dir={prepared}",
        "--set", f"data.local_path={raw}",
        "--set", f"data.raw_dir={raw}",
        "--set", "data.split.val_fraction=0.5",
    ]
    result = run(
        [str(SCRIPTS / "make_synthetic_dataset.py"), "--out", str(raw), "--games", "2",
         "--clips-per-game", "2", "--frames-per-clip", "8", "--width", "128", "--height", "96"]
    )
    assert result.returncode == 0, result.stderr
    result = run([str(SCRIPTS / "prepare_dataset.py"), *common, "--overwrite"])
    assert result.returncode == 0, result.stderr

    manifest_path = prepared / "manifest.json"
    data = json.loads(manifest_path.read_text())
    game = data["clips"][0]["game"]
    same_game = [c for c in data["clips"] if c["game"] == game]
    same_game[0]["split"], same_game[1]["split"] = "train", "val"
    manifest_path.write_text(json.dumps(data))

    result = run(
        [str(SCRIPTS / "train_ball_tracker.py"), *common, "--allow-local-training", "--epochs", "1",
         "--run-name", "leaky"],
        timeout=300,
    )
    output = result.stdout + result.stderr
    assert result.returncode != 0
    assert "refusing to train" in output
    assert "leakage" in output
    assert not (tmp_path / "artifacts" / "runs" / "leaky" / "checkpoints" / "last.pt").exists()


def test_check_environment_require_cuda_fails_without_a_gpu():
    result = run([str(SCRIPTS / "check_environment.py"), "--device", "cpu", "--require-cuda"])
    assert result.returncode != 0
    assert "CUDA is required" in (result.stderr + result.stdout)


def test_check_environment_succeeds_on_cpu_without_require_cuda():
    result = run([str(SCRIPTS / "check_environment.py"), "--device", "cpu"])
    assert result.returncode == 0, result.stderr
    assert "torch" in result.stdout


def test_prepare_reports_a_clear_error_without_data(tmp_path):
    result = run(
        [
            str(SCRIPTS / "prepare_dataset.py"),
            *TINY,
            "--set", f"data.raw_dir={tmp_path / 'missing'}",
            "--set", f"data.local_path={tmp_path / 'missing'}",
            "--set", f"data.prepared_dir={tmp_path / 'prepared'}",
        ]
    )
    assert result.returncode != 0
    assert "does not exist" in result.stderr or "no clips found" in result.stderr
