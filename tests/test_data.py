from __future__ import annotations

from pathlib import Path

import pytest
import torch

from rallyvision.data.dataset import build_dataloader, build_dataset
from rallyvision.data.labels import LabelError, parse_label_csv
from rallyvision.data.manifest import DatasetManifest
from rallyvision.data.prepare import (
    PrepareError,
    assign_splits,
    game_from_clip_path,
    prepare_dataset,
    reassign_manifest_splits,
)
from rallyvision.data.transforms import AugmentConfig, StackAugmentor


# -- label parsing --------------------------------------------------------
def test_parses_tracknet_header(tmp_path: Path):
    csv = tmp_path / "Label.csv"
    csv.write_text(
        "file name,visibility,x-coordinate,y-coordinate,status\n"
        "0000.jpg,1,100,200,0\n"
        "0001.jpg,0,,,0\n"
    )
    rows = parse_label_csv(csv)
    assert len(rows) == 2
    assert rows[0] == {"file": "0000.jpg", "visible": True, "x": 100.0, "y": 200.0, "status": 0}
    assert rows[1]["visible"] is False and rows[1]["x"] is None


def test_parses_alternative_header_spellings(tmp_path: Path):
    csv = tmp_path / "Label.csv"
    csv.write_text("Filename,Visible,X,Y\nf0.png,1,12.5,7.5\n")
    rows = parse_label_csv(csv)
    assert rows[0]["file"] == "f0.png"
    assert rows[0]["x"] == 12.5


def test_rejects_unrecognisable_header(tmp_path: Path):
    csv = tmp_path / "Label.csv"
    csv.write_text("a,b,c\n1,2,3\n")
    with pytest.raises(LabelError, match="could not find"):
        parse_label_csv(csv)


# -- preparation ----------------------------------------------------------
def test_download_reuses_existing_labelled_dataset(base_config, synthetic_raw):
    from rallyvision.data.sources import download_dataset

    base_config["data.source"] = "gdrive"
    base_config["data.gdrive_id"] = "should-not-be-fetched"
    base_config["data.raw_dir"] = str(synthetic_raw)
    base_config["data.local_path"] = str(synthetic_raw)
    root = download_dataset(base_config)
    assert Path(root) == Path(synthetic_raw)


def test_empty_raw_dir_error_lists_the_folder(base_config, tmp_path):
    empty = tmp_path / "empty_raw"
    empty.mkdir()
    base_config["data.raw_dir"] = str(empty)
    base_config["data.local_path"] = str(empty)
    with pytest.raises(PrepareError, match="no clips found"):
        prepare_dataset(base_config, raw_root=empty, prepared_root=tmp_path / "prepared", overwrite=True)


def test_prepare_indexes_every_clip(prepared_dataset: DatasetManifest):
    assert len(prepared_dataset.clips) == 4
    assert all(len(c.files) == 10 for c in prepared_dataset.clips)
    assert all(len(c.files) == len(c.visible) == len(c.x) == len(c.y) for c in prepared_dataset.clips)


def test_prepare_writes_a_reloadable_manifest(prepared_dataset: DatasetManifest):
    reloaded = DatasetManifest.load(Path(prepared_dataset.prepared_root))
    assert len(reloaded.clips) == len(prepared_dataset.clips)
    assert reloaded.input_height == prepared_dataset.input_height
    assert Path(reloaded.prepared_root).resolve() == Path(prepared_dataset.prepared_root).resolve()


def test_manifest_rebinds_prepared_root_when_the_folder_is_moved(prepared_dataset: DatasetManifest, tmp_path):
    """Drive remounts / copied folders must not keep the original absolute path."""
    import shutil

    moved = tmp_path / "drive" / "prepared"
    shutil.copytree(prepared_dataset.prepared_root, moved)
    reloaded = DatasetManifest.load(moved)
    assert Path(reloaded.prepared_root).resolve() == moved.resolve()
    assert len(reloaded.clips) == len(prepared_dataset.clips)


def test_split_is_disjoint_by_game(prepared_dataset: DatasetManifest):
    train_games = {c.game for c in prepared_dataset.split_clips("train")}
    val_games = {c.game for c in prepared_dataset.split_clips("val")}
    assert train_games and val_games
    assert not (train_games & val_games)


def test_stats_are_computed_not_hardcoded(prepared_dataset: DatasetManifest):
    stats = prepared_dataset.stats(num_frames=3, frame_stride=1)
    assert stats["frames"] == sum(len(c) for c in prepared_dataset.clips)
    # 10 frames per clip, 3-frame stacks -> 8 usable targets per clip.
    assert stats["usable_samples"] == 8 * len(prepared_dataset.clips)
    assert stats["splits"]["train"]["clips"] + stats["splits"]["val"]["clips"] == 4


def test_prepare_is_idempotent(base_config, synthetic_raw, tmp_path):
    prepared = tmp_path / "prepared2"
    first = prepare_dataset(base_config, raw_root=synthetic_raw, prepared_root=prepared, overwrite=True)
    second = prepare_dataset(base_config, raw_root=synthetic_raw, prepared_root=prepared)
    assert [c.id for c in first.clips] == [c.id for c in second.clips]


def test_frame_caching_resizes_and_rescales_labels(base_config, synthetic_raw, tmp_path):
    prepared = tmp_path / "cached"
    manifest = prepare_dataset(
        base_config, raw_root=synthetic_raw, prepared_root=prepared, cache_frames=True, overwrite=True
    )
    assert manifest.cached
    clip = manifest.clips[0]
    assert clip.cached_dir is not None
    assert (prepared / clip.cached_dir / clip.files[0]).is_file()
    assert clip.width == base_config.data.input_width
    for x, y, visible in zip(clip.x, clip.y, clip.visible):
        if visible:
            assert 0 <= x < clip.width and 0 <= y < clip.height


def test_overwrite_reuses_cached_frames_and_still_rescales_labels(base_config, synthetic_raw, tmp_path):
    prepared = tmp_path / "cached"
    first = prepare_dataset(
        base_config, raw_root=synthetic_raw, prepared_root=prepared, cache_frames=True, overwrite=True
    )
    clip = first.clips[0]
    frame = prepared / clip.cached_dir / clip.files[0]
    mtime = frame.stat().st_mtime_ns
    missing = prepared / clip.cached_dir / clip.files[1]
    missing.unlink()

    second = prepare_dataset(
        base_config, raw_root=synthetic_raw, prepared_root=prepared, cache_frames=True, overwrite=True
    )
    assert frame.stat().st_mtime_ns == mtime  # not re-written
    assert missing.is_file()  # gaps are filled in
    for a, b in zip(first.clips, second.clips):
        assert (a.x, a.y, a.width, a.height) == (b.x, b.y, b.width, b.height)


def test_explicit_val_games_override_fraction(prepared_dataset: DatasetManifest):
    clips = prepared_dataset.clips
    assign_splits(clips, val_games=["game2"])
    assert {c.game for c in clips if c.split == "val"} == {"game2"}


def test_split_problems_flags_empty_val_and_leakage(prepared_dataset: DatasetManifest):
    assert prepared_dataset.split_problems(num_frames=3, frame_stride=1) == []

    for clip in prepared_dataset.clips:
        clip.split = "train"
    assert any("val split has no usable samples" in p for p in prepared_dataset.split_problems(3, 1))

    game = prepared_dataset.clips[0].game
    same_game = [c for c in prepared_dataset.clips if c.game == game]
    same_game[0].split, same_game[1].split = "train", "val"
    assert any("leakage" in p for p in prepared_dataset.split_problems(3, 1))

    for clip in prepared_dataset.clips:
        clip.game = "game"
    assert any("2+" in p for p in prepared_dataset.split_problems(3, 1))


def test_game_id_is_the_clip_parent_folder():
    assert game_from_clip_path("game1/Clip1") == "game1"
    assert game_from_clip_path("Dataset/game1/Clip1") == "game1"
    assert game_from_clip_path("Clip1") == "game"


def _toy_clips(n: int, game: str = "game") -> list:
    from rallyvision.data.manifest import ClipRecord

    clips = []
    for i in range(n):
        clips.append(
            ClipRecord(
                id=f"{game}/Clip{i+1}",
                game=game,
                name=f"Clip{i+1}",
                source_dir=f"{game}/Clip{i+1}",
                files=[f"{j:04d}.jpg" for j in range(10)],
                visible=[1] * 10,
                x=[1.0] * 10,
                y=[1.0] * 10,
                width=64,
                height=64,
            )
        )
    return clips


def test_single_game_split_still_gets_a_val_set():
    clips = _toy_clips(5, game="game")
    assign_splits(clips, strategy="by_game", val_fraction=0.2, seed=1337)
    assert any(c.split == "val" for c in clips)
    assert any(c.split == "train" for c in clips)


def test_nested_wrapper_folder_splits_by_real_game():
    from rallyvision.data.manifest import ClipRecord

    clips = []
    for game, n in (("game1", 2), ("game2", 2)):
        for i in range(n):
            rel = f"Dataset/{game}/Clip{i+1}"
            clips.append(
                ClipRecord(
                    id=rel,
                    game="Dataset",
                    name=f"Clip{i+1}",
                    source_dir=rel,
                    files=[f"{j:04d}.jpg" for j in range(10)],
                    visible=[1] * 10,
                    x=[1.0] * 10,
                    y=[1.0] * 10,
                    width=64,
                    height=64,
                    split="train",
                )
            )
    for clip in clips:
        clip.game = game_from_clip_path(clip.source_dir)
    assign_splits(clips, strategy="by_game", val_fraction=0.5, seed=1337)
    assert {c.game for c in clips} == {"game1", "game2"}
    assert any(c.split == "val" for c in clips)
    assert any(c.split == "train" for c in clips)


def test_reassign_splits_repairs_an_all_train_manifest(base_config, tmp_path):
    from rallyvision.data.manifest import ClipRecord, DatasetManifest

    prepared = tmp_path / "prepared"
    prepared.mkdir()
    clips = []
    for game, n in (("game1", 2), ("game2", 2)):
        for i in range(n):
            rel = f"Dataset/{game}/Clip{i+1}"
            clips.append(
                ClipRecord(
                    id=rel,
                    game="Dataset",
                    name=f"Clip{i+1}",
                    source_dir=rel,
                    files=[f"{j:04d}.jpg" for j in range(10)],
                    visible=[1] * 10,
                    x=[1.0] * 10,
                    y=[1.0] * 10,
                    width=64,
                    height=64,
                    split="train",
                )
            )
    manifest = DatasetManifest(
        raw_root=str(tmp_path / "raw"),
        prepared_root=str(prepared),
        clips=clips,
    )
    manifest.save(prepared / "manifest.json")
    assert all(c.split == "train" for c in DatasetManifest.load(prepared).clips)

    repaired = reassign_manifest_splits(DatasetManifest.load(prepared), base_config)
    assert {c.game for c in repaired.clips} == {"game1", "game2"}
    assert any(c.split == "val" for c in repaired.clips)
    assert any(c.split == "train" for c in repaired.clips)
    reloaded = DatasetManifest.load(prepared)
    assert any(c.split == "val" for c in reloaded.clips)


# -- dataset --------------------------------------------------------------
def test_sample_shapes_and_types(base_config, prepared_dataset):
    dataset = build_dataset(base_config, prepared_dataset, split="train")
    assert len(dataset) > 0
    sample = dataset[0]
    channels = int(base_config.data.num_frames) * 3
    assert sample["frames"].shape == (channels, base_config.data.input_height, base_config.data.input_width)
    assert sample["heatmap"].shape == (1, base_config.data.input_height, base_config.data.input_width)
    assert sample["frames"].dtype == torch.float32
    assert 0.0 <= float(sample["frames"].min()) and float(sample["frames"].max()) <= 1.0


def test_heatmap_peak_matches_the_label(base_config, prepared_dataset):
    dataset = build_dataset(base_config, prepared_dataset, split="train")
    visible = next(i for i in range(len(dataset)) if dataset[i]["visible"].item() > 0.5)
    sample = dataset[visible]
    heatmap = sample["heatmap"][0].numpy()
    peak = divmod(int(heatmap.argmax()), heatmap.shape[1])
    target = sample["target_xy"].tolist()
    assert abs(peak[1] - target[0]) <= 1.5
    assert abs(peak[0] - target[1]) <= 1.5


def test_invisible_samples_have_an_empty_heatmap(base_config, prepared_dataset):
    dataset = build_dataset(base_config, prepared_dataset, split="train")
    hidden = [i for i in range(len(dataset)) if dataset[i]["visible"].item() < 0.5]
    if not hidden:
        pytest.skip("this synthetic dataset has no occluded frames")
    sample = dataset[hidden[0]]
    assert float(sample["heatmap"].max()) == 0.0
    assert sample["target_xy"].tolist() == [-1.0, -1.0]


def test_drop_invisible_removes_those_samples(base_config, prepared_dataset):
    keep = build_dataset(base_config, prepared_dataset, split="train")
    base_config["data.drop_invisible"] = True
    dropped = build_dataset(base_config, prepared_dataset, split="train")
    assert len(dropped) <= len(keep)
    assert all(dropped[i]["visible"].item() > 0.5 for i in range(min(5, len(dropped))))


def test_train_and_val_datasets_share_no_clips(base_config, prepared_dataset):
    train = build_dataset(base_config, prepared_dataset, split="train")
    val = build_dataset(base_config, prepared_dataset, split="val")
    assert {c.id for c in train.clips}.isdisjoint({c.id for c in val.clips})


def test_dataloader_batches_tensors_and_keeps_strings(base_config, prepared_dataset):
    dataset = build_dataset(base_config, prepared_dataset, split="train")
    loader = build_dataloader(base_config, dataset, split="train", device_kind="cpu")
    batch = next(iter(loader))
    assert batch["frames"].shape[0] == base_config.data.loader.batch_size
    assert isinstance(batch["clip_id"], list)
    assert isinstance(batch["clip_id"][0], str)


def test_to_original_scale_maps_back_to_source_resolution(base_config, prepared_dataset):
    dataset = build_dataset(base_config, prepared_dataset, split="val")
    sample = dataset[0]
    clip = dataset.clips[0]
    sx, sy = sample["to_original"].tolist()
    assert sx == pytest.approx(clip.width / base_config.data.input_width)
    assert sy == pytest.approx(clip.height / base_config.data.input_height)


# -- augmentation ---------------------------------------------------------
def test_flip_moves_the_point_consistently():
    import numpy as np

    frames = [np.zeros((10, 20, 3), dtype=np.uint8) for _ in range(3)]
    for f in frames:
        f[4, 3] = 255
    aug = StackAugmentor(AugmentConfig(enabled=True, horizontal_flip=1.0, brightness=0, contrast=0, translate=0, scale=0))
    out, point = aug(frames, (3.0, 4.0))
    assert point == (16.0, 4.0)
    assert all(f[4, 16].max() == 255 for f in out)


def test_disabled_augmentation_is_identity():
    import numpy as np

    frames = [np.full((8, 8, 3), 7, dtype=np.uint8) for _ in range(3)]
    aug = StackAugmentor(AugmentConfig(enabled=False))
    out, point = aug(frames, (1.0, 2.0))
    assert point == (1.0, 2.0)
    assert all(np.array_equal(a, b) for a, b in zip(frames, out))


def test_validation_datasets_are_never_augmented(base_config, prepared_dataset):
    base_config["data.augment.enabled"] = True
    val = build_dataset(base_config, prepared_dataset, split="val")
    assert val.augment_enabled is False
