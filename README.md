<a name="readme-top"></a>

---



## About The Project

**RallyVision: TrackNet-Style Tennis Ball Tracking with On-Device Core ML Deployment**

This project implements a **heatmap-based tiny-object detector** for tennis ball tracking, built from scratch in PyTorch, trained on Colab GPU, and deployed to Apple Silicon via Core ML:

- **Detection**: a TrackNet-style (Huang et al., 2019) VGG encoder/decoder consumes a stack of 3 consecutive RGB frames and predicts a **full-resolution** single-channel heatmap — no stride-4 output, because the ball is 5–10 pixels wide and motion-blurred.
- **Tracking**: peak extraction + a physics-aware tracker that rejects implausible jumps and interpolates short occlusion gaps.
- **Deployment**: Core ML export (fp32/fp16) with numerical validation against PyTorch, then a compute-unit benchmark sweep on the Apple Neural Engine, GPU and CPU.

The project treats **honest evaluation as a first-class feature**: the train/val split is held out **by game**, so no frame from a validation match is ever seen during training, and the pipeline hard-refuses to train on a leaky split. A full trace of a data-leakage bug that invalidated a first training run — and the guardrails added to make it unrepeatable — is in [Split Integrity](#split-integrity).

**Key Results (held-out validation: 3 unseen matches, 5,429 samples)**

| Metric | Value |
|---|---|
| Accuracy@5px | **0.7725** |
| Accuracy@10px | 0.7754 |
| Accuracy@20px | 0.7771 |
| Median localization error | **1.29 px** |
| p90 localization error | 2.29 px |
| Precision / Recall / F1 | 0.9783 / 0.7865 / **0.8720** |

All pixel errors, including the median error and accuracy@Npx, are measured at the **640×360 evaluation resolution**, not in 720p source pixels (1 px here ≈ 2 px at 1280×720). Accuracy@Npx is the fraction of **visible-ball** frames where the detected peak lies within N px of ground truth; a missed detection counts as a failure. Accuracy@5px ≈ accuracy@20px, which is the headline finding: **when the model fires, it is essentially always sub-pixel-to-2px accurate; every remaining error is a missed detection, not a mislocalization.** See [Threshold and Confidence Analysis](#threshold-and-confidence-analysis).

**On-device inference (Apple M3 Pro, Core ML fp16, 1×9×360×640)**

| Compute units | Median latency | Throughput | Model size |
|---|---|---|---|
| **ALL (Neural Engine)** | **18.74 ms** | **52.5 FPS** | 20.2 MB |
| CPU_AND_GPU | 53.86 ms | 18.6 FPS | 20.2 MB |
| CPU_ONLY | 124.75 ms | 8.0 FPS | 20.2 MB |

fp16 on the Neural Engine is **3.2× faster and half the size** of fp32, with a maximum output deviation of 0.0006 from PyTorch and **zero peak-location shift** — real-time headroom for a 30 FPS camera feed on a laptop-class chip.

**Core Features:**

- **Full-resolution heatmap output** — 360×640 in, 360×640 out, so localization is not quantized by a decoder stride
- **Temporal frame stacking** — 3 frames on the channel axis (9×360×640), giving the network the motion cue that separates a ball from a white line or logo
- **Penalty-reduced focal loss** — CornerNet/CenterNet formulation for a target that is ~0.01% positive pixels; plain BCE collapses to "predict zero everywhere"
- **Stack-consistent augmentation** — flip / translate / scale / brightness sampled once per *stack*, never per frame, so inter-frame motion survives augmentation; ball coordinates are transformed alongside the pixels
- **Leakage-proof splitting** — by-game held-out split, with a manifest-level integrity check that refuses to train if any game appears in both splits or if val is empty
- **Sub-pixel metrics** — connected-component peak extraction, errors reported in original-frame pixels at 5/10/20 px tolerances
- **Physics-aware tracker** — jump rejection with a gap-scaled budget, linear interpolation across occlusions up to 4 frames
- **Core ML pipeline** — export → numerical validation on real frames → compute-unit benchmark, fully separated from training (Colab trains, Mac deploys)
- **160 unit tests** — covering the split logic, the leakage guards, heatmap rendering, metrics, the tracker, and Core ML export

**Cloud Training:** Trained on Google Colab using an **NVIDIA A100 40GB** — 25 epochs over 14,216 training samples in **1 hour 49 minutes** (~4.4 min/epoch at batch size 32 with bf16 mixed precision).

**Demo — Alcaraz vs. Sinner (4K broadcast footage, never seen in training):**

![Ball tracking demo](./pics/alcaraz_vs_sinner_demo.gif)

**Best predictions (smallest localization error) — green circle = ground truth, right column = predicted heatmap:**

![Best predictions](./pics/rallyvision_visual.png)

**Failure cases (missed detections):**

![Failure cases](./pics/rallyvison_visual_2.png)

Every failure in the second grid is a **low-confidence miss** (peak score 0.06–0.27, below the 0.5 threshold) rather than a wrong location — and all six come from the same dim, low-contrast match. This is the visual counterpart of the precision/recall gap discussed in [Threshold and Confidence Analysis](#threshold-and-confidence-analysis).

I hope this project demonstrates a clean, end-to-end computer vision pipeline — from raw broadcast frames to a benchmarked on-device model — including how an evaluation bug gets caught, root-caused, and permanently fenced off. Fight on! ✌️

<p align="right">(<a href="#readme-top">back to top</a>)</p>

### Built With

* Python
* PyTorch
* Core ML / coremltools
* OpenCV
* NumPy
* TrackNet tennis dataset
* Google Colab (A100)
* pytest

<p align="right">(<a href="#readme-top">back to top</a>)</p>



## Project Structure

```sh
rallyvision/
├── pics/
│   ├── alcaraz_vs_sinner_demo.gif     # tracked broadcast footage
│   ├── rallyvision_visual.png         # best predictions grid
│   └── rallyvison_visual_2.png        # failure-case grid
│
├── config.yaml                        # single source of truth for every default
│
├── rallyvision/
│   ├── data/
│   │   ├── sources.py                 # dataset acquisition (local / gdrive / url / kaggle)
│   │   ├── labels.py                  # Label.csv parsing, visibility flags
│   │   ├── prepare.py                 # clip indexing, by-game split, frame cache
│   │   ├── manifest.py                # manifest I/O, split stats, leakage detection
│   │   ├── dataset.py                 # frame-stack Dataset + DataLoader factory
│   │   ├── transforms.py              # stack-consistent augmentation
│   │   ├── heatmap.py                 # clamped gaussian rendering + peak extraction
│   │   └── synthetic.py               # generated clips for tests and smoke runs
│   │
│   ├── models/
│   │   └── tracknet.py                # VGG encoder/decoder, full-res heatmap head
│   │
│   ├── engine/
│   │   ├── optim.py                   # AdamW param groups, cosine+warmup, AMP scaler
│   │   ├── trainer.py                 # training loop, best/last checkpointing, resume
│   │   └── evaluator.py               # split evaluation + metrics writing
│   │
│   ├── export/
│   │   ├── coreml_export.py           # torch.jit.trace -> .mlpackage (fp32 / fp16)
│   │   ├── coreml_validate.py         # torch vs Core ML numerical comparison
│   │   └── benchmark.py               # compute-unit latency sweep
│   │
│   ├── inference/
│   │   └── video.py                   # video decode -> stack -> heatmap -> annotate
│   │
│   ├── losses.py                      # focal / weighted-BCE / BCE / MSE heatmap losses
│   ├── metrics.py                     # pixel-error accumulator, accuracy@Npx, PR/F1
│   ├── tracker.py                     # jump rejection + gap interpolation
│   ├── checkpoint.py                  # versioned save/load with full run state
│   ├── paths.py                       # run directory layout
│   ├── device.py                      # device selection, AMP policy, cudnn flags
│   ├── artifacts.py                   # bundle best.pt + config + metrics for handoff
│   ├── config.py                      # YAML config with dotted-key overrides
│   ├── environment.py                 # environment capture / CUDA assertions
│   ├── viz.py                         # prediction grids
│   └── logging_utils.py               # console logging + banners
│
├── scripts/
│   ├── download_dataset.py            # fetch the TrackNet tennis clips
│   ├── prepare_dataset.py             # build manifest.json + cache resized frames
│   ├── train_ball_tracker.py          # training entrypoint (refuses leaky splits)
│   ├── evaluate_ball_tracker.py       # evaluate a checkpoint on a split
│   ├── visualize_predictions.py       # best/worst prediction grids
│   ├── infer_video.py                 # annotate a video + write track JSON
│   ├── export_coreml.py               # Core ML export (Mac only)
│   ├── validate_coreml.py             # numerical validation (Mac only)
│   ├── benchmark_coreml.py            # compute-unit benchmark (Mac only)
│   ├── collect_artifacts.py           # zip a run for handoff
│   ├── check_environment.py           # device / dependency preflight
│   └── make_synthetic_dataset.py      # tiny generated dataset for local tests
│
├── tests/                             # 160 tests (data, split guards, metrics, export)
│
├── requirements.txt                   # shared runtime
├── requirements-mac.txt               # + coremltools (Mac only)
├── requirements-dev.txt               # + pytest, ruff
└── .gitignore
```

<p align="right">(<a href="#readme-top">back to top</a>)</p>



## System Architecture

**Data Pipeline**

1. **Download**: fetch the community TrackNet tennis set (10 games, 95 clips, 19,835 labelled frames) — `game*/Clip*/` frame directories plus a per-clip `Label.csv`
2. **Index**: parse each `Label.csv` into `(visibility, x, y)` per frame; the **game id is the clip's parent folder**, deliberately *not* the top-level extraction directory
3. **Split**: group clips by game, hold out whole games for validation (`val_fraction 0.2`) — 7 games train / 3 games val
4. **Cache**: write 640×360 JPEG copies of every frame once, so training never re-decodes 1280×720 source frames
5. **Verify**: `manifest.split_problems()` checks for <2 games, empty train/val, and any game present in both splits; training aborts if any check fails
6. **Sample**: a training sample is frames `[i-2, i-1, i]` with the label from frame `i`; indexing starts at frame 2 of each clip, so no stack ever crosses a clip boundary

**Model Pipeline**

1. **Input**: 3 frames × 3 channels concatenated → `[N, 9, 360, 640]`, scaled to `[0, 1]`
2. **Encoder**: 4 VGG-style stages (2/2/3/3 conv blocks, 64→128→256→512 channels), MaxPool between stages → stride 8
3. **Decoder**: 3 nearest-neighbour upsample + conv stages mirroring the encoder, back to 360×640
4. **Head**: 1×1 conv → single channel, bias initialized to **−4.0** (a near-empty heatmap prior)
5. **Loss**: penalty-reduced focal loss on logits, normalized by the positive-pixel count
6. **Target**: a clamped gaussian (σ=2.5, radius 8) at the ball centre; all-zero heatmap when the ball is not visible

**Inference Pipeline**

1. Video decoded frame by frame; a rolling 3-frame buffer is resized to 640×360
2. Forward pass → sigmoid heatmap at full resolution
3. Peak extraction: threshold at 0.5, keep connected components ≥2 px, take the brightest centroid
4. Coordinates scaled back to original video resolution
5. `BallTracker` rejects jumps beyond a gap-scaled budget (100 px × gap), interpolates gaps ≤4 frames
6. Annotated MP4 + track JSON written

**Data Flow**

```
TrackNet tennis clips (game*/Clip*/*.jpg + Label.csv)
    ↓
prepare_dataset.py (index, by-game split, cache 640×360 frames)
    ↓
manifest.json ──► split_problems() ──► ABORT if leaky / empty
    ↓
BallDataset (3-frame stack, stack-consistent augmentation, gaussian target)
    ↓
TrackNet encoder (VGG, stride 8) ──► decoder (nearest upsample) ──► 1×1 head
    ↓
full-resolution heatmap [N, 1, 360, 640]
    ↓
focal loss (train)          │          peak extraction (eval / inference)
                            ↓
                  accuracy@Npx, PR/F1  │  BallTracker (jump reject + interpolate)
                            ↓                          ↓
                      best.pt (val-selected)     annotated video + track JSON
                            ↓
              Core ML export (fp32/fp16) → validate → benchmark on ANE/GPU/CPU
```

<p align="right">(<a href="#readme-top">back to top</a>)</p>



## Architecture Components

**Frame Stack Dataset** (`rallyvision/data/dataset.py`)

- One sample = `num_frames` consecutive frames on the channel axis; the label is the ball position in the **last** frame
- Sample indices start at `(num_frames−1) × frame_stride` within each clip, so a stack never spans two clips
- Frames with no visible ball are **kept** by default (`drop_invisible: false`) with an all-zero target — this teaches the model to output an empty heatmap instead of always hallucinating a ball
- Reads from the cached 640×360 frames when available, falling back to the raw frames transparently

**Stack-Consistent Augmentation** (`rallyvision/data/transforms.py`)

- Every transform is sampled **once per stack**, not per frame: the model's entire signal is inter-frame motion, so augmenting frames independently would destroy the thing it is meant to learn
- Horizontal flip, affine translate+scale (single `warpAffine` matrix), brightness/contrast
- Ball coordinates are transformed by the same matrix; a ball pushed outside the frame is correctly relabelled **invisible** rather than clamped to the border
- Validation is never augmented

**Heatmap Target and Peak Extraction** (`rallyvision/data/heatmap.py`)

- Gaussian clamped to a radius of 8 px, so rendering cost is independent of frame size
- Peak extraction: threshold → connected components → minimum blob area → brightest component's centroid, giving sub-pixel localization

**TrackNet Model** (`rallyvision/models/tracknet.py`)

- `ConvBlock` = Conv3×3 → ReLU → BatchNorm (the original TrackNet's unusual ordering, kept deliberately)
- Encoder stages `[64,64] / [128,128] / [256,256,256] / [512,512,512]`, decoder mirrors it with nearest-neighbour upsampling
- **Full-resolution output** — the ball is only a few pixels wide, so a stride-4 heatmap would quantize away most of the accuracy this model achieves
- Kaiming-normal init; **head bias = −4.0** so training starts from "mostly empty heatmap" instead of spending epochs learning that the target is 99.99% zeros
- `width` multiplier scales all channels (0.25 gives a fast model for smoke tests); `check_input_size` rejects dimensions not divisible by 8
- Every op (3×3 conv, BatchNorm, ReLU, MaxPool, nearest upsample) converts cleanly to Core ML with no Neural Engine fallback — 10,556,033 parameters

**Focal Heatmap Loss** (`rallyvision/losses.py`)

- Penalty-reduced pixel-wise focal loss (CornerNet eq. 1): `α=2` focuses on hard examples, `β=4` discounts negatives near the gaussian peak so near-misses are not punished as full errors
- `pos_threshold=0.9` rather than 1.0, because the rendered gaussian peak falls slightly below 1.0 when the ball centre lands between pixels
- Normalized by positive-pixel count; for frames with **no** visible ball (which contribute only negatives) it falls back to batch-size normalization, keeping their gradient comparable to a positive frame's

**Split Assignment and Integrity** (`rallyvision/data/prepare.py`, `rallyvision/data/manifest.py`)

- `game_from_clip_path` derives the game id from the clip's **parent folder**, immune to however many wrapper directories a download tool adds
- `assign_splits` guarantees at least one train group, falls back to by-clip splitting (with a warning) when only one game exists, and raises if either side ends up empty
- `split_problems()` returns a list of human-readable problems — fewer than 2 games, an empty usable-sample count on either side, or any game in both splits — and is called by the training script before a single batch is loaded

**Trainer** (`rallyvision/engine/trainer.py`)

- AdamW with parameter groups — BatchNorm and bias tensors excluded from weight decay
- Cosine decay to `min_lr`, with warmup folded into a single `LambdaLR` so there is only ever one scheduler object to checkpoint
- bf16 autocast on Ampere+ (no `GradScaler` needed), fp16 + `GradScaler` on older CUDA, AMP force-disabled on MPS/CPU
- Gradient clipping at norm 5.0
- Writes **`best.pt` first** (selected on `val/accuracy_at_5px`), then `last.pt`; raises if validation ran but the monitored metric is missing, and raises at the end of training if a val loader existed but no `best.pt` was ever written
- Full resume: model, optimizer, scheduler, scaler, RNG states, and metric history

**Metrics** (`rallyvision/metrics.py`)

- Errors are scaled by the per-clip factors stored in each sample; with cached frames (as in the reported run) those factors are 1, so errors are in **640×360 evaluation pixels**
- `accuracy@Npx` is computed over frames where the ball is **visible** only: a frame counts as correct when the detected peak is within N px of ground truth, and a missed detection counts as a failure
- Precision/recall/F1 at a 10 px tolerance, plus mean/median/p90/max error and RMSE
- Per-sample records are retained for the best/worst prediction grids and a CSV dump

**Ball Tracker** (`rallyvision/tracker.py`)

- Rejects a detection when its distance from the last accepted point exceeds `max_jump_px × min(gap, max_gap_frames+1)` — the budget scales with the gap, so a ball unseen for 3 frames is allowed to have travelled further
- Interpolates linearly across gaps of at most 4 frames, since a ball occluded by a player was still physically present
- Each output point is tagged `detected` / `interpolated` / `rejected` / `none` for downstream auditing

**Core ML Export and Validation** (`rallyvision/export/`)

- `torch.jit.trace` on a fixed `1×9×360×640` input → `.mlpackage` via coremltools, `iOS17` deployment target
- Validation compares PyTorch and Core ML outputs on **real validation frames** (not random noise): max absolute difference, mean difference, worst peak shift in pixels, and detection agreement
- Benchmark sweeps `ALL / CPU_AND_GPU / CPU_AND_NE / CPU_ONLY` with warmup, reporting median and p95 latency

<p align="right">(<a href="#readme-top">back to top</a>)</p>



## Usage

**Prerequisites**

```bash
git clone https://github.com/rayzhao27/rallyvision.git
cd rallyvision
```

```bash
conda create -n rallyvision python=3.9
conda activate rallyvision
```

```bash
# training / evaluation / inference
pip install -r requirements.txt

# Apple Silicon Mac only, adds coremltools
pip install -r requirements-mac.txt
```

**Check the environment**

```bash
python scripts/check_environment.py --device cuda --require-cuda
```

**Download and Prepare Data**

```bash
python scripts/download_dataset.py

python scripts/prepare_dataset.py --cache-frames
```

`--cache-frames` writes 640×360 copies once and is strongly recommended. Re-running is safe: the manifest is rebuilt but cached frames are reused. Use `--recache-frames` only after changing the input resolution.

```bash
# inspect an existing split without touching anything
python scripts/prepare_dataset.py --stats-only
```

**Train**

```bash
python scripts/train_ball_tracker.py \
  --device cuda \
  --run-name v3 \
  --epochs 30 \
  --batch-size 32 \
  --lr 1e-3 \
  --seed 1337 \
  --num-workers 8
```

The script logs train/val sample counts **and the game ids on each side**, then refuses to start if the split is empty or leaky. `best.pt` is written whenever `val/accuracy_at_5px` improves.

```bash
# resume after a disconnect
python scripts/train_ball_tracker.py --device cuda --run-name v3 --resume

# 3-batch smoke test (no real training)
python scripts/train_ball_tracker.py --device cuda --run-name smoke --smoke --smoke-batches 3
```

**Evaluate the val-selected checkpoint**

```bash
python scripts/evaluate_ball_tracker.py \
  --device cuda \
  --run v3 \
  --checkpoint artifacts/runs/v3/checkpoints/best.pt \
  --split val \
  --use-run-config
```

**Visualize best and worst predictions**

```bash
python scripts/visualize_predictions.py --device cuda --run v3 --num 6
```

**Annotate a video**

```bash
python scripts/infer_video.py \
  --device cuda \
  --run v3 \
  --video match.mov \
  --save-track track.json
```

**Export to Core ML (Apple Silicon Mac only)**

```bash
python scripts/export_coreml.py \
  --checkpoint checkpoint/best.pt \
  --precision both \
  --validate
```

```bash
python scripts/validate_coreml.py \
  --model artifacts/coreml/rallyvision_ball_fp16.mlpackage \
  --checkpoint checkpoint/best.pt \
  --run v3
```

```bash
python scripts/benchmark_coreml.py --model-dir artifacts/coreml
```

**Bundle a run for handoff**

```bash
python scripts/collect_artifacts.py --run v3 --zip
```

**Run the tests**

```bash
pip install -r requirements-dev.txt
python -m pytest -q
```

**Override any config value from the CLI**

Every script accepts repeated `--set key=value` flags using dotted paths into `config.yaml`:

```bash
python scripts/train_ball_tracker.py \
  --set data.loader.batch_size=16 \
  --set loss.name=wbce \
  --set data.augment.enabled=false
```

<p align="right">(<a href="#readme-top">back to top</a>)</p>



## Configuration

**Model Configuration**

| Parameter | Value |
|---|---|
| architecture | TrackNet (VGG encoder/decoder) |
| input | 3 frames × 3 channels = 9×360×640 |
| output | 1×360×640 heatmap (full resolution) |
| encoder stages | [64,64] / [128,128] / [256,256,256] / [512,512,512] |
| encoder stride | 8 (3× MaxPool) |
| decoder | nearest-neighbour upsample + mirrored conv stages |
| head | 1×1 conv, bias init −4.0 |
| width multiplier | 1.0 |
| sigmoid in forward | true (self-contained Core ML export) |
| dropout | 0.0 |
| total parameters | 10,556,033 |

**Data Configuration**

| Parameter | Value |
|---|---|
| input resolution | 640×360 (from 1280×720) |
| num_frames | 3 |
| frame_stride | 1 |
| heatmap downsample | 1 (full resolution) |
| gaussian sigma | 2.5 |
| gaussian radius | 8 px |
| drop_invisible | false |
| split strategy | by_game |
| val_fraction | 0.2 |
| cached frame JPEG quality | 92 |

**Augmentation Configuration**

| Parameter | Value |
|---|---|
| horizontal_flip | 0.5 |
| brightness | ±0.2 |
| contrast | ±0.2 |
| translate | ±5% of frame size |
| scale | ±5% |
| sampled per | stack (not per frame) |
| applied to val | never |

**Training Configuration**

| Parameter | Value |
|---|---|
| optimizer | AdamW (parameter groups) |
| learning_rate | 1e-3 |
| weight_decay | 1e-4 (non-BN/bias only) |
| lr_schedule | cosine decay to 1e-5, 1 warmup epoch |
| grad_clip | 5.0 |
| batch_size | 32 |
| epochs | 30 (best at 25) |
| loss | focal (α=2, β=4, pos_threshold=0.9) |
| mixed precision | bf16 autocast (CUDA only) |
| monitor | val/accuracy_at_5px (max) |
| seed | 1337 |

**Evaluation Configuration**

| Parameter | Value |
|---|---|
| detection_threshold | 0.5 |
| min_blob_area | 2 px |
| tolerances_px | 5, 10, 20 |
| pr_tolerance_px | 10 |
| error units | 640×360 evaluation pixels |

**Tracker Configuration**

| Parameter | Value |
|---|---|
| max_jump_px | 100 (scaled by gap length) |
| max_gap_frames | 4 |
| interpolate_gaps | true |
| smooth_window | 0 |

**Core ML Configuration**

| Parameter | Value |
|---|---|
| precisions exported | fp32, fp16 |
| deployment target | iOS17 |
| export batch size | 1 (fixed shape) |
| validation samples | 8 real val frames |
| max_abs_diff tolerance | 0.02 |
| max_peak_distance tolerance | 2.0 px |
| benchmark iters | 100 (10 warmup) |

**Dataset Statistics**

| Stat | Total | Train | Val (held out) |
|---|---|---|---|
| Games | 10 | 7 | 3 |
| Clips | 95 | 62 | 33 |
| Frames | 19,835 | 14,340 | 5,495 |
| Usable samples | 19,645 | 14,216 | 5,429 |
| Game ids | — | 1, 2, 3, 4, 6, 7, 10 | **5, 8, 9** |
| Visible-ball val frames | — | — | 5,240 (96.5%) |

<p align="right">(<a href="#readme-top">back to top</a>)</p>



## Cloud Training

Training was offloaded to Google Colab; Core ML work stays on the Mac. This split is enforced in code — `require_cuda_for_training` refuses to start a real training run on MPS or CPU, and the Core ML scripts are the only ones that import `coremltools`.

**GPU Selection**

This workload is **convolution-bound at high spatial resolution**: a 9×360×640 input through a full-resolution decoder means every layer operates on large feature maps, and activation memory dominates. An A100 40GB was selected because batch 32 peaks at ~29.5 GB of activations — an L4 (22 GB) caps out around batch 16 and roughly doubles wall-clock time.

**Data Loading**

The prepared frames live on Google Drive so they survive a disconnect, but reading three JPEGs per sample **through the Drive mount starved the GPU** — utilization sat near zero. Staging the prepared directory onto Colab's local SSD once per session (about 3 minutes for 19,837 files) and raising the loader to 8 workers fixed it. Checkpoints still write to Drive, so a disconnect loses nothing.

**Setup**

```bash
!git clone https://github.com/rayzhao27/rallyvision.git /content/rallyvision
!pip install -q opencv-python-headless "PyYAML>=6" tqdm pandas matplotlib scipy gdown

!python /content/rallyvision/scripts/download_dataset.py --set data.raw_dir=/content/drive/MyDrive/rallyvision/data/raw
!python /content/rallyvision/scripts/prepare_dataset.py --cache-frames --set data.prepared_dir=/content/drive/MyDrive/rallyvision/data/prepared

!python /content/rallyvision/scripts/train_ball_tracker.py \
  --device cuda \
  --run-name v3 \
  --epochs 30 \
  --batch-size 32 \
  --lr 1e-3 \
  --seed 1337 \
  --prepared-dir /content/prepared_local \
  --num-workers 8 \
  --set project.output_root=/content/drive/MyDrive/rallyvision/artifacts
```

**Training cost:** 25 epochs × ~4.4 min = **1 h 49 min** on a single A100.

<p align="right">(<a href="#readme-top">back to top</a>)</p>



## Results

**Held-out validation (3 unseen matches — game5, game8, game9; 5,429 samples; pixel errors at 640×360)**

| Metric | Value |
|---|---|
| samples | 5,429 |
| visible-ball frames | 5,240 |
| localized | 4,137 |
| mean error | 3.93 px |
| **median error** | **1.29 px** |
| p90 error | 2.29 px |
| accuracy@5px | 0.7725 |
| accuracy@10px | 0.7754 |
| accuracy@20px | 0.7771 |
| precision | 0.9783 |
| recall | 0.7865 |
| F1 | 0.8720 |
| counts | tp=4,063  fp=90  fn=1,103  tn=173 |

Mean error (3.93 px) is **3× the median** (1.29 px) because a handful of catastrophic confusions — a white shoe, a line junction, a bright logo — land hundreds of pixels away; the worst is 358 px. The p90 of 2.29 px is the honest description of typical behaviour.

**Training curve (val accuracy@5px on held-out games)**

| Epoch | train loss | val loss | accuracy@5px | recall | precision | localized |
|---|---|---|---|---|---|---|
| 1 | 562.07 | 4.07 | 0.0000 | 0.000 | — | 0 |
| 5 | 4.171 | 4.278 | 0.0000 | 0.000 | — | 0 |
| 10 | 4.098 | 4.287 | 0.0000 | 0.000 | — | 0 |
| 15 | 3.945 | 4.429 | 0.0000 | 0.000 | — | 0 |
| 17 | 2.751 | 2.861 | 0.0000 | 0.000 | — | 0 |
| 18 | 2.126 | 2.998 | 0.1002 | 0.132 | 0.962 | 714 |
| 19 | 1.339 | 1.443 | 0.3176 | 0.320 | 0.981 | 1,700 |
| 20 | 0.985 | 1.174 | 0.6183 | 0.633 | 0.980 | 3,341 |
| 21 | 0.821 | 1.036 | 0.7177 | 0.739 | 0.964 | 3,906 |
| 22 | 0.738 | 1.645 | 0.3865 | 0.389 | 0.979 | 2,064 |
| 23 | 0.653 | 0.883 | 0.7395 | 0.754 | 0.979 | 3,968 |
| 24 | 0.599 | 1.528 | 0.4195 | 0.424 | 0.982 | 2,240 |
| **25** | **0.570** | **0.775** | **0.7725** | **0.786** | **0.978** | **4,137** |

Two things stand out, and both are analyzed in the next section: the model produced **zero detections for its first 17 epochs** while the loss was steadily improving, and after it broke through, accuracy **oscillated violently** (0.72 → 0.39 → 0.74 → 0.42 → 0.77) while precision never moved off ~0.98.

**The run ended at epoch 25 of a planned 30, and the best epoch was the last one.** The model had not converged — val loss was still at its minimum and falling. The reported numbers are therefore a **floor**, not this architecture's ceiling.

**Core ML on Apple M3 Pro** (median of 100 iterations, 10 warmup, input 1×9×360×640)

| Model | Compute units | Median | p95 | FPS | Size |
|---|---|---|---|---|---|
| fp16 | **ALL** | **18.74 ms** | 20.37 ms | **52.5** | 20.2 MB |
| fp16 | CPU_AND_NE | 18.76 ms | 18.98 ms | 53.2 | 20.2 MB |
| fp16 | CPU_AND_GPU | 53.86 ms | 54.18 ms | 18.6 | 20.2 MB |
| fp16 | CPU_ONLY | 124.75 ms | 125.18 ms | 8.0 | 20.2 MB |
| fp32 | ALL | 59.30 ms | 59.50 ms | 16.9 | 40.3 MB |
| fp32 | CPU_AND_GPU | 59.31 ms | 60.07 ms | 16.8 | 40.3 MB |
| fp32 | CPU_AND_NE | 242.28 ms | 245.13 ms | 4.1 | 40.3 MB |
| fp32 | CPU_ONLY | 244.38 ms | 247.91 ms | 4.1 | 40.3 MB |

Three observations:

1. **fp16 is the only configuration that reaches the Neural Engine.** fp32 `CPU_AND_NE` (242.28 ms) is indistinguishable from fp32 `CPU_ONLY` (244.38 ms) — allowing the ANE buys nothing, because it does not execute fp32 and the graph falls back to CPU. The same units run fp16 **13× faster**. Quantization is not a nice-to-have here; it is the difference between shipping and not.
2. **Compute-unit requests are hints, and the precision decides what they mean.** fp16 `ALL` and `CPU_AND_NE` are identical (18.74 vs 18.76 ms), so `ALL` sends the whole graph to the ANE. fp32 `ALL` and `CPU_AND_GPU` are also identical (59.30 vs 59.31 ms) — with the ANE unusable, `ALL` falls back to the GPU instead.
3. **52.5 FPS at 360×640 leaves real headroom.** A 30 FPS feed needs 33 ms per frame; the model uses 19 ms, leaving budget for decode, tracking and rendering on the same core.

**Numerical fidelity (PyTorch vs Core ML, 8 real validation frames)**

| Precision | max abs diff | mean abs diff | worst peak shift | detection mismatches |
|---|---|---|---|---|
| fp32 | 0.000000 | 0.000000 | 0.000 px | 0 |
| fp16 | 0.000581 | 0.000073 | 0.000 px | 0 |

fp16 halves the model and gives a 3.2× speedup for a maximum output deviation of 0.0006 (tolerance 0.02) and **no change at all in where the peak lands**.

**Broadcast footage demo** (4K Alcaraz vs. Sinner clip, 2704×1696 @ 54.5 FPS, entirely outside the training distribution)

| Stat | Value |
|---|---|
| frames processed | 371 |
| detected | 273 |
| interpolated by tracker | 23 |
| rejected as implausible jumps | 0 |
| **coverage** | **79.8%** |
| mean frame-to-frame step | 17.7 px |
| max accepted step | 107.4 px |

Coverage on unseen 4K broadcast footage (79.8%) lands close to validation recall (78.7%) — the model generalizes off its training distribution without retraining, despite a different resolution, aspect ratio, broadcast overlay style and frame rate. Zero jump rejections indicates the detections were temporally coherent rather than flickering between false positives.

<p align="right">(<a href="#readme-top">back to top</a>)</p>



## Threshold and Confidence Analysis

The final model has **precision 0.978 but recall 0.787**. Nearly every error is a *miss*, not a mistake. Three independent pieces of evidence point at the same cause: the model's peak confidence sits close to the 0.5 detection threshold, so the metric measures calibration as much as localization.

**Evidence 1 — tolerance curves are flat.** accuracy@5px = 0.7725, @10px = 0.7754, @20px = 0.7771. Widening the tolerance **fourfold** buys 0.46 percentage points. In a normal detector this curve rises steeply, because loosening the tolerance forgives near-misses. Here there are almost no near-misses to forgive: of 4,137 localized samples, the p90 error is 2.29 px. **The model either nails the ball to within ~2 px or does not report it at all.**

**Evidence 2 — failure cases are all low-confidence, not mislocalized.** Every panel in the [failure grid](./pics/rallyvison_visual_2.png) is labelled `MISS (no detection)` with peak scores of **0.06, 0.08, 0.11, 0.24, 0.17, 0.27** — all below the 0.5 threshold, and several with a visible faint blob in the predicted heatmap at approximately the right place. The model frequently *knows* where the ball is while reporting confidence too low to count.

**Evidence 3 — accuracy oscillates while precision does not.** Across epochs 21–25, accuracy@5px swung 0.72 → 0.39 → 0.74 → 0.42 → 0.77 while precision stayed within 0.964–0.982 and median error stayed near 1.3 px. An epoch that "lost" 35 points of accuracy had not forgotten how to localize — the count of localized samples halved (3,906 → 2,064) while the ones it did report stayed just as accurate. **What oscillates is the confidence scale, not the spatial skill.**

**Why the first 17 epochs detected nothing.** The head bias is initialized to −4.0 (σ(−4) ≈ 0.018) and focal loss with β=4 aggressively discounts negatives near the gaussian peak. Together these hold the entire heatmap well under 0.5 while the network learns spatial structure. Val loss improved from 4.07 to 2.86 over those epochs with accuracy pinned at exactly 0.0000, then detections appeared at epoch 18 and accuracy reached 0.77 seven epochs later. **A monitored metric of 0.0 for two-thirds of a run was healthy progress, not a stall** — and `early_stopping_patience` on any reasonable value would have killed the run before it produced a single detection. This is why the default is `0` (off).

**Interpretation (hypothesis, not fully proven).** Under focal loss the positive class is a handful of pixels per 230,400, so the loss is dominated by keeping negatives near zero; the gradient pushing peak confidence *above* 0.5 is comparatively weak and only sharpens late in training. The resulting model appears well-localized but under-confident, with the 0.5 threshold cutting into the body of its confidence distribution rather than its tail. Fully confirming this requires a **precision/recall sweep over the detection threshold** on the saved `val_per_sample.csv` — that sweep is the single highest-value next experiment and is flagged as future work.

**What this implies for the reported numbers.** accuracy@5px of 0.7725 is a *thresholded* metric at 0.5. Lowering the threshold would trade precision (currently 0.978, with plenty of margin) for recall (0.787, the binding constraint). Since 90 false positives versus 1,103 false negatives is a badly lopsided operating point for a tracker that can post-hoc reject implausible jumps, **0.5 is very likely not the optimal threshold for this model** — the broadcast demo, where the tracker rejected 0 of 273 detections, supports that.

**Takeaways.**

1. Report the tolerance curve, not a single tolerance. A flat accuracy@5/10/20 curve is a fingerprint that says "my errors are misses, not mislocalizations", and it points at a completely different fix than a steep curve would.
2. Do not early-stop on a metric that is structurally zero early in training. With a near-empty-heatmap prior and focal loss, the detection metric is uninformative until the confidence scale calibrates; monitor loss during that phase.
3. Tune the decision threshold as a deployment parameter, separately from training. It is one scalar, it needs no retraining, and here it plausibly controls more of the headline metric than any architectural change would.

<p align="right">(<a href="#readme-top">back to top</a>)</p>



## Split Integrity

The first training run of this project reported **accuracy@5px of 0.957 and a 1.16 px mean error**. Those numbers were meaningless, and the bug is worth documenting because it produced a *better-looking* result than the correct pipeline does.

**The bug.** Splitting is done by game so that no frame from a validation match is ever trained on. The game id was derived from the clip path by taking the **first** path component. The dataset download tool wrapped everything in one extra directory, so every clip's path began `Dataset/gameN/ClipM` and every clip resolved to the same game, `"Dataset"`. With one group, the by-game split put **all 95 clips in train and left validation empty**.

**Why it went unnoticed.** An empty validation split is silently self-consistent: validation "passes" instantly with no samples, `best.pt` is never written because no metric is ever computed, and the run completes normally leaving only `last.pt`. The leaked metrics came from scoring that checkpoint on frames belonging to games it had trained on — with every game in train, no honest evaluation split existed to score it against.

**The fix, and the guardrails.** The game id now comes from the clip's **parent** folder (`game_from_clip_path`), which is invariant to wrapper directories. Four independent checks were added so this class of failure cannot silently recur:

1. `assign_splits` raises if either side of the split ends up empty, and falls back to by-clip splitting with an explicit warning when the dataset genuinely contains only one game.
2. `manifest.split_problems()` reports fewer than 2 games, zero usable samples on either side, or **any game present in both splits**, and the training script aborts on any of them before loading a batch.
3. The trainer writes `best.pt` *before* `last.pt`, raises if validation ran without producing the monitored metric, and raises at the end of training if a validation loader existed but no `best.pt` was ever saved — making "training finished but there is no val-selected checkpoint" a hard error instead of a quiet outcome.
4. Evaluation warns loudly when it falls back to `last.pt`, and the reported numbers here come from `best.pt` on games the model never saw.

The corrected split is printed and asserted before every run:

```
unique games : 10
train games  : game1, game10, game2, game3, game4, game6, game7  (62 clips, 14,216 samples)
val games    : game5, game8, game9                               (33 clips,  5,429 samples)
OK: 3 held-out val game(s) are not in train.
```

**Takeaways.**

1. A metric that improves after a pipeline change deserves the same scrutiny as one that regresses. 0.957 → 0.772 was the pipeline getting *more* correct, not less.
2. Empty is the most dangerous value for a data split, because it makes every downstream check vacuously pass. Assert non-emptiness explicitly and early.
3. Infer grouping keys from the **end** of a path, not the beginning. Leading path components belong to whoever unzipped the archive; trailing ones belong to the dataset.
4. Selection and reporting should use the same held-out games, and the code should refuse to proceed when they cannot.

<p align="right">(<a href="#readme-top">back to top</a>)</p>



## Design Decisions

**Why a full-resolution heatmap instead of the usual stride-4 output?** A tennis ball in a 1280×720 broadcast frame is roughly 5–10 px across. At stride 4 on a 640×360 grid, one output cell covers 8×8 original pixels — larger than the object being located, which would quantize away the 1.29 px median error this model achieves. The decoder cost is real but buys the accuracy that makes the system useful.

**Why stack 3 frames instead of detecting on a single frame?** A stationary white blob on a green court is ambiguous: line junctions, logos, shoes and distant balls all look like candidates. Motion disambiguates them, and stacking on the channel axis gives the network that cue with no recurrence, no optical flow, and no dynamic shapes — which also keeps the Core ML graph trivially convertible.

**Why focal loss over weighted BCE?** The target heatmap is ~0.01% positive pixels. Plain BCE converges to "predict zero everywhere", and weighted BCE requires hand-tuning `pos_weight` per resolution while punishing near-misses as harshly as gross errors. The penalty-reduced focal loss discounts negatives *by their proximity to the gaussian peak* (β=4), so a prediction one pixel off is treated as nearly correct — exactly the right inductive bias for sub-pixel localization.

**Why initialize the head bias to −4.0?** With a neutral init the first epochs are spent learning nothing but "the answer is almost always zero". Starting at σ(−4) ≈ 0.018 encodes that prior directly and lets training spend its budget on spatial structure. The cost is documented in [Threshold and Confidence Analysis](#threshold-and-confidence-analysis): detections do not appear until the confidence scale catches up.

**Why keep frames where the ball is invisible?** Dropping them (`drop_invisible: true`) yields a model that has never been rewarded for outputting an empty heatmap, so it hallucinates a ball during occlusions and between rallies. Keeping them costs some effective training signal but produces the 0.978 precision that makes the tracker's jump rejection unnecessary in practice.

**Why augment per stack rather than per frame?** The entire signal this model relies on is inter-frame motion. Flipping frame 2 but not frames 1 and 3 would fabricate motion that no camera could produce. Sampling one transform per stack preserves the temporal relationship while still varying the data.

**Why split by game and not by clip or frame?** Clips from the same match share court surface, camera angle, lighting, broadcast overlay and often the same players. A frame-level or clip-level split lets the model memorize a court instead of learning a ball, and inflates metrics accordingly. Whole-match hold-out is the only split that answers the deployment question: *does this work on a match I have never seen?*

**Why ReLU before BatchNorm in `ConvBlock`?** This inverts the modern convention and is kept deliberately to match the original TrackNet, so results remain comparable to that line of work. Changing it is a one-line experiment worth running, not an unnoticed bug.

**Why cosine decay with warmup folded into a single `LambdaLR`?** Two scheduler objects means two things to checkpoint, restore and get subtly wrong on resume. Folding warmup into the same lambda leaves exactly one scheduler in the checkpoint, and resume is verified by a test rather than by inspection.

**Why bf16 rather than fp16 on Ampere and newer?** bf16 has fp32's exponent range, so it needs no `GradScaler` and cannot silently produce inf/NaN in a loss that divides by a positive-pixel count that may be zero. fp16 with a scaler is used on older CUDA where bf16 is unavailable.

**Why export fp32 as well, when fp16 is strictly better on device?** fp32 is the numerical reference: validating fp16 against PyTorch alone conflates conversion error with quantization error. fp32 matched PyTorch to 0.000000, which isolates fp16's 0.000581 as pure quantization — and made it safe to ship the smaller model.

**Why is Core ML work confined to the Mac and training confined to Colab?** They have incompatible dependency requirements (coremltools pins vs. Colab's CUDA torch build) and incompatible hardware needs. `require_cuda_for_training` refuses real training on MPS/CPU and the Core ML scripts are the only importers of `coremltools`, so the boundary is enforced by code rather than by discipline.

<p align="right">(<a href="#readme-top">back to top</a>)</p>



## Reference

Huang, Y.-C., Liao, I.-N., Chen, C.-H., İk, T.-U., & Peng, W.-C. (2019). TrackNet: A Deep Learning Network for Tracking High-speed and Tiny Objects in Sports Applications. *AVSS 2019*. https://arxiv.org/abs/1907.03698

Law, H., & Deng, J. (2018). CornerNet: Detecting Objects as Paired Keypoints. *ECCV 2018*. https://arxiv.org/abs/1808.01244

Zhou, X., Wang, D., & Krähenbühl, P. (2019). Objects as Points. https://arxiv.org/abs/1904.07850

Lin, T.-Y., Goyal, P., Girshick, R., He, K., & Dollár, P. (2017). Focal Loss for Dense Object Detection. *ICCV 2017*. https://arxiv.org/abs/1708.02002

Loshchilov, I., & Hutter, F. (2019). Decoupled Weight Decay Regularization. *ICLR 2019*. https://arxiv.org/abs/1711.05101

Simonyan, K., & Zisserman, A. (2015). Very Deep Convolutional Networks for Large-Scale Image Recognition. *ICLR 2015*. https://arxiv.org/abs/1409.1556

Labelled tennis dataset: https://github.com/yastrebksv/TrackNet
