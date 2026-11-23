#!/usr/bin/env python
"""Find the training batch size that uses the GPU best, before a long run.

    python scripts/sweep_batch_size.py --set data.input_height=720 --set data.input_width=1280

For each batch size: a few warm-up steps, then timed training steps (forward, loss,
backward, optimizer) on random inputs of the configured size, with the same mixed
precision as training. Reports samples/s and peak GPU memory, stops at the first
out-of-memory, and suggests the largest batch that stays under 85% of GPU memory
while running within 5% of the best throughput (more samples per step for the
same speed; the margin leaves room for the real data and the validation pass).
GPU only: the data loader is timed separately by the notebook.
"""

from __future__ import annotations

import argparse
import time

import torch

from rallyvision.common.cli import add_config_args, build_config
from rallyvision.common.device import autocast_context, resolve_device_from_config
from rallyvision.ball_tracking.model.losses import build_loss
from rallyvision.ball_tracking.model import build_model


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_config_args(ap)
    ap.add_argument("--sizes", type=int, nargs="+", default=[8, 16, 24, 32, 48, 64, 96])
    ap.add_argument("--steps", type=int, default=15)
    args = ap.parse_args()

    cfg = build_config(args)
    device = resolve_device_from_config(cfg, "cuda")
    if not device.is_cuda:
        raise SystemExit("needs a CUDA GPU")
    torch.backends.cudnn.benchmark = True
    h, w = int(cfg.data.input_height), int(cfg.data.input_width)
    c_in = 3 * int(cfg.data.num_frames)
    total = torch.cuda.get_device_properties(0).total_memory
    print(f"{torch.cuda.get_device_name(0)}, {total / 2**30:.0f} GB; input {w}x{h}, amp {device.amp_dtype}")

    results = []
    for bs in args.sizes:
        model = build_model(cfg).to(device.device).train()
        opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
        loss_fn = build_loss(cfg)
        x = torch.rand(bs, c_in, h, w, device=device.device)
        y = torch.zeros(bs, 1, h, w, device=device.device)
        y[:, :, h // 2, w // 2] = 1.0
        torch.cuda.reset_peak_memory_stats()
        try:
            for i in range(args.steps + 3):
                if i == 3:
                    torch.cuda.synchronize()
                    t0 = time.time()
                with autocast_context(device):
                    loss = loss_fn(model.forward_logits(x), y)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
            torch.cuda.synchronize()
        except torch.cuda.OutOfMemoryError:
            print(f"batch {bs:3d}: out of memory")
            break
        finally:
            del model, opt, x, y
        rate = bs * args.steps / (time.time() - t0)
        peak = torch.cuda.max_memory_allocated() / total
        results.append((bs, rate, peak))
        print(f"batch {bs:3d}: {rate:7.1f} samples/s, peak memory {peak:5.1%}")
        torch.cuda.empty_cache()

    ok = [r for r in results if r[2] <= 0.85]
    if not ok:
        raise SystemExit("no batch size fits under 85% of GPU memory")
    best = max(r[1] for r in ok)
    pick = max(r[0] for r in ok if r[1] >= 0.95 * best)
    print(f"\nsuggested batch size: {pick}  (best throughput {best:.1f} samples/s)")


if __name__ == "__main__":
    main()
