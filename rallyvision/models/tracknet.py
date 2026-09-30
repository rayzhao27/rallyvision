"""TrackNet-style ball detector.

A VGG-shaped encoder/decoder that maps a stack of ``num_frames`` RGB frames to
a single-channel heatmap at full input resolution. The ball is a handful of
pixels wide and often motion-blurred, so the design keeps a full-resolution
output rather than the usual stride-4 heatmap, and relies on the temporal stack
for the "which bright blob is moving like a ball" signal.

Every op used here converts cleanly to Core ML: 3x3 convolutions, BatchNorm,
ReLU, MaxPool and nearest-neighbour upsampling. No dynamic shapes, no ops that
force a CPU fallback on the Neural Engine.
"""

from __future__ import annotations

from typing import List, Sequence

import torch
import torch.nn as nn

# Downsampling factor of the encoder; input dims must be divisible by this.
STRIDE = 8


def _scaled(channels: int, width: float) -> int:
    """Scale a channel count by `width`, rounded to a multiple of 8, min 8."""
    value = int(round(channels * width / 8.0)) * 8
    return max(8, value)


class ConvBlock(nn.Module):
    """Conv3x3 -> ReLU -> BatchNorm, as in the original TrackNet."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.norm = nn.BatchNorm2d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(self.relu(self.conv(x)))


def _stage(in_channels: int, channels: Sequence[int]) -> nn.Sequential:
    layers: List[nn.Module] = []
    prev = in_channels
    for out_channels in channels:
        layers.append(ConvBlock(prev, out_channels))
        prev = out_channels
    return nn.Sequential(*layers)


class TrackNet(nn.Module):
    """Encoder/decoder heatmap regressor.

    Args:
        num_frames: consecutive frames stacked on the channel axis.
        in_channels_per_frame: 3 for RGB.
        out_channels: heatmap channels (1 = ball position in the last frame).
        width: channel multiplier; 0.25 gives a fast model for smoke tests.
        sigmoid_output: apply sigmoid in `forward`. Kept on so the exported
            Core ML model is self-contained; the loss uses `forward_logits`.
    """

    def __init__(
        self,
        num_frames: int = 3,
        in_channels_per_frame: int = 3,
        out_channels: int = 1,
        width: float = 1.0,
        sigmoid_output: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.num_frames = int(num_frames)
        self.in_channels_per_frame = int(in_channels_per_frame)
        self.in_channels = self.num_frames * self.in_channels_per_frame
        self.out_channels = int(out_channels)
        self.width = float(width)
        self.sigmoid_output = bool(sigmoid_output)

        c1, c2, c3, c4 = (_scaled(c, width) for c in (64, 128, 256, 512))

        self.enc1 = _stage(self.in_channels, [c1, c1])
        self.pool1 = nn.MaxPool2d(2, 2)
        self.enc2 = _stage(c1, [c2, c2])
        self.pool2 = nn.MaxPool2d(2, 2)
        self.enc3 = _stage(c2, [c3, c3, c3])
        self.pool3 = nn.MaxPool2d(2, 2)
        self.enc4 = _stage(c3, [c4, c4, c4])

        self.up3 = nn.Upsample(scale_factor=2, mode="nearest")
        self.dec3 = _stage(c4, [c3, c3, c3])
        self.up2 = nn.Upsample(scale_factor=2, mode="nearest")
        self.dec2 = _stage(c3, [c2, c2])
        self.up1 = nn.Upsample(scale_factor=2, mode="nearest")
        self.dec1 = _stage(c2, [c1, c1])

        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.head = nn.Conv2d(c1, self.out_channels, kernel_size=1)

        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
        # Start with a near-empty heatmap: the target is ~0.01% positive pixels,
        # so a neutral init spends the first epochs just learning "mostly zero".
        nn.init.constant_(self.head.bias, -4.0)

    def forward_logits(self, x: torch.Tensor) -> torch.Tensor:
        """Raw logits, shaped ``[N, out_channels, H, W]``."""
        x = self.enc1(x)
        x = self.enc2(self.pool1(x))
        x = self.enc3(self.pool2(x))
        x = self.enc4(self.pool3(x))
        x = self.dec3(self.up3(x))
        x = self.dec2(self.up2(x))
        x = self.dec1(self.up1(x))
        return self.head(self.dropout(x))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.forward_logits(x)
        return torch.sigmoid(logits) if self.sigmoid_output else logits

    @torch.no_grad()
    def predict_heatmap(self, x: torch.Tensor) -> torch.Tensor:
        """Probabilities in [0, 1], regardless of `sigmoid_output`."""
        logits = self.forward_logits(x)
        return torch.sigmoid(logits)

    def expected_input_shape(self, height: int, width: int) -> tuple:
        return (1, self.in_channels, height, width)

    def extra_repr(self) -> str:
        return (
            f"num_frames={self.num_frames}, in_channels={self.in_channels}, "
            f"out_channels={self.out_channels}, width={self.width}"
        )


def check_input_size(height: int, width: int) -> None:
    """Raise if the encoder's three pooling stages would not divide evenly."""
    if height % STRIDE or width % STRIDE:
        raise ValueError(
            f"input {width}x{height} must be divisible by {STRIDE} "
            f"(the model pools three times); try {width // STRIDE * STRIDE}x"
            f"{height // STRIDE * STRIDE}"
        )
