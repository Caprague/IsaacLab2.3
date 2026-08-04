# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import torch.nn as nn

from rsl_rl.utils import resolve_nn_activation


class StudentDepthCNN(nn.Module):
    """A lightweight CNN encoder for depth images, designed for GRU distillation.

    This encoder processes 360° LiDAR depth images with horizontal cyclic padding
    to handle the wrap-around nature of panoramic data. It uses a 3-layer
    convolutional architecture with global average pooling to produce a compact
    32-dimensional feature vector, aligned with the teacher encoder output.

    Architecture::

        Input: [B, 1, 32, 180] (NCHW)
        ── horizontal cyclic pad (2 cols each side) ──┐
        Conv2d(1→16, k=5, s=1) → ELU → [B, 16, 32, 180]
        ── horizontal cyclic pad (2 cols each side) ──┐
        Conv2d(16→32, k=5, s=2) → ELU → [B, 32, 16, 90]
        ── horizontal cyclic pad (2 cols each side) ──┐
        Conv2d(32→64, k=5, s=2) → ELU → [B, 64, 8, 45]
        AdaptiveAvgPool2d(1) → [B, 64, 1, 1] → flatten → [B, 64]
        Linear(64→32) → [B, 32]

    The horizontal cyclic padding concatenates 2 columns from the opposite side
    on each horizontal edge, allowing the convolution kernel to naturally wrap
    around the 360° field of view. Vertical padding is handled by the Conv2d
    ``padding=(2, 0)`` argument.

    Args:
        activation: Name of the activation function to use after each conv layer.
            Defaults to ``"elu"``. Supported values are those accepted by
            :func:`rsl_rl.utils.resolve_nn_activation`.

    """

    def __init__(self, activation: str = "elu"):
        super().__init__()

        act_fn = resolve_nn_activation(activation)

        # ── Layer 1: 1 → 16 channels, stride 1 ──
        # padding=(2, 0): 2px vertical padding; horizontal padded via cyclic cat
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=(2, 0))

        # ── Layer 2: 16 → 32 channels, stride 2 (downsample) ──
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2, padding=(2, 0))

        # ── Layer 3: 32 → 64 channels, stride 2 (downsample) ──
        self.conv3 = nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=(2, 0))

        # Shared activation (stateless, safe to reuse across layers)
        self.act = act_fn

        # Global average pooling → scalar per channel
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Final projection to 32-dim (aligned with teacher encoder output)
        self.fc = nn.Linear(64, 32)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with horizontal cyclic padding at each conv stage.

        Args:
            x: Input depth image tensor of shape ``[B, 1, 32, 180]`` in NCHW
                format, where the last dimension corresponds to 180 angular bins
                covering a 360° horizontal field of view.

        Returns:
            Output feature vector of shape ``[B, 32]``.

        """
        # ── Conv block 1 ──
        # Cyclic pad: wrap 2 columns from each horizontal edge
        x = torch.cat([x[..., -2:], x, x[..., :2]], dim=-1)
        x = self.act(self.conv1(x))

        # ── Conv block 2 ──
        x = torch.cat([x[..., -2:], x, x[..., :2]], dim=-1)
        x = self.act(self.conv2(x))

        # ── Conv block 3 ──
        x = torch.cat([x[..., -2:], x, x[..., :2]], dim=-1)
        x = self.act(self.conv3(x))

        # ── Global pooling + projection ──
        x = self.global_pool(x).flatten(1)  # [B, 64]
        x = self.fc(x)                       # [B, 32]

        return x
