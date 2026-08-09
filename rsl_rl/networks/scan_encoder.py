# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Scan encoder for the teacher policy (Parkour-style feature injection).

Encodes the 187-dim height scan (reshaped to [B, 1, 11, 17]) into a compact
32-dim latent via 2D convolutions. It is trained end-to-end inside the teacher
actor during PPO, and its output ``scan_latent`` defines the feature injection
slot that the student's ``depth_latent`` is aligned to during distillation.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from rsl_rl.utils import resolve_nn_activation


class ScanEncoder(nn.Module):
    """Compact 2D-conv encoder for height scan features (187 -> latent_dim)."""

    def __init__(self, latent_dim: int = 32, activation: str = "elu"):
        super().__init__()

        self.latent_dim = latent_dim
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, padding=2),
            resolve_nn_activation(activation),
            nn.Conv2d(16, latent_dim, kernel_size=5, padding=2),
            resolve_nn_activation(activation),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode a flat height scan into a compact latent.

        Args:
            x: Input tensor of shape ``[B, 187]``.

        Returns:
            Latent tensor of shape ``[B, latent_dim]``.
        """
        return self.encoder(x.reshape(-1, 1, 11, 17))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Alias for :meth:`encode`."""
        return self.encode(x)
