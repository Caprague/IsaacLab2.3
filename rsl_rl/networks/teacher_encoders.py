# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import torch.nn as nn

from rsl_rl.utils import resolve_nn_activation


class HeightScanEncoder(nn.Module):
    """AutoEncoder for height scan features.

    Encodes a 187-dimensional height scan (reshaped to [1, 11, 17]) into a
    compact latent representation via 2D convolutions, and decodes it back
    for reconstruction.

    Encoder architecture:
        [B, 187] → reshape [B, 1, 11, 17]
        → Conv2d(1→16, k=5, p=2) → ELU
        → Conv2d(16→latent_dim, k=5, p=2) → ELU
        → AdaptiveAvgPool2d(1) → flatten → [B, latent_dim]

    Decoder architecture:
        [B, latent_dim] → Linear(latent_dim→128) → ELU
        → Linear(128→187) → [B, 187]

    Args:
        latent_dim: Dimension of the latent space. Defaults to 32.
        activation: Activation function name. Defaults to ``"elu"``.

    """

    def __init__(self, latent_dim: int = 32, activation: str = "elu"):
        super().__init__()

        self.latent_dim = latent_dim

        # Encoder: expects [B, 1, 11, 17], outputs [B, latent_dim]
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, padding=2),
            resolve_nn_activation(activation),
            nn.Conv2d(16, latent_dim, kernel_size=5, padding=2),
            resolve_nn_activation(activation),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )

        # Decoder: expects [B, latent_dim], outputs [B, 187]
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            resolve_nn_activation(activation),
            nn.Linear(128, 187),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode height scan into latent representation.

        Args:
            x: Input tensor of shape ``[B, 187]``.

        Returns:
            Latent tensor of shape ``[B, latent_dim]``.

        """
        x = x.reshape(-1, 1, 11, 17)
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent representation back to height scan.

        Args:
            z: Latent tensor of shape ``[B, latent_dim]``.

        Returns:
            Reconstructed tensor of shape ``[B, 187]``.

        """
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Full autoencoder forward pass: encode then decode.

        Args:
            x: Input tensor of shape ``[B, 187]``.

        Returns:
            Reconstructed tensor of shape ``[B, 187]``.

        """
        return self.decode(self.encode(x))


class PrivilegeEncoder(nn.Module):
    """AutoEncoder for privilege (teacher) features.

    Encodes a multi-dimensional privilege vector into a compact latent
    representation via MLP layers, and decodes it back for reconstruction.

    Encoder architecture:
        [B, input_dim] → Linear(input_dim→128) → ELU
        → Linear(128→64) → ELU
        → Linear(64→latent_dim) → [B, latent_dim]

    Decoder architecture:
        [B, latent_dim] → Linear(latent_dim→64) → ELU
        → Linear(64→128) → ELU
        → Linear(128→input_dim) → [B, input_dim]

    Args:
        input_dim: Dimension of the input privilege vector. Defaults to 60.
        latent_dim: Dimension of the latent space. Defaults to 32.
        activation: Activation function name. Defaults to ``"elu"``.

    """

    def __init__(self, input_dim: int = 60, latent_dim: int = 32, activation: str = "elu"):
        super().__init__()

        self.input_dim = input_dim
        self.latent_dim = latent_dim

        # Encoder: expects [B, input_dim], outputs [B, latent_dim]
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            resolve_nn_activation(activation),
            nn.Linear(128, 64),
            resolve_nn_activation(activation),
            nn.Linear(64, latent_dim),
        )

        # Decoder: expects [B, latent_dim], outputs [B, input_dim]
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            resolve_nn_activation(activation),
            nn.Linear(64, 128),
            resolve_nn_activation(activation),
            nn.Linear(128, input_dim),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode privilege vector into latent representation.

        Args:
            x: Input tensor of shape ``[B, input_dim]``.

        Returns:
            Latent tensor of shape ``[B, latent_dim]``.

        """
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent representation back to privilege vector.

        Args:
            z: Latent tensor of shape ``[B, latent_dim]``.

        Returns:
            Reconstructed tensor of shape ``[B, input_dim]``.

        """
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Full autoencoder forward pass: encode then decode.

        Args:
            x: Input tensor of shape ``[B, input_dim]``.

        Returns:
            Reconstructed tensor of shape ``[B, input_dim]``.

        """
        return self.decode(self.encode(x))
