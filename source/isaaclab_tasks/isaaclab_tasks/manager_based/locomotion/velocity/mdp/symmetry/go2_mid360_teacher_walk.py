# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


"""Functions to specify left-right symmetry for Go2 Mid360 teacher-stage observations.

This module handles the 4 observation groups used by the Mid360 config's teacher policy:
    - proprioception (history=5, 47 dims/frame × 5 = 235 total)
    - mapScans (history=1, 187 total)
    - privileged (history=3, 18 dims/frame × 3 = 54 total)
    - headProximity (history=1, 32 total)

Observation layout (concatenate_terms=True + history_length):
    Each group is a flat tensor of [frame_0, frame_1, ..., frame_{H-1}],
    where each frame is the concatenation of all terms for that time step.
    Transformations operate per-frame to be robust to the underlying storage order.
"""

from __future__ import annotations

import torch
from tensordict import TensorDict
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedRLEnv

# specify the functions that are available for import
__all__ = ["compute_symmetric_states"]


@torch.no_grad()
def compute_symmetric_states(
    env: ManagerBasedRLEnv,
    obs: TensorDict | None = None,
    actions: torch.Tensor | None = None,
):
    """Augments observations and actions with left-right symmetry transformations.

    Applies a single symmetry (original + left-right mirrored), doubling the batch.

    Args:
        env: The environment instance.
        obs: The original observation tensor dictionary. Defaults to None.
        actions: The original actions tensor. Defaults to None.

    Returns:
        Augmented observations and actions tensors, or None if input was None.
    """

    # observations
    if obs is not None:
        batch_size = obs.batch_size[0]
        obs_aug = obs.repeat(2)

        # --- proprioception (history=5, 235 dims) ---
        obs_aug["proprioception"][:batch_size] = obs["proprioception"][:]
        obs_aug["proprioception"][batch_size:] = _transform_proprioception_left_right(
            obs["proprioception"]
        )

        # --- mapScans (history=1, 187 dims) ---
        obs_aug["mapScans"][:batch_size] = obs["mapScans"][:]
        obs_aug["mapScans"][batch_size:] = _transform_mapScans_left_right(
            obs["mapScans"]
        )

        # --- privileged (history=3, 54 dims) ---
        obs_aug["privileged"][:batch_size] = obs["privileged"][:]
        obs_aug["privileged"][batch_size:] = _transform_privileged_left_right(
            obs["privileged"]
        )

        # --- headProximity (history=1, 32 dims) ---
        obs_aug["headProximity"][:batch_size] = obs["headProximity"][:]
        obs_aug["headProximity"][batch_size:] = _transform_headProximity_left_right(
            obs["headProximity"]
        )

    else:
        obs_aug = None

    # actions
    if actions is not None:
        batch_size = actions.shape[0]
        actions_aug = torch.zeros(batch_size * 2, actions.shape[1], device=actions.device)
        actions_aug[:batch_size] = actions[:]
        actions_aug[batch_size:] = _transform_actions_left_right(actions)
    else:
        actions_aug = None

    return obs_aug, actions_aug


# ============================================================================================
# Proprioception: 47 dims/frame × history=5 = 235 total
#
# Per-frame layout:
#   phase(2)  vel_cmd(3)  imu(3)  grav(3)  joint_pos(12)  joint_vel(12)  actions(12)
#   | 0-1  |  |  2-4  |  | 5-7 |  | 8-10 |  |   11-22   |  |   23-34   |  |  35-46  |
#
# Phase negation:
#   In trot gait, diagonal pair 0 (FL+RR) and pair 1 (FR+RL) alternate.
#   Under left-right mirror (FL↔FR, RL↔RR), the pairs swap roles, advancing
#   the phase by half a cycle: φ' = (φ + 0.5) mod 1.0.
#     sin(2π·φ') = sin(2πφ + π) = -sin(2πφ)
#     cos(2π·φ') = cos(2πφ + π) = -cos(2πφ)
#   Therefore both sin and cos are negated.
# ============================================================================================

PROPRIO_DIMS_PER_FRAME = 47

def _transform_proprioception_left_right(obs: torch.Tensor) -> torch.Tensor:
    """Left-right symmetry for proprioception (235 dims, history=5, 47/frame)."""
    obs = obs.clone()
    B = obs.shape[0]
    n_steps = obs.shape[1] // PROPRIO_DIMS_PER_FRAME
    data = obs.view(B, n_steps, PROPRIO_DIMS_PER_FRAME)  # (B, H, 47)

    # phase: dims 0-1  [sin→-sin, cos→-cos]
    data[:, :, 0:2] *= -1.0

    # velocity commands: dims 2-4  [vx→vx, vy→-vy, ωz→-ωz]
    data[:, :, 2] *= 1.0
    data[:, :, 3] *= -1.0
    data[:, :, 4] *= -1.0

    # imu ang vel: dims 5-7  [ωx→-ωx, ωy→ωy, ωz→-ωz]
    data[:, :, 5] *= -1.0
    data[:, :, 6] *= 1.0
    data[:, :, 7] *= -1.0

    # projected gravity: dims 8-10  [gx→gx, gy→-gy, gz→gz]
    data[:, :, 8] *= 1.0
    data[:, :, 9] *= -1.0
    data[:, :, 10] *= 1.0

    # joint positions: dims 11-22 (12 values)
    data[:, :, 11:23] = _switch_go2_joints_left_right_vectorized(
        data[:, :, 11:23].reshape(B * n_steps, 12), repeat=1
    ).reshape(B, n_steps, 12)

    # joint velocities: dims 23-34 (12 values)
    data[:, :, 23:35] = _switch_go2_joints_left_right_vectorized(
        data[:, :, 23:35].reshape(B * n_steps, 12), repeat=1
    ).reshape(B, n_steps, 12)

    # last actions: dims 35-46 (12 values)
    data[:, :, 35:47] = _switch_go2_joints_left_right_vectorized(
        data[:, :, 35:47].reshape(B * n_steps, 12), repeat=1
    ).reshape(B, n_steps, 12)

    return data.reshape(obs.shape)


# ============================================================================================
# MapScans: 187 dims (11×17 grid, flip rows=y for left-right mirroring)
#
# GridPatternCfg(size=(1.6, 1.0), resolution=0.1, ordering="xy"):
#   meshgrid(x(17), y(11), "xy") → (11 rows=y, 17 cols=x)
#   Flatten is row-major: 11 rows of 17 columns.
#   Left-right mirror: y→-y → flip rows → flip dim 1 of (batch, 11, 17).
# ============================================================================================

def _transform_mapScans_left_right(obs: torch.Tensor) -> torch.Tensor:
    """Left-right symmetry for mapScans: flip the width (y) dimension = flip rows."""
    obs = obs.clone()
    obs[:, :] = obs[:, :].view(-1, 11, 17).flip(dims=[1]).reshape(obs.shape[0], -1)
    return obs


# ============================================================================================
# Privileged: 18 dims/frame × history=3 = 54 total
#
# Per-frame layout:
#   gait(2)  contact(4)  lin_vel(3)  feet_dist(4)  base_h(1)  foot_h(4)
#   | 0-1 |  |  2-5   |  |  6-8  |  |   9-12   |  |  13  |  | 14-17 |
# ============================================================================================

PRIV_DIMS_PER_FRAME = 18

def _transform_privileged_left_right(obs: torch.Tensor) -> torch.Tensor:
    """Left-right symmetry for privileged (54 dims, history=3, 18/frame)."""
    obs = obs.clone()
    B = obs.shape[0]
    n_steps = obs.shape[1] // PRIV_DIMS_PER_FRAME
    data = obs.view(B, n_steps, PRIV_DIMS_PER_FRAME)  # (B, H, 18)

    # gait_trot_mask: dims 0-1  swap [pair0, pair1] ↔ [pair1, pair0]
    data[:, :, 0:2] = _switch_gait_mask_left_right_vectorized(
        data[:, :, 0:2].reshape(B * n_steps, 2), repeat=1
    ).reshape(B, n_steps, 2)

    # feet_contact_mask: dims 2-5  [FL,FR,RL,RR] → [FR,FL,RR,RL]
    data[:, :, 2:6] = _switch_go2_feets_left_right_vectorized(
        data[:, :, 2:6].reshape(B * n_steps, 4), repeat=1
    ).reshape(B, n_steps, 4)

    # base_lin_vel: dims 6-8  [vx→vx, vy→-vy, vz→vz]
    data[:, :, 6] *= 1.0
    data[:, :, 7] *= -1.0
    data[:, :, 8] *= 1.0

    # feet_distance: dims 9-12  [FL,FR,RL,RR] → [FR,FL,RR,RL]
    data[:, :, 9:13] = _switch_go2_feets_left_right_vectorized(
        data[:, :, 9:13].reshape(B * n_steps, 4), repeat=1
    ).reshape(B, n_steps, 4)

    # base_height: dim 13  unchanged (height is symmetric)

    # foot heights: dims 14-17  [FL,FR,RL,RR] → [FR,FL,RR,RL]
    data[:, :, 14:18] = _switch_go2_feets_left_right_vectorized(
        data[:, :, 14:18].reshape(B * n_steps, 4), repeat=1
    ).reshape(B, n_steps, 4)

    return data.reshape(obs.shape)


# ============================================================================================
# HeadProximity: 32 dims (4 zenith rows × 8 azimuth cols)
#
# head_proximity_pattern: meshgrid(azimuth(8), zenith(4), "xy") → (4, 8) grid.
# Flatten is row-major: 4 rows (zenith) of 8 columns (azimuth).
# Left-right mirror: azimuth φ → -φ → flip columns (dim 2).
#   az = linspace(0, 2π, 8): [0, π/4, π/2, ..., 2π]
#   Simple flip gives [2π, ..., π/2, π/4, 0]. az=0 and az=2π are the same
#   physical direction (forward), so the simple flip is physically correct.
# ============================================================================================

def _transform_headProximity_left_right(obs: torch.Tensor) -> torch.Tensor:
    """Left-right symmetry for headProximity: flip azimuth (width) columns."""
    obs = obs.clone()
    obs[:, :] = obs[:, :].view(-1, 4, 8).flip(dims=[2]).reshape(obs.shape[0], -1)
    return obs


# ============================================================================================
# Actions: 12 dims  swap left-right joints + hip negation
# ============================================================================================

def _transform_actions_left_right(actions: torch.Tensor) -> torch.Tensor:
    """Left-right symmetry for actions (12 dims)."""
    actions = actions.clone()
    actions[:] = _switch_go2_joints_left_right_vectorized(actions[:], repeat=1)
    return actions


# ============================================================================================
# Helper functions
#
# Joint ordering in Isaac Sim for Go2:
#   [FL_hip, FR_hip, RL_hip, RR_hip,
#    FL_thigh, FR_thigh, RL_thigh, RR_thigh,
#    FL_calf, FR_calf, RL_calf, RR_calf]
#
#   FL = left front  →  [0, 4, 8]
#   FR = right front →  [1, 5, 9]
#   RL = left hind   →  [2, 6, 10]
#   RR = right hind  →  [3, 7, 11]
#
# Left-right swap: FL↔FR, RL↔RR → left_indices=[0,4,8,2,6,10], right_indices=[1,5,9,3,7,11]
# Hip joints additionally negate their values (sign flip for hip angle direction)
# ============================================================================================

def _switch_go2_joints_left_right_vectorized(joint_data: torch.Tensor, repeat: int) -> torch.Tensor:
    """Swap left-right joints and negate hip values, vectorized over repeat steps."""
    original_shape = joint_data.shape
    joint_data_reshaped = joint_data.view(*original_shape[:-1], repeat, 12)

    left_indices = torch.tensor([0, 4, 8, 2, 6, 10], device=joint_data.device)
    right_indices = torch.tensor([1, 5, 9, 3, 7, 11], device=joint_data.device)

    joint_data_switched = torch.empty_like(joint_data_reshaped)
    joint_data_switched[..., left_indices] = joint_data_reshaped[..., right_indices]
    joint_data_switched[..., right_indices] = joint_data_reshaped[..., left_indices]

    # Hip joints (indices 0,1,2,3) negate under left-right mirror
    hip_indices = torch.tensor([0, 1, 2, 3], device=joint_data.device)
    joint_data_switched[..., hip_indices] = joint_data_switched[..., hip_indices] * -1.0

    return joint_data_switched.view(original_shape)


def _switch_go2_feets_left_right_vectorized(feet_data: torch.Tensor, repeat: int) -> torch.Tensor:
    """Swap left-right feet in per-foot data, vectorized over repeat steps."""
    original_shape = feet_data.shape
    feet_data_reshaped = feet_data.view(*original_shape[:-1], repeat, 4)

    left_indices = torch.tensor([0, 2], device=feet_data.device)
    right_indices = torch.tensor([1, 3], device=feet_data.device)

    feet_data_switched = torch.empty_like(feet_data_reshaped)
    feet_data_switched[..., left_indices] = feet_data_reshaped[..., right_indices]
    feet_data_switched[..., right_indices] = feet_data_reshaped[..., left_indices]

    return feet_data_switched.view(original_shape)


def _switch_gait_mask_left_right_vectorized(gait_data: torch.Tensor, repeat: int) -> torch.Tensor:
    """Swap diagonal pair masks [pair0, pair1] ↔ [pair1, pair0], vectorized over repeat steps.

    In a trot gait, pair 0 (FL+RR) and pair 1 (FR+RL) swap roles under left-right mirroring.
    """
    original_shape = gait_data.shape
    gait_data_reshaped = gait_data.view(*original_shape[:-1], repeat, 2)

    gait_data_switched = torch.empty_like(gait_data_reshaped)
    gait_data_switched[..., 0] = gait_data_reshaped[..., 1]
    gait_data_switched[..., 1] = gait_data_reshaped[..., 0]

    return gait_data_switched.view(original_shape)
