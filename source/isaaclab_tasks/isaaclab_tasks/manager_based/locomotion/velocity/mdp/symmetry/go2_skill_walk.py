# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


"""Functions to specify left-right symmetry for Go2 skill-walk teacher-stage observations.

This module handles the 3 observation groups used by ``go2_loco_skill_walk_cfg.py``:
    - proprioception (history=5, 47 dims/frame × 5 = 235 total)
    - mapScans (history=1, 187 total)
    - privileged (history=3, 18 dims/frame × 3 = 54 total)

Observation layout is identical to ``go2_mid360_teacher_walk.py`` (term-major):
    every observation term has its own history buffer (oldest -> newest), is flattened
    to (H * d), and the term blocks are concatenated in term-definition order.
    Transformations operate directly on the term-major flat layout (NOT per-frame).
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
# Proprioception: 235 total, term-major (history=5)
#   term blocks: phase(10) vel_cmd(15) imu(15) grav(15) jpos(60) jvel(60) act(60)
#
# Phase negation:
#   In trot gait, left-right mirror advances phase by half a cycle:
#     φ' = (φ + 0.5) mod 1.0
#     sin(2π·φ') = -sin(2πφ),  cos(2π·φ') = -cos(2πφ)
# ============================================================================================

def _transform_proprioception_left_right(obs: torch.Tensor) -> torch.Tensor:
    """Left-right symmetry for proprioception (235 dims, term-major, history=5)."""
    obs = obs.clone()
    B = obs.shape[0]
    if obs.shape[1] != 235:
        raise ValueError(f"Expected 235-dim proprioception, got {obs.shape[1]}")

    # phase block [0:10] (2/frame x 5): sin,cos both negated (half-cycle advance)
    obs[:, 0:10] *= -1.0

    # velocity_commands block [10:25] (3/frame x 5): [vx, vy, wz] -> vy, wz negated
    obs[:, 11:25:3] *= -1.0  # vy positions: 11,14,17,20,23
    obs[:, 12:25:3] *= -1.0  # wz positions: 12,15,18,21,24

    # imu_ang_vel block [25:40] (3/frame x 5): [wx, wy, wz] -> wx, wz negated
    obs[:, 25:40:3] *= -1.0  # wx positions: 25,28,31,34,37
    obs[:, 27:40:3] *= -1.0  # wz positions: 27,30,33,36,39

    # projected_gravity block [40:55] (3/frame x 5): [gx, gy, gz] -> gy negated
    obs[:, 41:55:3] *= -1.0  # gy positions: 41,44,47,50,53

    # joint_pos_rel block [55:115] (12/frame x 5): swap left-right joints + hip negation
    obs[:, 55:115] = _switch_go2_joints_left_right_vectorized(
        obs[:, 55:115].reshape(B * 5, 12), repeat=1
    ).reshape(B, 60)

    # joint_vel_rel block [115:175]
    obs[:, 115:175] = _switch_go2_joints_left_right_vectorized(
        obs[:, 115:175].reshape(B * 5, 12), repeat=1
    ).reshape(B, 60)

    # actions block [175:235]
    obs[:, 175:235] = _switch_go2_joints_left_right_vectorized(
        obs[:, 175:235].reshape(B * 5, 12), repeat=1
    ).reshape(B, 60)

    return obs


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
# Privileged: 54 total, term-major (history=3)
#   term blocks: gait(6) contact(12) lin_vel(9) feet_dist(12) base_h(3)
#                FL_foot_h(3) FR_foot_h(3) RL_foot_h(3) RR_foot_h(3)
# ============================================================================================

def _transform_privileged_left_right(obs: torch.Tensor) -> torch.Tensor:
    """Left-right symmetry for privileged (54 dims, term-major, history=3)."""
    obs = obs.clone()
    B = obs.shape[0]
    if obs.shape[1] != 54:
        raise ValueError(f"Expected 54-dim privileged, got {obs.shape[1]}")

    # gait_trot_mask block [0:6] (2/frame x 3): swap [pair0, pair1] per frame
    obs[:, 0:6] = _switch_gait_mask_left_right_vectorized(
        obs[:, 0:6].reshape(B * 3, 2), repeat=1
    ).reshape(B, 6)

    # feet_contact_mask block [6:18] (4/frame x 3): [FL,FR,RL,RR] -> [FR,FL,RR,RL]
    obs[:, 6:18] = _switch_go2_feets_left_right_vectorized(
        obs[:, 6:18].reshape(B * 3, 4), repeat=1
    ).reshape(B, 12)

    # base_lin_vel block [18:27] (3/frame x 3): [vx, vy, vz] -> vy negated
    obs[:, 19:27:3] *= -1.0  # vy positions: 19,22,25

    # feet_distance block [27:39] (4/frame x 3): swap feet
    obs[:, 27:39] = _switch_go2_feets_left_right_vectorized(
        obs[:, 27:39].reshape(B * 3, 4), repeat=1
    ).reshape(B, 12)

    # base_height block [39:42] (3): unchanged (height is symmetric)

    # foot-height blocks: FL[42:45] FR[45:48] RL[48:51] RR[51:54]
    # under left-right mirror: FL<->FR, RL<->RR (swap whole term blocks)
    fl = obs[:, 42:45].clone()
    fr = obs[:, 45:48].clone()
    rl = obs[:, 48:51].clone()
    rr = obs[:, 51:54].clone()
    obs[:, 42:45] = fr
    obs[:, 45:48] = fl
    obs[:, 48:51] = rr
    obs[:, 51:54] = rl

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
