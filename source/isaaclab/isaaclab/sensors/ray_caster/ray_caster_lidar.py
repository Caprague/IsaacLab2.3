# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
from isaaclab.utils.math import quat_apply, yaw_quat

from .multi_mesh_ray_caster import MultiMeshRayCaster
from .ray_caster_box_data import RayCasterBoxData

if TYPE_CHECKING:
    from .ray_caster_lidar_cfg import RayCasterLidarCfg

from .data_collector import SimulationDataSaver


class RayCasterLidar(MultiMeshRayCaster):
    """A ray-casting LiDAR sensor with multi-mesh support.

    Extends :class:`MultiMeshRayCaster` with LiDAR-specific features:

    - Dynamic scan pattern updating (e.g., for Livox Mid-360)
    - Bounding box clipping and normalization of point cloud data
    - Local frame transformation (world / yaw / base)
    - Point cloud data collection and saving
    - Yaw inversion for upside-down mounted LiDARs
    """

    cfg: RayCasterLidarCfg
    """The configuration parameters."""

    def __init__(self, cfg: RayCasterLidarCfg):
        """Initializes the ray-caster LiDAR object.

        Args:
            cfg: The configuration parameters.
        """
        super().__init__(cfg)
        # override data container with LiDAR-specific one
        self._data = RayCasterBoxData()

        # check the data box clip params setting
        if self.cfg.data_box_clip is not None:
            for i, val in enumerate(self.cfg.data_box_clip):
                assert isinstance(val, (int, float)) and val > 0, (
                    f"Element {i} of data_box_clip must be a positive number, got {val}"
                )

        # check the data collection mode flag
        if cfg.data_collection:
            assert cfg.data_save_path is not None, "Must set data_save_path while data_collection is True !!!"
            self.pc_data_saver = SimulationDataSaver(
                save_path_root=cfg.data_save_path,
                data_type=cfg.pc_data_saver_cfg.data_type,
                sub_dir_name=cfg.pc_data_saver_cfg.sub_dir_name,
                max_sequence=cfg.pc_data_saver_cfg.max_sequence,
                T_max=cfg.pc_data_saver_cfg.T_max,
            )
            self.pose_data_saver = SimulationDataSaver(
                save_path_root=cfg.data_save_path,
                data_type=cfg.pose_data_saver_cfg.data_type,
                sub_dir_name=cfg.pose_data_saver_cfg.sub_dir_name,
                max_sequence=cfg.pose_data_saver_cfg.max_sequence,
                T_max=cfg.pose_data_saver_cfg.T_max,
            )

    def __str__(self) -> str:
        """Returns: A string containing information about the instance."""
        return (
            f"Ray-caster LiDAR @ '{self.cfg.prim_path}': \n"
            f"\tview type            : {self._view.__class__}\n"
            f"\tupdate period (s)    : {self.cfg.update_period}\n"
            f"\tnumber of meshes     : {self._num_envs} x {sum(self._num_meshes_per_env.values())} \n"
            f"\tnumber of sensors    : {self._view.count}\n"
            f"\tnumber of rays/sensor: {self.num_rays}\n"
            f"\ttotal number of rays : {self.num_rays * self._view.count}"
        )

    """
    Properties
    """

    @property
    def data(self) -> RayCasterBoxData:
        # update sensors if needed
        self._update_outdated_buffers()
        # return the data
        return self._data

    @property
    def frame(self) -> torch.Tensor:
        """Frame number when the measurement took place."""
        return self._frame

    """
    Operations.
    """

    def reset(self, env_ids: Sequence[int] | None = None):
        super().reset(env_ids)
        # resolve None
        if env_ids is None:
            env_ids = slice(None)
        # Reset the frame count
        self._frame[env_ids] = 0
        # data collection mode
        if self.cfg.data_collection:
            self.pc_data_saver.reset_input_counter(env_ids)
            self.pose_data_saver.reset_input_counter(env_ids)

    """
    Implementation.
    """

    def _apply_angle_noise_to_directions(self, directions: torch.Tensor) -> torch.Tensor:
        """对射线方向向量添加角度噪声。

        Args:
            directions: 射线方向向量，形状 (R, 3) 或 (N, R, 3)

        Returns:
            添加噪声后的归一化射线方向向量
        """
        x, y, z = directions[..., 0], directions[..., 1], directions[..., 2]
        theta = torch.atan2(y, x)
        phi = torch.acos(torch.clamp(z, min=-1.0, max=1.0))

        std_rad = torch.deg2rad(torch.tensor(self.cfg.noise_cfg.angle_noise_std_deg, device=self._device))
        theta_noisy = theta + torch.randn_like(theta) * std_rad
        phi_noisy = phi + torch.randn_like(phi) * std_rad

        x_noisy = torch.cos(theta_noisy) * torch.sin(phi_noisy)
        y_noisy = torch.sin(theta_noisy) * torch.sin(phi_noisy)
        z_noisy = torch.cos(phi_noisy)

        return torch.stack([x_noisy, y_noisy, z_noisy], dim=-1)

    def _initialize_impl(self):
        super()._initialize_impl()

        # initialize frame counter
        self._frame = torch.zeros(self._num_envs, device=self.device)
        
        # initialize LiDAR-specific data buffers
        self._data.sensor_pos_w = torch.zeros(self._num_envs, 3, device=self.device)
        self._data.ray_hits_b = torch.zeros(self._num_envs, self.num_rays, 3, device=self.device)
        self._data.ray_hits_mask = torch.zeros(self._num_envs, self.num_rays, dtype=torch.bool, device=self.device)

        if self.cfg.data_collection:
            self.pc_data_saver.set_num_envs(self._view.count)
            self.pose_data_saver.set_num_envs(self._view.count)

    def _initialize_rays_impl(self):
        super()._initialize_rays_impl()
        # re-initialize the data container to LiDAR-specific type after parent call
        # note: parent sets _data to MultiMeshRayCasterData, we need to keep RayCasterBoxData
        ray_hits_w = self._data.ray_hits_w
        pos_w = self._data.pos_w
        quat_w = self._data.quat_w
        self._data = RayCasterBoxData()
        self._data.pos_w = pos_w
        self._data.quat_w = quat_w
        self._data.ray_hits_w = ray_hits_w
        self._data.sensor_pos_w = torch.zeros(self._num_envs, 3, device=self.device)
        self._data.ray_hits_b = torch.zeros(self._num_envs, self.num_rays, 3, device=self.device)
        self._data.ray_hits_mask = torch.zeros(self._num_envs, self.num_rays, dtype=torch.bool, device=self.device)
        self._data.frame_id = torch.zeros(self._num_envs, device=self.device)
        if self.cfg.return_distance:
            self._data.ray_distance = torch.zeros(self._num_envs, self.num_rays, device=self.device)

        self._original_ray_directions = self.ray_directions.clone()

    def _update_buffers_impl(self, env_ids: Sequence[int]):
        """Fills the buffers of the sensor data."""
        # increment frame count
        self._frame[env_ids] += 1
        # Update ray pattern if dynamic pattern is enabled
        if self.cfg.dynamic_pattern:
            new_ray_starts, new_ray_directions = self.cfg.pattern_cfg.func(self.cfg.pattern_cfg, self._device)
            offset_pos = torch.tensor(list(self.cfg.offset.pos), device=self._device)
            offset_quat = torch.tensor(list(self.cfg.offset.rot), device=self._device)
            new_ray_directions = quat_apply(offset_quat.repeat(len(new_ray_directions), 1), new_ray_directions)
            new_ray_starts += offset_pos

            self.ray_starts[:, : len(new_ray_starts)] = new_ray_starts.unsqueeze(0).repeat(self._num_envs, 1, 1)
            self.ray_directions[:, : len(new_ray_directions)] = new_ray_directions.unsqueeze(0).repeat(
                self._num_envs, 1, 1
            )

            if self.cfg.noise_cfg.enable_angle_noise:
                self.ray_directions[env_ids, : len(new_ray_directions)] = self._apply_angle_noise_to_directions(
                    self.ray_directions[env_ids, : len(new_ray_directions)]
                )
        else:
            if self.cfg.noise_cfg.enable_angle_noise:
                self.ray_directions[env_ids] = self._apply_angle_noise_to_directions(
                    self._original_ray_directions[env_ids]
                )

        # --- call parent to do multi-mesh ray casting ---
        super()._update_buffers_impl(env_ids)

        # --- LiDAR-specific post-processing ---

        current_pos_w = self._data.pos_w[env_ids]  # (N, 3)
        current_quat_w = self._data.quat_w[env_ids]  # (N, 4) - wxyz
        offset_pos = torch.tensor(list(self.cfg.offset.pos), device=self._device)  # (3,)
        offset_quat = torch.tensor(list(self.cfg.offset.rot), device=self._device)  # (4,)
        offset_pos_w = quat_apply(current_quat_w, offset_pos.unsqueeze(0).repeat(len(env_ids), 1))  # (N, 3)

        # sensor world position (base + offset)
        sensor_pos_w = current_pos_w + offset_pos_w  # (N, 3)
        self._data.sensor_pos_w[env_ids] = sensor_pos_w

        # --- apply measurement noise ---
        noise_cfg = self.cfg.noise_cfg
        if noise_cfg.enable_range_noise:
            ray_hits_w = self._data.ray_hits_w[env_ids]  # (N, R, 3)
            distances = self._data.ray_distance[env_ids]  # (N, R) - use pre-computed distance
            valid_mask = ~torch.isinf(distances)  # (N, R)

            noise_std = noise_cfg.range_noise_std_base + noise_cfg.range_noise_std_factor * distances
            noise = torch.randn_like(distances) * noise_std
            noisy_distances = distances + noise
            noisy_distances = torch.max(noisy_distances, torch.zeros_like(noisy_distances))
            directions = (ray_hits_w - sensor_pos_w.unsqueeze(1)) / distances.unsqueeze(-1)
            noisy_hits = sensor_pos_w.unsqueeze(1) + directions * noisy_distances.unsqueeze(-1)
            ray_hits_w = torch.where(valid_mask.unsqueeze(-1), noisy_hits, ray_hits_w)

            self._data.ray_hits_w[env_ids] = ray_hits_w
        # sensor world orientation (base rotation * offset rotation)
        sensor_quat_w = math_utils.quat_mul(
            current_quat_w, offset_quat.unsqueeze(0).repeat(len(env_ids), 1)
        )  # (N, 4)

        if self.cfg.ray_alignment == "world":
            local_hits = self._data.ray_hits_w[env_ids] - self._data.sensor_pos_w[env_ids].unsqueeze(1)
        elif self.cfg.ray_alignment == "yaw":
            local_hits = self._data.ray_hits_w[env_ids] - self._data.sensor_pos_w[env_ids].unsqueeze(1)
            sensor_yaw_quat = yaw_quat(sensor_quat_w)  # (N, 4)
            if self.cfg.yaw_inv:
                yaw_flip = torch.zeros_like(sensor_yaw_quat)
                yaw_flip[:, 3] = 1.0  # z=1 → 180° yaw rotation
                sensor_yaw_quat = math_utils.quat_mul(sensor_yaw_quat, yaw_flip)
            sensor_yaw_quat_inv = math_utils.quat_inv(sensor_yaw_quat)  # (N, 4)
            local_hits = quat_apply(
                sensor_yaw_quat_inv.repeat(1, self.num_rays).view(-1, 4),
                local_hits.view(-1, 3),
            ).view(local_hits.shape)
        elif self.cfg.ray_alignment == "base":
            local_hits = self._data.ray_hits_w[env_ids] - self._data.sensor_pos_w[env_ids].unsqueeze(1)
            sensor_quat_inv = math_utils.quat_inv(sensor_quat_w)  # (N, 4)
            if self.cfg.yaw_inv:
                sensor_quat_inv = math_utils.quat_mul(
                    torch.tensor([0.0, 0.0, 0.0, 1.0], dtype=torch.float32, device=local_hits.device).repeat(len(env_ids), 1), 
                    sensor_quat_inv,
                )  # (N, 4)
            local_hits = quat_apply(
                sensor_quat_inv.repeat(1, self.num_rays).view(-1, 4),
                local_hits.view(-1, 3),
            ).view(local_hits.shape)
        else:
            raise RuntimeError(
                f"Unsupported ray_alignment type: {self.cfg.ray_alignment}. Cannot transform to local frame."
            )

        # bounding box clipping
        if self.cfg.data_box_clip is not None:
            clip_bounds = torch.tensor(self.cfg.data_box_clip, dtype=local_hits.dtype, device=local_hits.device)
            half_clip_bounds = clip_bounds / 2.0
            clip_mask = torch.all(torch.abs(local_hits) < half_clip_bounds, dim=-1)  # (N, B)

            hit_mask = ~torch.any(torch.isinf(self._data.ray_hits_w[env_ids]), dim=-1)  # (N, B)
            final_mask = hit_mask & clip_mask  # (N, B)

            if self.cfg.data_normalization:
                normalized_local_hits = local_hits / clip_bounds  # (N, B, 3)
                local_hits = normalized_local_hits
        else:
            final_mask = ~torch.any(torch.isinf(self._data.ray_hits_w[env_ids]), dim=-1)  # (N, B)

        self._data.ray_hits_b[env_ids] = local_hits
        self._data.ray_hits_mask[env_ids] = final_mask

        # data collection mode
        if self.cfg.data_collection:
            self.pc_data_saver.save_data(
                env_ids, self._data.ray_hits_b[env_ids], self._data.ray_hits_mask[env_ids]
            )
            self.pose_data_saver.save_data(
                env_ids, self._data.sensor_pos_w[env_ids], self._data.quat_w[env_ids]
            )
        
        # update frame_id in data container
        self._data.frame_id = self._frame

    def _debug_vis_callback(self, event):
        viz_points = self._data.ray_hits_w[self._data.ray_hits_mask].view(-1, 3)
        viz_points = viz_points[~torch.any(torch.isinf(viz_points), dim=1)]
        if viz_points is not None and viz_points.shape[0] >= 1:
            self.ray_visualizer.visualize(viz_points)
