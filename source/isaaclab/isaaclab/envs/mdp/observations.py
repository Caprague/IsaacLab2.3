# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to create observation terms.

The functions can be passed to the :class:`isaaclab.managers.ObservationTermCfg` object to enable
the observation introduced by the function.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers.manager_base import ManagerTermBase
from isaaclab.managers.manager_term_cfg import ObservationTermCfg
from isaaclab.sensors import Camera, Imu, RayCaster, RayCasterCamera, TiledCamera

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv

from isaaclab.envs.utils.io_descriptors import (
    generic_io_descriptor,
    record_body_names,
    record_dtype,
    record_joint_names,
    record_joint_pos_offsets,
    record_joint_vel_offsets,
    record_shape,
)


"""
Cycle Time Phase.
"""
def cycle_phase(env: ManagerBasedRLEnv, cycle_period: torch.Tensor) -> torch.Tensor:
    """Terminate the episode when the episode length exceeds the maximum episode length."""   
    if not hasattr(env, "episode_length_buf"):
        return torch.zeros((env.num_envs, 2), device=env.device).unsqueeze(-1)

    phase = (env.episode_length_buf * env.step_dt) % cycle_period / cycle_period
    sin_phase = torch.sin(2 * torch.pi * phase).unsqueeze(-1)
    cos_phase = torch.cos(2 * torch.pi * phase).unsqueeze(-1)
    return torch.cat((sin_phase, cos_phase), dim=1)


def cycle_phase_vel(env: ManagerBasedRLEnv, cycle_period: tuple[torch.Tensor, torch.Tensor], velocity_limit: tuple[torch.Tensor, torch.Tensor], command_name: str) -> torch.Tensor:
    """Terminate the episode when the episode length exceeds the maximum episode length."""   
    if not hasattr(env, "episode_length_buf"):
        return torch.zeros((env.num_envs, 2), device=env.device).unsqueeze(-1)

    vel_cmd = torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1).clamp(min=velocity_limit[0], max=velocity_limit[1])
    cyc_val = cycle_period[1]*torch.ones_like(vel_cmd) - (vel_cmd - velocity_limit[0])*(cycle_period[1] - cycle_period[0])/(velocity_limit[1] - velocity_limit[0])
    phase = (env.episode_length_buf * env.step_dt) % cyc_val / cyc_val
    sin_phase = torch.sin(2 * torch.pi * phase).unsqueeze(-1)
    cos_phase = torch.cos(2 * torch.pi * phase).unsqueeze(-1)

    # print(f"vel_cmd: {vel_cmd}")
    # print(f"cyc_val: {cyc_val}")
    # print(f"phase: {phase}")

    return torch.cat((sin_phase.abs(), cos_phase.abs()), dim=1)


"""
Gait Phase Mask
"""
def gait_trot_mask(env: ManagerBasedEnv, cycle_period: torch.Tensor) -> torch.Tensor:
    """Gait phase mask.

    Returns:
        A tensor of shape (num_envs, 2) with the gait phase mask.
    """
    if not hasattr(env, "episode_length_buf"):
        return torch.zeros((env.num_envs, 2), device=env.device).unsqueeze(-1)

    # return float mask 1 is stance, 0 is swing
    phase = (env.episode_length_buf * env.step_dt) % cycle_period / cycle_period
    stance_mask = torch.zeros((env.num_envs, 2), device=env.device)
    stance_mask[:, 0] = (phase < 0.5).float()
    stance_mask[:, 1] = (phase > 0.5).float()
    return stance_mask


def gait_trot_mask_vel(env: ManagerBasedEnv, 
                       cycle_period: tuple[torch.Tensor, torch.Tensor], 
                       velocity_limit: tuple[torch.Tensor, torch.Tensor], 
                       command_name: str,
                       binary_threshold: float = 0.5) -> torch.Tensor:
    """Gait phase mask.

    Returns:
        A tensor of shape (num_envs, 2) with the gait phase mask.
    """
    if not hasattr(env, "episode_length_buf"):
        return torch.zeros((env.num_envs, 2), device=env.device).unsqueeze(-1)

    # return float mask 1 is stance, 0 is swing
    vel_cmd = torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1).clamp(min=velocity_limit[0], max=velocity_limit[1])
    cyc_val = cycle_period[1]*torch.ones_like(vel_cmd) - (vel_cmd - velocity_limit[0])*(cycle_period[1] - cycle_period[0])/(velocity_limit[1] - velocity_limit[0])
    phase = (env.episode_length_buf * env.step_dt) % cyc_val / cyc_val
    sin_phase_abs = torch.sin(2 * torch.pi * phase).abs()
    cos_phase_abs = torch.cos(2 * torch.pi * phase).abs()
    stance_mask = torch.zeros((env.num_envs, 2), device=env.device)
    stance_mask[:, 0] = (cos_phase_abs > binary_threshold).float()
    stance_mask[:, 1] = (sin_phase_abs > binary_threshold).float()

    # print(f"stance_mask: {stance_mask}")

    return stance_mask


def feet_contact_mask(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, threshold: float = 5.0) -> torch.Tensor:
    """Contact mask for the feet.

    Returns:
        A tensor of shape (num_envs, 4) with the contact mask for the feet.
    """
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the contact mask
    return (contact_sensor.data.net_forces_w_history[:, 0, sensor_cfg.body_ids, 2] > threshold).float()


def feet_distance(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """计算左右同侧足与前后同侧足间的距离惩罚"""
    asset: RigidObject = env.scene[asset_cfg.name]
    
    # 获取四个足端位置 (假设顺序为: 左前, 右前, 左后, 右后)
    feet_pos = asset.data.body_pos_w[:, -4:, :]
    
    # 计算左右同侧足距离 (左前-左后, 右前-右后)
    left_pair_dist = torch.norm(feet_pos[:, 0] - feet_pos[:, 2], dim=1)
    right_pair_dist = torch.norm(feet_pos[:, 1] - feet_pos[:, 3], dim=1)
    
    # 计算前后同侧足距离 (左前-右前, 左后-右后)
    front_pair_dist = torch.norm(feet_pos[:, 0] - feet_pos[:, 1], dim=1)
    rear_pair_dist = torch.norm(feet_pos[:, 2] - feet_pos[:, 3], dim=1)

    return torch.cat((left_pair_dist.unsqueeze(-1), 
                      right_pair_dist.unsqueeze(-1), 
                      front_pair_dist.unsqueeze(-1), 
                      rear_pair_dist.unsqueeze(-1)), 
                      dim=1)


"""
Root state.
"""


@generic_io_descriptor(units="m", axes=["Z"], observation_type="RootState", on_inspect=[record_shape, record_dtype])
def base_pos_z(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Root height in the simulation world frame."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.root_pos_w[:, 2].unsqueeze(-1)


@generic_io_descriptor(
    units="m/s", axes=["X", "Y", "Z"], observation_type="RootState", on_inspect=[record_shape, record_dtype]
)
def base_lin_vel(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Root linear velocity in the asset's root frame."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    return asset.data.root_lin_vel_b


@generic_io_descriptor(
    units="rad/s", axes=["X", "Y", "Z"], observation_type="RootState", on_inspect=[record_shape, record_dtype]
)
def base_ang_vel(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Root angular velocity in the asset's root frame."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    return asset.data.root_ang_vel_b


@generic_io_descriptor(
    units="m/s^2", axes=["X", "Y", "Z"], observation_type="RootState", on_inspect=[record_shape, record_dtype]
)
def projected_gravity(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Gravity projection on the asset's root frame."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    return asset.data.projected_gravity_b


@generic_io_descriptor(
    units="m", axes=["X", "Y", "Z"], observation_type="RootState", on_inspect=[record_shape, record_dtype]
)
def root_pos_w(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Asset root position in the environment frame."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    return asset.data.root_pos_w - env.scene.env_origins


@generic_io_descriptor(
    units="unit", axes=["W", "X", "Y", "Z"], observation_type="RootState", on_inspect=[record_shape, record_dtype]
)
def root_quat_w(
    env: ManagerBasedEnv, make_quat_unique: bool = False, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Asset root orientation (w, x, y, z) in the environment frame.

    If :attr:`make_quat_unique` is True, then returned quaternion is made unique by ensuring
    the quaternion has non-negative real component. This is because both ``q`` and ``-q`` represent
    the same orientation.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]

    quat = asset.data.root_quat_w
    # make the quaternion real-part positive if configured
    return math_utils.quat_unique(quat) if make_quat_unique else quat


@generic_io_descriptor(
    units="m/s", axes=["X", "Y", "Z"], observation_type="RootState", on_inspect=[record_shape, record_dtype]
)
def root_lin_vel_w(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Asset root linear velocity in the environment frame."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    return asset.data.root_lin_vel_w


@generic_io_descriptor(
    units="rad/s", axes=["X", "Y", "Z"], observation_type="RootState", on_inspect=[record_shape, record_dtype]
)
def root_ang_vel_w(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Asset root angular velocity in the environment frame."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    return asset.data.root_ang_vel_w


"""
Body state
"""


@generic_io_descriptor(observation_type="BodyState", on_inspect=[record_shape, record_dtype, record_body_names])
def body_pose_w(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """The flattened body poses of the asset w.r.t the env.scene.origin.

    Note: Only the bodies configured in :attr:`asset_cfg.body_ids` will have their poses returned.

    Args:
        env: The environment.
        asset_cfg: The SceneEntity associated with this observation.

    Returns:
        The poses of bodies in articulation [num_env, 7 * num_bodies]. Pose order is [x,y,z,qw,qx,qy,qz].
        Output is stacked horizontally per body.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]

    # access the body poses in world frame
    pose = asset.data.body_pose_w[:, asset_cfg.body_ids, :7]
    if isinstance(asset_cfg.body_ids, (slice, int)):
        pose = pose.clone()  # if slice or int, make a copy to avoid modifying original data
    pose[..., :3] = pose[..., :3] - env.scene.env_origins.unsqueeze(1)
    return pose.reshape(env.num_envs, -1)


@generic_io_descriptor(observation_type="BodyState", on_inspect=[record_shape, record_dtype, record_body_names])
def body_projected_gravity_b(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """The direction of gravity projected on to bodies of an Articulation.

    Note: Only the bodies configured in :attr:`asset_cfg.body_ids` will have their poses returned.

    Args:
        env: The environment.
        asset_cfg: The Articulation associated with this observation.

    Returns:
        The unit vector direction of gravity projected onto body_name's frame. Gravity projection vector order is
        [x,y,z]. Output is stacked horizontally per body.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]

    body_quat = asset.data.body_quat_w[:, asset_cfg.body_ids]
    gravity_dir = asset.data.GRAVITY_VEC_W.unsqueeze(1)
    return math_utils.quat_apply_inverse(body_quat, gravity_dir).view(env.num_envs, -1)


"""
Joint state.
"""


@generic_io_descriptor(
    observation_type="JointState", on_inspect=[record_joint_names, record_dtype, record_shape], units="rad"
)
def joint_pos(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """The joint positions of the asset.

    Note: Only the joints configured in :attr:`asset_cfg.joint_ids` will have their positions returned.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.joint_pos[:, asset_cfg.joint_ids]


@generic_io_descriptor(
    observation_type="JointState",
    on_inspect=[record_joint_names, record_dtype, record_shape, record_joint_pos_offsets],
    units="rad",
)
def joint_pos_rel(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """The joint positions of the asset w.r.t. the default joint positions.

    Note: Only the joints configured in :attr:`asset_cfg.joint_ids` will have their positions returned.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]


@generic_io_descriptor(observation_type="JointState", on_inspect=[record_joint_names, record_dtype, record_shape])
def joint_pos_limit_normalized(
    env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """The joint positions of the asset normalized with the asset's joint limits.

    Note: Only the joints configured in :attr:`asset_cfg.joint_ids` will have their normalized positions returned.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    return math_utils.scale_transform(
        asset.data.joint_pos[:, asset_cfg.joint_ids],
        asset.data.soft_joint_pos_limits[:, asset_cfg.joint_ids, 0],
        asset.data.soft_joint_pos_limits[:, asset_cfg.joint_ids, 1],
    )


@generic_io_descriptor(
    observation_type="JointState", on_inspect=[record_joint_names, record_dtype, record_shape], units="rad/s"
)
def joint_vel(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    """The joint velocities of the asset.

    Note: Only the joints configured in :attr:`asset_cfg.joint_ids` will have their velocities returned.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.joint_vel[:, asset_cfg.joint_ids]


@generic_io_descriptor(
    observation_type="JointState",
    on_inspect=[record_joint_names, record_dtype, record_shape, record_joint_vel_offsets],
    units="rad/s",
)
def joint_vel_rel(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    """The joint velocities of the asset w.r.t. the default joint velocities.

    Note: Only the joints configured in :attr:`asset_cfg.joint_ids` will have their velocities returned.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.joint_vel[:, asset_cfg.joint_ids] - asset.data.default_joint_vel[:, asset_cfg.joint_ids]


@generic_io_descriptor(
    observation_type="JointState", on_inspect=[record_joint_names, record_dtype, record_shape], units="N.m"
)
def joint_effort(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """The joint applied effort of the robot.

    NOTE: Only the joints configured in :attr:`asset_cfg.joint_ids` will have their effort returned.

    Args:
        env: The environment.
        asset_cfg: The SceneEntity associated with this observation.

    Returns:
        The joint effort (N or N-m) for joint_names in asset_cfg, shape is [num_env,num_joints].
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.applied_torque[:, asset_cfg.joint_ids]


"""
Sensors.
"""


def height_scan(env: ManagerBasedEnv, sensor_cfg: SceneEntityCfg, offset: float = 0.5) -> torch.Tensor:
    """Height scan from the given sensor w.r.t. the sensor's frame.

    The provided offset (Defaults to 0.5) is subtracted from the returned values.
    """
    # extract the used quantities (to enable type-hinting)
    sensor: RayCaster = env.scene.sensors[sensor_cfg.name]
    # height scan: height = sensor_height - hit_point_z - offset
    return sensor.data.pos_w[:, 2].unsqueeze(1) - sensor.data.ray_hits_w[..., 2] - offset

    # height = sensor.data.pos_w[:, 2].unsqueeze(1) - sensor.data.ray_hits_w[..., 2] - offset
    # print("height scan:", height)
    # return height


def height_scan_delay(
    env: ManagerBasedEnv,
    sensor_cfg: SceneEntityCfg,
    offset: float = 0.5,
    delay_range: tuple[int, int] = (0, 5),
    resample_interval_range: tuple[float, float] = (0.5, 10.0),
    dt: float = 0.02,
    downsample_freq_scale: int = 2,
) -> torch.Tensor:
    """Height scan with communication delay simulation.

    The function simulates communication delay by:
    1. Maintaining a history buffer of height observations
    2. Randomly sampling delay times for each environment
    3. Returning delayed observations from the buffer

    Args:
        env: The environment instance.
        sensor_cfg: The sensor configuration.
        offset: Height offset to subtract (Defaults to 0.5).
        delay_range: Range of delay steps (min, max) for random sampling.
        resample_interval_range: Range of time intervals (in seconds) for resampling delay values.
        dt: Time step duration in seconds.
        downsample_freq_scale: Downsampling factor for buffer updates. For example, 2 means
            update buffer every 2 steps (step 0: new data, step 1: cached data, step 2: new data...).
            Uses current_episode_length to determine the sampling timing.

    Returns:
        The height scan observations with random delay applied.
    """

    # Skip initialization if episode_length_buf doesn't exist (e.g., during setup)
    if not hasattr(env, "episode_length_buf"):
        sensor: RayCaster = env.scene.sensors[sensor_cfg.name]
        current_height = sensor.data.pos_w[:, 2].unsqueeze(1) - sensor.data.ray_hits_w[..., 2] - offset
        return current_height

    # Get current height scan observations
    sensor: RayCaster = env.scene.sensors[sensor_cfg.name]
    current_height = sensor.data.pos_w[:, 2].unsqueeze(1) - sensor.data.ray_hits_w[..., 2] - offset

    # Initialize buffer and state if not exists
    buffer_key = "_height_scan_delay_buffer"
    delays_key = "_height_scan_delay_delays"
    time_since_key = "_height_scan_delay_time_since"
    next_resample_key = "_height_scan_delay_next_resample"
    is_initialized_key = "_height_scan_delay_initialized"
    last_observation_key = "_height_scan_delay_last_observation"

    if not hasattr(env, buffer_key):
        max_delay = delay_range[1]
        setattr(env, buffer_key, torch.zeros((max_delay, env.num_envs, current_height.shape[1]), device=env.device))
        setattr(env, delays_key, torch.zeros(env.num_envs, dtype=torch.int, device=env.device))
        setattr(env, time_since_key, torch.zeros(env.num_envs, device=env.device))
        setattr(env, is_initialized_key, torch.zeros(env.num_envs, dtype=torch.bool, device=env.device))
        setattr(env, last_observation_key, current_height.clone())

        random_intervals = torch.rand(env.num_envs, device=env.device)
        random_intervals = random_intervals * (resample_interval_range[1] - resample_interval_range[0])
        random_intervals = random_intervals + resample_interval_range[0]
        setattr(env, next_resample_key, random_intervals)

        random_delays = torch.randint(delay_range[0], delay_range[1], (env.num_envs,), device=env.device, dtype=torch.int)
        getattr(env, delays_key).copy_(random_delays)

    buffer = getattr(env, buffer_key)
    delays = getattr(env, delays_key)
    time_since = getattr(env, time_since_key)
    next_resample = getattr(env, next_resample_key)
    is_initialized = getattr(env, is_initialized_key)
    last_observation = getattr(env, last_observation_key)

    # Detect reset: episode length is 0 (reset just happened)
    current_episode_length = env.episode_length_buf
    reset_mask = current_episode_length == 0

    if torch.any(reset_mask):
        reset_ids = torch.where(reset_mask)[0]
        is_initialized[reset_ids] = False

    # Update time tracking
    time_since += dt

    # Check which environments need resampling (either time-based or reset-triggered)
    should_resample = (time_since >= next_resample) | reset_mask

    if torch.any(should_resample):
        env_ids = torch.where(should_resample)[0]
        random_intervals = torch.rand(len(env_ids), device=env.device)
        random_intervals = random_intervals * (resample_interval_range[1] - resample_interval_range[0])
        random_intervals = random_intervals + resample_interval_range[0]
        next_resample[env_ids] = random_intervals
        time_since[env_ids] = 0.0

        random_delays = torch.randint(delay_range[0], delay_range[1], (len(env_ids),), device=env.device, dtype=delays.dtype)
        delays[env_ids] = random_delays

    # Initialize buffer for new/reset environments (vectorized)
    if not torch.all(is_initialized):
        uninitialized_ids = torch.where(~is_initialized)[0]
        if len(uninitialized_ids) > 0:
            buffer[:, uninitialized_ids] = current_height[uninitialized_ids].unsqueeze(0).expand(buffer.shape[0], -1, -1)
            is_initialized[uninitialized_ids] = True
            last_observation[uninitialized_ids] = current_height[uninitialized_ids].clone()

    # Determine which environments should use new observation vs cached observation (downsampling)
    # Use current_episode_length as the step counter (it tracks total steps in current episode)
    should_use_new_obs = (current_episode_length % downsample_freq_scale) == 0

    # Update last observation for environments using new data
    if torch.any(should_use_new_obs):
        new_obs_ids = torch.where(should_use_new_obs)[0]
        last_observation[new_obs_ids] = current_height[new_obs_ids].clone()

    # Use cached observation for downsampling, new observation otherwise
    observation_to_store = torch.where(should_use_new_obs.unsqueeze(1), current_height, last_observation)

    # Shift buffer and add observation (vectorized)
    buffer = torch.roll(buffer, shifts=1, dims=0)
    buffer[0] = observation_to_store
    setattr(env, buffer_key, buffer)
    setattr(env, last_observation_key, last_observation)

    # Get delayed observations (vectorized)
    result = current_height.clone()
    delay_mask = delays > 0

    if torch.any(delay_mask):
        delay_ids = torch.where(delay_mask)[0]
        result[delay_ids] = buffer[delays[delay_ids], delay_ids]

    return result


def height_scan_3d(env: ManagerBasedEnv, sensor_cfg: SceneEntityCfg, offset: float = 0.0) -> torch.Tensor:
    """Height scan from the given sensor w.r.t. the sensor's frame.

    The provided offset (Defaults to 0.5) is subtracted from the returned values.
    """
    # extract the used quantities (to enable type-hinting)
    sensor: RayCaster = env.scene.sensors[sensor_cfg.name]

    resolution = sensor.cfg.pattern_cfg.resolution
    size = sensor.cfg.pattern_cfg.size
    x = torch.arange(start=-size[0] / 2, end=size[0] / 2 + 1.0e-9, step=resolution, device=env.device)
    y = torch.arange(start=-size[1] / 2, end=size[1] / 2 + 1.0e-9, step=resolution, device=env.device)
    grid_x, grid_y = torch.meshgrid(x, y, indexing="xy")
    height_x = grid_x.reshape(1, -1, 1).repeat(env.num_envs, 1, 1)
    height_y = grid_y.reshape(1, -1, 1).repeat(env.num_envs, 1, 1)

    height_z = (sensor.data.pos_w[:, 2].unsqueeze(1) - sensor.data.ray_hits_w[..., 2] - offset).unsqueeze(-1)
    height_3d = torch.cat((height_x, height_y, height_z), dim=-1).reshape(env.num_envs, -1)

    # print(f"height_x shape: {height_x.shape}")
    # print(f"height_y shape: {height_y.shape}")
    # print(f"height_z shape: {height_z.shape}")
    # print(f"height_3d shape: {height_3d.shape}")
    # print("height scan 3d:", height_3d)

    return height_3d


def mid360_structured_depth_image(
    env: ManagerBasedEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("head_mid360_scanner"),
    width: int = 180,
    height: int = 32,
    min_range_m: float = 0.1,
    max_range_m: float = 10.0,
    min_elevation_deg: float = -7.0,
    max_elevation_deg: float = 52.0,
    aggregation_method: str = "mean",
    log_k: float = 10.0,
) -> torch.Tensor:
    """
    Convert Mid360 LiDAR point cloud to structured depth image.

    This function follows the same conversion approach as mid360_to_nvblox_bridge:
    1. Cartesian coordinates -> Spherical coordinates (range, azimuth, elevation)
    2. Spherical coordinates -> Grid indices (u, v)
    3. Clip range to [min_range_m, max_range_m], apply log mapping
    4. Aggregate multiple points per grid cell (min/mean/max)

    Args:
        env: The environment instance.
        sensor_cfg: The sensor configuration.
        width: Horizontal resolution (default: 180, 2° per pixel).
        height: Vertical resolution (default: 32).
        min_range_m: Minimum valid range in meters. Points below this are set to 0.
        max_range_m: Maximum valid range in meters. Points above this are clipped.
        min_elevation_deg: Minimum elevation angle in degrees.
        max_elevation_deg: Maximum elevation angle in degrees.
        aggregation_method: Aggregation method for multiple points in same cell.
            Options: "min", "max", "mean". Default: "mean".
        log_k: Log mapping parameter. When > 0, applies log(1+k*d)/log(1+k*max)
            to compress distant range and enhance close range distinction.
            Default: 10.0. Set to 0 to disable log mapping.

    Returns:
        Structured depth image tensor. Shape: (num_envs, height, width, 1).
            Invalid pixels are set to 0.0.
    """
    sensor: RayCaster = env.scene.sensors[sensor_cfg.name]

    points_b = sensor.data.ray_hits_b

    N = env.num_envs

    x, y, z = points_b[..., 0], points_b[..., 1], points_b[..., 2]

    ranges = torch.sqrt(x**2 + y**2 + z**2)
    azimuths = torch.atan2(y, x)
    elevations = torch.asin(z / (ranges + 1e-8))

    min_elev_rad = torch.tensor(min_elevation_deg * torch.pi / 180.0, device=env.device)
    max_elev_rad = torch.tensor(max_elevation_deg * torch.pi / 180.0, device=env.device)
    azimuth_res_rad = 2.0 * torch.pi / width
    elevation_res_rad = (max_elev_rad - min_elev_rad) / (height - 1)

    u_idx = ((azimuths + torch.pi) / azimuth_res_rad).long()
    v_idx = ((elevations - min_elev_rad) / elevation_res_rad).long()

    u_idx = torch.clamp(u_idx, 0, width - 1)
    v_idx = torch.clamp(v_idx, 0, height - 1)

    ranges = torch.where(ranges > max_range_m, torch.tensor(max_range_m, device=env.device), ranges)
    ranges = torch.where(ranges < min_range_m, torch.tensor(0.0, device=env.device), ranges)
    valid_mask = ranges > 0

    if log_k > 0:
        log_denominator = torch.log(torch.tensor(1 + log_k * max_range_m, device=env.device))
        ranges = torch.where(
            valid_mask,
            torch.log(1 + log_k * ranges) / log_denominator * max_range_m,
            ranges
        )

    depth_map = torch.full((N, height * width), float("inf"), device=env.device)

    flat_idx = v_idx * width + u_idx

    if aggregation_method == "min":
        filtered_ranges = torch.where(valid_mask, ranges, float("inf"))
        depth_map.scatter_reduce_(1, flat_idx, filtered_ranges, reduce="min", include_self=False)
    elif aggregation_method == "max":
        filtered_ranges = torch.where(valid_mask, ranges, -float("inf"))
        depth_map.scatter_reduce_(1, flat_idx, filtered_ranges, reduce="max", include_self=False)
    elif aggregation_method == "mean":
        count_map = torch.zeros((N, height * width), device=env.device)
        count_map.scatter_add_(1, flat_idx, valid_mask.float())
        sum_map = torch.zeros((N, height * width), device=env.device)
        sum_map.scatter_add_(1, flat_idx, ranges * valid_mask.float())
        depth_map = torch.where(count_map > 0, sum_map / count_map, float("inf"))
    else:
        raise ValueError(f"Unknown aggregation method: {aggregation_method}")

    depth_map = depth_map.view(N, height, width).unsqueeze(-1)
    depth_map = torch.where(depth_map >= float("inf"), torch.tensor(0.0, device=env.device), depth_map)

    return depth_map


def depth_image_age(
    env: ManagerBasedEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("head_mid360_scanner"),
    max_age: int = 5,
) -> torch.Tensor:
    """
    返回深度图的存留帧数标记，用于多速率同步场景。

    当深度图更新频率低于 policy 频率时（如深度图10Hz，policy 50Hz），
    此函数返回当前深度图已被使用的帧数，归一化到 [0, 1]。

    Args:
        env: The environment instance.
        sensor_cfg: The sensor configuration for the depth image source.
        max_age: Maximum expected age in frames before depth image updates.
            Default: 5 (for 50Hz policy with 10Hz depth image).

    Returns:
        Tensor of shape (num_envs, 1) with values in [0, 1].
            0.0: 深度图刚更新
            1.0: 深度图即将过期
    """
    sensor = env.scene.sensors[sensor_cfg.name]
    
    state_key = f"_depth_age_{sensor_cfg.name}"
    if not hasattr(env, state_key):
        setattr(env, state_key, {
            'counter': torch.zeros(env.num_envs, device=env.device),
            'last_frame': torch.full((env.num_envs,), -1, device=env.device),
        })
    
    state = getattr(env, state_key)
    current_frame = sensor.data.frame_id
    
    mask_new = current_frame != state['last_frame']
    state['counter'] = torch.where(mask_new, torch.tensor(0, device=env.device), state['counter'] + 1)
    state['last_frame'] = current_frame
    
    return (state['counter'] / max_age).clamp(0.0, 1.0).view(-1, 1)


def body_incoming_wrench(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Incoming spatial wrench on bodies of an articulation in the simulation world frame.

    This is the 6-D wrench (force and torque) applied to the body link by the incoming joint force.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # obtain the link incoming forces in world frame
    body_incoming_joint_wrench_b = asset.data.body_incoming_joint_wrench_b[:, asset_cfg.body_ids]
    return body_incoming_joint_wrench_b.view(env.num_envs, -1)


def imu_orientation(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("imu")) -> torch.Tensor:
    """Imu sensor orientation in the simulation world frame.

    Args:
        env: The environment.
        asset_cfg: The SceneEntity associated with an IMU sensor. Defaults to SceneEntityCfg("imu").

    Returns:
        Orientation in the world frame in (w, x, y, z) quaternion form. Shape is (num_envs, 4).
    """
    # extract the used quantities (to enable type-hinting)
    asset: Imu = env.scene[asset_cfg.name]
    # return the orientation quaternion
    return asset.data.quat_w


def imu_projected_gravity(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("imu")) -> torch.Tensor:
    """Imu sensor orientation w.r.t the env.scene.origin.

    Args:
        env: The environment.
        asset_cfg: The SceneEntity associated with an Imu sensor.

    Returns:
        Gravity projected on imu_frame, shape of torch.tensor is (num_env,3).
    """

    asset: Imu = env.scene[asset_cfg.name]
    return asset.data.projected_gravity_b


def imu_ang_vel(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("imu")) -> torch.Tensor:
    """Imu sensor angular velocity w.r.t. environment origin expressed in the sensor frame.

    Args:
        env: The environment.
        asset_cfg: The SceneEntity associated with an IMU sensor. Defaults to SceneEntityCfg("imu").

    Returns:
        The angular velocity (rad/s) in the sensor frame. Shape is (num_envs, 3).
    """
    # extract the used quantities (to enable type-hinting)
    asset: Imu = env.scene[asset_cfg.name]
    # return the angular velocity
    return asset.data.ang_vel_b


def imu_lin_acc(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("imu")) -> torch.Tensor:
    """Imu sensor linear acceleration w.r.t. the environment origin expressed in sensor frame.

    Args:
        env: The environment.
        asset_cfg: The SceneEntity associated with an IMU sensor. Defaults to SceneEntityCfg("imu").

    Returns:
        The linear acceleration (m/s^2) in the sensor frame. Shape is (num_envs, 3).
    """
    asset: Imu = env.scene[asset_cfg.name]
    return asset.data.lin_acc_b


def image(
    env: ManagerBasedEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("tiled_camera"),
    data_type: str = "rgb",
    convert_perspective_to_orthogonal: bool = False,
    normalize: bool = True,
) -> torch.Tensor:
    """Images of a specific datatype from the camera sensor.

    If the flag :attr:`normalize` is True, post-processing of the images are performed based on their
    data-types:

    - "rgb": Scales the image to (0, 1) and subtracts with the mean of the current image batch.
    - "depth" or "distance_to_camera" or "distance_to_plane": Replaces infinity values with zero.

    Args:
        env: The environment the cameras are placed within.
        sensor_cfg: The desired sensor to read from. Defaults to SceneEntityCfg("tiled_camera").
        data_type: The data type to pull from the desired camera. Defaults to "rgb".
        convert_perspective_to_orthogonal: Whether to orthogonalize perspective depth images.
            This is used only when the data type is "distance_to_camera". Defaults to False.
        normalize: Whether to normalize the images. This depends on the selected data type.
            Defaults to True.

    Returns:
        The images produced at the last time-step
    """
    # extract the used quantities (to enable type-hinting)
    sensor: TiledCamera | Camera | RayCasterCamera = env.scene.sensors[sensor_cfg.name]

    # obtain the input image
    images = sensor.data.output[data_type]

    # depth image conversion
    if (data_type == "distance_to_camera") and convert_perspective_to_orthogonal:
        images = math_utils.orthogonalize_perspective_depth(images, sensor.data.intrinsic_matrices)

    # rgb/depth/normals image normalization
    if normalize:
        if data_type == "rgb":
            images = images.float() / 255.0
            mean_tensor = torch.mean(images, dim=(1, 2), keepdim=True)
            images -= mean_tensor
        elif "distance_to" in data_type or "depth" in data_type:
            images[images == float("inf")] = 0
        elif "normals" in data_type:
            images = (images + 1.0) * 0.5

    return images.clone()


class image_features(ManagerTermBase):
    """Extracted image features from a pre-trained frozen encoder.

    This term uses models from the model zoo in PyTorch and extracts features from the images.

    It calls the :func:`image` function to get the images and then processes them using the model zoo.

    A user can provide their own model zoo configuration to use different models for feature extraction.
    The model zoo configuration should be a dictionary that maps different model names to a dictionary
    that defines the model, preprocess and inference functions. The dictionary should have the following
    entries:

    - "model": A callable that returns the model when invoked without arguments.
    - "reset": A callable that resets the model. This is useful when the model has a state that needs to be reset.
    - "inference": A callable that, when given the model and the images, returns the extracted features.

    If the model zoo configuration is not provided, the default model zoo configurations are used. The default
    model zoo configurations include the models from Theia :cite:`shang2024theia` and ResNet :cite:`he2016deep`.
    These models are loaded from `Hugging-Face transformers <https://huggingface.co/docs/transformers/index>`_ and
    `PyTorch torchvision <https://pytorch.org/vision/stable/models.html>`_ respectively.

    Args:
        sensor_cfg: The sensor configuration to poll. Defaults to SceneEntityCfg("tiled_camera").
        data_type: The sensor data type. Defaults to "rgb".
        convert_perspective_to_orthogonal: Whether to orthogonalize perspective depth images.
            This is used only when the data type is "distance_to_camera". Defaults to False.
        model_zoo_cfg: A user-defined dictionary that maps different model names to their respective configurations.
            Defaults to None. If None, the default model zoo configurations are used.
        model_name: The name of the model to use for inference. Defaults to "resnet18".
        model_device: The device to store and infer the model on. This is useful when offloading the computation
            from the environment simulation device. Defaults to the environment device.
        inference_kwargs: Additional keyword arguments to pass to the inference function. Defaults to None,
            which means no additional arguments are passed.

    Returns:
        The extracted features tensor. Shape is (num_envs, feature_dim).

    Raises:
        ValueError: When the model name is not found in the provided model zoo configuration.
        ValueError: When the model name is not found in the default model zoo configuration.
    """

    def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedEnv):
        # initialize the base class
        super().__init__(cfg, env)

        # extract parameters from the configuration
        self.model_zoo_cfg: dict = cfg.params.get("model_zoo_cfg")  # type: ignore
        self.model_name: str = cfg.params.get("model_name", "resnet18")  # type: ignore
        self.model_device: str = cfg.params.get("model_device", env.device)  # type: ignore

        # List of Theia models - These are configured through `_prepare_theia_transformer_model` function
        default_theia_models = [
            "theia-tiny-patch16-224-cddsv",
            "theia-tiny-patch16-224-cdiv",
            "theia-small-patch16-224-cdiv",
            "theia-base-patch16-224-cdiv",
            "theia-small-patch16-224-cddsv",
            "theia-base-patch16-224-cddsv",
        ]
        # List of ResNet models - These are configured through `_prepare_resnet_model` function
        default_resnet_models = ["resnet18", "resnet34", "resnet50", "resnet101"]

        # Check if model name is specified in the model zoo configuration
        if self.model_zoo_cfg is not None and self.model_name not in self.model_zoo_cfg:
            raise ValueError(
                f"Model name '{self.model_name}' not found in the provided model zoo configuration."
                " Please add the model to the model zoo configuration or use a different model name."
                f" Available models in the provided list: {list(self.model_zoo_cfg.keys())}."
                "\nHint: If you want to use a default model, consider using one of the following models:"
                f" {default_theia_models + default_resnet_models}. In this case, you can remove the"
                " 'model_zoo_cfg' parameter from the observation term configuration."
            )
        if self.model_zoo_cfg is None:
            if self.model_name in default_theia_models:
                model_config = self._prepare_theia_transformer_model(self.model_name, self.model_device)
            elif self.model_name in default_resnet_models:
                model_config = self._prepare_resnet_model(self.model_name, self.model_device)
            else:
                raise ValueError(
                    f"Model name '{self.model_name}' not found in the default model zoo configuration."
                    f" Available models: {default_theia_models + default_resnet_models}."
                )
        else:
            model_config = self.model_zoo_cfg[self.model_name]

        # Retrieve the model, preprocess and inference functions
        self._model = model_config["model"]()
        self._reset_fn = model_config.get("reset")
        self._inference_fn = model_config["inference"]

    def reset(self, env_ids: torch.Tensor | None = None):
        # reset the model if a reset function is provided
        # this might be useful when the model has a state that needs to be reset
        # for example: video transformers
        if self._reset_fn is not None:
            self._reset_fn(self._model, env_ids)

    def __call__(
        self,
        env: ManagerBasedEnv,
        sensor_cfg: SceneEntityCfg = SceneEntityCfg("tiled_camera"),
        data_type: str = "rgb",
        convert_perspective_to_orthogonal: bool = False,
        model_zoo_cfg: dict | None = None,
        model_name: str = "resnet18",
        model_device: str | None = None,
        inference_kwargs: dict | None = None,
    ) -> torch.Tensor:
        # obtain the images from the sensor
        image_data = image(
            env=env,
            sensor_cfg=sensor_cfg,
            data_type=data_type,
            convert_perspective_to_orthogonal=convert_perspective_to_orthogonal,
            normalize=False,  # we pre-process based on model
        )
        # store the device of the image
        image_device = image_data.device
        # forward the images through the model
        features = self._inference_fn(self._model, image_data, **(inference_kwargs or {}))

        # move the features back to the image device
        return features.detach().to(image_device)

    """
    Helper functions.
    """

    def _prepare_theia_transformer_model(self, model_name: str, model_device: str) -> dict:
        """Prepare the Theia transformer model for inference.

        Args:
            model_name: The name of the Theia transformer model to prepare.
            model_device: The device to store and infer the model on.

        Returns:
            A dictionary containing the model and inference functions.
        """
        from transformers import AutoModel

        def _load_model() -> torch.nn.Module:
            """Load the Theia transformer model."""
            model = AutoModel.from_pretrained(f"theaiinstitute/{model_name}", trust_remote_code=True).eval()
            return model.to(model_device)

        def _inference(model, images: torch.Tensor) -> torch.Tensor:
            """Inference the Theia transformer model.

            Args:
                model: The Theia transformer model.
                images: The preprocessed image tensor. Shape is (num_envs, height, width, channel).

            Returns:
                The extracted features tensor. Shape is (num_envs, feature_dim).
            """
            # Move the image to the model device
            image_proc = images.to(model_device)
            # permute the image to (num_envs, channel, height, width)
            image_proc = image_proc.permute(0, 3, 1, 2).float() / 255.0
            # Normalize the image
            mean = torch.tensor([0.485, 0.456, 0.406], device=model_device).view(1, 3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225], device=model_device).view(1, 3, 1, 1)
            image_proc = (image_proc - mean) / std

            # Taken from Transformers; inference converted to be GPU only
            features = model.backbone.model(pixel_values=image_proc, interpolate_pos_encoding=True)
            return features.last_hidden_state[:, 1:]

        # return the model, preprocess and inference functions
        return {"model": _load_model, "inference": _inference}

    def _prepare_resnet_model(self, model_name: str, model_device: str) -> dict:
        """Prepare the ResNet model for inference.

        Args:
            model_name: The name of the ResNet model to prepare.
            model_device: The device to store and infer the model on.

        Returns:
            A dictionary containing the model and inference functions.
        """
        from torchvision import models

        def _load_model() -> torch.nn.Module:
            """Load the ResNet model."""
            # map the model name to the weights
            resnet_weights = {
                "resnet18": "ResNet18_Weights.IMAGENET1K_V1",
                "resnet34": "ResNet34_Weights.IMAGENET1K_V1",
                "resnet50": "ResNet50_Weights.IMAGENET1K_V1",
                "resnet101": "ResNet101_Weights.IMAGENET1K_V1",
            }

            # load the model
            model = getattr(models, model_name)(weights=resnet_weights[model_name]).eval()
            return model.to(model_device)

        def _inference(model, images: torch.Tensor) -> torch.Tensor:
            """Inference the ResNet model.

            Args:
                model: The ResNet model.
                images: The preprocessed image tensor. Shape is (num_envs, channel, height, width).

            Returns:
                The extracted features tensor. Shape is (num_envs, feature_dim).
            """
            # move the image to the model device
            image_proc = images.to(model_device)
            # permute the image to (num_envs, channel, height, width)
            image_proc = image_proc.permute(0, 3, 1, 2).float() / 255.0
            # normalize the image
            mean = torch.tensor([0.485, 0.456, 0.406], device=model_device).view(1, 3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225], device=model_device).view(1, 3, 1, 1)
            image_proc = (image_proc - mean) / std

            # forward the image through the model
            return model(image_proc)

        # return the model, preprocess and inference functions
        return {"model": _load_model, "inference": _inference}


"""
Actions.
"""


@generic_io_descriptor(dtype=torch.float32, observation_type="Action", on_inspect=[record_shape])
def last_action(env: ManagerBasedEnv, action_name: str | None = None) -> torch.Tensor:
    """The last input action to the environment.

    The name of the action term for which the action is required. If None, the
    entire action tensor is returned.
    """
    if action_name is None:
        return env.action_manager.action
    else:
        return env.action_manager.get_term(action_name).raw_actions


"""
Commands.
"""


@generic_io_descriptor(dtype=torch.float32, observation_type="Command", on_inspect=[record_shape])
def generated_commands(env: ManagerBasedRLEnv, command_name: str | None = None) -> torch.Tensor:
    """The generated command from command term in the command manager with the given name."""
    return env.command_manager.get_command(command_name)


"""
Time.
"""


def current_time_s(env: ManagerBasedRLEnv) -> torch.Tensor:
    """The current time in the episode (in seconds)."""
    return env.episode_length_buf.unsqueeze(1) * env.step_dt


def remaining_time_s(env: ManagerBasedRLEnv) -> torch.Tensor:
    """The maximum time remaining in the episode (in seconds)."""
    return env.max_episode_length_s - env.episode_length_buf.unsqueeze(1) * env.step_dt
