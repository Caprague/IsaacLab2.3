# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

"""Pretrain teacher encoder networks via AutoEncoder reconstruction.

This script runs a trained teacher PPO policy in the environment to collect
observations (mapScans and privileged), then trains HeightScanEncoder and
PrivilegeEncoder autoencoders on reconstruction to learn meaningful latent
representations for downstream student distillation.

Workflow:
    1. Launch Isaac Sim and create the environment.
    2. Load a Stage3 teacher PPO checkpoint.
    3. Run the teacher policy to collect mapScans / privileged obs into a buffer.
    4. Periodically train both autoencoders via MSE reconstruction loss.
    5. Save the encoder weights (decoder discarded) for Stage4 student training.
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(
    description="Pretrain teacher encoders via AutoEncoder reconstruction."
)
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent",
    type=str,
    default="rsl_rl_cfg_entry_point",
    help="Name of the RL agent configuration entry point.",
)
parser.add_argument(
    "--checkpoint",
    type=str,
    default=None,
    help="Direct path to teacher checkpoint (.pt).",
)
parser.add_argument(
    "--encoder_checkpoint",
    type=str,
    default=None,
    help="Path to existing encoder weights for warm start.",
)
parser.add_argument(
    "--load_run",
    type=str,
    default=None,
    help="Regex matching run directory name for checkpoint lookup.",
)
parser.add_argument(
    "--load_checkpoint",
    type=str,
    default=".*",
    help="Regex matching checkpoint filename for lookup.",
)
parser.add_argument(
    "--source_experiment",
    type=str,
    default="Go2-Loco-Skill-Walk-Mid360Depth-10Hz",
    help="Source experiment name containing teacher checkpoints.",
)
parser.add_argument(
    "--num_envs", type=int, default=None, help="Number of environments to simulate."
)
parser.add_argument(
    "--total_steps",
    type=int,
    default=2000,
    help="Total environment rollout steps.",
)
parser.add_argument(
    "--train_every",
    type=int,
    default=100,
    help="Train encoders every N environment steps.",
)
parser.add_argument(
    "--train_iters",
    type=int,
    default=10,
    help="Number of training iterations per train event.",
)
parser.add_argument(
    "--batch_size",
    type=int,
    default=4096,
    help="Training batch size.",
)
parser.add_argument(
    "--save_every",
    type=int,
    default=5,
    help="Save checkpoint every N training iterations.",
)
parser.add_argument(
    "--push_interval",
    type=int,
    default=300,
    help="Push perturbation interval. 0 = disabled.",
)
parser.add_argument(
    "--seed", type=int, default=None, help="Seed for the environment."
)
# append RSL-RL cli arguments (required by update_rsl_rl_cfg)
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import os
from dataclasses import dataclass, field
from datetime import datetime

import gymnasium as gym
import torch
import torch.nn as nn
from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.assets import retrieve_file_path

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

from rsl_rl.networks.teacher_encoders import HeightScanEncoder, PrivilegeEncoder

# PLACEHOLDER: Extension template (do not remove this comment)


@dataclass
class ObsBuffer:
    """Fixed-capacity circular buffer for teacher observation collection.

    Stores mapScans and privileged observations from the teacher policy rollout
    for subsequent autoencoder training.

    Args:
        capacity: Maximum number of samples to store.

    """

    capacity: int
    map_scans: torch.Tensor = field(init=False)
    privileged: torch.Tensor = field(init=False)
    write_ptr: int = field(default=0, init=False)
    _total_written: int = field(default=0, init=False)
    full: bool = field(default=False, init=False)

    def __post_init__(self):
        """Allocate the fixed-size buffers."""
        self.map_scans = torch.zeros(self.capacity, 187)
        self.privileged = torch.zeros(self.capacity, 60)

    def add(self, ms: torch.Tensor, priv: torch.Tensor) -> None:
        """Add a batch of observations to the circular buffer.

        Handles wrap-around correctly when writing past the buffer end.

        Args:
            ms: mapScans tensor of shape ``[N, 187]`` (CPU).
            priv: privileged tensor of shape ``[N, 60]`` (CPU).

        """
        n = ms.shape[0]
        space_to_end = self.capacity - self.write_ptr
        if n <= space_to_end:
            # Fits within remaining space without wrapping
            self.map_scans[self.write_ptr : self.write_ptr + n] = ms
            self.privileged[self.write_ptr : self.write_ptr + n] = priv
        else:
            # Wraps around: write first part to end, remainder to beginning
            self.map_scans[self.write_ptr :] = ms[:space_to_end]
            self.map_scans[: n - space_to_end] = ms[space_to_end:]
            self.privileged[self.write_ptr :] = priv[:space_to_end]
            self.privileged[: n - space_to_end] = priv[space_to_end:]
        self.write_ptr = (self.write_ptr + n) % self.capacity
        self._total_written += n
        if not self.full and self._total_written >= self.capacity:
            self.full = True

    def get_valid(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return all valid data from the circular buffer.

        Returns:
            Tuple of ``(map_scans, privileged)`` each of shape
            ``[valid_samples, dim]``.

        """
        n = self.capacity if self.full else self.write_ptr
        return self.map_scans[:n], self.privileged[:n]


def _apply_push_perturbation(
    env: RslRlVecEnvWrapper,
    step_counter: int,
    last_push_step: int,
    push_interval: int,
) -> int:
    """Apply a mild random velocity push to robots for data diversity.

    Args:
        env: The wrapped simulation environment.
        step_counter: Current environment step count.
        last_push_step: Step count when push was last applied.
        push_interval: Minimum steps between successive pushes.

    Returns:
        Updated ``last_push_step`` (unchanged if no push was applied).

    """
    if push_interval <= 0:
        return last_push_step
    if step_counter - last_push_step < push_interval:
        return last_push_step

    num_envs = env.num_envs
    device = env.unwrapped.device
    rand_vel = torch.stack(
        [
            torch.rand(num_envs, device=device) * 0.5 - 0.25,  # x: [-0.25, 0.25]
            torch.rand(num_envs, device=device) * 0.5 - 0.25,  # y: [-0.25, 0.25]
            torch.rand(num_envs, device=device) * 0.3,          # z: [0, 0.3]
            torch.rand(num_envs, device=device) * 0.2 - 0.1,   # roll: [-0.1, 0.1]
            torch.rand(num_envs, device=device) * 0.2 - 0.1,   # pitch: [-0.1, 0.1]
            torch.rand(num_envs, device=device) * 0.3 - 0.15,  # yaw: [-0.15, 0.15]
        ],
        dim=1,
    )
    env.unwrapped.root_physx_view.set_root_velocities(
        rand_vel, env.unwrapped._robot_actor_indices
    )
    return step_counter


@hydra_task_config(args_cli.task, args_cli.agent)
def main(
    env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg,
    agent_cfg: RslRlBaseRunnerCfg,
):
    """Pretrain teacher encoders for student distillation.

    Args:
        env_cfg: The environment configuration.
        agent_cfg: The RL agent configuration.

    """
    # ------------------------------------------------------------------
    # 1. Resolve checkpoint path
    # ------------------------------------------------------------------
    if args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        log_root_path = os.path.join(
            "logs", "rsl_rl", args_cli.source_experiment
        )
        log_root_path = os.path.abspath(log_root_path)
        resume_path = get_checkpoint_path(
            log_root_path, args_cli.load_run, args_cli.load_checkpoint
        )
    print(f"[INFO] Loading teacher checkpoint from: {resume_path}")

    # ------------------------------------------------------------------
    # 2. Create environment
    # ------------------------------------------------------------------
    # override configurations with non-hydra CLI arguments
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = (
        args_cli.num_envs
        if args_cli.num_envs is not None
        else env_cfg.scene.num_envs
    )

    # set the environment seed
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = (
        args_cli.device if args_cli.device is not None else env_cfg.sim.device
    )
    # set the log directory for the environment
    env_cfg.log_dir = os.path.dirname(resume_path)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    # ------------------------------------------------------------------
    # 3. Load teacher policy
    # ------------------------------------------------------------------
    runner = OnPolicyRunner(
        env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device
    )
    runner.load(resume_path)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    # Reset recurrent states by resetting the actor-critic after load
    try:
        policy_nn = runner.alg.policy
    except AttributeError:
        policy_nn = runner.alg.actor_critic

    # ------------------------------------------------------------------
    # 4. Prepare output directory
    # ------------------------------------------------------------------
    output_experiment = "Go2-Loco-Skill-Walk-Mid360Depth-10Hz-GRU"
    output_root = os.path.join("logs", "rsl_rl", output_experiment)
    output_run = (
        f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_pretrain_teacher"
    )
    output_dir = os.path.join(output_root, output_run)
    os.makedirs(os.path.join(output_dir, "params"), exist_ok=True)
    print(f"[INFO] Output directory: {output_dir}")

    # ------------------------------------------------------------------
    # 5. Create AutoEncoder models + optimizers
    # ------------------------------------------------------------------
    device = env.unwrapped.device

    hs_ae = HeightScanEncoder().to(device)
    priv_ae = PrivilegeEncoder().to(device)

    # warm start from existing encoder weights
    if args_cli.encoder_checkpoint:
        print(
            f"[INFO] Loading encoder weights from: {args_cli.encoder_checkpoint}"
        )
        enc_state = torch.load(
            args_cli.encoder_checkpoint, weights_only=True, map_location=device
        )
        if "height_scan_encoder" in enc_state:
            hs_ae.encoder.load_state_dict(enc_state["height_scan_encoder"])
        else:
            hs_ae.encoder.load_state_dict(enc_state)
        if "privilege_encoder" in enc_state:
            priv_ae.encoder.load_state_dict(enc_state["privilege_encoder"])

    hs_ae.train()
    priv_ae.train()

    hs_opt = torch.optim.Adam(hs_ae.parameters(), lr=1e-3)
    priv_opt = torch.optim.Adam(priv_ae.parameters(), lr=1e-3)
    mse = nn.MSELoss()

    # ------------------------------------------------------------------
    # 6. Create observation buffer
    # ------------------------------------------------------------------
    buffer = ObsBuffer(capacity=200_000)

    # ------------------------------------------------------------------
    # 7. Play + Train alternating loop
    # ------------------------------------------------------------------
    obs = env.get_observations()

    total_steps = args_cli.total_steps
    train_every = args_cli.train_every
    train_iters = args_cli.train_iters
    batch_size = args_cli.batch_size
    save_every = args_cli.save_every
    push_interval = args_cli.push_interval

    train_count = 0
    last_push_step = 0

    print(
        f"[INFO] Starting Play+Train loop: {total_steps} steps, "
        f"train every {train_every} steps."
    )

    for step in range(total_steps):
        # --- Play: teacher policy inference (no grad) ---
        with torch.inference_mode():
            actions = policy(obs)
            obs, _, dones, _ = env.step(actions)
            policy_nn.reset(dones)

        # --- Collect observations to CPU buffer ---
        buffer.add(obs["mapScans"].cpu(), obs["privileged"].cpu())

        # --- Apply push perturbation ---
        last_push_step = _apply_push_perturbation(
            env, step, last_push_step, push_interval
        )

        # --- Train: every train_every steps ---
        if step > 0 and step % train_every == 0:
            ms_data, priv_data = buffer.get_valid()
            if ms_data.shape[0] < batch_size:
                print(
                    f"[WARN] Buffer has {ms_data.shape[0]} samples, "
                    f"need {batch_size}. Skipping training at step {step}."
                )
                continue

            # Train HeightScanEncoder
            hs_loss_sum = 0.0
            for _ in range(train_iters):
                idx = torch.randperm(ms_data.shape[0])[:batch_size]
                batch = ms_data[idx].to(device)
                loss = mse(hs_ae(batch), batch)
                hs_opt.zero_grad()
                loss.backward()
                hs_opt.step()
                hs_loss_sum += loss.item()
            avg_hs_loss = hs_loss_sum / train_iters

            # Train PrivilegeEncoder
            priv_loss_sum = 0.0
            for _ in range(train_iters):
                idx = torch.randperm(priv_data.shape[0])[:batch_size]
                batch = priv_data[idx].to(device)
                loss = mse(priv_ae(batch), batch)
                priv_opt.zero_grad()
                loss.backward()
                priv_opt.step()
                priv_loss_sum += loss.item()
            avg_priv_loss = priv_loss_sum / train_iters

            train_count += 1
            print(
                f"[Train {train_count:04d} | step {step:05d}] "
                f"HS loss: {avg_hs_loss:.6f}, "
                f"Priv loss: {avg_priv_loss:.6f}, "
                f"buffer: {ms_data.shape[0]} samples"
            )

            # --- Save intermediate checkpoints ---
            if train_count % save_every == 0:
                torch.save(
                    hs_ae.encoder.state_dict(),
                    os.path.join(output_dir, "height_scan_encoder.pt"),
                )
                torch.save(
                    priv_ae.encoder.state_dict(),
                    os.path.join(output_dir, "privilege_encoder.pt"),
                )
                print(
                    f"[INFO] Saved intermediate encoder checkpoints "
                    f"at train_count={train_count}"
                )

    # ------------------------------------------------------------------
    # 8. Final save: merge teacher checkpoint + encoder weights
    # ------------------------------------------------------------------
    print("[INFO] Saving final combined model_teacher.pt ...")
    teacher_ckpt = torch.load(
        resume_path, weights_only=False, map_location="cpu"
    )
    combined = {
        "model_state_dict": teacher_ckpt["model_state_dict"],
        "optimizer_state_dict": teacher_ckpt.get(
            "optimizer_state_dict", {}
        ),
        "iter": teacher_ckpt.get("iter", 0),
        "infos": teacher_ckpt.get("infos", {}),
        "height_scan_encoder": hs_ae.encoder.state_dict(),
        "privilege_encoder": priv_ae.encoder.state_dict(),
        "encoder_iter": train_count,
        "source_checkpoint": str(resume_path),
    }
    torch.save(combined, os.path.join(output_dir, "model_teacher.pt"))
    torch.save(
        hs_ae.encoder.state_dict(),
        os.path.join(output_dir, "height_scan_encoder.pt"),
    )
    torch.save(
        priv_ae.encoder.state_dict(),
        os.path.join(output_dir, "privilege_encoder.pt"),
    )
    print(
        f"[INFO] Final outputs saved to: {output_dir}\n"
        f"  - model_teacher.pt\n"
        f"  - height_scan_encoder.pt\n"
        f"  - privilege_encoder.pt"
    )

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
