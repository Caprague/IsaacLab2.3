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

Usage:
    ./isaaclab.sh -p scripts/tools/encoder_pretrain/pretrain_teacher_encoders.py \
        --task Go2-Loco-Skill-Walk-Mid360Depth-10Hz-PretrainTeacher \
        --agent rsl_rl_cfg_entry_point \
        --checkpoint logs/rsl_rl/Go2-Loco-Skill-Walk-Mid360Depth-10Hz/{run}/model_16000.pt \
        --headless --num_envs 1024 --total_steps 20000 --train_every 100

    ./isaaclab.sh -p scripts/tools/encoder_pretrain/pretrain_teacher_encoders.py \
        --task Go2-Loco-Skill-Walk-Mid360Depth-10Hz-PretrainTeacher \
        --agent rsl_rl_cfg_entry_point \
        --checkpoint logs/rsl_rl/Go2-Loco-Skill-Walk-Mid360Depth-10Hz/2026-07-31_00-15-36/model_16000.pt \
        --headless --num_envs 1024 --total_steps 20000 --train_every 100

See Also:
    docs/go2_analysis_docs/go2_gru_student_policy_architecture_redesign.md
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import os
import shutil
import sys

from isaaclab.app import AppLauncher

# local imports (add rsl_rl dir to path to avoid ROS2 "scripts" package collision)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "reinforcement_learning", "rsl_rl"))
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
# --checkpoint and --load_run are already defined by cli_args.add_rsl_rl_args()
parser.add_argument(
    "--encoder_checkpoint",
    type=str,
    default=None,
    help="Path to existing encoder weights for warm start.",
)

parser.add_argument(
    "--num_envs", type=int, default=1024, help="Number of environments to simulate."
)
parser.add_argument(
    "--total_steps",
    type=int,
    default=20000,
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
    default=2048,
    help="Training batch size.",
)
parser.add_argument(
    "--save_every",
    type=int,
    default=5,
    help="Save checkpoint every N training iterations.",
)
parser.add_argument(
    "--seed", type=int, default=None, help="Seed for the environment."
)
# --gui: override default headless mode to show the simulator window
parser.add_argument(
    "--gui", action="store_true", help="Show simulator GUI (default: headless mode for pretraining)."
)
# append RSL-RL cli arguments (required by update_rsl_rl_cfg)
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# Pretraining defaults to headless mode for performance;
# use --gui to show the simulator window
if not args_cli.gui:
    args_cli.headless = True

# validate required arguments before launching Isaac Sim
if not args_cli.task:
    parser.error("--task is required. Example: --task Go2-Loco-Skill-Walk-Mid360Depth-10Hz-PretrainTeacher")
if not args_cli.checkpoint:
    parser.error("--checkpoint is required. Example: --checkpoint logs/rsl_rl/.../model_6000.pt")

# ---- DEBUG: CLI arguments ----
print("=" * 60)
print("[DEBUG] CLI Arguments:")
for k, v in sorted(vars(args_cli).items()):
    print(f"  {k}: {v}")
print("=" * 60)

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
    _initialized: bool = field(default=False, init=False)
    full: bool = field(default=False, init=False)

    def __post_init__(self):
        """Allocation is deferred until the first add() call to infer dims."""

    def _init_buffers(self, ms_dim: int, priv_dim: int) -> None:
        """Allocate fixed-size buffers based on observed dimensions."""
        self.map_scans = torch.zeros(self.capacity, ms_dim)
        self.privileged = torch.zeros(self.capacity, priv_dim)
        self._initialized = True

    def add(self, ms: torch.Tensor, priv: torch.Tensor) -> None:
        """Add a batch of observations to the circular buffer.

        Handles wrap-around correctly when writing past the buffer end.
        On first call, lazily allocates buffers based on observed dims.

        Args:
            ms: mapScans tensor of shape ``[N, ms_dim]`` (CPU).
            priv: privileged tensor of shape ``[N, priv_dim]`` (CPU).

        """
        if not self._initialized:
            self._init_buffers(ms.shape[1], priv.shape[1])
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
    if not args_cli.checkpoint:
        raise ValueError("--checkpoint is required. Please specify the path to the teacher checkpoint.")
    resume_path = retrieve_file_path(args_cli.checkpoint)
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

    # ---- DEBUG: Environment info ----
    print("=" * 60)
    print(f"[DEBUG] Environment:")
    print(f"  num_envs:       {env.num_envs}")
    print(f"  device:         {env.unwrapped.device}")
    print(f"  env_cfg.seed:   {env_cfg.seed}")
    print(f"  task:           {args_cli.task}")
    print(f"  action_space:   {env.unwrapped.single_action_space}")
    print("=" * 60)

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

    # ---- DEBUG: Teacher checkpoint info ----
    print("=" * 60)
    print(f"[DEBUG] Teacher Checkpoint:")
    print(f"  checkpoint_path: {resume_path}")
    # read iteration from checkpoint file
    ckpt = torch.load(resume_path, weights_only=False, map_location="cpu")
    print(f"  checkpoint_iter: {ckpt.get('iter', 'N/A')}")
    print(f"  policy type:     {type(policy_nn).__name__}")
    teacher_params = sum(p.numel() for p in policy_nn.parameters())
    print(f"  teacher params:  {teacher_params:,}")
    print("=" * 60)

    # ------------------------------------------------------------------
    # 4. Prepare output directory
    # ------------------------------------------------------------------
    output_experiment = "Go2-Loco-Skill-Walk-Mid360Depth-10Hz-GRU"
    output_root = os.path.join("logs", "rsl_rl", output_experiment)
    output_run = (
        f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_pretrain_teacher"
    )
    output_dir = os.path.join(output_root, output_run)

    # Clean up old incomplete run folders (those without model_teacher.pt)
    # so the log directory doesn't accumulate empty folders from killed runs
    if os.path.isdir(output_root):
        for entry in os.listdir(output_root):
            if not entry.endswith("_pretrain_teacher"):
                continue
            old_dir = os.path.join(output_root, entry)
            if os.path.isdir(old_dir) and not os.path.exists(
                os.path.join(old_dir, "model_teacher.pt")
            ):
                shutil.rmtree(old_dir)
                print(f"[INFO] Cleaned up incomplete run folder: {entry}")

    os.makedirs(os.path.join(output_dir, "params"), exist_ok=True)
    print(f"[INFO] Output directory: {output_dir}")

    # ------------------------------------------------------------------
    # 5. Create AutoEncoder models + optimizers
    # ------------------------------------------------------------------
    device = env.unwrapped.device
    mse = nn.MSELoss()

    hs_ae = HeightScanEncoder().to(device)
    priv_ae = PrivilegeEncoder().to(device)

    # warm start from existing encoder weights
    if args_cli.encoder_checkpoint:
        print(f"[INFO] Loading encoder weights from: {args_cli.encoder_checkpoint}")
        enc_state = torch.load(args_cli.encoder_checkpoint, weights_only=True, map_location=device)
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

    # ---- DEBUG: Encoder model info ----
    print("=" * 60)
    print("[DEBUG] Encoder Models:")
    hs_params = sum(p.numel() for p in hs_ae.parameters())
    priv_params = sum(p.numel() for p in priv_ae.parameters())
    print(f"  HeightScanEncoder: {hs_params:,} params, latent_dim={hs_ae.latent_dim}")
    print(f"  PrivilegeEncoder:  {priv_params:,} params, latent_dim={priv_ae.latent_dim}, input_dim={priv_ae.input_dim}")
    print("=" * 60)

    # ------------------------------------------------------------------
    # 6. Create observation buffer
    # ------------------------------------------------------------------
    buffer = ObsBuffer(capacity=200_000)

    # ------------------------------------------------------------------
    # 7. Play + Train alternating loop
    # ------------------------------------------------------------------
    obs = env.get_observations()

    # ---- DEBUG: Observation shapes ----
    print("=" * 60)
    print("[DEBUG] Observation TensorDict keys and shapes:")
    for key, val in obs.items():
        if val is not None and isinstance(val, torch.Tensor):
            print(f"  {key:20s}: {list(val.shape)}  (dtype={val.dtype}, device={val.device})")
        elif val is not None:
            print(f"  {key:20s}: {type(val).__name__} (non-tensor)")
        else:
            print(f"  {key:20s}: None")
    print(f"  buffer capacity: {buffer.capacity:,}, batch_size: {args_cli.batch_size}")
    print("=" * 60)

    total_steps = args_cli.total_steps
    train_every = args_cli.train_every
    train_iters = args_cli.train_iters
    batch_size = args_cli.batch_size
    save_every = args_cli.save_every

    train_count = 0

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

        # ---- DEBUG: step summary (every 50 steps) ----
        if step % 50 == 0:
            print(f"[DEBUG] step={step:05d}: "
                  f"action_range=[{actions.min().item():.4f}, {actions.max().item():.4f}], "
                  f"buffer_size={buffer._total_written:,}")

        # --- Collect observations to CPU buffer ---
        buffer.add(obs["mapScans"].cpu(), obs["privileged"].cpu())

        # --- Train: every train_every steps ---
        if step > 0 and step % train_every == 0:
            ms_data, priv_data = buffer.get_valid()
            if ms_data.shape[0] < batch_size:
                print(
                    f"[WARN] Buffer has {ms_data.shape[0]} samples, "
                    f"need {batch_size}. Skipping training at step {step}."
                )
                continue

            # ---- DEBUG: first training batch dimensions ----
            if train_count == 0:
                print("=" * 60)
                print(f"[DEBUG] First training event at step={step}:")
                print(f"  ms_data shape:   {list(ms_data.shape)}")
                print(f"  priv_data shape: {list(priv_data.shape)}")
                print(f"  batch_size:      {batch_size}")
                print(f"  train_iters:     {train_iters}")
                # test forward pass
                test_batch = ms_data[:2].to(device)
                with torch.no_grad():
                    recon = hs_ae(test_batch)
                print(f"  HS test: input={list(test_batch.shape)} -> output={list(recon.shape)}")
                test_batch = priv_data[:2].to(device)
                with torch.no_grad():
                    recon = priv_ae(test_batch)
                print(f"  Priv test: input={list(test_batch.shape)} -> output={list(recon.shape)}")
                print("=" * 60)

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

    # ---- DEBUG: final save summary ----
    print("=" * 60)
    print("[DEBUG] Final Save Summary:")
    print(f"  train_count:        {train_count}")
    print(f"  output_dir:         {output_dir}")
    hs_sd = hs_ae.encoder.state_dict()
    priv_sd = priv_ae.encoder.state_dict()
    print(f"  HS encoder keys:    {list(hs_sd.keys())[:4]}... ({len(hs_sd)} total)")
    print(f"  Priv encoder keys:  {list(priv_sd.keys())[:4]}... ({len(priv_sd)} total)")
    print(f"  model_teacher.pt keys: {list(combined.keys())}")
    # show a sample weight shape
    first_key = list(hs_sd.keys())[0]
    print(f"  HS '{first_key}' shape: {list(hs_sd[first_key].shape)}")
    first_key = list(priv_sd.keys())[0]
    print(f"  Priv '{first_key}' shape: {list(priv_sd[first_key].shape)}")
    print("=" * 60)

    print(
        f"[INFO] Final outputs saved to: {output_dir}\n"
        f"  - model_teacher.pt\n"
        f"  - height_scan_encoder.pt\n"
        f"  - privilege_encoder.pt"
    )

    # close the simulator
    try:
        env.close()
    except Exception as e:
        print(f"[WARN] Error during env.close(): {e}")


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app — wrapped in try/except because Isaac Sim may segfault
    # when X11 connection is lost (e.g. tmux detach, SSH disconnect)
    try:
        simulation_app.close()
    except Exception as e:
        print(f"[WARN] Error during simulation_app.close(): {e}")
    # force clean exit to avoid segfault from dangling C extension cleanup
    import os as _os
    _os._exit(0)
