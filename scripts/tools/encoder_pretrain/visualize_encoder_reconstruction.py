# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import sys
# Isaac Sim's bundled Python lacks _tkinter; borrow from system Python 3.11
sys.path.insert(0, "/usr/lib/python3.11/lib-dynload")

"""Visualize AutoEncoder reconstruction quality for teacher encoders.

Loads a trained teacher PPO policy, runs it in headless simulation, and displays
a real-time matplotlib window comparing original observations against encoder-decoder
reconstructions, with per-element error percentages.

Layout (3 columns × 2 rows):
    Left:    Original observation (height scan heatmap + privileged bars)
    Middle:  Decoder-reconstructed data
    Right:   Per-element reconstruction error percentage

Usage:
    ./isaaclab.sh -p scripts/tools/encoder_pretrain/visualize_encoder_reconstruction.py \
        --task Go2-Loco-Skill-Walk-Mid360Depth-10Hz-PretrainTeacher \
        --checkpoint logs/rsl_rl/Go2-Loco-Skill-Walk-Mid360Depth-10Hz/{run}/model_16000.pt \
        --encoder_checkpoint logs/rsl_rl/Go2-Loco-Skill-Walk-Mid360Depth-10Hz-GRU/{run}_pretrain_teacher/model_teacher.pt \
        --num_envs 4 --env_id 0

    ./isaaclab.sh -p scripts/tools/encoder_pretrain/visualize_encoder_reconstruction.py \
        --task Go2-Loco-Skill-Walk-Mid360Depth-10Hz-PretrainTeacher \
        --checkpoint logs/rsl_rl/Go2-Loco-Skill-Walk-Mid360Depth-10Hz/2026-07-31_00-15-36/model_16000.pt \
        --encoder_checkpoint logs/rsl_rl/Go2-Loco-Skill-Walk-Mid360Depth-10Hz-GRU/2026-08-04_23-24-32_pretrain_teacher/model_teacher.pt \
        --num_envs 4 --env_id 0

Keyboard controls:
    Space / Enter  — step forward
    q / Esc        — quit
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import os
import sys

from isaaclab.app import AppLauncher

# local imports (add rsl_rl dir to path to avoid ROS2 "scripts" package collision)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "reinforcement_learning", "rsl_rl"))
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(
    description="Visualize AutoEncoder reconstruction quality for teacher encoders."
)
parser.add_argument("--task", type=str, default="Go2-Loco-Skill-Walk-Mid360Depth-10Hz-PretrainTeacher",
                    help="Name of the task.")
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point",
                    help="Name of the RL agent configuration entry point.")
parser.add_argument("--encoder_checkpoint", type=str, default=None,
                    help="Path to pretrained encoder weights (model_teacher.pt or encoder .pt files).")
parser.add_argument("--num_envs", type=int, default=4, help="Number of environments.")
parser.add_argument("--env_id", type=int, default=0, help="Which env index to visualize.")
parser.add_argument("--fps", type=int, default=10, help="Visualization update rate.")
# append RSL-RL cli arguments (required by update_rsl_rl_cfg, includes --checkpoint)
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# validate required arguments
if not args_cli.checkpoint:
    parser.error("--checkpoint is required.")
if not args_cli.encoder_checkpoint:
    parser.error("--encoder_checkpoint is required.")

# force headless for this visualization tool
args_cli.headless = True

# ---- DEBUG: CLI arguments ----
print("=" * 60)
print("[DEBUG] CLI Arguments:")
for k, v in sorted(vars(args_cli).items()):
    print(f"  {k}: {v}")
print("=" * 60)

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app (headless)
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import numpy as np
import torch
from rsl_rl.runners import OnPolicyRunner

import gymnasium as gym
import matplotlib
matplotlib.use("TkAgg")  # GUI window via X11 forwarding (MobaXterm)
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

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


# ------------------------------------------------------------------
# Privileged observation term labels (18 terms, each with 3-frame history = 54 dims)
# ------------------------------------------------------------------
_PRIV_TERM_NAMES = [
    # (name, n_dims_per_frame)
    ("gait_trot_mask",     2),   # idx  0-1
    ("feet_contact_mask",  4),   # idx  2-5
    ("base_lin_vel",       3),   # idx  6-8
    ("feet_distance",      4),   # idx  9-12
    ("base_height",        1),   # idx 13
    ("FL_foot_height",     1),   # idx 14
    ("FR_foot_height",     1),   # idx 15
    ("RL_foot_height",     1),   # idx 16
    ("RR_foot_height",     1),   # idx 17
]

_HISTORY_LENGTH = 3


def _build_priv_labels() -> list[str]:
    """Build per-dimension labels for the 54-dim privileged vector."""
    labels = []
    for name, ndim in _PRIV_TERM_NAMES:
        for h in range(_HISTORY_LENGTH):
            suffix = f"_h{h}" if _HISTORY_LENGTH > 1 else ""
            if ndim == 1:
                labels.append(f"{name}{suffix}")
            else:
                for i in range(ndim):
                    labels.append(f"{name}[{i}]{suffix}")
    return labels


PRIV_LABELS = _build_priv_labels()  # 54 strings


def _load_encoder_weights(
    checkpoint_path: str, hs_ae: HeightScanEncoder, priv_ae: PrivilegeEncoder, device: torch.device
) -> None:
    """Load encoder weights from a checkpoint file.

    Supports:
        - model_teacher.pt (combined checkpoint with keys "height_scan_encoder", "privilege_encoder")
        - Individual height_scan_encoder.pt / privilege_encoder.pt files

    """
    state = torch.load(checkpoint_path, weights_only=True, map_location=device)

    if "height_scan_encoder" in state:
        hs_ae.encoder.load_state_dict(state["height_scan_encoder"])
        print("[INFO] Loaded height_scan_encoder from model_teacher.pt")
    else:
        hs_ae.encoder.load_state_dict(state)
        print("[INFO] Loaded height_scan_encoder from standalone .pt")

    if "privilege_encoder" in state:
        priv_ae.encoder.load_state_dict(state["privilege_encoder"])
        print("[INFO] Loaded privilege_encoder from model_teacher.pt")
    elif "height_scan_encoder" in state:
        # model_teacher.pt but no privilege_encoder key → try separate file
        priv_path = os.path.join(os.path.dirname(checkpoint_path), "privilege_encoder.pt")
        if os.path.exists(priv_path):
            priv_state = torch.load(priv_path, weights_only=True, map_location=device)
            priv_ae.encoder.load_state_dict(priv_state)
            print(f"[INFO] Loaded privilege_encoder from: {priv_path}")
        else:
            print("[WARN] privilege_encoder not found in checkpoint or directory.")
    else:
        print("[WARN] Could not determine encoder weight format.")


# ------------------------------------------------------------------
# Matplotlib visualization setup
# ------------------------------------------------------------------


def _setup_figure() -> tuple[plt.Figure, dict[str, plt.Axes]]:
    """Create the 3×2 visualization figure (TkAgg backend).

    Returns:
        Tuple of (figure, dict of named axes).

    """
    plt.ion()
    fig = plt.figure("Encoder Reconstruction Visualizer", figsize=(18, 10))
    gs = GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.30)

    axes = {
        "hs_orig":   fig.add_subplot(gs[0, 0]),
        "hs_recon":  fig.add_subplot(gs[0, 1]),
        "hs_error":  fig.add_subplot(gs[0, 2]),
        "priv_orig":  fig.add_subplot(gs[1, 0]),
        "priv_recon": fig.add_subplot(gs[1, 1]),
        "priv_error": fig.add_subplot(gs[1, 2]),
    }

    axes["hs_orig"].set_title("Height Scan — Original")
    axes["hs_recon"].set_title("Height Scan — Reconstructed")
    axes["hs_error"].set_title("Height Scan — Error %")
    axes["priv_orig"].set_title("Privileged — Original")
    axes["priv_recon"].set_title("Privileged — Reconstructed")
    axes["priv_error"].set_title("Privileged — Error %")

    return fig, axes


def _update_heatmap(
    ax: plt.Axes, data: np.ndarray, im: object | None, title: str, vmin: float, vmax: float,
    cmap: str = "viridis", is_error: bool = False,
) -> object:
    """Update a 2D heatmap subplot.

    Args:
        ax: Matplotlib axes.
        data: 2D array of shape ``(11, 17)``.
        im: Existing ``AxesImage``, or None for first draw.
        title: Subplot title.
        vmin, vmax: Colorbar range.
        cmap: Colormap name.
        is_error: If True, annotate cells with percentage values.

    Returns:
        Updated ``AxesImage``.

    """
    if im is None:
        im = ax.imshow(data, aspect="auto", origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    else:
        im.set_data(data)
        im.set_clim(vmin, vmax)
    ax.set_title(title, fontsize=9)

    if is_error:
        # remove old text annotations
        for txt in getattr(ax, "_error_texts", []):
            txt.remove()
        ax._error_texts = []
        for r in range(data.shape[0]):
            for c in range(data.shape[1]):
                txt = ax.text(c, r, f"{data[r, c]:.1f}", ha="center", va="center",
                              fontsize=5, color="white",
                              bbox=dict(boxstyle="round,pad=0.1", facecolor="black", alpha=0.4))
                ax._error_texts.append(txt)

    return im


def _update_bar(
    ax: plt.Axes, values: np.ndarray, bars: object | None, title: str, ylim: tuple,
    color: str = "steelblue",
) -> object:
    """Update a bar chart subplot.

    Args:
        ax: Matplotlib axes.
        values: 1D array of shape ``(54,)``.
        bars: Existing ``BarContainer``, or None.
        title: Subplot title.
        ylim: Y-axis range ``(min, max)``.
        color: Bar color.

    Returns:
        Updated ``BarContainer``.

    """
    if bars is None:
        x = np.arange(len(values))
        bars = ax.bar(x, values, color=color, width=0.8)
        ax.set_xticks(x[::6])  # label every 6th to avoid crowding
        ax.set_xticklabels([PRIV_LABELS[i] for i in x[::6]], rotation=90, fontsize=5)
        ax.set_ylim(*ylim)
    else:
        for bar, val in zip(bars, values):
            bar.set_height(val)
    ax.set_title(title, fontsize=9)
    return bars


def _compute_error_pct(orig: np.ndarray, recon: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Compute per-element absolute percentage error.

    Uses the original value range as the denominator to avoid division by zero.

    Args:
        orig: Original values.
        recon: Reconstructed values.
        eps: Small constant to avoid division by zero.

    Returns:
        Error percentage array, same shape as input.

    """
    denom = np.maximum(np.abs(orig), eps)
    return np.abs(orig - recon) / denom * 100.0


@hydra_task_config(args_cli.task, args_cli.agent)
def main(
    env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg,
    agent_cfg: RslRlBaseRunnerCfg,
):
    """Visualize encoder reconstruction quality in real time.

    Args:
        env_cfg: The environment configuration.
        agent_cfg: The RL agent configuration.

    """
    # ------------------------------------------------------------------
    # 1. Resolve checkpoint paths
    # ------------------------------------------------------------------
    resume_path = retrieve_file_path(args_cli.checkpoint)
    encoder_path = retrieve_file_path(args_cli.encoder_checkpoint)
    print(f"[INFO] Teacher checkpoint:  {resume_path}")
    print(f"[INFO] Encoder checkpoint:  {encoder_path}")

    # ------------------------------------------------------------------
    # 2. Create environment
    # ------------------------------------------------------------------
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    env_cfg.log_dir = os.path.dirname(resume_path)

    env = gym.make(args_cli.task, cfg=env_cfg)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    # ---- DEBUG: Environment info ----
    print("=" * 60)
    print("[DEBUG] Environment:")
    print(f"  num_envs:  {env.num_envs}")
    print(f"  device:    {env.unwrapped.device}")
    print(f"  task:      {args_cli.task}")
    print(f"  env_id:    {args_cli.env_id}")
    print("=" * 60)

    # ------------------------------------------------------------------
    # 3. Load teacher policy
    # ------------------------------------------------------------------
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(resume_path)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    try:
        policy_nn = runner.alg.policy
    except AttributeError:
        policy_nn = runner.alg.actor_critic

    ckpt = torch.load(resume_path, weights_only=False, map_location="cpu")
    print(f"[INFO] Teacher checkpoint iter: {ckpt.get('iter', 'N/A')}")
    teacher_params = sum(p.numel() for p in policy_nn.parameters())
    print(f"[INFO] Teacher params: {teacher_params:,}")

    # ------------------------------------------------------------------
    # 4. Load encoder weights
    # ------------------------------------------------------------------
    device = env.unwrapped.device
    hs_ae = HeightScanEncoder().to(device)
    priv_ae = PrivilegeEncoder().to(device)
    _load_encoder_weights(encoder_path, hs_ae, priv_ae, device)
    hs_ae.eval()
    priv_ae.eval()

    print("[DEBUG] Encoder Models:")
    print(f"  HeightScanEncoder: {sum(p.numel() for p in hs_ae.parameters()):,} params, latent_dim={hs_ae.latent_dim}")
    print(f"  PrivilegeEncoder:  {sum(p.numel() for p in priv_ae.parameters()):,} params, latent_dim={priv_ae.latent_dim}, input_dim={priv_ae.input_dim}")

    # ------------------------------------------------------------------
    # 5. Setup matplotlib figure
    # ------------------------------------------------------------------
    fig, axes = _setup_figure()

    hs_im_orig = None
    hs_im_recon = None
    hs_im_error = None
    priv_bars_orig = None
    priv_bars_recon = None
    priv_bars_error = None

    env_id = args_cli.env_id
    if env_id >= env.num_envs:
        raise ValueError(f"--env_id={env_id} >= num_envs={env.num_envs}")

    # ------------------------------------------------------------------
    # 6. Main visualization loop
    # ------------------------------------------------------------------
    obs = env.get_observations()
    step = 0
    auto_run = False

    print("\n[INFO] Visualization ready.")
    print("       Space/Enter = step  |  r = toggle auto-run  |  q/Esc = quit\n")

    def _step_simulation():
        """Advance simulation by one step."""
        nonlocal step, obs
        with torch.inference_mode():
            actions = policy(obs)
            obs, _, dones, _ = env.step(actions)
            policy_nn.reset(dones)
        step += 1

    def _update_plots():
        """Recompute reconstructions and refresh all subplots."""
        nonlocal hs_im_orig, hs_im_recon, hs_im_error
        nonlocal priv_bars_orig, priv_bars_recon, priv_bars_error

        ms_cpu = obs["mapScans"][env_id].cpu().numpy()
        priv_cpu = obs["privileged"][env_id].cpu().numpy()

        ms_t = obs["mapScans"][env_id:env_id + 1].to(device)
        priv_t = obs["privileged"][env_id:env_id + 1].to(device)

        with torch.no_grad():
            ms_recon = hs_ae(ms_t).squeeze(0).cpu().numpy()
            priv_recon = priv_ae(priv_t).squeeze(0).cpu().numpy()

        ms_2d = ms_cpu.reshape(11, 17)
        ms_recon_2d = ms_recon.reshape(11, 17)
        hs_error_2d = _compute_error_pct(ms_2d, ms_recon_2d)
        priv_error_1d = _compute_error_pct(priv_cpu, priv_recon)

        hs_mse = np.mean((ms_2d - ms_recon_2d) ** 2)
        priv_mse = np.mean((priv_cpu - priv_recon) ** 2)

        hs_im_orig = _update_heatmap(
            axes["hs_orig"], ms_2d, hs_im_orig,
            f"Height Scan Orig (step {step})", vmin=-1.5, vmax=1.5,
        )
        hs_im_recon = _update_heatmap(
            axes["hs_recon"], ms_recon_2d, hs_im_recon,
            f"Height Scan Recon (step {step})", vmin=-1.5, vmax=1.5,
        )
        hs_im_error = _update_heatmap(
            axes["hs_error"], hs_error_2d, hs_im_error,
            f"Height Scan Error % (step {step})", vmin=0, vmax=100,
            cmap="hot", is_error=True,
        )
        priv_bars_orig = _update_bar(
            axes["priv_orig"], priv_cpu, priv_bars_orig,
            f"Privileged Orig (step {step})", ylim=(-12, 12), color="steelblue",
        )
        priv_bars_recon = _update_bar(
            axes["priv_recon"], priv_recon, priv_bars_recon,
            f"Privileged Recon (step {step})", ylim=(-12, 12), color="darkorange",
        )
        priv_bars_error = _update_bar(
            axes["priv_error"], priv_error_1d, priv_bars_error,
            f"Privileged Error % (step {step})", ylim=(0, 100), color="crimson",
        )

        fig.suptitle(
            f"Step {step} | "
            f"HS MSE: {hs_mse:.4f}  (±{np.mean(hs_error_2d):.1f}%) | "
            f"Priv MSE: {priv_mse:.4f}  (±{np.mean(priv_error_1d):.1f}%) | "
            f"Mode: {'AUTO' if auto_run else 'STEP'}",
            fontsize=11, fontweight="bold",
        )


# ------------------------------------------------------------------
# 6. Main visualization loop
# ------------------------------------------------------------------

    print("\n[INFO] Opening TkAgg visualization window (via X11 forwarding)...")
    print("       Space = step  |  r = toggle auto-run  |  q = quit\n")

    # key-press callback for matplotlib TkAgg
    _last_key = [None]

    def _on_key(event):
        _last_key[0] = event.key

    fig.canvas.mpl_connect("key_press_event", _on_key)

    # initial draw
    _update_plots()

    while simulation_app.is_running():
        key = _last_key[0]
        _last_key[0] = None

        if key is not None:
            key = key.lower()
            if key in ("q", "escape"):
                break
            elif key == "r":
                auto_run = not auto_run
                mode_str = "AUTO" if auto_run else "STEP"
                print(f"[INFO] Toggled to {mode_str} mode")
                continue
            elif key in (" ", "space", "enter"):
                _step_simulation()
                _update_plots()
                continue

        if auto_run:
            _step_simulation()
            _update_plots()
            plt.pause(0.03)  # ~33 FPS
        else:
            plt.pause(0.1)  # idle

    env.close()
    print("\n[INFO] Done.")


if __name__ == "__main__":
    main()
    simulation_app.close()
