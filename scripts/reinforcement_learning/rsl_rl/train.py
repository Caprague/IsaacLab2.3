# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to train RL agent with RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
parser.add_argument(
    "--distributed", action="store_true", default=False, help="Run training with multiple GPUs or nodes."
)
parser.add_argument("--export_io_descriptors", action="store_true", default=False, help="Export IO descriptors.")
parser.add_argument(
    "--ray-proc-id", "-rid", type=int, default=None, help="Automatically configured by Ray integration, otherwise None."
)
parser.add_argument(
    "--verify_obs_layout",
    action="store_true",
    default=False,
    help="Temporary runtime probe: dump real observation layout (term-major vs frame-major) and exit.",
)
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Check for minimum supported RSL-RL version."""

import importlib.metadata as metadata
import platform

from packaging import version

# check minimum supported rsl-rl version
RSL_RL_VERSION = "3.0.1"
installed_version = metadata.version("rsl-rl-lib")
if version.parse(installed_version) < version.parse(RSL_RL_VERSION):
    if platform.system() == "Windows":
        cmd = [r".\isaaclab.bat", "-p", "-m", "pip", "install", f"rsl-rl-lib=={RSL_RL_VERSION}"]
    else:
        cmd = ["./isaaclab.sh", "-p", "-m", "pip", "install", f"rsl-rl-lib=={RSL_RL_VERSION}"]
    print(
        f"Please install the correct version of RSL-RL.\nExisting version is: '{installed_version}'"
        f" and required version is: '{RSL_RL_VERSION}'.\nTo install the correct version, run:"
        f"\n\n\t{' '.join(cmd)}\n"
    )
    exit(1)

"""Rest everything follows."""

import logging
import os
import time
from datetime import datetime

import gymnasium as gym
import torch
from rsl_rl.runners import DistillationRunner, OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_yaml

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

# import logger
logger = logging.getLogger(__name__)

# PLACEHOLDER: Extension template (do not remove this comment)

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


def _classify_proprio_layout(t: torch.Tensor) -> None:
    """Phase-pair energy test: (sin*3)^2 + (cos*3)^2 == 9 for real phase pairs."""
    n = t.shape[-1]
    if n != 235:
        print(f"[PROBE] proprioception dims {n} != 235; skip phase classifier")
        return
    h, per_frame = 5, 47
    term_major_idx = [(2 * k, 2 * k + 1) for k in range(h)]  # term-major: phase block flat[0:10]
    frame_major_idx = [(k * per_frame, k * per_frame + 1) for k in range(h)]  # frame-major: step starts

    def energy(idxs):
        vals = [float(t[0, i] ** 2 + t[0, j] ** 2) for i, j in idxs]
        return sum(vals) / len(vals)

    e_tm, e_fm = energy(term_major_idx), energy(frame_major_idx)
    verdict = "TERM-MAJOR" if abs(e_tm - 9.0) < abs(e_fm - 9.0) else "FRAME-MAJOR"
    print(
        "[PROBE] proprioception phase-pair energy: "
        f"term-major={e_tm:.3f} frame-major={e_fm:.3f} (phase scale=3 -> expect 9.0)"
    )
    print(f"[PROBE] >>> proprioception layout verdict: {verdict}")


def _classify_privileged_layout(t: torch.Tensor) -> None:
    """Contact-mask binary test: feet_contact_mask values are exactly 0/3 (scale=3)."""
    n = t.shape[-1]
    if n != 54:
        print(f"[PROBE] privileged dims {n} != 54; skip contact classifier")
        return

    def binary_fraction(slices):
        vals = torch.cat([t[0, sl] for sl in slices])
        return float(((vals == 0.0) | (vals == 3.0)).float().mean())

    # term-major: contact block flat[6:18] (4/frame x 3 frames)
    f_tm = binary_fraction([slice(6, 18)])
    # frame-major: contact at flat[2:6], [20:24], [38:42]
    f_fm = binary_fraction([slice(2, 6), slice(20, 24), slice(38, 42)])
    verdict = "TERM-MAJOR" if f_tm > f_fm else "FRAME-MAJOR"
    print(
        "[PROBE] privileged contact-mask binary fraction: "
        f"term-major={f_tm:.3f} frame-major={f_fm:.3f}"
    )
    print(f"[PROBE] >>> privileged layout verdict: {verdict}")


def _verify_obs_layout(env, probe_steps: int = 12) -> None:
    """Temporary runtime probe: dump real observation layout and classify term/frame-major.

    Only used for debugging via ``--verify_obs_layout``; exits before training.
    """
    print("\n" + "=" * 90)
    print("[PROBE] Observation-layout verification (temporary runtime probe)")
    device = env.device

    # 1) Fill history buffers with a few dummy steps
    obs = env.get_observations()
    for _ in range(probe_steps):
        dummy = torch.zeros(env.num_envs, env.num_actions, device=device)
        obs, _, _, _ = env.step(dummy)
    print(f"[PROBE] Filled history with {probe_steps} dummy steps.")

    # 2) Observation-manager config summary (expected layout)
    mgr = getattr(env.unwrapped, "observation_manager", None)
    if mgr is not None:
        try:
            term_names = getattr(mgr, "group_obs_term_names", None) or getattr(
                mgr, "_group_obs_term_names", {}
            )
            term_dims = getattr(mgr, "group_obs_term_dim", None) or getattr(
                mgr, "_group_obs_term_dim", {}
            )
            concat = getattr(mgr, "group_obs_concatenate", None) or getattr(
                mgr, "_group_obs_concatenate", {}
            )
            term_cfgs = getattr(mgr, "_group_obs_term_cfgs", {})
        except Exception as err:
            print(f"[PROBE] cannot read manager internals: {err}")
            term_names, term_dims, concat, term_cfgs = {}, {}, {}, {}
        for group_name in ("proprioception", "privileged", "proprioception_noised"):
            if group_name not in term_names:
                continue
            print(f"[PROBE] config[{group_name}] terms={term_names[group_name]}")
            print(f"          per-term dims={term_dims.get(group_name)}")
            print(f"          concatenate_terms={concat.get(group_name)}")
            if group_name in term_cfgs:
                flags = [
                    (getattr(c, "flatten_history_dim", "?"), getattr(c, "history_length", "?"))
                    for c in term_cfgs[group_name]
                ]
                print(f"          per-term (flatten_history_dim, history_length)={flags}")
    else:
        print("[PROBE] observation_manager not found (non-manager env); skip config summary.")

    # 3) Real flat value samples
    for group_name in ("proprioception", "privileged", "proprioception_noised"):
        if group_name in obs:
            t = obs[group_name]
            print(f"[PROBE] obs[{group_name}] shape={tuple(t.shape)}")
            if t.dim() == 2:
                print(f"          first 24 values: {t[0, :24].tolist()}")
            elif t.dim() == 3:
                print(f"          last frame first 24: {t[0, -1, :24].tolist()}")

    # 4) Empirical layout classification
    if "proprioception" in obs and obs["proprioception"].dim() == 2:
        _classify_proprio_layout(obs["proprioception"])
    if "privileged" in obs and obs["privileged"].dim() == 2:
        _classify_privileged_layout(obs["privileged"])
    print("[PROBE] Done.")
    print("=" * 90)


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Train with RSL-RL agent."""
    # override configurations with non-hydra CLI arguments
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    agent_cfg.max_iterations = (
        args_cli.max_iterations if args_cli.max_iterations is not None else agent_cfg.max_iterations
    )

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    # check for invalid combination of CPU device with distributed training
    if args_cli.distributed and args_cli.device is not None and "cpu" in args_cli.device:
        raise ValueError(
            "Distributed training is not supported when using CPU device. "
            "Please use GPU device (e.g., --device cuda) for distributed training."
        )

    # multi-gpu training configuration
    if args_cli.distributed:
        env_cfg.sim.device = f"cuda:{app_launcher.local_rank}"
        agent_cfg.device = f"cuda:{app_launcher.local_rank}"

        # set seed to have diversity in different threads
        seed = agent_cfg.seed + app_launcher.local_rank
        env_cfg.seed = seed
        agent_cfg.seed = seed

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    # specify directory for logging runs: {time-stamp}_{run_name}
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # The Ray Tune workflow extracts experiment name using the logging line below, hence, do not
    # change it (see PR #2346, comment-2819298849)
    print(f"Exact experiment name requested from command line: {log_dir}")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)

    # set the IO descriptors export flag if requested
    if isinstance(env_cfg, ManagerBasedRLEnvCfg):
        env_cfg.export_io_descriptors = args_cli.export_io_descriptors
    else:
        logger.warning(
            "IO descriptors are only supported for manager based RL environments. No IO descriptors will be exported."
        )

    # set the log directory for the environment (works for all environment types)
    env_cfg.log_dir = log_dir

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # save resume path before creating a new log_dir
    if agent_cfg.resume or agent_cfg.algorithm.class_name == "Distillation":
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    start_time = time.time()

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    # TEMP: runtime observation-layout verification probe (exits before training)
    if args_cli.verify_obs_layout:
        _verify_obs_layout(env)
        print("[PROBE] Exiting without training (--verify_obs_layout).")
        return

    # create runner from rsl-rl
    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    # write git state to logs
    runner.add_git_repo_to_log(__file__)
    # load the checkpoint
    if agent_cfg.resume or agent_cfg.algorithm.class_name == "Distillation":
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        # load previously trained model
        runner.load(resume_path)

    # dump the configuration into log-directory
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)

    # run training
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)

    print(f"Training time: {round(time.time() - start_time, 2)} seconds")

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
