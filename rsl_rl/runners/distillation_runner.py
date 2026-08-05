# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import os
import time
import torch
from collections import deque
from tensordict import TensorDict

import rsl_rl
from rsl_rl.algorithms import Distillation
from rsl_rl.env import VecEnv
from rsl_rl.modules import StudentTeacher, StudentTeacherRecurrent, StudentTeacherDepthImage, StudentTeacherDepthImageRecurrent
from rsl_rl.runners import OnPolicyRunner
from rsl_rl.utils import resolve_obs_groups, store_code_state


class DistillationRunner(OnPolicyRunner):
    """On-policy runner for training and evaluation of teacher-student training."""

    def __init__(self, env: VecEnv, train_cfg: dict, log_dir: str | None = None, device: str = "cpu") -> None:
        self.cfg = train_cfg
        self.alg_cfg = train_cfg["algorithm"]
        self.policy_cfg = train_cfg["policy"]
        self.device = device
        self.env = env

        # Check if multi-GPU is enabled
        self._configure_multi_gpu()

        # Store training configuration
        self.num_steps_per_env = self.cfg["num_steps_per_env"]
        self.save_interval = self.cfg["save_interval"]

        # Teacher driving configuration (DAgger-style)
        self.teacher_driving = self.cfg.get("teacher_driving", False)
        self.teacher_driving_switch_iter = self.cfg.get("teacher_driving_switch_iter", 0)

        # Query observations from environment for algorithm construction
        obs = self.env.get_observations()
        self.cfg["obs_groups"] = resolve_obs_groups(obs, self.cfg["obs_groups"], default_sets=["teacher"])

        # Create the algorithm
        self.alg = self._construct_algorithm(obs)

        # Decide whether to disable logging
        # Note: We only log from the process with rank 0 (main process)
        self.disable_logs = self.is_distributed and self.gpu_global_rank != 0

        # Logging
        self.log_dir = log_dir
        self.writer = None
        self.tot_timesteps = 0
        self.tot_time = 0
        self.current_learning_iteration = 0
        self.git_status_repos = [rsl_rl.__file__]

    def load(self, path: str, load_optimizer: bool = True, map_location: str | None = None) -> dict:
        """Load model checkpoint, including pretrained teacher encoder weights.

        Extends ``OnPolicyRunner.load()`` to also load ``HeightScanEncoder`` and
        ``PrivilegeEncoder`` weights from the checkpoint (e.g., ``model_teacher.pt``
        produced by ``pretrain_teacher_encoders.py``). These weights are loaded and
        frozen for distillation training.

        When loading from a teacher PPO checkpoint (``resumed_training == False``),
        the encoder weights are loaded from the top-level checkpoint keys.
        When resuming from a distillation checkpoint (``resumed_training == True``),
        the encoders are already part of ``model_state_dict`` and need no separate
        loading.

        Args:
            path: Path to the checkpoint file.
            load_optimizer: Whether to load the optimizer state. Defaults to ``True``.
            map_location: Device to map tensors to. Defaults to ``None``.

        Returns:
            The ``infos`` dict from the checkpoint.

        """
        loaded_dict = torch.load(path, weights_only=False, map_location=map_location)

        # Load model (teacher MLP from PPO, or full student from distillation checkpoint)
        resumed_training = self.alg.policy.load_state_dict(loaded_dict["model_state_dict"])

        # Load optimizer if resuming from a distillation checkpoint
        if load_optimizer and resumed_training and "optimizer_state_dict" in loaded_dict:
            self.alg.optimizer.load_state_dict(loaded_dict["optimizer_state_dict"])
        if resumed_training:
            self.current_learning_iteration = loaded_dict["iter"]

        # Load pretrained teacher encoder weights (only when loading from a teacher
        # checkpoint, not resuming from a student checkpoint where encoders are
        # already in model_state_dict)
        if not resumed_training and hasattr(self.alg.policy, "height_scan_encoder"):
            enc_loaded = False
            if "height_scan_encoder" in loaded_dict:
                self.alg.policy.height_scan_encoder.load_state_dict(
                    loaded_dict["height_scan_encoder"]
                )
                enc_loaded = True
            if "privilege_encoder" in loaded_dict:
                self.alg.policy.privilege_encoder.load_state_dict(
                    loaded_dict["privilege_encoder"]
                )
                enc_loaded = True
            if enc_loaded:
                # Freeze encoder parameters for distillation
                for p in self.alg.policy.height_scan_encoder.parameters():
                    p.requires_grad = False
                for p in self.alg.policy.privilege_encoder.parameters():
                    p.requires_grad = False
                self.alg.policy.height_scan_encoder.eval()
                self.alg.policy.privilege_encoder.eval()
                print("[INFO] Loaded and froze pretrained teacher encoders from checkpoint")
            else:
                print("[WARN] Checkpoint does not contain pretrained encoder weights!")
                print("[WARN] Teacher encoders will remain randomly initialized.")
                print("[WARN] Run pretrain_teacher_encoders.py to generate encoder weights first.")

        return loaded_dict.get("infos", {})

    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False) -> None:
        # Initialize writer
        self._prepare_logging_writer()
        # Check if teacher is loaded
        if not self.alg.policy.loaded_teacher:
            raise ValueError("Teacher model parameters not loaded. Please load a teacher model to distill.")

        # Randomize initial episode lengths (for exploration)
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf, high=int(self.env.max_episode_length)
            )

        # Start learning
        obs = self.env.get_observations().to(self.device)
        self.train_mode()  # switch to train mode (for dropout for example)

        # Book keeping
        ep_infos = []
        rewbuffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        # Ensure all parameters are in-synced
        if self.is_distributed:
            print(f"Synchronizing parameters for rank {self.gpu_global_rank}...")
            self.alg.broadcast_parameters()

        # Start training
        start_iter = self.current_learning_iteration
        tot_iter = start_iter + num_learning_iterations
        for it in range(start_iter, tot_iter):
            start = time.time()
            # Switch logging for teacher-to-student driving transition
            if self.teacher_driving and it == self.teacher_driving_switch_iter:
                print(f"[INFO] Switching from teacher driving to student driving at iteration {it}")
            # Rollout
            with torch.inference_mode():
                for _ in range(self.num_steps_per_env):
                    # Sample actions
                    if self.teacher_driving and it < self.teacher_driving_switch_iter:
                        actions = self.alg.teacher_act(obs)
                    else:
                        actions = self.alg.act(obs)
                    # Step the environment
                    obs, rewards, dones, extras = self.env.step(actions.to(self.env.device))
                    # Move to device
                    obs, rewards, dones = (obs.to(self.device), rewards.to(self.device), dones.to(self.device))
                    # Process the step
                    self.alg.process_env_step(obs, rewards, dones, extras)
                    # Book keeping
                    if self.log_dir is not None:
                        if "episode" in extras:
                            ep_infos.append(extras["episode"])
                        elif "log" in extras:
                            ep_infos.append(extras["log"])
                        # Update rewards
                        cur_reward_sum += rewards
                        # Update episode length
                        cur_episode_length += 1
                        # Clear data for completed episodes
                        new_ids = (dones > 0).nonzero(as_tuple=False)
                        rewbuffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                        lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                        cur_reward_sum[new_ids] = 0
                        cur_episode_length[new_ids] = 0

                stop = time.time()
                collection_time = stop - start
                start = stop

            # Update policy
            loss_dict = self.alg.update()

            stop = time.time()
            learn_time = stop - start
            self.current_learning_iteration = it

            if self.log_dir is not None and not self.disable_logs:
                # Log information
                self.log(locals())
                # Save model
                if it % self.save_interval == 0:
                    self.save(os.path.join(self.log_dir, f"model_{it}.pt"))

            # Clear episode infos
            ep_infos.clear()
            # Save code state
            if it == start_iter and not self.disable_logs:
                # Obtain all the diff files
                git_file_paths = store_code_state(self.log_dir, self.git_status_repos)
                # If possible store them to wandb or neptune
                if self.logger_type in ["wandb", "neptune"] and git_file_paths:
                    for path in git_file_paths:
                        self.writer.save_file(path)

        # Save the final model after training
        if self.log_dir is not None and not self.disable_logs:
            self.save(os.path.join(self.log_dir, f"model_{self.current_learning_iteration}.pt"))

    def _construct_algorithm(self, obs: TensorDict) -> Distillation:
        """Construct the distillation algorithm."""
        # Initialize the policy
        student_teacher_class = eval(self.policy_cfg.pop("class_name"))
        student_teacher: StudentTeacher | StudentTeacherRecurrent | StudentTeacherDepthImage | StudentTeacherDepthImageRecurrent = student_teacher_class(
            obs, self.cfg["obs_groups"], self.env.num_actions, **self.policy_cfg
        ).to(self.device)

        # Initialize the algorithm
        alg_class = eval(self.alg_cfg.pop("class_name"))
        alg: Distillation = alg_class(
            student_teacher, device=self.device, **self.alg_cfg, multi_gpu_cfg=self.multi_gpu_cfg
        )

        # Initialize the storage
        alg.init_storage(
            "distillation",
            self.env.num_envs,
            self.num_steps_per_env,
            obs,
            [self.env.num_actions],
        )

        return alg
