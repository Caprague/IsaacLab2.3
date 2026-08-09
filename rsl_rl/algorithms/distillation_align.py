# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Distillation algorithm with explicit teacher-student latent alignment.

Extends :class:`Distillation` with two latent-alignment losses (Parkour-style
feature alignment), combined with the behavior-cloning loss:

    total_loss = behavior_loss
               + w_depth * MSE(student_depth_latent,   teacher_scan_latent.detach())
               + w_priv  * MSE(student_privilege_latent, teacher_privilege_latent.detach())

The teacher latents are frozen (detached) so gradients only flow into the
student (CNN + GRU + latents + student MLP). This makes the distillation
target more stable and controllable than pure behavior cloning.

The policy must expose ``get_student_latents()`` and ``get_teacher_latents(obs)``
(implemented by :class:`StudentTeacherDepthImageRecurrent`).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from rsl_rl.algorithms import Distillation


class DistillationAlign(Distillation):
    """Distillation with behavior cloning + teacher-student latent alignment."""

    def __init__(
        self,
        policy,
        num_learning_epochs: int = 1,
        gradient_length: int = 15,
        learning_rate: float = 1e-3,
        max_grad_norm: float | None = None,
        loss_type: str = "mse",
        optimizer: str = "adam",
        device: str = "cpu",
        align_weight_depth: float = 1.0,
        align_weight_privilege: float = 1.0,
        multi_gpu_cfg: dict | None = None,
    ) -> None:
        super().__init__(
            policy=policy,
            num_learning_epochs=num_learning_epochs,
            gradient_length=gradient_length,
            learning_rate=learning_rate,
            max_grad_norm=max_grad_norm,
            loss_type=loss_type,
            optimizer=optimizer,
            device=device,
            multi_gpu_cfg=multi_gpu_cfg,
        )

        self.align_weight_depth = align_weight_depth
        self.align_weight_privilege = align_weight_privilege
        print(
            f"[DistillationAlign] align_weight_depth={align_weight_depth}, "
            f"align_weight_privilege={align_weight_privilege}"
        )

        assert hasattr(policy, "get_student_latents"), (
            "DistillationAlign requires the policy to implement get_student_latents()."
        )
        assert hasattr(policy, "get_teacher_latents"), (
            "DistillationAlign requires the policy to implement get_teacher_latents(obs)."
        )
        print("[DistillationAlign] policy latent alignment interface: OK")

    def update(self) -> dict[str, float]:
        """Update the student policy with BC + latent-alignment losses."""
        self.num_updates += 1
        mean_behavior_loss = 0
        mean_align_depth = 0
        mean_align_privilege = 0
        loss = 0
        cnt = 0

        for epoch in range(self.num_learning_epochs):
            self.policy.reset(hidden_states=self.last_hidden_states)
            self.policy.detach_hidden_states()
            for obs, _, privileged_actions, dones in self.storage.generator():
                # Inference of the student for gradient computation
                actions = self.policy.act_inference(obs)

                # Behavior cloning loss
                behavior_loss = self.loss_fn(actions, privileged_actions)

                # Latent alignment losses (teacher latents detached inside the policy)
                student_depth, student_priv = self.policy.get_student_latents()
                teacher_depth, teacher_priv = self.policy.get_teacher_latents(obs)
                align_depth = self.align_weight_depth * self.loss_fn(student_depth, teacher_depth)
                align_privilege = self.align_weight_privilege * self.loss_fn(student_priv, teacher_priv)

                # Total loss
                loss = loss + behavior_loss + align_depth + align_privilege
                mean_behavior_loss += behavior_loss.item()
                mean_align_depth += align_depth.item()
                mean_align_privilege += align_privilege.item()
                cnt += 1

                # Gradient step
                if cnt % self.gradient_length == 0:
                    self.optimizer.zero_grad()
                    loss.backward()
                    if self.is_multi_gpu:
                        self.reduce_parameters()
                    if self.max_grad_norm:
                        nn.utils.clip_grad_norm_(self.policy.student.parameters(), self.max_grad_norm)
                    self.optimizer.step()
                    self.policy.detach_hidden_states()
                    loss = 0

                # Reset dones
                self.policy.reset(dones.view(-1))
                self.policy.detach_hidden_states(dones.view(-1))

        mean_behavior_loss /= cnt
        mean_align_depth /= cnt
        mean_align_privilege /= cnt
        self.storage.clear()
        self.last_hidden_states = self.policy.get_hidden_states()
        self.policy.detach_hidden_states()

        # Construct the loss dictionary
        loss_dict = {
            "behavior": mean_behavior_loss,
            "align_depth": mean_align_depth,
            "align_privilege": mean_align_privilege,
        }

        return loss_dict
