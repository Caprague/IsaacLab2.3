# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Student-Teacher policy with depth image CNN + GRU temporal fusion + teacher distillation."""

from __future__ import annotations

import torch
import torch.nn as nn
import warnings
from tensordict import TensorDict
from torch.distributions import Normal
from typing import Any, NoReturn

from rsl_rl.networks import MLP, EmpiricalNormalization, HiddenState, Memory, ScanEncoder, PrivilegeEncoder
from rsl_rl.networks.student_depth_cnn import StudentDepthCNN


class StudentTeacherDepthImageRecurrent(nn.Module):
    """Student-Teacher policy with depth image CNN + GRU temporal fusion + teacher distillation.

    This module combines three key components:

    1. **Depth CNN encoder** (:class:`StudentDepthCNN`): Processes 360° LiDAR depth images
       of shape ``[B, 1, 32, 180]`` into a compact 32-dimensional latent vector.

    2. **GRU temporal fusion**: Fuses the latest proprioceptive frame with the depth latent
       through a GRU-based memory module, producing two compact latent codes:
       ``depth_latent`` (32, aligned to the teacher's ``scan_latent``) and
       ``privilege_latent`` (32, aligned to the teacher's ``privilege_latent``).
       Auxiliary depth channels (e.g., ``depth_image_age``, the 10Hz/50Hz frame-age
       clock signal) are concatenated into the GRU input as well.

    3. **Teacher-student distillation with latent alignment**: A frozen teacher policy
       (actor MLP + scan encoder + privilege encoder) provides privileged-information
       action targets. During distillation the student's ``depth_latent``/``privilege_latent``
       are regressed to the teacher's frozen ``scan_latent``/``privilege_latent``
       (detached), and combined with the behavior-cloning loss (see ``DistillationAlign``).

    Architecture::

        Depth image [B, 1, 32, 180] ──► StudentDepthCNN ──► [B, 32] depth_latent
                                                                          │
        Proprio latest [B, P] ────────────────────────────────────────────┤
            │                                                             │
            └──► Concat [B, P + 32] ──► gru_input_mlp ──► [B, 64] ──► Memory(GRU)
                                                                              │
                                                                              ▼
                                                                      gru_output_mlp
                                                                            │
                                                   depth_latent [B,32] + privilege_latent [B,32]
                                                                             │
                                                     Concat with prop_all ──► Student MLP ──► actions

    .. note::
        The teacher scan/privilege encoders are trained jointly inside the teacher policy
        during PPO (see ``ActorCriticScan``). During distillation their weights are loaded
        from the teacher checkpoint (keys ``scan_encoder.*``/``privilege_encoder.*``) and
        frozen together with the teacher MLP.

    Args:
        obs: Example observation TensorDict for inferring dimensions.
        obs_groups: Dictionary mapping observation group names to lists of observation keys.
            Must contain ``"policy"`` and ``"teacher"`` keys.
        num_actions: Number of action dimensions.
        student_obs_normalization: Whether to normalize student observations.
            Defaults to ``False``.
        teacher_obs_normalization: Whether to normalize teacher observations.
            Defaults to ``False``.
        student_hidden_dims: Hidden layer dimensions for the student MLP.
            Defaults to ``[256, 256, 256]``.
        teacher_hidden_dims: Hidden layer dimensions for the teacher MLP.
            Defaults to ``[256, 256, 256]``.
        activation: Activation function name. Defaults to ``"elu"``.
        init_noise_std: Initial standard deviation for action noise.
            Defaults to ``0.1``.
        noise_std_type: Type of action noise parameterization (``"scalar"`` or ``"log"``).
            Defaults to ``"scalar"``.
        rnn_type: Type of recurrent network (``"gru"`` or ``"lstm"``).
            Defaults to ``"lstm"``.
        rnn_hidden_dim: Hidden dimension for the GRU/Memory module.
            Defaults to ``256``.
        rnn_num_layers: Number of recurrent layers. Defaults to ``1``.
        teacher_recurrent: Whether the teacher also uses a recurrent memory.
            Defaults to ``False``.
        **kwargs: Additional keyword arguments (ignored with a warning).

    Attributes:
        is_recurrent: Class-level flag indicating this policy is recurrent. Always ``True``.

    """

    is_recurrent: bool = True

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        num_actions: int,
        student_obs_normalization: bool = False,
        teacher_obs_normalization: bool = False,
        student_hidden_dims: tuple[int] | list[int] = [256, 256, 256],
        teacher_hidden_dims: tuple[int] | list[int] = [256, 256, 256],
        activation: str = "elu",
        init_noise_std: float = 0.1,
        noise_std_type: str = "scalar",
        rnn_type: str = "lstm",
        rnn_hidden_dim: int = 256,
        rnn_num_layers: int = 1,
        teacher_recurrent: bool = False,
        **kwargs: dict[str, Any],
    ) -> None:
        # ── Handle deprecated arguments ──
        if "rnn_hidden_size" in kwargs:
            warnings.warn(
                "The argument `rnn_hidden_size` is deprecated and will be removed in a future version. "
                "Please use `rnn_hidden_dim` instead.",
                DeprecationWarning,
            )
            if rnn_hidden_dim == 256:  # Only override if the new argument is at its default
                rnn_hidden_dim = kwargs.pop("rnn_hidden_size")
        if kwargs:
            print(
                "StudentTeacherDepthImageRecurrent.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs])
            )
        super().__init__()

        # ── Flags ──
        self.loaded_teacher = False
        self.teacher_recurrent = teacher_recurrent
        self._last_student_depth_latent = None
        self._last_student_privilege_latent = None

        # ── Save obs groups ──
        self.obs_groups = obs_groups

        # ── Depth image observation group ──
        self.depth_obs_group = "mid360_depth"
        self.depth_height = 32
        self.depth_width = 180
        self.depth_channels = 1
        self.depth_flat_dim = self.depth_height * self.depth_width * self.depth_channels  # 5760

        # ── Student proprioceptive observation groups (exclude depth) ──
        self.student_prop_groups = [
            g for g in obs_groups["policy"] if g != self.depth_obs_group
        ]
        assert len(self.student_prop_groups) > 0, (
            "Student observation groups must contain at least one non-depth group. "
            f"Got policy groups: {obs_groups['policy']}"
        )

        num_student_basic_obs = 0
        for g in self.student_prop_groups:
            assert len(obs[g].shape) == 3, (
                f"Student proprioceptive observation group '{g}' must be frame-major "
                f"(shape [B, H, D]) so the latest frame can be extracted correctly. "
                f"Set flatten_history_dim=False on the group in the env config. "
                f"Got shape {obs[g].shape}"
            )
            num_student_basic_obs += obs[g].numel() // obs[g].shape[0]

        # ── Dynamically compute proprioceptive features per frame ──
        # The proprioceptive observations must be frame-major (B, H, D): a time-major
        # flatten (B, H*D) then places the latest frame in the last `proprio_per_frame`
        # dims. This requires the env obs group to set flatten_history_dim=False; with
        # the default per-term flatten the group is term-major and a tail slice would
        # NOT yield the latest frame.
        history_length = 6  # Hard-coded: 6-frame history stack
        assert obs[self.student_prop_groups[0]].shape[1] == history_length, (
            f"Frame-major prop group history length ({obs[self.student_prop_groups[0]].shape[1]}) "
            f"must match the hard-coded history length ({history_length})"
        )
        assert num_student_basic_obs > 0, (
            f"Student proprioceptive observation dimension must be > 0, got {num_student_basic_obs}"
        )
        assert num_student_basic_obs % history_length == 0, (
            f"Student proprioceptive observation dimension ({num_student_basic_obs}) "
            f"must be divisible by the history length ({history_length})"
        )
        self.proprio_per_frame = num_student_basic_obs // history_length

        print(f"[StudentTeacherDepthImageRecurrent] num_student_basic_obs: {num_student_basic_obs}")
        print(f"[StudentTeacherDepthImageRecurrent] proprio_per_frame: {self.proprio_per_frame}")

        # Auxiliary depth channels (e.g., depth_image_age for 10Hz/50Hz sync) are appended
        # after the flat depth image. They are fed into the GRU as a clock signal so the
        # recurrent module knows how stale the current depth frame is.
        self.depth_aux_dim = max(0, obs[self.depth_obs_group].shape[-1] - self.depth_flat_dim)
        if self.depth_aux_dim > 0:
            print(f"[StudentTeacherDepthImageRecurrent] depth_aux_dim: {self.depth_aux_dim}")

        # ── Student: Depth CNN ──
        self.depth_cnn = StudentDepthCNN(activation=activation)
        print(f"[StudentTeacherDepthImageRecurrent] Depth CNN: {self.depth_cnn}")

        # ── Student: GRU input MLP (proprio_latest + depth_aux + depth_latent → 64) ──
        # Input: proprio_per_frame (latest frame) + depth_aux_dim (age) + 32 (depth_latent)
        # Output: 64 (compact feature for GRU)
        self.gru_input_mlp = MLP(
            input_dim=self.proprio_per_frame + self.depth_aux_dim + 32,
            output_dim=64,
            hidden_dims=[128],
            activation=activation,
        )
        print(f"[StudentTeacherDepthImageRecurrent] GRU input MLP: {self.gru_input_mlp}")

        # ── Student: Memory (GRU/LSTM) ──
        self.memory_s = Memory(
            input_size=64,
            hidden_dim=rnn_hidden_dim,
            num_layers=rnn_num_layers,
            type=rnn_type,
        )
        print(f"[StudentTeacherDepthImageRecurrent] Memory: {self.memory_s}")

        # ── Student: GRU output MLP (rnn_hidden_dim → depth_latent + privilege_latent) ──
        # The GRU outputs two 32-dim latent codes aligned to the teacher's
        # scan_latent / privilege_latent during distillation (DistillationAlign).
        self.gru_output_mlp = MLP(
            input_dim=rnn_hidden_dim,
            output_dim=64,
            hidden_dims=[128],
            activation=activation,
        )
        print(f"[StudentTeacherDepthImageRecurrent] GRU output MLP: {self.gru_output_mlp}")

        # ── Student: Policy MLP (prop_all + depth_latent + privilege_latent → actions) ──
        num_student_input = num_student_basic_obs + 32 + 32  # prop_all(history) + two latents
        print(f"[StudentTeacherDepthImageRecurrent] num_student_input: {num_student_input}")
        self.student = MLP(
            input_dim=num_student_input,
            output_dim=num_actions,
            hidden_dims=student_hidden_dims,
            activation=activation,
        )
        print(f"[StudentTeacherDepthImageRecurrent] Student MLP: {self.student}")

        # ── Student observation normalization ──
        self.student_obs_normalization = student_obs_normalization
        if student_obs_normalization:
            self.student_obs_normalizer = EmpiricalNormalization(num_student_basic_obs)
        else:
            self.student_obs_normalizer = torch.nn.Identity()

        # ── Teacher: Observation dimensions and scan boundaries ──
        self.teacher_groups = obs_groups["teacher"]
        self.teacher_scan_obs_group = "mapScans"
        self.teacher_privilege_obs_group = "privileged"
        num_teacher_obs = 0
        self._teacher_scan_start: int | None = None
        self._teacher_scan_end: int | None = None
        self._teacher_priv_start: int | None = None
        self._teacher_priv_end: int | None = None
        for obs_group in self.teacher_groups:
            assert len(obs[obs_group].shape) == 2, (
                f"Teacher observation group '{obs_group}' must be 1D (shape [B, D]), "
                f"got shape {obs[obs_group].shape}"
            )
            dim = obs[obs_group].shape[-1]
            if obs_group == self.teacher_scan_obs_group:
                self._teacher_scan_start = num_teacher_obs
                self._teacher_scan_end = num_teacher_obs + dim
            if obs_group == self.teacher_privilege_obs_group:
                self._teacher_priv_start = num_teacher_obs
                self._teacher_priv_end = num_teacher_obs + dim
            num_teacher_obs += dim
        assert self._teacher_scan_start is not None and self._teacher_scan_end is not None, (
            f"Teacher scan observation group '{self.teacher_scan_obs_group}' not found in "
            f"teacher groups {self.teacher_groups}"
        )
        assert self._teacher_priv_start is not None and self._teacher_priv_end is not None, (
            f"Teacher privilege observation group '{self.teacher_privilege_obs_group}' not found in "
            f"teacher groups {self.teacher_groups}"
        )
        print(f"[StudentTeacherDepthImageRecurrent] num_teacher_obs: {num_teacher_obs}")
        num_teacher_obs_encoded = num_teacher_obs - (
            self._teacher_scan_end - self._teacher_scan_start
        ) - (self._teacher_priv_end - self._teacher_priv_start) + 32 + 32
        print(f"[StudentTeacherDepthImageRecurrent] num_teacher_obs_encoded: {num_teacher_obs_encoded}")

        # ── Teacher: Scan encoder (mirror of ActorCriticScan.scan_encoder) ──
        self.teacher_scan_encoder = ScanEncoder(latent_dim=32, activation=activation)
        print(f"[StudentTeacherDepthImageRecurrent] Teacher ScanEncoder: {self.teacher_scan_encoder}")

        # ── Teacher: Privilege encoder (mirror of ActorCriticScan.privilege_encoder) ──
        self.teacher_privilege_encoder = PrivilegeEncoder(
            input_dim=self._teacher_priv_end - self._teacher_priv_start,
            latent_dim=32,
            activation=activation,
        )
        print(
            f"[StudentTeacherDepthImageRecurrent] Teacher PrivilegeEncoder: {self.teacher_privilege_encoder}"
        )

        # ── Teacher: Recurrent memory (optional) ──
        if self.teacher_recurrent:
            self.memory_t = Memory(
                input_size=num_teacher_obs_encoded,
                hidden_dim=rnn_hidden_dim,
                num_layers=rnn_num_layers,
                type=rnn_type,
            )
            print(f"[StudentTeacherDepthImageRecurrent] Teacher Memory: {self.memory_t}")

        # ── Teacher: Policy MLP (proprio + scan_latent + privilege_latent → actions) ──
        self.teacher = MLP(
            input_dim=num_teacher_obs_encoded,
            output_dim=num_actions,
            hidden_dims=teacher_hidden_dims,
            activation=activation,
        )
        self.teacher.eval()
        print(f"[StudentTeacherDepthImageRecurrent] Teacher MLP: {self.teacher}")

        # ── Teacher observation normalization ──
        self.teacher_obs_normalization = teacher_obs_normalization
        if teacher_obs_normalization:
            self.teacher_obs_normalizer = EmpiricalNormalization(num_teacher_obs_encoded)
        else:
            self.teacher_obs_normalizer = torch.nn.Identity()

        # ── Teacher is always frozen during distillation ──
        for param in self.teacher.parameters():
            param.requires_grad = False
        for param in self.teacher_scan_encoder.parameters():
            param.requires_grad = False
        for param in self.teacher_privilege_encoder.parameters():
            param.requires_grad = False

        # ── Action noise ──
        self.noise_std_type = noise_std_type
        if self.noise_std_type == "scalar":
            self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        elif self.noise_std_type == "log":
            self.log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(num_actions)))
        else:
            raise ValueError(
                f"Unknown standard deviation type: {self.noise_std_type}. "
                f"Should be 'scalar' or 'log'"
            )

        # ── Action distribution ──
        # Note: Populated in _update_distribution
        self.distribution = None

        # ── Disable args validation for speedup ──
        Normal.set_default_validate_args(False)

    # ═══════════════════════════════════════════════════════════════════════════
    #  Lifecycle & State Management
    # ═══════════════════════════════════════════════════════════════════════════

    def reset(
        self,
        dones: torch.Tensor | None = None,
        hidden_states: tuple[HiddenState, HiddenState] = (None, None),
    ) -> None:
        """Reset the hidden states of the memory modules.

        Args:
            dones: Boolean tensor indicating which environments have terminated.
                If ``None``, resets all hidden states unconditionally.
            hidden_states: Tuple of ``(student_hidden_state, teacher_hidden_state)``
                to use when resetting unconditionally.
        """
        self.memory_s.reset(dones, hidden_states[0])
        if self.teacher_recurrent:
            self.memory_t.reset(dones, hidden_states[1])

    def forward(self) -> NoReturn:
        """Forward pass is not implemented.

        Raises:
            NotImplementedError: Always, since this module uses explicit
                :meth:`act` / :meth:`act_inference` methods.
        """
        raise NotImplementedError

    # ═══════════════════════════════════════════════════════════════════════════
    #  Properties
    # ═══════════════════════════════════════════════════════════════════════════

    @property
    def action_mean(self) -> torch.Tensor:
        """Mean of the most recent action distribution.

        Returns:
            Mean action tensor. Returns a zero tensor if the distribution has
            not been computed yet.
        """
        if self.distribution is None:
            return torch.zeros(1)
        return self.distribution.mean

    @property
    def action_std(self) -> torch.Tensor:
        """Standard deviation of the most recent action distribution.

        Returns:
            Standard deviation tensor. Returns the current noise parameter if
            the distribution has not been computed yet.
        """
        if self.distribution is None:
            if self.noise_std_type == "scalar":
                return self.std.detach()
            elif self.noise_std_type == "log":
                return torch.exp(self.log_std).detach()
        return self.distribution.stddev

    @property
    def entropy(self) -> torch.Tensor:
        """Entropy of the most recent action distribution.

        Returns:
            Entropy tensor summed over action dimensions. Returns a zero tensor
            if the distribution has not been computed yet.
        """
        if self.distribution is None:
            return torch.zeros(1)
        return self.distribution.entropy().sum(dim=-1)

    # ═══════════════════════════════════════════════════════════════════════════
    #  Distribution
    # ═══════════════════════════════════════════════════════════════════════════

    def _update_distribution(self, action_mean: torch.Tensor) -> None:
        """Update the action distribution with the given mean.

        The standard deviation is determined by :attr:`noise_std_type`.

        Args:
            action_mean: Predicted action mean tensor of shape ``[B, num_actions]``.
        """
        if self.noise_std_type == "scalar":
            std = self.std.expand_as(action_mean)
        elif self.noise_std_type == "log":
            std = torch.exp(self.log_std).expand_as(action_mean)
        else:
            raise ValueError(
                f"Unknown standard deviation type: {self.noise_std_type}. "
                f"Should be 'scalar' or 'log'"
            )
        self.distribution = Normal(action_mean, std)

    # ═══════════════════════════════════════════════════════════════════════════
    #  Observation Extraction
    # ═══════════════════════════════════════════════════════════════════════════

    def get_student_obs(self, obs: TensorDict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Extract and organize student observations from the TensorDict.

        Args:
            obs: Observation TensorDict containing depth image and proprioceptive data.

        Returns:
            A tuple of ``(proprio_latest, depth_img, prop_all)`` where:

            - ``proprio_latest``: Latest proprioceptive frame of shape ``[B, proprio_per_frame]``.
            - ``depth_img``: Depth image of shape ``[B, 1, 32, 180]`` in NCHW format.
            - ``prop_all``: Full proprioceptive history of shape ``[B, num_student_basic_obs]``
              (unnormalized, for normalization update).

        Note:
            Requires the proprioceptive obs groups to be frame-major (shape ``[B, H, D]``,
            i.e. ``flatten_history_dim=False`` in the env config). The group is flattened
            time-major here so that the last ``proprio_per_frame`` dims are exactly the
            latest frame.
        """
        # ── Depth image ──
        depth_flat = obs[self.depth_obs_group]  # [B, 5761] (5760 pixels + 1 extra dim)
        depth_img = depth_flat[:, : self.depth_flat_dim].reshape(
            -1, self.depth_channels, self.depth_height, self.depth_width
        )  # [B, 1, 32, 180]

        # ── Proprioceptive: concatenate all prop groups ──
        prop_list = [obs[g] for g in self.student_prop_groups]
        # Frame-major (B, H, D) → time-major (B, H*D): frames are ordered oldest→newest,
        # so the last `proprio_per_frame` dims are the latest frame.
        prop_all = torch.cat(prop_list, dim=-1).reshape(prop_list[0].shape[0], -1)  # [B, num_student_basic_obs]

        # ── Latest frame only ──
        prop_latest = prop_all[:, -self.proprio_per_frame :]  # [B, proprio_per_frame]

        return prop_latest, depth_img, prop_all

    def get_teacher_obs(self, obs: TensorDict) -> torch.Tensor:
        """Extract teacher observations from the TensorDict.

        Args:
            obs: Observation TensorDict containing teacher observation groups.

        Returns:
            Concatenated teacher observation tensor of shape ``[B, num_teacher_obs]``.
        """
        obs_list = [obs[obs_group] for obs_group in self.obs_groups["teacher"]]
        return torch.cat(obs_list, dim=-1)

    # ═══════════════════════════════════════════════════════════════════════════
    #  Core Computation
    # ═══════════════════════════════════════════════════════════════════════════

    def _compute_latents_and_action(
        self, obs: TensorDict, sample: bool
    ) -> torch.Tensor:
        """Core computation: depth CNN → GRU fusion → student MLP → action.

        Args:
            obs: Observation TensorDict.
            sample: If ``True``, samples from the action distribution. If ``False``,
                returns the deterministic action mean.

        Returns:
            Action tensor of shape ``[B, num_actions]``.
        """
        prop_latest, depth_img, prop_all = self.get_student_obs(obs)

        # ── Depth CNN: extract 32-dim latent ──
        depth_latent = self.depth_cnn(depth_img)  # [B, 32]

        # ── Depth auxiliary channels (e.g., depth_image_age): clock signal for 10Hz/50Hz sync ──
        depth_aux = obs[self.depth_obs_group][:, self.depth_flat_dim:]  # [B, depth_aux_dim]

        # ── GRU fusion: combine latest proprio + depth age + depth latent ──
        gru_input = torch.cat([prop_latest, depth_aux, depth_latent], dim=-1)  # [B, proprio_per_frame + aux + 32]
        gru_feat = self.gru_input_mlp(gru_input)                     # [B, 64]
        gru_out = self.memory_s(gru_feat).squeeze(0)                 # [B, rnn_hidden_dim]
        latents = self.gru_output_mlp(gru_out)                       # [B, 64]

        # Extract the two latent codes: depth_latent (aligned to teacher scan_latent)
        # and privilege_latent (aligned to teacher privilege_latent).
        depth_latent = latents[:, :32]         # [B, 32]
        privilege_latent = latents[:, 32:]     # [B, 32]
        self._last_student_depth_latent = depth_latent
        self._last_student_privilege_latent = privilege_latent

        # ── Student MLP: combine full proprio history + two latents ──
        prop_all_norm = self.student_obs_normalizer(prop_all)  # [B, num_student_basic_obs]
        student_input = torch.cat(
            [prop_all_norm, depth_latent, privilege_latent], dim=-1
        )  # [B, num_student_basic_obs + 64]
        action_mean = self.student(student_input)  # [B, num_actions]

        if sample:
            self._update_distribution(action_mean)
            return self.distribution.sample()
        return action_mean

    # ═══════════════════════════════════════════════════════════════════════════
    #  Action Methods
    # ═══════════════════════════════════════════════════════════════════════════

    def act(self, obs: TensorDict) -> torch.Tensor:
        """Compute a stochastic action during training / exploration.

        Args:
            obs: Observation TensorDict.

        Returns:
            Sampled action tensor of shape ``[B, num_actions]``.
        """
        return self._compute_latents_and_action(obs, sample=True)

    def act_inference(self, obs: TensorDict) -> torch.Tensor:
        """Compute a deterministic action during inference.

        Args:
            obs: Observation TensorDict.

        Returns:
            Deterministic action mean tensor of shape ``[B, num_actions]``.
        """
        return self._compute_latents_and_action(obs, sample=False)

    def evaluate(self, obs: TensorDict) -> torch.Tensor:
        """Evaluate the teacher policy for distillation targets.

        Args:
            obs: Observation TensorDict containing teacher observation groups.

        Returns:
            Teacher action tensor of shape ``[B, num_actions]``.
        """
        teacher_flat = self.get_teacher_obs(obs)  # [B, num_teacher_obs]
        scan = teacher_flat[:, self._teacher_scan_start : self._teacher_scan_end]
        scan_latent = self.teacher_scan_encoder.encode(scan)  # [B, 32]
        priv = teacher_flat[:, self._teacher_priv_start : self._teacher_priv_end]
        privilege_latent = self.teacher_privilege_encoder.encode(priv)  # [B, 32]
        teacher_in = torch.cat(
            [
                teacher_flat[:, : self._teacher_scan_start],
                scan_latent,
                teacher_flat[:, self._teacher_scan_end : self._teacher_priv_start],
                privilege_latent,
                teacher_flat[:, self._teacher_priv_end :],
            ],
            dim=-1,
        )  # [B, num_teacher_obs_encoded]
        teacher_in = self.teacher_obs_normalizer(teacher_in)
        with torch.no_grad():
            if self.teacher_recurrent:
                self.memory_t.eval()
                teacher_in = self.memory_t(teacher_in).squeeze(0)
            return self.teacher(teacher_in)

    def get_student_latents(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the student's latest ``(depth_latent, privilege_latent)``.

        Populated during :meth:`act_inference` / :meth:`act`; used by the
        alignment losses in ``DistillationAlign``.
        """
        return self._last_student_depth_latent, self._last_student_privilege_latent

    def get_teacher_latents(self, obs: TensorDict) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute the frozen teacher's ``(scan_latent, privilege_latent)`` as alignment targets.

        Args:
            obs: Observation TensorDict containing teacher observation groups.

        Returns:
            Tuple of teacher ``scan_latent`` and ``privilege_latent`` (no grad).
        """
        teacher_flat = self.get_teacher_obs(obs)  # [B, num_teacher_obs]
        scan = teacher_flat[:, self._teacher_scan_start : self._teacher_scan_end]
        priv = teacher_flat[:, self._teacher_priv_start : self._teacher_priv_end]
        with torch.no_grad():
            scan_latent = self.teacher_scan_encoder.encode(scan)
            privilege_latent = self.teacher_privilege_encoder.encode(priv)
        return scan_latent, privilege_latent

    # ═══════════════════════════════════════════════════════════════════════════
    #  Hidden State Management
    # ═══════════════════════════════════════════════════════════════════════════

    def get_hidden_states(self) -> tuple[HiddenState, HiddenState]:
        """Get the current hidden states of both student and teacher memory modules.

        Returns:
            A tuple of ``(student_hidden_state, teacher_hidden_state)``.
            If :attr:`teacher_recurrent` is ``False``, the teacher hidden state is ``None``.
        """
        if self.teacher_recurrent:
            return self.memory_s.hidden_state, self.memory_t.hidden_state
        return self.memory_s.hidden_state, None

    def detach_hidden_states(self, dones: torch.Tensor | None = None) -> None:
        """Detach hidden states from the computation graph for truncated BPTT.

        Args:
            dones: Boolean tensor indicating which environments have terminated.
                If ``None``, detaches all hidden states unconditionally.
        """
        self.memory_s.detach_hidden_state(dones)
        if self.teacher_recurrent:
            self.memory_t.detach_hidden_state(dones)

    # ═══════════════════════════════════════════════════════════════════════════
    #  Training & Normalization
    # ═══════════════════════════════════════════════════════════════════════════

    def train(self, mode: bool = True) -> None:
        """Set the module to training or evaluation mode.

        The teacher and teacher encoders are always kept in evaluation mode
        since they are frozen during distillation training.

        Args:
            mode: If ``True``, sets to training mode; otherwise evaluation mode.
        """
        super().train(mode)
        # Teacher and its components are always frozen during distillation
        self.teacher.eval()
        self.teacher_obs_normalizer.eval()
        self.teacher_scan_encoder.eval()
        self.teacher_privilege_encoder.eval()

    def update_normalization(self, obs: TensorDict) -> None:
        """Update the running statistics of the student observation normalizer.

        Only active when :attr:`student_obs_normalization` is ``True``.

        Args:
            obs: Observation TensorDict.
        """
        if self.student_obs_normalization:
            _, _, prop_all = self.get_student_obs(obs)
            self.student_obs_normalizer.update(prop_all)

    # ═══════════════════════════════════════════════════════════════════════════
    #  Checkpoint Loading
    # ═══════════════════════════════════════════════════════════════════════════

    def load_state_dict(self, state_dict: dict, strict: bool = True) -> bool:
        """Load the parameters of the student and teacher networks.

        Supports two loading modes:

        1. **Teacher-only loading**: If the state dict keys contain ``"actor"``,
           the teacher MLP (``actor.*``), its scan encoder (``scan_encoder.*``) and
           its normalizer (``actor_obs_normalizer.*``) are loaded with key remapping.
           If :attr:`teacher_recurrent` is ``True``, the teacher memory
           (``"memory_a."`` → ``"memory_t."``) is also loaded.

        2. **Full student loading**: If the state dict keys contain ``"student"``,
           the complete student state dict (including CNN, GRU, and student MLP) is loaded.

        Args:
            state_dict: State dictionary of the model.
            strict: Whether to strictly enforce that the keys in ``state_dict`` match
                the keys returned by this module's :meth:`state_dict` function.

        Returns:
            ``True`` if loading a full student checkpoint (training resumes);
            ``False`` if only the teacher was loaded.

        Raises:
            ValueError: If the state dict contains neither ``"actor"`` nor ``"student"`` keys.
        """
        if any("actor" in key for key in state_dict):
            # ── Teacher-only loading (from RL training checkpoint) ──
            teacher_state_dict = {}
            teacher_scan_state_dict = {}
            teacher_privilege_state_dict = {}
            teacher_obs_normalizer_state_dict = {}
            for key, value in state_dict.items():
                if "actor." in key:
                    teacher_state_dict[key.replace("actor.", "")] = value
                elif "scan_encoder." in key:
                    teacher_scan_state_dict[key.replace("scan_encoder.", "")] = value
                elif "privilege_encoder." in key:
                    teacher_privilege_state_dict[key.replace("privilege_encoder.", "")] = value
                if "actor_obs_normalizer." in key:
                    teacher_obs_normalizer_state_dict[key.replace("actor_obs_normalizer.", "")] = value
            self.teacher.load_state_dict(teacher_state_dict, strict=False)
            if teacher_scan_state_dict:
                self.teacher_scan_encoder.load_state_dict(teacher_scan_state_dict, strict=False)
                print("[INFO] Loaded teacher scan encoder weights from checkpoint.")
            else:
                print(
                    "[WARN] Checkpoint does not contain 'scan_encoder.*' weights! "
                    "Teacher scan encoder stays randomly initialized."
                )
            if teacher_privilege_state_dict:
                self.teacher_privilege_encoder.load_state_dict(teacher_privilege_state_dict, strict=False)
                print("[INFO] Loaded teacher privilege encoder weights from checkpoint.")
            else:
                print(
                    "[WARN] Checkpoint does not contain 'privilege_encoder.*' weights! "
                    "Teacher privilege encoder stays randomly initialized."
                )
            self.teacher_obs_normalizer.load_state_dict(teacher_obs_normalizer_state_dict, strict=False)

            # Also load teacher recurrent memory if applicable
            if self.teacher_recurrent:
                memory_t_state_dict = {}
                for key, value in state_dict.items():
                    if "memory_a." in key:
                        memory_t_state_dict[key.replace("memory_a.", "")] = value
                self.memory_t.load_state_dict(memory_t_state_dict, strict=strict)

            self.loaded_teacher = True
            self.teacher.eval()
            self.teacher_obs_normalizer.eval()
            return False  # Training does not resume

        elif any("student" in key for key in state_dict):
            # ── Full student loading (from distillation checkpoint) ──
            # Backward compatibility: checkpoints saved before the student memory was renamed
            # from "memory" to "memory_s" use the "memory." key prefix. Remap these keys so
            # existing distillation checkpoints can still be resumed without retraining.
            if any(key.startswith("memory.") for key in state_dict):
                state_dict = {
                    (f"memory_s.{key[len('memory.'):]}" if key.startswith("memory.") else key): value
                    for key, value in state_dict.items()
                }
            super().load_state_dict(state_dict, strict=strict)
            self.loaded_teacher = True
            self.teacher.eval()
            self.teacher_obs_normalizer.eval()
            return True  # Training resumes

        else:
            raise ValueError(
                "state_dict does not contain student or teacher parameters. "
                "Expected keys containing 'actor' (teacher from RL training) "
                "or 'student' (student from distillation training)."
            )
