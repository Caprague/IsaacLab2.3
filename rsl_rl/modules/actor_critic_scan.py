# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Actor-Critic policy with end-to-end scan + privilege encoders (Parkour-style teacher).

The teacher actor encodes the privileged height scan (``mapScans``) into a
compact ``scan_latent`` and the privileged state (``privileged``) into a
``privilege_latent`` before feeding them to the actor/critic MLPs:

    flat obs [B, D] ──► slice scan ──► ScanEncoder ──► scan_latent(32)
                        │
    flat obs [B, D] ──► slice priv ──► PrivilegeEncoder ──► privilege_latent(32)
                        │
                        └──► concat [proprio, scan_latent, privilege_latent] ──► actor / critic

Both encoders are trained jointly with the policy during PPO, so the trained
teacher checkpoint natively contains their weights. These latents define the
feature-injection slots that the student's ``depth_latent``/``privilege_latent``
align to during distillation.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from tensordict import TensorDict
from torch.distributions import Normal
from typing import Any, NoReturn

from rsl_rl.networks import MLP, EmpiricalNormalization, ScanEncoder, PrivilegeEncoder


class ActorCriticScan(nn.Module):
    """Actor-Critic with a trainable scan encoder (Parkour-style teacher)."""

    is_recurrent: bool = False

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        num_actions: int,
        actor_obs_normalization: bool = False,
        critic_obs_normalization: bool = False,
        actor_hidden_dims: tuple[int] | list[int] = [256, 256, 256],
        critic_hidden_dims: tuple[int] | list[int] = [256, 256, 256],
        activation: str = "elu",
        init_noise_std: float = 1.0,
        noise_std_type: str = "scalar",
        state_dependent_std: bool = False,
        scan_obs_group: str = "mapScans",
        scan_latent_dim: int = 32,
        privilege_obs_group: str = "privileged",
        privilege_latent_dim: int = 32,
        **kwargs: dict[str, Any],
    ) -> None:
        if kwargs:
            print(
                "ActorCriticScan.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs])
            )
        if state_dependent_std:
            raise NotImplementedError("ActorCriticScan does not support state_dependent_std.")
        super().__init__()

        # ── Save observation groups and compute flat-obs boundaries of scan/privilege groups ──
        self.obs_groups = obs_groups
        self.scan_obs_group = scan_obs_group
        self.scan_latent_dim = scan_latent_dim
        self.privilege_obs_group = privilege_obs_group
        self.privilege_latent_dim = privilege_latent_dim

        num_actor_obs = 0
        self._scan_start: int | None = None
        self._scan_end: int | None = None
        self._priv_start: int | None = None
        self._priv_end: int | None = None
        for obs_group in obs_groups["policy"]:
            assert len(obs[obs_group].shape) == 2, "The ActorCriticScan module only supports 1D observations."
            dim = obs[obs_group].shape[-1]
            if obs_group == scan_obs_group:
                self._scan_start = num_actor_obs
                self._scan_end = num_actor_obs + dim
            if obs_group == privilege_obs_group:
                self._priv_start = num_actor_obs
                self._priv_end = num_actor_obs + dim
            num_actor_obs += dim
        assert self._scan_start is not None and self._scan_end is not None, (
            f"Scan observation group '{scan_obs_group}' not found in policy groups {obs_groups['policy']}"
        )
        assert self._priv_start is not None and self._priv_end is not None, (
            f"Privilege observation group '{privilege_obs_group}' not found in policy groups {obs_groups['policy']}"
        )
        num_actor_obs_encoded = (
            num_actor_obs
            - (self._scan_end - self._scan_start)
            - (self._priv_end - self._priv_start)
            + scan_latent_dim
            + privilege_latent_dim
        )
        print(
            f"ActorCriticScan obs boundaries: scan=[{self._scan_start}, {self._scan_end}) "
            f"priv=[{self._priv_start}, {self._priv_end}) encoded_actor_in={num_actor_obs_encoded}"
        )

        # ── Scan encoder (trained end-to-end with the policy) ──
        self.scan_encoder = ScanEncoder(latent_dim=scan_latent_dim, activation=activation)
        print(f"ActorCriticScan scan_encoder: {self.scan_encoder}")

        # ── Privilege encoder (trained end-to-end with the policy) ──
        self.privilege_encoder = PrivilegeEncoder(
            input_dim=self._priv_end - self._priv_start,
            latent_dim=privilege_latent_dim,
            activation=activation,
        )
        print(f"ActorCriticScan privilege_encoder: {self.privilege_encoder}")

        # ── Actor / Critic ──
        self.actor = MLP(num_actor_obs_encoded, num_actions, actor_hidden_dims, activation)
        print(f"Actor MLP: {self.actor}")
        self.critic = MLP(num_actor_obs_encoded, 1, critic_hidden_dims, activation)
        print(f"Critic MLP: {self.critic}")

        # ── Observation normalization ──
        self.actor_obs_normalization = actor_obs_normalization
        if actor_obs_normalization:
            self.actor_obs_normalizer = EmpiricalNormalization(num_actor_obs_encoded)
        else:
            self.actor_obs_normalizer = torch.nn.Identity()
        self.critic_obs_normalization = critic_obs_normalization
        if critic_obs_normalization:
            self.critic_obs_normalizer = EmpiricalNormalization(num_actor_obs_encoded)
        else:
            self.critic_obs_normalizer = torch.nn.Identity()

        # ── Action noise ──
        self.noise_std_type = noise_std_type
        if self.noise_std_type == "scalar":
            self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        elif self.noise_std_type == "log":
            self.log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(num_actions)))
        else:
            raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")

        self.distribution = None
        Normal.set_default_validate_args(False)

    # ── Observation encoding ──
    def get_actor_obs(self, obs: TensorDict) -> torch.Tensor:
        """Concatenate policy obs and replace scan/privilege groups with their encoded latents.

        Output layout: [groups before scan, scan_latent, groups between scan and
        privilege, privilege_latent, groups after privilege].
        """
        obs_list = [obs[obs_group] for obs_group in self.obs_groups["policy"]]
        obs_flat = torch.cat(obs_list, dim=-1)
        scan = obs_flat[:, self._scan_start : self._scan_end]
        scan_latent = self.scan_encoder.encode(scan)
        priv = obs_flat[:, self._priv_start : self._priv_end]
        privilege_latent = self.privilege_encoder.encode(priv)
        return torch.cat(
            [
                obs_flat[:, : self._scan_start],
                scan_latent,
                obs_flat[:, self._scan_end : self._priv_start],
                privilege_latent,
                obs_flat[:, self._priv_end :],
            ],
            dim=-1,
        )

    def get_critic_obs(self, obs: TensorDict) -> torch.Tensor:
        return self.get_actor_obs(obs)

    # ── Distribution ──
    def _update_distribution(self, obs: torch.Tensor) -> None:
        mean = self.actor(obs)
        if self.noise_std_type == "scalar":
            std = torch.clamp(self.std, min=1e-3).expand_as(mean)
        elif self.noise_std_type == "log":
            std = torch.exp(self.log_std).expand_as(mean)
        else:
            raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")
        self.distribution = Normal(mean, std)

    # ── Policy interface (mirrors rsl_rl ActorCritic) ──
    def reset(self, dones: torch.Tensor | None = None) -> None:
        pass

    def forward(self) -> NoReturn:
        raise NotImplementedError

    @property
    def action_mean(self) -> torch.Tensor:
        if self.distribution is None:
            return torch.zeros(1)
        return self.distribution.mean

    @property
    def action_std(self) -> torch.Tensor:
        if self.distribution is None:
            if self.noise_std_type == "scalar":
                return self.std.detach()
            return torch.exp(self.log_std).detach()
        return self.distribution.stddev

    @property
    def entropy(self) -> torch.Tensor:
        if self.distribution is None:
            return torch.zeros(1)
        return self.distribution.entropy().sum(dim=-1)

    def act(self, obs: TensorDict, **kwargs: dict[str, Any]) -> torch.Tensor:
        obs = self.get_actor_obs(obs)
        obs = self.actor_obs_normalizer(obs)
        self._update_distribution(obs)
        return self.distribution.sample()

    def act_inference(self, obs: TensorDict) -> torch.Tensor:
        obs = self.get_actor_obs(obs)
        obs = self.actor_obs_normalizer(obs)
        return self.actor(obs)

    def evaluate(self, obs: TensorDict, **kwargs: dict[str, Any]) -> torch.Tensor:
        obs = self.get_critic_obs(obs)
        obs = self.critic_obs_normalizer(obs)
        return self.critic(obs)

    def get_actions_log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        return self.distribution.log_prob(actions).sum(dim=-1)

    def update_normalization(self, obs: TensorDict) -> None:
        if self.actor_obs_normalization:
            self.actor_obs_normalizer.update(self.get_actor_obs(obs))
        if self.critic_obs_normalization:
            self.critic_obs_normalizer.update(self.get_critic_obs(obs))

    def get_hidden_states(self) -> None:
        return None

    def detach_hidden_states(self, dones: torch.Tensor | None = None) -> None:
        pass

    def load_state_dict(self, state_dict: dict, strict: bool = True) -> bool:
        super().load_state_dict(state_dict, strict=strict)
        return True
