from __future__ import annotations

import torch
import torch.nn as nn
from tensordict import TensorDict
from torch.distributions import Normal
from typing import Any, NoReturn

from rsl_rl.networks import MLP, EmpiricalNormalization, DepthImageEncoder, HiddenState


class StudentTeacherDepthImage(nn.Module):
    is_recurrent: bool = False

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
        **kwargs: dict[str, Any],
    ) -> None:
        if kwargs:
            print(
                "StudentTeacherDepthImage.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs])
            )
        super().__init__()

        self.loaded_teacher = False

        self.obs_groups = obs_groups
        self.depth_obs_group = "mid360_depth"

        self.depth_height = 32
        self.depth_width = 180
        self.depth_channels = 1
        self.depth_flat_dim = self.depth_height * self.depth_width * self.depth_channels

        self.student_basic_obs_groups = [
            g for g in obs_groups["policy"] if g != self.depth_obs_group
        ]

        num_student_basic_obs = 0
        for g in self.student_basic_obs_groups:
            assert len(obs[g].shape) == 2, "Student basic observations must be 1D."
            num_student_basic_obs += obs[g].shape[-1]

        self.depth_encoder = DepthImageEncoder(activation=activation)

        with torch.no_grad():
            dummy_depth = torch.randn(1, self.depth_height, self.depth_width, self.depth_channels)
            self.depth_feature_dim = self.depth_encoder(dummy_depth).shape[-1]

        num_student_obs = num_student_basic_obs + self.depth_feature_dim

        num_teacher_obs = 0
        for obs_group in obs_groups["teacher"]:
            assert len(obs[obs_group].shape) == 2, "Teacher observations must be 1D."
            num_teacher_obs += obs[obs_group].shape[-1]

        self.student = MLP(num_student_obs, num_actions, student_hidden_dims, activation)
        print(f"Student MLP: {self.student}")

        self.student_obs_normalization = student_obs_normalization
        if student_obs_normalization:
            self.student_obs_normalizer = EmpiricalNormalization(num_student_basic_obs)
        else:
            self.student_obs_normalizer = torch.nn.Identity()

        self.teacher = MLP(num_teacher_obs, num_actions, teacher_hidden_dims, activation)
        self.teacher.eval()
        print(f"Teacher MLP: {self.teacher}")

        self.teacher_obs_normalization = teacher_obs_normalization
        if teacher_obs_normalization:
            self.teacher_obs_normalizer = EmpiricalNormalization(num_teacher_obs)
        else:
            self.teacher_obs_normalizer = torch.nn.Identity()

        self.noise_std_type = noise_std_type
        if self.noise_std_type == "scalar":
            self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        elif self.noise_std_type == "log":
            self.log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(num_actions)))
        else:
            raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")

        self.distribution = None

        Normal.set_default_validate_args(False)



    def reset(
        self, dones: torch.Tensor | None = None, hidden_states: tuple[HiddenState, HiddenState] = (None, None)
    ) -> None:
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
            elif self.noise_std_type == "log":
                return torch.exp(self.log_std).detach()
        return self.distribution.stddev

    @property
    def entropy(self) -> torch.Tensor:
        if self.distribution is None:
            return torch.zeros(1)
        return self.distribution.entropy().sum(dim=-1)

    def _update_distribution(self, obs: torch.Tensor) -> None:
        mean = self.student(obs)
        if self.noise_std_type == "scalar":
            std = self.std.expand_as(mean)
        elif self.noise_std_type == "log":
            std = torch.exp(self.log_std).expand_as(mean)
        else:
            raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")
        self.distribution = Normal(mean, std)

    def act(self, obs: TensorDict) -> torch.Tensor:
        obs = self.get_student_obs(obs)
        obs = self.student_obs_normalizer(obs)
        self._update_distribution(obs)
        return self.distribution.sample()

    def act_inference(self, obs: TensorDict) -> torch.Tensor:
        obs = self.get_student_obs(obs)
        obs = self.student_obs_normalizer(obs)
        return self.student(obs)

    def evaluate(self, obs: TensorDict) -> torch.Tensor:
        obs = self.get_teacher_obs(obs)
        obs = self.teacher_obs_normalizer(obs)
        with torch.no_grad():
            return self.teacher(obs)

    def get_student_obs(self, obs: TensorDict) -> torch.Tensor:
        depth_flat = obs[self.depth_obs_group]

        if depth_flat.dim() == 2:
            depth_img = depth_flat[:, :self.depth_flat_dim].view(
                -1, self.depth_height, self.depth_width, self.depth_channels
            )
        else:
            depth_img = depth_flat

        depth_features = self.depth_encoder(depth_img)

        basic_obs_list = [obs[g] for g in self.student_basic_obs_groups]
        basic_obs = torch.cat(basic_obs_list, dim=-1)

        return torch.cat([basic_obs, depth_features], dim=-1)

    def get_teacher_obs(self, obs: TensorDict) -> torch.Tensor:
        obs_list = [obs[obs_group] for obs_group in self.obs_groups["teacher"]]
        return torch.cat(obs_list, dim=-1)

    def get_hidden_states(self) -> tuple[HiddenState, HiddenState]:
        return None, None

    def detach_hidden_states(self, dones: torch.Tensor | None = None) -> None:
        pass

    def train(self, mode: bool = True) -> None:
        super().train(mode)
        self.teacher.eval()
        self.teacher_obs_normalizer.eval()

    def update_normalization(self, obs: TensorDict) -> None:
        if self.student_obs_normalization:
            student_obs = self.get_student_obs(obs)
            self.student_obs_normalizer.update(student_obs)

    def load_state_dict(self, state_dict: dict, strict: bool = True) -> bool:
        if any("actor" in key for key in state_dict):
            teacher_state_dict = {}
            teacher_obs_normalizer_state_dict = {}
            for key, value in state_dict.items():
                if "actor." in key:
                    teacher_state_dict[key.replace("actor.", "")] = value
                if "actor_obs_normalizer." in key:
                    teacher_obs_normalizer_state_dict[key.replace("actor_obs_normalizer.", "")] = value
            self.teacher.load_state_dict(teacher_state_dict, strict=strict)
            self.teacher_obs_normalizer.load_state_dict(teacher_obs_normalizer_state_dict, strict=strict)
            self.loaded_teacher = True
            self.teacher.eval()
            self.teacher_obs_normalizer.eval()
            return False
        elif any("student" in key for key in state_dict):
            super().load_state_dict(state_dict, strict=strict)
            self.loaded_teacher = True
            self.teacher.eval()
            self.teacher_obs_normalizer.eval()
            return True
        else:
            raise ValueError("state_dict does not contain student or teacher parameters")