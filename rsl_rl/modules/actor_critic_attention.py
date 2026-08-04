# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Normal
from typing import Optional

from rsl_rl.utils import resolve_nn_activation
from rsl_rl.networks import MHA, ScanCNNEncoder, EmpiricalNormalization, MLP


class ActorNetworkMHA(nn.Module):
    def __init__(
        self, 
        num_basic_obs, 
        num_scan_obs, 
        num_actions, 
        mha_embed_dim,                          # 64
        mha_num_heads,                          # 16
        mha_dropout,                            # 0.1
        scan_size_x,                            # 17
        scan_size_y,                            # 11
        basic_obs_dims_list,
        basic_obs_history_length,
        actor_hidden_dims=[256, 256, 256], 
        activation="elu"
    ):
        super().__init__()
        self.num_basic_obs = num_basic_obs                    
        self.num_scan_obs = num_scan_obs                        
        self.num_actions = num_actions                         
        self.num_actor_obs = mha_embed_dim + num_basic_obs    
        self.mha_embed_dim = mha_embed_dim                      # 64
        self.mha_num_heads = mha_num_heads                      # 16
        self.mha_dropout = mha_dropout                          # 0.1
        self.scan_size_x = scan_size_x                          # 17
        self.scan_size_y = scan_size_y                          # 11
        self.basic_obs_dims_list = basic_obs_dims_list
        self.basic_obs_history_length = basic_obs_history_length
        activation_layer = resolve_nn_activation(activation)

        # 根据 basic 各项的维度与历史长度，计算拼接后，各项的最新观测所处的索引位置
        self.basic_obs_indices = []
        index = 0
        for obs_term_size in self.basic_obs_dims_list:
            index += obs_term_size * self.basic_obs_history_length
            self.basic_obs_indices.append((index - obs_term_size, index))
        self.basic_lastest_obs_starts = torch.tensor([idx[0] for idx in self.basic_obs_indices])
        self.basic_lastest_obs_ends = torch.tensor([idx[1] for idx in self.basic_obs_indices])

        # Map Scan 的 CNN 前置特征提取模块，输出通道数为 64-3=61
        # 输入形状为 (例)[batch_size, 1, 11, 17] 或 [batch_size, 11, 17, 1]
        # 输出形状为 (例)[batch_size, 61, 11, 17]
        self.ScanCNNEncoder = ScanCNNEncoder(d=mha_embed_dim, keep_d=3, activation=activation)

        # ScanCNNEncoder -> Concat -> Map Scan Feature1
        # 1. Concat 特征拼接，Map Scan CNN 输出的特征(仅处理z值) + Map Scan 原始3D坐标(x、y、z值)， 形状 [batch_size, 61, 11, 17] -> [batch_size, 64, 11, 17]
        # 2. 重塑形状，[batch_size, 64, 11, 17] -> [batch_size, 11*17, 64]

        # MHA 模块，将 Map Scan Feature1 作为 K-V，将 basic 作为 Q
        # 输入 K-V 形状 [batch_size, 11*17, 64]，输入 Q 形状 [batch_size, 1, 47]
        # 注：针对 Q 形状，单独加 Linear 层，将其映射到 64 维，形状 [batch_size, 1, 47] -> [batch_size, 1, 64]
        # 输出 Map Scan Feature2，形状 [batch_size, 1, 64]
        self.BasicLinear = nn.Linear(self.num_basic_obs // self.basic_obs_history_length, mha_embed_dim)
        self.MHA = MHA(d=mha_embed_dim, num_heads=mha_num_heads, dropout=mha_dropout)

        # 中间变量，存储 MHA 输出的注意力权重
        self.register_buffer('_attention_storage', torch.zeros(1, 1), persistent=False)

        # Actor 主网络，将 Map Scan Feature2 和 basic x num_hist 拼接后，输入到 3 层全连接网络，输出 action
        actor_layers = []
        actor_layers.append(nn.Linear(self.num_actor_obs, actor_hidden_dims[0]))
        actor_layers.append(activation_layer)
        for layer_index in range(len(actor_hidden_dims)):
            if layer_index == len(actor_hidden_dims) - 1:
                actor_layers.append(nn.Linear(actor_hidden_dims[layer_index], num_actions))
            else:
                actor_layers.append(nn.Linear(actor_hidden_dims[layer_index], actor_hidden_dims[layer_index + 1]))
                actor_layers.append(activation_layer)
        self.actor_network = nn.Sequential(*actor_layers)

    @property
    def mha_attention_weights(self) -> torch.Tensor:
        return self._attention_storage

    def forward(self, basic_obs, scan_obs):
        # # 断言检查观测维度
        # assert basic_obs.shape[1] == self.num_basic_obs, f"basic_obs 维度错误，期望 {self.num_basic_obs}，但 got {basic_obs.shape[1]}"
        # assert scan_obs.shape[1] == self.num_scan_obs, f"scan_obs 维度错误，期望 {self.num_scan_obs}，但 got {scan_obs.shape[1]}"

        # 扫描感知重塑形状，[batch_size, 11*17*3] -> [batch_size, 11, 17, 3] -> [batch_size, 3, 11, 17]
        scan_obs = scan_obs.reshape(-1, self.scan_size_y, self.scan_size_x, 3).permute(0, 3, 1, 2)
        # 提取扫描感知的 z 值，形状 [batch_size, 1, 11, 17]
        scan_z_obs = scan_obs[:, 2, :, :].unsqueeze(1)
        # 经由 ScanCNNEncoder 提取特征，形状 [batch_size, 61, 11, 17]
        scan_feature1 = self.ScanCNNEncoder(scan_z_obs)
        # 拼接扫描感知原始3D坐标(x、y、z值)，拼接后形状 [batch_size, 64, 11, 17]
        scan_feature1 = torch.cat([scan_obs, scan_feature1], dim=1)
        # 重塑形状，[batch_size, 64, 11, 17] -> [batch_size, 11*17, 64]
        scan_feature1 = scan_feature1.permute(0, 2, 3, 1).reshape(-1, self.scan_size_y * self.scan_size_x, self.mha_embed_dim)

        # 提取最新 basic 特征，形状 [batch_size, 47] -> 添加批次维度，形状 [batch_size, 1, 47]
        basic = torch.cat([basic_obs[:, start:end] for start, end in zip(self.basic_lastest_obs_starts, self.basic_lastest_obs_ends)], dim=1).unsqueeze(1)
        # # 断言检查 basic 维度
        # assert basic.shape[-1] == (self.num_basic_obs // self.basic_obs_history_length), f"basic 维度错误，期望 {(self.num_basic_obs // self.basic_obs_history_length)}，但 got {basic.shape[-1]}"

        # 经由 MHA 模块，将 Map Scan Feature1 作为 K-V，将 basic 作为 Q
        # 输入 K-V 形状 [batch_size, 11*17, 64]，输入 Q 形状 [batch_size, 1, 47]
        # 输出 Map Scan Feature2，形状 [batch_size, 1, 64]
        # 直接接收两个返回值，不需要检查长度
        scan_feature2, attn_weights = self.MHA(self.BasicLinear(basic), scan_feature1)
        if attn_weights is not None:
            self._attention_storage = attn_weights.clone().data

        # 拼接 Map Scan Feature2 和 basic，形状 [batch_size, 470+64]
        actor_obs = torch.cat([basic_obs, scan_feature2.squeeze(1)], dim=1)
        # # 断言检查 actor_obs 维度
        # assert actor_obs.shape[1] == self.num_actor_obs, f"actor_obs 维度错误，期望 {self.num_actor_obs}，但 got {actor_obs.shape[1]}"
        # 经由 Actor 主网络，输出 action
        return self.actor_network(actor_obs)


class ActorCriticAttention(nn.Module):
    is_recurrent = False

    def __init__(
        self,
        obs,
        obs_groups,
        num_actions,
        # **policy_cfg <- agent_cfg.policy: RslRlPpoActorCriticCfg
        mha_embed_dim,                          # 64
        mha_num_heads,                          # 16
        mha_dropout,                            # 0.1
        scan_size_x,                            # 17
        scan_size_y,                            # 11
        basic_obs_dims_list,
        basic_obs_history_length,
        actor_hidden_dims=[256, 256, 256],
        critic_hidden_dims=[256, 256, 256],
        basic_obs_normalization=False,
        scan_obs_normalization=False,
        critic_obs_normalization=False,
        activation="elu",
        init_noise_std=1.0,
        noise_std_type: str = "scalar",
        **kwargs,
    ):
        if kwargs:
            print(
                "ActorCritic.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs.keys()])
            )
        super().__init__()
        self.obs_groups = obs_groups

        # basic observation dimensions
        num_basic_obs = 0
        for obs_group in obs_groups["basic"]:
            assert len(obs[obs_group].shape) == 2, "The ActorCritic module only supports 1D observations."
            num_basic_obs += obs[obs_group].shape[-1]
        # scan observation dimensions
        num_scan_obs = 0
        for obs_group in obs_groups["scan"]:
            assert len(obs[obs_group].shape) == 2, "The ActorCritic module only supports 1D observations."
            num_scan_obs += obs[obs_group].shape[-1]
        # critic observation dimensions
        num_critic_obs = 0
        for obs_group in obs_groups["critic"]:
            assert len(obs[obs_group].shape) == 2, "The ActorCritic module only supports 1D observations."
            num_critic_obs += obs[obs_group].shape[-1]

        # actor
        self.actor = ActorNetworkMHA(
            num_basic_obs, 
            num_scan_obs, 
            num_actions, 
            mha_embed_dim,
            mha_num_heads,
            mha_dropout,
            scan_size_x,
            scan_size_y,
            basic_obs_dims_list,
            basic_obs_history_length,
            actor_hidden_dims,
            activation,
        )
        print(f"Actor Network MHA: {self.actor}")
        # critic
        self.critic = MLP(num_critic_obs, 1, critic_hidden_dims, activation)
        print(f"Critic MLP: {self.critic}")

        # basic observation normalization
        self.basic_obs_normalization = basic_obs_normalization
        if basic_obs_normalization:
            self.basic_obs_normalizer = EmpiricalNormalization(num_basic_obs)
        else:
            self.basic_obs_normalizer = torch.nn.Identity()
        # scan observation normalization
        self.scan_obs_normalization = scan_obs_normalization
        if scan_obs_normalization:
            self.scan_obs_normalizer = EmpiricalNormalization(num_scan_obs)
        else:
            self.scan_obs_normalizer = torch.nn.Identity()
        # critic observation normalization
        self.critic_obs_normalization = critic_obs_normalization
        if critic_obs_normalization:
            self.critic_obs_normalizer = EmpiricalNormalization(num_critic_obs)
        else:
            self.critic_obs_normalizer = torch.nn.Identity()

        # Action noise
        self.noise_std_type = noise_std_type
        if self.noise_std_type == "scalar":
            self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        elif self.noise_std_type == "log":
            self.log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(num_actions)))
        else:
            raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")

        # Action distribution (populated in update_distribution)
        self.distribution = None
        # disable args validation for speedup
        Normal.set_default_validate_args(False)

    def reset(self, dones=None):
        pass

    def forward(self):
        raise NotImplementedError

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def update_distribution(self, basic_obs, scan_obs):
        # compute mean
        mean = self.actor(basic_obs, scan_obs)
        # compute standard deviation
        if self.noise_std_type == "scalar":
            std = self.std.expand_as(mean)
        elif self.noise_std_type == "log":
            std = torch.exp(self.log_std).expand_as(mean)
        else:
            raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")
        # create distribution
        self.distribution = Normal(mean, std)

    def act(self, obs, **kwargs):
        basic_obs = self.get_basic_obs(obs)
        basic_obs = self.basic_obs_normalizer(basic_obs)
        scan_obs = self.get_scan_obs(obs)
        scan_obs = self.scan_obs_normalizer(scan_obs)
        self.update_distribution(basic_obs, scan_obs)
        return self.distribution.sample()

    def act_inference(self, obs):
        basic_obs = self.get_basic_obs(obs)
        basic_obs = self.basic_obs_normalizer(basic_obs)
        scan_obs = self.get_scan_obs(obs)
        scan_obs = self.scan_obs_normalizer(scan_obs)
        return self.actor(basic_obs, scan_obs)
    
    @property
    def mha_attention_weights(self):
        return self.actor.mha_attention_weights

    def evaluate(self, obs, **kwargs):
        obs = self.get_critic_obs(obs)
        obs = self.critic_obs_normalizer(obs)
        return self.critic(obs)

    def get_basic_obs(self, obs):
        obs_list = []
        for obs_group in self.obs_groups["basic"]:
            obs_list.append(obs[obs_group])
        return torch.cat(obs_list, dim=-1)

    def get_scan_obs(self, obs):
        obs_list = []
        for obs_group in self.obs_groups["scan"]:
            obs_list.append(obs[obs_group])
        return torch.cat(obs_list, dim=-1)

    def get_critic_obs(self, obs):
        obs_list = []
        for obs_group in self.obs_groups["critic"]:
            obs_list.append(obs[obs_group])
        return torch.cat(obs_list, dim=-1)

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def update_normalization(self, obs):
        if self.basic_obs_normalization:
            basic_obs = self.get_basic_obs(obs)
            self.basic_obs_normalizer.update(basic_obs)
        if self.scan_obs_normalization:
            scan_obs = self.get_scan_obs(obs)
            self.scan_obs_normalizer.update(scan_obs)
        if self.critic_obs_normalization:
            critic_obs = self.get_critic_obs(obs)
            self.critic_obs_normalizer.update(critic_obs)

    def load_state_dict(self, state_dict, strict=True):
        """Load the parameters of the actor-critic model.

        Args:
            state_dict (dict): State dictionary of the model.
            strict (bool): Whether to strictly enforce that the keys in state_dict match the keys returned by this
                           module's state_dict() function.

        Returns:
            bool: Whether this training resumes a previous training. This flag is used by the `load()` function of
                  `OnPolicyRunner` to determine how to load further parameters (relevant for, e.g., distillation).
        """

        super().load_state_dict(state_dict, strict=strict)
        return True  # training resumes