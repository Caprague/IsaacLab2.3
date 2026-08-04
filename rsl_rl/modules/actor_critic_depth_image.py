import torch
import torch.nn as nn

from rsl_rl.networks import MLP, EmpiricalNormalization, DepthImageEncoder
from rsl_rl.modules import ActorCritic


class ActorCriticDepthImage(ActorCritic):
    is_recurrent = False

    def __init__(
        self,
        obs,
        obs_groups,
        num_actions,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
        init_noise_std=1.0,
        **kwargs,
    ):
        nn.Module.__init__(self)

        self.obs_groups = obs_groups
        self.depth_obs_group = "mid360_depth"

        self.depth_height = 32
        self.depth_width = 180
        self.depth_channels = 1
        self.depth_flat_dim = self.depth_height * self.depth_width * self.depth_channels

        self.basic_obs_groups = [
            g for g in obs_groups["policy"] if g != self.depth_obs_group
        ]

        num_basic_obs = 0
        for g in self.basic_obs_groups:
            num_basic_obs += obs[g].shape[-1]

        self.depth_encoder = DepthImageEncoder(activation=activation)

        with torch.no_grad():
            dummy_depth = torch.randn(1, self.depth_height, self.depth_width, self.depth_channels)
            self.depth_feature_dim = self.depth_encoder(dummy_depth).shape[-1]

        num_actor_obs = num_basic_obs + self.depth_feature_dim

        self.actor = MLP(num_actor_obs, num_actions, actor_hidden_dims, activation)
        self.critic = MLP(num_actor_obs, 1, critic_hidden_dims, activation)

        self.obs_normalizer = EmpiricalNormalization(num_basic_obs)

        self.actor_obs = obs
        self.num_actions = num_actions

        if init_noise_std is not None:
            self.init_noise_std = init_noise_std
            self.std = nn.Parameter(
                torch.ones(num_actions) * init_noise_std, requires_grad=True
            )
        else:
            self.std = None

    def get_actor_obs(self, obs):
        depth_flat = obs[self.depth_obs_group]

        if depth_flat.dim() == 2:
            depth_img = depth_flat[:, :self.depth_flat_dim].view(
                -1, self.depth_height, self.depth_width, self.depth_channels
            )
        else:
            depth_img = depth_flat

        depth_features = self.depth_encoder(depth_img)

        basic_obs_list = [obs[g] for g in self.basic_obs_groups]
        basic_obs = torch.cat(basic_obs_list, dim=-1)
        basic_obs = self.obs_normalizer(basic_obs)

        return torch.cat([basic_obs, depth_features], dim=-1)

    def get_critic_obs(self, obs):
        return self.get_actor_obs(obs)