# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg, RslRlSymmetryCfg

from isaaclab_tasks.manager_based.locomotion.velocity.mdp.symmetry import go2_mid360_teacher_walk


@configclass
class UnitreeGo2LocoSkillPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 32
    max_iterations: int = 10
    save_interval = 250
    experiment_name = "Go2-Loco-Skill-Walk-Mid360Depth-10Hz-GRU"

    env_stage = "stage1"        # stage1(前向+DR) / stage2(全向+DR) / stage3(distillation专用)

    def __post_init__(self):
        if self.env_stage == "stage1":
            self.max_iterations = 6001
        elif self.env_stage == "stage2":
            self.max_iterations = 6001
        elif self.env_stage == "stage3":
            raise ValueError("stage3 requires distillation training, use <--agent rsl_rl_distillation_cfg_entry_point> !")
        else:
            raise ValueError(f"Unknown stage: {self.env_stage}, choose from: stage1, stage2, stage3")
   
    obs_groups = {
        "policy": ["proprioception", "mapScans", "privileged"],
        "critic": ["proprioception", "mapScans", "privileged"],
    }
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
        noise_std_type="scalar",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        rnd_cfg=None,
        symmetry_cfg=RslRlSymmetryCfg(
            use_data_augmentation=True, 
            data_augmentation_func=go2_mid360_teacher_walk.compute_symmetric_states,
            use_mirror_loss=True,
            mirror_loss_coeff=0.1,
        ),
    )
