# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import (
    RslRlDistillationAlignAlgorithmCfg,
    RslRlDistillationRunnerCfg,
    RslRlDistillationStudentTeacherDepthImageRecurrentCfg,
)


@configclass
class UnitreeGo2LocoSkillDistillationRunnerCfg(RslRlDistillationRunnerCfg):
    num_steps_per_env = 120
    max_iterations = 8001
    teacher_driving = True
    teacher_driving_mode = "mixed"
    teacher_driving_ratio = 0.15
    save_interval = 250
    experiment_name = "Go2-Loco-Skill-Walk-Mid360Depth-10Hz-GRU"
    obs_groups = {
        "policy": ["proprioception_noised", "mid360_depth"],
        "teacher": ["proprioception", "mapScans", "privileged"],
    }
    policy = RslRlDistillationStudentTeacherDepthImageRecurrentCfg(
        init_noise_std=0.05,
        noise_std_type="scalar",
        student_obs_normalization=False,
        teacher_obs_normalization=False,
        student_hidden_dims=[512, 256, 128],
        teacher_hidden_dims=[512, 256, 128],
        activation="elu",
        rnn_type="gru",
        rnn_hidden_dim=256,
        rnn_num_layers=1,
        teacher_recurrent=False,
    )
    algorithm = RslRlDistillationAlignAlgorithmCfg(
        num_learning_epochs=2,
        learning_rate=1.0e-3,
        gradient_length=10,
        optimizer="adam",
        loss_type="mse",
        align_weight_depth=1.0,
        align_weight_privilege=1.0,
    )
