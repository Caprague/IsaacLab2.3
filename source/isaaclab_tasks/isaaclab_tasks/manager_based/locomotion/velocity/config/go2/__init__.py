# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##

gym.register(
    id="Isaac-Velocity-Flat-Unitree-Go2-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:UnitreeGo2FlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UnitreeGo2FlatPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_flat_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Velocity-Flat-Unitree-Go2-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:UnitreeGo2FlatEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UnitreeGo2FlatPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_flat_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Velocity-Rough-Unitree-Go2-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rough_env_cfg:UnitreeGo2RoughEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UnitreeGo2RoughPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_rough_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Velocity-Rough-Unitree-Go2-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rough_env_cfg:UnitreeGo2RoughEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UnitreeGo2RoughPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_rough_ppo_cfg.yaml",
    },
)

# -------------------------------------------------------------------------------------------------------
# Blind Walk
gym.register(
    id="Go2-Loco-Blind-Walk",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.go2_loco_blind_walk_cfg:Go2LocomotionSkillEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg_blind_walk:UnitreeGo2LocoSkillPPORunnerCfg",
    },
)

gym.register(
    id="Go2-Loco-Blind-Walk-Play",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.go2_loco_blind_walk_cfg:Go2LocomotionSkillEnvCfg_Play",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg_blind_walk:UnitreeGo2LocoSkillPPORunnerCfg",
    },
)

# ------------------------------------------------------------------------------------------------------
# Skill Walk
gym.register(
    id="Go2-Loco-Skill-Walk",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.go2_loco_skill_walk_cfg:Go2LocomotionSkillEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg_walk:UnitreeGo2LocoSkillPPORunnerCfg",
        "rsl_rl_distillation_cfg_entry_point": f"{agents.__name__}.rsl_rl_distillation_cfg_walk:UnitreeGo2LocoSkillDistillationRunnerCfg",
    },
)

gym.register(
    id="Go2-Loco-Skill-Walk-Play",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.go2_loco_skill_walk_cfg:Go2LocomotionSkillEnvCfg_Play",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg_walk:UnitreeGo2LocoSkillPPORunnerCfg",
        "rsl_rl_distillation_cfg_entry_point": f"{agents.__name__}.rsl_rl_distillation_cfg_walk:UnitreeGo2LocoSkillDistillationRunnerCfg",
    },
)

# ------------------------------------------------------------------------------------------------------
# Skill Walk Perception Data Collection
gym.register(
    id="Go2-Loco-Skill-Walk-Data-Collection",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.go2_loco_skill_walk_data_collection_cfg:Go2LocomotionSkillEnvCfg_Play",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg_walk:UnitreeGo2LocoSkillPPORunnerCfg",
        "rsl_rl_distillation_cfg_entry_point": f"{agents.__name__}.rsl_rl_distillation_cfg_walk:UnitreeGo2LocoSkillDistillationRunnerCfg",
    },
)

# -------------------------------------------------------------------------------------------------------
# Attention Walk
gym.register(
    id="Go2-Loco-Attention-Walk",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.go2_loco_attention_walk_cfg:Go2LocomotionSkillEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg_attention_walk:UnitreeGo2LocoSkillPPORunnerCfg",
    },
)

gym.register(
    id="Go2-Loco-Attention-Walk-Play",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.go2_loco_attention_walk_cfg:Go2LocomotionSkillEnvCfg_Play",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg_attention_walk:UnitreeGo2LocoSkillPPORunnerCfg",
    },
)

# -------------------------------------------------------------------------------------------------------
# Zero Agent
gym.register(
    id="Go2-Zero-Agent",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.go2_zero_agent_cfg:Go2LocomotionSkillEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg_zero_agent:UnitreeGo2LocoSkillPPORunnerCfg",
    },
)

gym.register(
    id="Go2-Zero-Agent-Play",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.go2_zero_agent_cfg:Go2LocomotionSkillEnvCfg_Play",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg_zero_agent:UnitreeGo2LocoSkillPPORunnerCfg",
    },
)

# -------------------------------------------------------------------------------------------------------
# Test Env
gym.register(
    id="Go2-Test",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.go2_test:Go2LocomotionSkillEnvCfg_Play",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg_walk:UnitreeGo2LocoSkillPPORunnerCfg",
        "rsl_rl_distillation_cfg_entry_point": f"{agents.__name__}.rsl_rl_distillation_cfg_walk:UnitreeGo2LocoSkillDistillationRunnerCfg",
    },
)

