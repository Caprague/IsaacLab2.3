# 总结：Go2-Mid360Depth-10Hz 训练管线现状

> **日期**: 2026-08-10
> **范围**: Go2-Loco-Skill-Walk-Mid360Depth-10Hz（base 版本，非 GRU）当前的配置、底层架构与训练流程
> **状态**: 已补齐 `depth_image_age` 并重训，学生 Stage4 地形可达等级收敛于 **4.7**（教师 ~5.3/有效上限 5.0）

---

## 一、任务注册与入口

`source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/go2/__init__.py`：

| 任务 ID | 环境入口 | RL 入口 |
|---------|----------|---------|
| `Go2-Loco-Skill-Walk-Mid360Depth-10Hz` | `go2_loco_skill_walk_mid360_depth_10hz_cfg.py:Go2LocomotionSkillEnvCfg` | PPO：`rsl_rl_ppo_cfg_walk_mid360_depth_10hz`；蒸馏：`rsl_rl_distillation_cfg_walk_mid360_depth_10hz` |
| `Go2-Loco-Skill-Walk-Mid360Depth-10Hz-Play` | `Go2LocomotionSkillEnvCfg_Play` | 同上 |

脚本入口：`scripts/reinforcement_learning/rsl_rl/train.py`（教师用 `rsl_rl_cfg_entry_point`，学生用 `rsl_rl_distillation_cfg_entry_point`）、`play.py`（回放）。

---

## 二、环境配置

### 2.1 阶段划分（`stage` 参数，默认 `stage1`）

| 阶段 | 内容 | 备注 |
|------|------|------|
| Stage1 | 低速向前，基础 Go2 模型，严格奖励 | 教师 PPO |
| Stage2 | 全向移动，基础模型 | 教师 PPO |
| Stage3 | 全向移动，mid360 模型 | 教师 PPO（特权观测收敛） |
| Stage4 | 学生蒸馏，启用 Mid360 深度图 | 学生训练 |

### 2.2 观测组与维度

| 组 | 用途 | 维度 |
|----|------|------|
| `proprioception` | 教师 | 47×5 = 235 |
| `mapScans` | 教师 | 187（17×11 高度扫描） |
| `privileged` | 教师 | 18×3 = 54 |
| `proprioception_noised` | 学生 | 47×6 = 282（默认 term 主序扁平，MLP 全量消费，布局无关） |
| `mid360_depth` | 学生 | 5760 深度像素 + 1 `depth_image_age` |

### 2.3 动作、地形、传感器

- 动作：12 关节位置（scale 0.125/0.25/0.25，clip ±5）；
- 地形：`SKILL_WALK_PLUS_TERRAINS_HARD_CFG`，`max_init_terrain_level=5`，课程学习；
- 传感器更新：IMU/接触 200Hz、高度扫描 50Hz、`head_mid360_scanner` 10Hz（Stage4，规则网格 5760 射线或 CSV 动态模式）；
- 关键配置：`depth_image_age`（10Hz/50Hz 帧龄标记）已在网络侧接入学生输入；`proprioception_noised` 保持默认 term 主序扁平（本架构全量喂 MLP，不涉及按帧切片）。

---

## 三、教师训练管线（Stage1–3，PPO）

### 3.1 网络架构：`ActorCritic`（原始观测直连）

`rsl_rl/modules/actor_critic.py`：

```
proprio(235) + mapScans(187) + privileged(54) = 476
  ──► actor MLP(512→256→128→12)
  ──► critic MLP(512→256→128→1)
```

- 教师为**原始观测直连 MLP**，无特征编码器（与 GRU 方案的新教师 `ActorCriticScan` 不同）；
- 支持对称性数据增强 + mirror loss（`go2_mid360_teacher_walk`），自适应 lr、`entropy_coef=0.01`。

### 3.2 配置参数

| 参数 | 值 |
|------|-----|
| `num_steps_per_env` | 32 |
| `max_iterations` | Stage1 6001 / Stage2 4001 / Stage3 6001 |
| actor/critic hidden | [512, 256, 128] |
| 归一化 | 关闭（输入已在 env 侧 scale/clip） |

### 3.3 教师 checkpoint 结构

`model_state_dict` 包含：`actor.*`、`critic.*`、`std`（无编码器权重）。

---

## 四、学生蒸馏管线（Stage4）

### 4.1 学生网络架构

`rsl_rl/modules/student_teacher_depth_image.py`：

```
depth(5760) ──► DepthImageEncoder ──► 192 维（avg+max+min 三池化，各 64）
proprio(282) + age(1) + depth_features(192) = 475 ──► student_MLP(512→256→128→12)
```

关键设计（详见 `Go2训练总结/优势分析-Go2-Mid360Depth-10Hz-CNN特征提取.md`）：
- 轴分解卷积（水平 1×3 方位 + 垂直 3×1 俯仰）+ 方位循环 padding，与 Mid360 球面投影几何对齐；
- avg/max/min 三统计池化 → 192 维紧凑富特征，CNN 仅 ~57K 参数；
- 无瓶颈直连 MLP，BC 梯度直达 CNN；
- `depth_image_age` 并入基础观测（475 维输入），提供 10Hz/50Hz 帧龄信息。

### 4.2 教师镜像与冻结

- 教师侧：`teacher` MLP（输入 476），`evaluate()` 直接拼接三组原始观测；
- 教师构造时 `eval()`，蒸馏全程冻结。

### 4.3 蒸馏算法与损失：`Distillation`（BC-only）

`rsl_rl/algorithms/distillation.py`：

```
total_loss = behavior_loss = MSE(student_action, teacher_action)
```

- 无显式 latent 对齐、无 yaw 辅助任务；
- `gradient_length=1` 对前馈学生网络无影响（每步独立）。

### 4.4 配置参数（`rsl_rl_distillation_cfg_walk_mid360_depth_10hz.py`）

| 参数 | 值 |
|------|-----|
| `num_steps_per_env` | 60 |
| `max_iterations` | 5001 |
| `gradient_length` | 1（前馈网络，无 BPTT 需求） |
| `num_learning_epochs` / lr | 2 / 1e-3 |
| 学生/教师 hidden | [512, 256, 128] |
| `teacher_driving` | False |

---

## 五、训练流程（端到端）

```
Stage1–2（基础模型 PPO，ActorCritic）
   │
Stage3（mid360 模型 PPO，特权观测收敛）
   │ checkpoint（actor.* + critic.* + std）
   ▼
Stage4（Distillation 学生蒸馏，BC-only）
   ├─ 加载教师 checkpoint（冻结）
   ├─ 每轮：深度 CNN + 学生 MLP 前向 → action MSE → 梯度更新
   └─ 产出学生策略（DepthImageEncoder 192 维 + 学生 MLP）
```

启动命令：

```bash
# 教师
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
    --task Go2-Loco-Skill-Walk-Mid360Depth-10Hz \
    --agent rsl_rl_cfg_entry_point --headless --num_envs 4096

# 学生
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
    --task Go2-Loco-Skill-Walk-Mid360Depth-10Hz \
    --agent rsl_rl_distillation_cfg_entry_point \
    --load_run ".*" --checkpoint "model_6000" --headless --num_envs 4096
```

---

## 六、涉及文件清单

| 文件 | 作用 |
|------|------|
| `.../go2/go2_loco_skill_walk_mid360_depth_10hz_cfg.py` | base 环境配置（阶段/观测/传感器/奖励，含 `Go2LocomotionSkillEnvCfg_PretrainTeacher` 遗留类） |
| `.../go2/agents/rsl_rl_ppo_cfg_walk_mid360_depth_10hz.py` | 教师 PPO 配置（ActorCritic） |
| `.../go2/agents/rsl_rl_distillation_cfg_walk_mid360_depth_10hz.py` | 学生蒸馏配置（Distillation） |
| `.../go2/__init__.py` | 任务注册 |
| `source/isaaclab_rl/isaaclab_rl/rsl_rl/distillation_cfg.py` | 蒸馏策略配置类（`RslRlDistillationStudentTeacherDepthImageCfg`） |
| `rsl_rl/modules/student_teacher_depth_image.py` | 学生-教师蒸馏模块 |
| `rsl_rl/networks/depth_image_encoder.py` | 深度 CNN（192 维，avg/max/min 池化） |
| `rsl_rl/algorithms/distillation.py` | 蒸馏算法（BC-only） |
| `rsl_rl/runners/distillation_runner.py`、`storage/rollout_storage.py` | 蒸馏 runner / rollout 存储 |

---

## 七、关键修复与演进

1. **`depth_image_age` 接入**：10Hz/50Hz 帧龄标记并入学生输入（学生 MLP 输入 474 → 475），重训后地形等级从 4.5 提升至 **4.7**；
2. **观测历史拼接问题**：base 架构全量扁平喂 MLP，不受 term 主序/帧主序布局影响（GRU 方案的取帧 bug 与 base 无关）；
3. **教师上限确认**：教师 Stage3 收敛于 ~5.3（有效上限 5.0 + 边缘平坦拓展），学生 4.7 距上限约 0.3，瓶颈属单帧 10Hz 感知的结构性上限而非网络容量。

---

## 八、当前可调参数与待办

- **剩余差距（4.7 → 5.0）**：建议先试共享权重双帧拼接（同一 CNN 处理最近两帧 + 各自 age，特征 192+192 进 MLP），或适度增大 CNN 输出（192→256）；
- **蒸馏侧**：可试 `teacher_driving=True`（DAgger 早期引导）与 huber 损失；
- **yaw 预测头**：未实现（如需可后续补充）；
- **注意**：`Go2LocomotionSkillEnvCfg_PretrainTeacher` 类仍保留在 base 环境 cfg 中（对应已废弃的 AE 预训练流程，可考虑清理，不影响训练）。

---

*关联文档：`Go2训练总结/优势分析-Go2-Mid360Depth-10Hz-CNN特征提取.md`、`Go2训练管线/总结-Go2-Mid360Depth-10Hz-GRU-训练管线.md`*
