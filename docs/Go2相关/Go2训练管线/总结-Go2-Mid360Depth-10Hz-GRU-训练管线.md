# 总结：Go2-Mid360Depth-10Hz-GRU 训练管线现状

> **日期**: 2026-08-10
> **范围**: Go2-Loco-Skill-Walk-Mid360Depth-10Hz-GRU 环境当前的配置、底层架构与训练流程
> **状态**: 已完成 Parkour 对齐改造（教师双编码器 + 学生双嵌入 + 显式特征对齐蒸馏），待训练主机验证

---

## 一、任务注册与入口

`source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/go2/__init__.py`：

| 任务 ID | 环境入口 | RL 入口 |
|---------|----------|---------|
| `Go2-Loco-Skill-Walk-Mid360Depth-10Hz-GRU` | `go2_loco_skill_walk_mid360_depth_10hz_gru_cfg.py:Go2LocomotionSkillEnvCfg` | PPO：`rsl_rl_ppo_cfg_walk_mid360_depth_10hz_gru`；蒸馏：`rsl_rl_distillation_cfg_walk_mid360_depth_10hz_gru` |
| `Go2-Loco-Skill-Walk-Mid360Depth-10Hz-GRU-Play` | `Go2LocomotionSkillEnvCfg_Play` | 同上 |

脚本入口：`scripts/reinforcement_learning/rsl_rl/train.py`（教师用 `rsl_rl_cfg_entry_point`，学生用 `rsl_rl_distillation_cfg_entry_point`）、`play.py`（回放）。

---

## 二、环境配置

### 2.1 阶段划分（`stage` 参数，默认 `stage4`）

| 阶段 | 内容 | 备注 |
|------|------|------|
| Stage1 | 低速向前，基础 Go2 模型，严格奖励 | 教师 PPO |
| Stage2 | 全向移动，基础模型 | 教师 PPO |
| Stage3 | 全向移动，mid360 模型 | 教师 PPO（新架构收敛） |
| Stage4 | 学生蒸馏，启用 Mid360 深度图 | 学生训练 |

### 2.2 观测组与维度

| 组 | 用途 | 维度 |
|----|------|------|
| `proprioception` | 教师 | 47×5 = 235 |
| `mapScans` | 教师 | 187（17×11 高度扫描） |
| `privileged` | 教师 | 18×3 = 54 |
| `proprioception_noised` | 学生 | 47×6 = 282，`flatten_history_dim=False` → `(B,6,47)` 帧主序 |
| `mid360_depth` | 学生 | 5760 深度像素 + 1 `depth_image_age` |

### 2.3 动作、地形、传感器

- 动作：12 关节位置（scale 0.125/0.25/0.25，clip ±5）；
- 地形：`SKILL_WALK_PLUS_TERRAINS_HARD_CFG`，`max_init_terrain_level=5`，课程学习；
- 传感器更新：IMU/接触 200Hz、高度扫描 50Hz、`head_mid360_scanner` 10Hz（Stage4，规则网格 5760 射线或 CSV 动态模式）；
- 关键配置：`proprioception_noised` 帧主序（组级 `history_length=6` + `flatten_history_dim=False`，保证取最新帧正确）；`depth_image_age` 已接入学生输入（10Hz/50Hz 时钟信号）。

---

## 三、教师训练管线（Stage1–3，PPO）

### 3.1 网络架构：`ActorCriticScan`（双编码器）

`rsl_rl/modules/actor_critic_scan.py`：

```
mapScans(187) ──► ScanEncoder ──► scan_latent(32)
privileged(54) ──► PrivilegeEncoder ──► privilege_latent(32)
proprio(235) + scan_latent(32) + privilege_latent(32) = 299 ──► actor MLP(512→256→128→12)
                                                            ──► critic MLP(512→256→128→1)
```

- 两个编码器随 PPO **端到端训练**，checkpoint 自带其权重；
- 支持对称性数据增强 + mirror loss（`go2_mid360_teacher_walk`），自适应 lr、`entropy_coef=0.01`。

### 3.2 配置参数

| 参数 | 值 |
|------|-----|
| `num_steps_per_env` | 32 |
| `max_iterations` | Stage1 6001 / Stage2 4001 / Stage3 6001 |
| actor/critic hidden | [512, 256, 128] |
| 编码器 latent | scan 32 / privilege 32（组：`mapScans` / `privileged`） |
| 归一化 | 关闭（输入已在 env 侧 scale/clip） |

### 3.3 教师 checkpoint 结构

`model_state_dict` 包含：`actor.*`、`critic.*`、`scan_encoder.encoder.*`、`privilege_encoder.encoder.*`、`std`。

---

## 四、学生蒸馏管线（Stage4）

### 4.1 学生网络架构

`rsl_rl/modules/student_teacher_depth_image_recurrent.py`：

```
depth(5760) ──► StudentDepthCNN ──► depth_latent(32)
prop_latest(47) + age(1) + depth_latent(32) = 80 ──► input_mlp(80→128→64)
  ──► GRU(64→256) ──► gru_output_mlp(256→128→64)
  ──► depth_latent(32) + privilege_latent(32)
  ──► student_MLP(282+32+32=346 → 512→256→128→12)
```

关键点：
- `prop_latest` 从帧主序组 `(B,6,47)` 经时间主序展平后取尾部 47 维，**最新帧提取正确**；
- GRU 输出两个 32 维嵌入：`depth_latent`（对齐教师 `scan_latent`）与 `privilege_latent`（对齐教师 `privilege_latent`）。

### 4.2 教师镜像与冻结

- 教师侧镜像：`teacher_scan_encoder` + `teacher_privilege_encoder` + teacher MLP（输入 299）；
- `evaluate()` 按 `[proprio, scan_latent, privilege_latent]` 拼接后推理；
- 教师 MLP 与两个编码器构造时即冻结（`requires_grad=False`），蒸馏全程不更新；
- checkpoint 加载：`actor.*`→teacher MLP、`scan_encoder.*`→teacher_scan_encoder、`privilege_encoder.*`→teacher_privilege_encoder（缺失时告警，不崩溃）。

### 4.3 蒸馏算法与损失：`DistillationAlign`

`rsl_rl/algorithms/distillation_align.py`（继承原 `Distillation`，新文件，原管线不受影响）：

```
total_loss = behavior_loss
           + w_depth · MSE(student_depth_latent,   teacher_scan_latent.detach())
           + w_priv  · MSE(student_privilege_latent, teacher_privilege_latent.detach())
```

- 教师 latent 在 `no_grad` 下计算，梯度只流向学生；
- 日志输出 `behavior` / `align_depth` / `align_privilege` 三个分量。

### 4.4 配置参数（`rsl_rl_distillation_cfg_walk_mid360_depth_10hz_gru.py`）

| 参数 | 值 |
|------|-----|
| `num_steps_per_env` | 120（对齐 Parkour，覆盖 12 个 10Hz 深度更新周期） |
| `max_iterations` | 8001 |
| `gradient_length` | 10（BPTT） |
| `num_learning_epochs` / lr | 2 / 1e-3 |
| `align_weight_depth` / `align_weight_privilege` | 1.0 / 1.0 |
| `teacher_driving` | False（可选） |
| rnn | gru，hidden 256，1 层，`teacher_recurrent=False` |

---

## 五、训练流程（端到端）

```
Stage1–2（基础模型 PPO，ActorCriticScan）
   │
Stage3（mid360 模型 PPO，scan/privilege 编码器随训练收敛）
   │ checkpoint（actor.* + scan_encoder.* + privilege_encoder.*）
   ▼
Stage4（DistillationAlign 学生蒸馏）
   ├─ 加载教师 checkpoint（冻结 MLP + 双编码器）
   ├─ 每轮：学生 GRU 前向 → BC + 双 latent 对齐损失 → BPTT（gradient_length=10）
   └─ 产出学生策略（深度 CNN + GRU + 双嵌入 + 学生 MLP）
```

启动命令：

```bash
# 教师
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
    --task Go2-Loco-Skill-Walk-Mid360Depth-10Hz-GRU \
    --agent rsl_rl_cfg_entry_point --headless --num_envs 4096

# 学生（--load_run 指向新教师 run）
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
    --task Go2-Loco-Skill-Walk-Mid360Depth-10Hz-GRU \
    --agent rsl_rl_distillation_cfg_entry_point \
    --load_run ".*" --checkpoint "model_6000" --headless --num_envs 4096
```

---

## 六、涉及文件清单

| 文件 | 作用 |
|------|------|
| `.../go2/go2_loco_skill_walk_mid360_depth_10hz_gru_cfg.py` | GRU 环境配置（阶段/观测/传感器/奖励） |
| `.../go2/agents/rsl_rl_ppo_cfg_walk_mid360_depth_10hz_gru.py` | 教师 PPO 配置（ActorCriticScan） |
| `.../go2/agents/rsl_rl_distillation_cfg_walk_mid360_depth_10hz_gru.py` | 学生蒸馏配置（DistillationAlign） |
| `.../go2/__init__.py` | 任务注册 |
| `source/isaaclab_rl/isaaclab_rl/rsl_rl/rl_cfg.py` | `RslRlPpoActorCriticScanCfg` |
| `source/isaaclab_rl/isaaclab_rl/rsl_rl/distillation_cfg.py` | 蒸馏策略/算法配置类（`...RecurrentCfg`、`RslRlDistillationAlignAlgorithmCfg`） |
| `rsl_rl/modules/actor_critic_scan.py` | 教师双编码器 Actor-Critic |
| `rsl_rl/modules/student_teacher_depth_image_recurrent.py` | 学生-教师蒸馏模块（GRU 双嵌入） |
| `rsl_rl/networks/scan_encoder.py` | 教师 scan 编码器 |
| `rsl_rl/networks/teacher_encoders.py` | 教师 privilege 编码器（纯编码器版） |
| `rsl_rl/networks/student_depth_cnn.py`、`memory.py` | 学生 CNN / GRU 记忆模块 |
| `rsl_rl/algorithms/distillation.py`、`distillation_align.py` | 原蒸馏算法 / 对齐蒸馏算法 |
| `rsl_rl/runners/distillation_runner.py`、`storage/rollout_storage.py` | 蒸馏 runner / rollout 存储 |

---

## 七、已实施的关键修复与演进时间线

1. **观测历史拼接 bug 修复**：`flatten_history_dim=False` 帧主序 + 时间主序展平，最新本体帧提取正确（详见 `Go2训练总结/重要经验-IsaacLab-观测历史拼接.md`）；
2. **`depth_image_age` 接入**：10Hz/50Hz 帧龄时钟信号进入学生输入（base 与 GRU 两模块）；
3. **`gradient_length` 1 → 10**：GRU 获得 BPTT 时序梯度；
4. **删除 AE 预训练**：`scripts/tools/encoder_pretrain/` 移除，改为教师 PPO 端到端训练编码器；
5. **教师双编码器**：`ActorCriticScan`（scan 187→32 + privilege 54→32，actor 输入 299）；
6. **学生双嵌入**：GRU 输出 depth_latent + privilege_latent（学生 MLP 输入 346）；
7. **显式特征对齐蒸馏**：`DistillationAlign`（BC + 加权 latent 对齐），yaw 辅助任务未纳入（无相关适配设定）。

---

## 八、当前可调参数与待办

- **对齐权重**：`align_weight_depth` / `align_weight_privilege`（当前 1.0/1.0），可按 loss 分量调；
- **`teacher_driving`**：可选开启（约 2500 iter 前教师驱动，DAgger 风格）；
- **yaw 预测头**：未实现（如需可后续补充环境 yaw 真值观测）；
- **验证**：需在训练主机重训 Stage1–3（新教师架构）→ Stage4'（对齐蒸馏），对比 base 学生 4.7 与教师上限 5.0；
- **checkpoint 兼容性**：新教师/学生结构与旧版不兼容，旧 GRU checkpoint 无法复用；base 环境管线不受影响。

---

*关联文档：`Go2参考改进/参考-Parkour-GRU方案训练改进.md`、`Go2参考改进/参考-Parkour-深度特征提取训练.md`、`Go2训练总结/失败分析-Go2-Mid360Depth-10Hz-GRU-时序融合.md`*
