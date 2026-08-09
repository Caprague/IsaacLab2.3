# 项目参考：Parkour 对齐的 GRU 方案训练改进

> **日期**: 2026-08-09
> **状态**: 方案设计，待评审后实施
> **适用**: Go2-Loco-Skill-Walk-Mid360Depth-10Hz-GRU 环境
> **一句话结论**: 当前 GRU 方案失败的根因之一是训练管线与 Parkour 的真实做法脱节——教师侧没有"融合训练的特征编码器"（scan_encoder），学生侧没有"注入点对齐"（depth_latent 顶替教师 scan_latent 槽位）和 yaw 辅助任务。改进方向是把 GRU 方案升级为严格的 Parkour 两阶段管线：教师 PPO 阶段端到端训练特征编码器，学生蒸馏阶段对齐特征注入点与策略行为。

---

## 一、当前 GRU 方案训练介绍

### 1.1 任务注册与入口

`source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/go2/__init__.py`：

- `Go2-Loco-Skill-Walk-Mid360Depth-10Hz-GRU` → `go2_loco_skill_walk_mid360_depth_10hz_gru_cfg.py:Go2LocomotionSkillEnvCfg`
  - PPO 入口：`rsl_rl_ppo_cfg_walk_mid360_depth_10hz_gru:UnitreeGo2LocoSkillPPORunnerCfg`
  - 蒸馏入口：`rsl_rl_distillation_cfg_walk_mid360_depth_10hz_gru:UnitreeGo2LocoSkillDistillationRunnerCfg`
- `Go2-Loco-Skill-Walk-Mid360Depth-10Hz-GRU-Play` → `Go2LocomotionSkillEnvCfg_Play`

训练入口脚本：`scripts/reinforcement_learning/rsl_rl/train.py`（`--agent rsl_rl_cfg_entry_point` 跑 Stage1–3 教师；`--agent rsl_rl_distillation_cfg_entry_point` 跑 Stage4 学生）；回放：`play.py`。

### 1.2 环境配置（GRU cfg）

`go2_loco_skill_walk_mid360_depth_10hz_gru_cfg.py`，`stage` 默认 `stage4`，四阶段：

| 阶段 | 内容 | 模型/传感器 |
|------|------|-------------|
| Stage1 | 低速向前，基础 Go2 模型，严格奖励 | 无 mid360 |
| Stage2 | 全向移动，基础模型 | 无 mid360 |
| Stage3 | 全向移动，mid360 模型（教师收敛） | 无深度图（教师用特权观测） |
| Stage4 | 学生蒸馏，启用 Mid360 深度图 | `head_mid360_scanner`（10Hz） |

观测组（维度已核对）：

| 组 | 用途 | 维度 |
|----|------|------|
| `proprioception` | 教师 | 47×5 = 235 |
| `mapScans` | 教师 | 187（17×11 高度扫描） |
| `privileged` | 教师 | 18×3 = 54（gait 2 + 接触 4 + 线速度 3 + feet_dist 4 + base 高 1 + 足高 4） |
| `proprioception_noised` | 学生 | 47×6 = 282，`flatten_history_dim=False` → `(B,6,47)` 帧主序 |
| `mid360_depth` | 学生 | 5760 深度像素 + 1 `depth_image_age` |

动作：12 关节位置（scale 0.125/0.25/0.25，clip ±5）。地形：`SKILL_WALK_PLUS_TERRAINS_HARD_CFG`，`max_init_terrain_level=5`，课程学习。传感器更新：IMU/接触 200Hz、高度扫描 50Hz、mid360 10Hz（Stage4）。

### 1.3 教师训练管线（Stage1–3，PPO）

`rsl_rl_ppo_cfg_walk_mid360_depth_10hz_gru.py`：

- `ActorCritic` 512/256/128，`obs_groups` = `["proprioception","mapScans","privileged"]`（policy/critic 相同，476 维直连）；
- 对称性数据增强 + mirror loss（`go2_mid360_teacher_walk`）；
- 自适应 lr、`entropy_coef=0.01`、`num_steps_per_env=32`；
- `env_stage` 参数切换 Stage1/2/3（max_iterations 6001/4001/6001）。

**关键事实**：当前教师是"原始观测直连 MLP"，**没有把高度扫描压缩成特征编码器**；教师策略本身没有 `scan_latent` 注入点。

### 1.4 学生蒸馏管线（Stage4）

`rsl_rl_distillation_cfg_walk_mid360_depth_10hz_gru.py`：

- `RslRlDistillationStudentTeacherDepthImageRecurrentCfg`：GRU（`rnn_type="gru"`、`rnn_hidden_dim=256`、1 层、`teacher_recurrent=False`），学生/教师 MLP 均为 512/256/128，ELU；
- 算法：2 epochs、lr=1e-3、`gradient_length=10`（已从 1 修正）、MSE、`teacher_driving=False`；
- `num_steps_per_env=60`、`max_iterations=8001`、`save_interval=250`。

### 1.5 网络结构（当前实现）

`rsl_rl/modules/student_teacher_depth_image_recurrent.py`：

```
学生:
  depth(5760) → StudentDepthCNN → depth_latent(32)
  prop_latest(47) + age(1) + depth_latent(32) = 80 → input_mlp(80→128→64)
  → GRU(64→256) → output_mlp(256→128→64) → hs(32)+priv(32)
  → student_MLP(282+32+32=346 → 512→256→128→12)
教师:
  teacher_MLP(235+187+54=476 → 512→256→128→12)，冻结
  HeightScanEncoder(187→32)、PrivilegeEncoder(54→32)：加载冻结，未参与前向/损失
```

### 1.6 涉及文件清单

| 文件 | 作用 |
|------|------|
| `.../go2/go2_loco_skill_walk_mid360_depth_10hz_gru_cfg.py` | GRU 环境配置（阶段/观测/传感器/奖励） |
| `.../go2/agents/rsl_rl_ppo_cfg_walk_mid360_depth_10hz_gru.py` | 教师 PPO 配置（Stage1–3） |
| `.../go2/agents/rsl_rl_distillation_cfg_walk_mid360_depth_10hz_gru.py` | 学生蒸馏配置（Stage4） |
| `.../go2/__init__.py` | 任务注册 |
| `source/isaaclab_rl/isaaclab_rl/rsl_rl/distillation_cfg.py` | 蒸馏策略配置类 |
| `rsl_rl/modules/student_teacher_depth_image_recurrent.py` | GRU 学生-教师模块 |
| `rsl_rl/networks/student_depth_cnn.py` | 学生深度 CNN（32 维） |
| `rsl_rl/networks/teacher_encoders.py` | 教师辅助编码器（当前未使用） |
| `rsl_rl/networks/memory.py` | GRU/LSTM 记忆模块 |
| `rsl_rl/algorithms/distillation.py` | 蒸馏算法（BC-only 损失） |
| `rsl_rl/runners/distillation_runner.py` | 蒸馏 runner（checkpoint 加载/冻结编码器） |
| `rsl_rl/storage/rollout_storage.py` | 蒸馏 rollout 存储 |
| `scripts/reinforcement_learning/rsl_rl/train.py`、`play.py` | 训练/回放入口 |

### 1.7 当前已修复/已调整项

- 本体感知组帧主序：`flatten_history_dim=False` → `(B,6,47)`，取最新帧逻辑正确（见 `重要经验-IsaacLab-观测历史拼接.md`）；
- `depth_image_age` 已并入学生输入（base 与 GRU 两模块）；
- `gradient_length` 已从 1 修正为 10；
- `scripts/tools/encoder_pretrain/`（AE 预训练脚本）已删除，`distillation_runner.py` 中相关提示文案已更新，GRU 环境 docstring 的 `--load_run` 示例已改为直接加载教师 checkpoint。

---

## 二、Parkour 参考要点回顾

详见 `docs/ref_analysis_docs/参考-Parkour-深度特征提取训练.md`，核心：

```
教师: proprio + scan_encoder(scan dots 132→32) + privileged → actor（PPO 端到端训练 scan_encoder）
学生: depth → CNN(32) → combination_mlp(prop+32→32) → GRU(512) → output(32 depth_latent + 2 yaw)
      depth_actor(obs_student, scandots_latent=depth_latent)  ← 注入点与教师 scan_latent 对齐
损失: depth_actor_loss(BC) + yaw_loss；只启用 update_depth_actor（纯特征蒸馏被注释）
配套: GRU hidden detach；update_interval=5 → rollout 120 步；yaw 用 delta_yaw_ok 门控
```

---

## 三、当前方案与 Parkour 的差距分析

| 维度 | Parkour | 当前 GRU 方案 | 差距性质 |
|------|---------|---------------|----------|
| 教师特征编码器 | scan_encoder 在教师 PPO 内**端到端训练**，输出 32 维 latent | 教师 MLP 直连 476 维原始观测；编码器是独立 AE（已删） | **结构性缺失** |
| 特征注入点 | 学生 `scandots_latent=depth_latent` 顶替教师 `scan_latent` 槽位 | 学生 hs/priv 两个无监督 latent，与教师无对应关系 | **结构性缺失** |
| 学生损失 | BC + yaw_loss | 仅 BC | 缺失 yaw 辅助任务 |
| GRU 输出 | 32 维 depth_latent（+2 yaw） | 64 维拆 hs/priv | 与注入点不匹配 |
| 时序训练 | hidden detach + rollout 120 | `gradient_length=10`（已对齐），rollout 60 | 基本对齐，rollout 可加长 |
| 教师驱动 | 学生阶段教师冻结 | 冻结（`teacher_driving=False`） | 可选 DAgger |
| 特权估计 | Estimator + DAgger 历史编码器 | 无（教师直接用 privileged） | 可选升级 |

**结论**：当前 GRU 方案的失败，本质是"时序外壳"（GRU）套在了一个既没有教师特征编码器、也没有注入点对齐的残缺蒸馏管线上。

---

## 四、改进方案：严格对齐 Parkour 的升级训练

### 4.1 总体架构升级图

```
教师阶段（Stage3'，PPO 端到端）:
  proprio(235) ──────────────┐
  mapScans(187) → ScanEncoder(187→32) ─→ scan_latent(32) ─→ concat(321) → actor_MLP → 教师动作
  privileged(54) ────────────┘
  （scan_encoder 随 PPO 一起训练，checkpoint 自带编码器权重）

学生阶段（Stage4'，蒸馏）:
  depth(5760)+age → StudentDepthCNN → depth_latent(32)
  prop_latest + depth_latent → input_mlp → GRU → output_mlp → depth_latent(32) [+ yaw(2)]
  student_actor(本体 282 + depth_latent(32) 注入) → 学生动作
  损失: MSE(student_act, teacher_act) + yaw_loss
  （depth_latent 的注入槽位 == 教师 scan_latent 槽位）
```

### 4.2 教师阶段：融合训练特征编码器

目标：让教师策略在 PPO 训练中端到端学习"高度扫描 → 32 维地形特征"的能力，替代已删除的 AE 预训练。

1. **新增教师 Actor-Critic 类**（如 `rsl_rl/modules/teacher_actor_critic_scan.py`）：
   - actor 输入 = `proprioception(235) + scan_latent(32) + privileged(54)` = 321；
   - `ScanEncoder`：复用 `HeightScanEncoder` 的 encoder 结构（187 → 32）或按 Parkour `scan_encoder_dims=[128,64,32]` 实现，**作为 actor 子模块随 PPO 端到端训练**；
   - critic 输入可保持 476 原始观测（或同样 321）；
2. **教师 PPO 配置**：`rsl_rl_ppo_cfg_walk_mid360_depth_10hz_gru.py` 的 `policy.class_name` 改为新类，`obs_groups` 不变（mapScans 仍保留，由网络内部编码）；
3. **训练**：Stage3' 重训（max_iterations 6001），checkpoint 直接包含 scan_encoder 权重；
4. **可选升级（暂缓）**：Estimator（proprio → privileged）+ DAgger 历史编码器——当前教师直接使用 privileged 输入，先不引入 RMA 复杂度。

### 4.3 学生阶段：对齐特征注入点与策略行为

1. **重构 `StudentTeacherDepthImageRecurrent`**：
   - GRU 输出改为 **32 维 `depth_latent`**（+ 可选 2 维 yaw），去掉 hs/priv 无监督拆分；
   - 学生 actor 输入 = 本体全量（282，保留时间主序展平）+ `depth_latent(32)` 注入槽位——与教师 `scan_latent` 槽位一一对应；
   - 删除/停用学生侧无用的 `HeightScanEncoder`/`PrivilegeEncoder`（教师 checkpoint 自带 scan_encoder，蒸馏时教师整体冻结即可）；
2. **增加 yaw 辅助任务**：
   - 学生 yaw 预测头输出 2 维；目标 = 真实偏航观测（需在环境观测组补充 yaw 真值，或从命令 heading/IMU 偏航导出）；
   - `yaw_loss = ‖yaw_teacher - yaw_student‖₂`，与 BC 一起优化；
3. **损失与优化器**：
   - 总损失 = BC(action MSE) + yaw_loss；**不引入显式 latent 损失**（与 Parkour 实际只启用 `update_depth_actor` 一致）；
   - `distillation.py` 优化器改为只含学生参数（当前包含冻结编码器，无害但需清理）；
4. **时序训练配置**：
   - `gradient_length=10`（已改）；
   - `num_steps_per_env` 60 → **120**（覆盖 12 个 10Hz 深度更新周期，对齐 Parkour）；
   - `teacher_driving` 可选开启（约 2500 iter 前教师驱动，DAgger 风格）。

### 4.4 训练流程与阶段划分

| 阶段 | 内容 | 产物 |
|------|------|------|
| Stage1–2 | 不变（基础模型 PPO） | 教师基础策略 |
| Stage3' | 新教师架构 PPO（scan_encoder 端到端） | 教师 checkpoint（含 scan_latent 能力） |
| Stage4' | 新学生蒸馏（注入点对齐 + BC + yaw） | 学生策略（GRU + depth_latent 注入） |

AE 预训练步骤整体移除（`encoder_pretrain/` 已删除）。

### 4.5 实施步骤与文件改动点

1. **新增** `rsl_rl/modules/teacher_actor_critic_scan.py`：教师 Actor-Critic + ScanEncoder（或复用 `rsl_rl/networks/teacher_encoders.py` 的 encoder 结构）；
2. **修改** `rsl_rl_ppo_cfg_walk_mid360_depth_10hz_gru.py`：`policy.class_name` → 新类，新增 `scan_encoder` 相关参数；
3. **修改** `rsl_rl/modules/student_teacher_depth_image_recurrent.py`：GRU 输出 32(+2)、学生注入点重构、移除 hs/priv；
4. **修改** `rsl_rl/algorithms/distillation.py`：增加 yaw_loss；优化器仅含学生参数；
5. **修改** `rsl_rl/runners/distillation_runner.py`：教师 checkpoint 加载逻辑适配新架构（整体冻结 actor，含 scan_encoder）；移除对顶层 `height_scan_encoder` key 的依赖（或保留兼容分支）；
6. **修改** `rsl_rl_distillation_cfg_walk_mid360_depth_10hz_gru.py`：yaw 标志、`num_steps_per_env=120`、`teacher_driving` 等；
7. **修改** 环境 cfg：如需 yaw 真值，在观测组补充偏航项；
8. **更新** GRU 环境 docstring 与蒸馏配置注释（已部分完成：`--load_run` 指向教师 checkpoint）。

### 4.6 兼容性与回退

- **checkpoint 不兼容**：教师（新架构）与学生（新输入）两侧均需重训，旧 GRU checkpoint 无法复用；
- **base 环境不受影响**：base 的教师/学生管线独立，继续用 192 维直连架构；
- **回退**：保留当前 `StudentTeacherDepthImageRecurrent` 实现，可加配置开关切换旧（hs/priv）与新（depth_latent 注入）学生结构，便于 A/B 对比。

---

## 五、风险与验证

### 风险

| 风险 | 等级 | 缓解 |
|------|:----:|------|
| 教师重训成本（Stage3' 6000 iter） | 中 | 可在现有 Stage3 checkpoint 基础上热启动（若网络结构兼容） |
| yaw 真值定义不明确 | 中 | 优先用 base 朝向/IMU 偏航或命令 heading；先做小实验验证可学性 |
| 注入点对齐后学生仍依赖多帧历史 | 低 | 保留 282 维本体历史作为额外输入，depth_latent 注入作为"地形特征槽位" |
| 学生 GRU 训练不稳定 | 中 | 保持 `gradient_length=10`，必要时先 BC warmup 再开 yaw |

### 验证指标

- 学生地形可达等级：目标 ≥ base 的 4.7，冲刺教师上限 5.0；
- 教师 scan_latent 与深度 depth_latent 的分布可视化（t-SNE/统计），确认注入点对齐有效；
- yaw 预测误差收敛曲线；
- 训练监控：BC loss、yaw loss、terrain level 曲线与 base 方案对照。

---

*关联文档：`docs/ref_analysis_docs/参考-Parkour-深度特征提取训练.md`、`docs/go2_analysis_docs/失败分析-Go2-Mid360Depth-10Hz-GRU-时序融合.md`、`docs/go2_analysis_docs/优势分析-Go2-Mid360Depth-10Hz-CNN特征提取.md`*
