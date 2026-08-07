# Go2 Mid360Depth 学生策略分析报告与方案A实施记录

> **日期**: 2026-08-08
> **范围**: Go2-Loco-Skill-Walk-Mid360Depth-10Hz（base）与 Go2-Loco-Skill-Walk-Mid360Depth-10Hz-GRU（GRU）两环境的 Stage4 学生蒸馏
> **状态**: 分析完成，方案A（接入 depth_image_age）已实施于根目录 `rsl_rl/`，待同步训练主机验证

---

## 一、背景与问题

- base 环境 Stage4 学生策略最大可达地形等级止步于 **4.5**（地形最大等级 5.0）；
- 为突破 4.5，参照 extreme Parkour 设计了 GRU 学生网络（见 `docs/go2_analysis_docs/go2_gru_student_policy_architecture_redesign.md`），并编写 `scripts/tools/encoder_pretrain/pretrain_teacher_encoders.py` 补偿性预训练教师特征编码器；
- 最终 GRU 方案训练出的学生策略最大可达地形**不升反降，止步于 3.0**。

本文件记录根因分析结论，以及方案A（以 base 特征提取方式为基线，补上 `depth_image_age` 观测输入）的实施细节。

---

## 二、现状核实：两个环境实际运行的网络

| 项目 | base（10Hz） | GRU（10Hz-GRU） |
|------|--------------|------------------|
| 学生深度编码器 | `DepthImageEncoder` → **192 维**（avg+max+min 三池化各 64） | `StudentDepthCNN` → **32 维** → MLP(79→64) → GRU(64→256) → MLP(256→64) → 拆分 hs(32)+priv(32) |
| 学生策略输入 | 本体 282 + 深度 192 = **474** | 本体 282 + hs 32 + priv 32 = **346** |
| 教师 | MLP(476) 直接吃原始观测（235+187+54） | 同左，另挂 `HeightScanEncoder`/`PrivilegeEncoder`（加载冻结，未参与前向） |
| 蒸馏损失 | action MSE（行为克隆） | action MSE（与 base **完全相同**） |
| `depth_image_age` | 被切片丢弃 | 被切片丢弃 |
| 历史帧顺序 | 全量 282 维喂 MLP，布局无关 | **GRU 的 `[:, -47:]` 取法错误**（term 主序拼接，详见 `go2_mid360_proprio_history_layout_analysis.md`，已修复） |

关键维度（与文档数字的差异）：
- 学生本体每帧 = phase 1 + 速度指令 4 + 角速度 3 + 重力投影 3 + 关节位置 12 + 关节速度 12 + 动作 12 = **47 维**（历史 6 → 282）；
- 教师 privileged 每帧 = gait mask **2** + 接触 mask 4 + 线速度 3 + feet_distance 4 + base 高度 1 + 四足高度 4 = **18 维**（历史 3 → 54）。设计文档写的"60 维（20/帧）"有误，代码默认 `PrivilegeEncoder(input_dim=54)` 与实际一致；
- 教师观测总计 = 235 + 187 + 54 = **476 维**（文档写 482，同样有误，但代码动态计算维度，不受影响）。

---

## 三、根因分析（按影响排序）

### 1. 核心问题：特征对齐损失从未实现（GRU 方案失败的根本原因）

重设计文档第 5.3 节规划的损失为：

```
总 Loss = BC_loss + α * hs_loss + β * priv_loss
```

但实际代码中：
- `rsl_rl/algorithms/distillation.py` 的 `update()` **只计算行为克隆损失**（MSE 学生动作 vs 教师动作）；
- `rsl_rl/modules/student_teacher_depth_image_recurrent.py` 中的 `height_scan_encoder`/`privilege_encoder` 只在 `__init__` 创建、`train()` 置 eval，**从未在任何前向计算或损失中被调用**；
- `rsl_rl/runners/distillation_runner.py` 仅负责从 `model_teacher.pt` 加载并冻结编码器权重。

因此：教师编码器预训练对 GRU 学生训练**零影响**（既无帮助也不致害）；学生 GRU 输出的 `hs_latent`/`priv_latent` 是无监督约束的任意内部特征。性能下降来自新架构本身在相同 BC 目标下更难优化，而非预训练方式好坏。

### 2. `gradient_length=1` 使 GRU 失去时序学习能力

两个蒸馏配置均设 `gradient_length=1`（`agents/rsl_rl_distillation_cfg_walk_mid360_depth_10hz_gru.py`）。对前馈网络无影响；对 GRU 意味着每次 backward 只回传 1 步（BPTT 截断），隐状态仅以"值"前传、得不到时间维度梯度。10Hz 深度图在 50Hz 策略下每帧重复 5 次，GRU 本应学会"识别重复/老化 → 保持或外推状态"，但缺乏时序梯度支撑。rsl_rl 循环蒸馏的常规取值为 10~24。

### 3. 32 维瓶颈 + 更长特征路径 + 无辅助信号，BC 梯度不足以维持地形信息

- base：192 维深度特征（avg+max+min）**直连**策略 MLP，路径短、信息冗余高；
- GRU：深度信息需穿过 32 维瓶颈 → 输入 MLP → GRU → 输出 MLP 才到达策略，且无 latent 监督。

BC 梯度经长路径回传至 CNN 时大幅衰减，学生更容易"忽略深度、只靠本体历史"——在陡峭地形上表现为地形等级骤降（4.5 → 3.0）。

### 4. 专为 10Hz/50Hz 同步设计的 `depth_image_age` 被两个网络同时丢弃

环境观测 `mid360_depth` = 5760 深度像素 + 1 维 `depth_image_age`（变更日志明确其用途为"深度图帧标记/多速率同步"），但两个网络的 `get_student_obs` 均使用 `depth_flat[:, :5760]` 切片，**age 标记被丢弃**。base 的 4.5 成绩是在无此信号下取得的；GRU 想用时序补偿感知陈旧性，却连"这帧多旧"的时钟信号都没有。**这是本次方案A的直接修复点。**

### 5. 预训练方式本身的次要问题（接入 latent 损失前必须解决）

- AE 重建最大化"输入信息量"而非"与动作/地形相关的信息量"；
- 数据采集使用 `Go2LocomotionSkillEnvCfg_PretrainTeacher`：`curriculum=False`、`max_init_terrain_level=None`，基本不覆盖高难度地形；
- 文档设计的推力扰动（`--push_interval`、`set_root_velocities` 注入）**在实际脚本中完全未实现**；
- 文档与代码已脱节（privileged 60 vs 54、teacher_driving 设计 True/2500 实际 False、yaw 预测头未实现）。

### 6. 前提检查：教师自身的上限

蒸馏上限 = 教师上限。若 Stage3 教师用特权观测跑 terrain 5.0 本身成功率低，则 4.5 的瓶颈在教师/课程/奖励，而非学生网络。任何网络改动前应先用 Stage3 teacher checkpoint 在 terrain 5.0 上做特权回放验证。

---

## 四、结论

1. **根因既不是"教师编码器训练方式不合理"，也不是"特征嵌入维度过低"（32 维与 Parkour 一致，不是问题）**，而是 GRU 方案的"特征对齐训练管线从未真正实现"，叠加 `gradient_length=1` 截断时序梯度、32 维瓶颈 + 长路径 + 无辅助监督、以及 age 信号缺失。
2. **改进方向**：以 base 环境的特征提取方式为基线收敛 4.5→5.0（低风险首选）；若确需时序能力，则按 Parkour 的真实做法（注入点对齐 + BC，或补显式 latent 损失并修复训练超参）完整实现 GRU 方案。

---

## 五、方案A实施记录（本次改动）

### 5.1 改动目标

把 `depth_image_age` 接入学生网络输入：
- base：`StudentTeacherDepthImage` —— age 并入学生基础观测向量（与 282 维本体一起送入策略 MLP）；
- GRU：`StudentTeacherDepthImageRecurrent` —— age 作为时钟信号并入 **GRU 输入**（`prop_latest + depth_aux + depth_latent`），供时序模块感知深度帧陈旧度。

### 5.2 修改文件与内容

#### a) `rsl_rl/modules/student_teacher_depth_image.py`

```python
# __init__：计算深度观测的辅助通道维度并计入学生基础观测
self.depth_aux_dim = max(0, obs[self.depth_obs_group].shape[-1] - self.depth_flat_dim)
num_student_basic_obs += self.depth_aux_dim

# get_student_obs：从深度观测尾部取出 aux 通道并入 basic_obs
depth_aux = depth_flat[:, self.depth_flat_dim:]
basic_obs_list.append(depth_aux)
```

#### b) `rsl_rl/modules/student_teacher_depth_image_recurrent.py`

```python
# __init__：计算 depth_aux_dim
self.depth_aux_dim = max(0, obs[self.depth_obs_group].shape[-1] - self.depth_flat_dim)

# GRU 输入 MLP 维度：proprio_per_frame + depth_aux_dim + 32
self.gru_input_mlp = MLP(input_dim=self.proprio_per_frame + self.depth_aux_dim + 32, ...)

# _compute_latents_and_action：把 age 拼入 GRU 输入
depth_aux = obs[self.depth_obs_group][:, self.depth_flat_dim:]
gru_input = torch.cat([prop_latest, depth_aux, depth_latent], dim=-1)
```

### 5.3 维度变化

| 项目 | 修改前 | 修改后 |
|------|--------|--------|
| base 学生策略输入 | 282 + 192 = 474 | 282 + 1 + 192 = **475** |
| GRU 输入 | 47 + 32 = 79 | 47 + 1 + 32 = **80** |
| GRU 学生策略输入 | 282 + 64 = 346 | 346（不变） |
| 教师输入 | 476 | 476（不变，teacher checkpoint 完全兼容） |

### 5.4 兼容性与注意事项

- **学生侧 checkpoint 不兼容**：学生 MLP/GRU 输入层维度变化，Stage4 需重新训练（旧蒸馏 checkpoint 无法直接加载）；
- **教师侧完全兼容**：`model_teacher.pt` 中教师 MLP（476 维）与编码器权重不受影响；
- **向后兼容**：`depth_aux_dim` 用 `max(0, ...)` 计算，若某环境观测组无 aux 通道（纯 5760 维），行为与原来完全一致；
- 若 `student_obs_normalization=True`，base 模块的归一化器维度已同步计入 aux（本仓库配置为 False，不受影响）。

### 5.5 训练主机上的验证步骤

```bash
# 1. 启动训练，确认日志出现维度打印
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
    --task Go2-Loco-Skill-Walk-Mid360Depth-10Hz \
    --agent rsl_rl_distillation_cfg_entry_point \
    --load_run <stage3_run> --checkpoint model_6000 \
    --headless --num_envs 4096
# 预期：Student MLP 输入 475（原 474）

# 2. 教师上限诊断（先于学生调参）
# 用 Stage3 teacher 在 terrain 5.0 上特权回放，确认教师自身能过 5.0

# 3. 监控 Stage4 训练：行为克隆 loss 收敛 + 地形等级曲线，对比 base 4.5 基线
```

---

## 六、方案B要点（GRU 完整实现，待定后实施）

若确需 GRU 时序能力，按 Parkour 参考实现（`docs/ref_analysis_docs/depth_feature_extraction_analysis.md`）补齐：

1. **特征对齐二选一**：
   - 推荐：Parkour 原版注入点对齐 —— 学生 MLP 输入 = 本体历史 + `depth_latent`，纯 BC 训练（参考代码中纯特征蒸馏 `update_depth_encoder` 实际被注释，仅用 `update_depth_actor`）；
   - 或按文档 5.3 补显式 latent 损失：`Distillation.update()` 中教师编码器冻结输出 `teacher_hs/priv`（detach），学生 `hs/priv` 回归，损失 = BC + α·hs + β·priv；同时优化器改为只含学生参数；
2. **`gradient_length` 改为 10~24**（GRU 成立的前提）；
3. **`depth_image_age` 已接入 GRU 输入**（本次改动已包含）；
4. **保留 32 维 latent**，不回退 192 维（瓶颈不是问题，缺监督才是）；
5. 教师 driving 早期引导（`teacher_driving=True`，约 2500 iter 切换）；
6. 若保留显式 latent 损失：预训练需覆盖 terrain 3–5 + 实现推力扰动；更优做法是任务对齐预训练（回归教师内部特征），而非纯 AE 重建；
7. 可选：补 Parkour 的 yaw 预测头（影响转向，非地形等级主因）。

---

## 七、后续验证清单

- [ ] Stage3 教师 terrain 5.0 特权回放成功率（判断瓶颈在教师还是学生）；
- [ ] base + age 信号 Stage4 重训，对比原 4.5 基线；
- [ ] 若仍 < 5.0：尝试深度双帧拼接（最近两帧 + 各自 age）、CNN 输出 192→256、`teacher_driving=True`、huber 损失；
- [ ] GRU 方案按方案B清单决策是否继续。
