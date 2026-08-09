# 项目参考：Parkour 深度图特征提取模块的训练分析

> **日期**: 2026-08-09（重塑自 `docs/ref_analysis_docs/depth_feature_extraction_analysis.md`）
> **适用对象**: 本项目学生视觉蒸馏（Stage4）参考 extreme Parkour 训练方法时的对照依据
> **一句话结论**: Parkour 的深度特征提取模块用**两阶段蒸馏**训练——先 PPO 训练带特权信息的教师，再以**动作级行为克隆（BC）为主**把深度感知蒸馏进学生；"特征对齐"靠的是**注入点对齐**（深度编码器输出 32 维，顶替教师 `scan_encoder` 的特征槽位），而不是显式的 latent 回归损失（纯特征蒸馏 `update_depth_encoder` 在代码中被注释、实际未用）。

---

## 一、模块定位与整体架构

该深度图特征提取模块属于 **Student 策略（视觉蒸馏策略）** 的核心组件，采用 **教师-学生蒸馏（Teacher-Student Distillation）** 范式。核心代码分布在：

| 文件 | 作用 |
|------|------|
| `rsl_rl/modules/depth_backbone.py` | 深度图编码器骨干网络定义 |
| `rsl_rl/runners/on_policy_runner.py` | 训练流程编排（`learn_vision`） |
| `rsl_rl/algorithms/ppo.py` | 损失计算与优化器（`update_depth_actor` 等） |
| `legged_gym/envs/base/legged_robot.py` | 深度图采集与预处理 |
| `legged_gym/envs/base/legged_robot_config.py` | 深度图与训练超参配置 |

模块由两个子网络组成：

```
DepthOnlyFCBackbone58x87 (CNN骨干)  →  RecurrentDepthBackbone (时序融合)
```

---

## 二、网络结构

### 1. DepthOnlyFCBackbone58x87 —— 空间特征提取（CNN）

`depth_backbone.py` L70-L101

| 层 | 操作 | 输出形状 | 说明 |
|----|------|----------|------|
| 输入 | — | `[N, 1, 58, 87]` | 单帧深度图，高度58、宽度87 |
| Conv2d | kernel=5, in=1, out=32 | `[N, 32, 54, 83]` | 5×5卷积提取局部纹理 |
| MaxPool2d | kernel=2, stride=2 | `[N, 32, 27, 41]` | 下采样 |
| ELU | 激活 | `[N, 32, 27, 41]` | — |
| Conv2d | kernel=3, in=32, out=64 | `[N, 64, 25, 39]` | 3×3卷积提取高层特征 |
| ELU | 激活 | `[N, 64, 25, 39]` | — |
| Flatten | — | `[N, 64×25×39=62400]` | 展平 |
| Linear | 62400→128 | `[N, 128]` | 全连接压缩 |
| ELU | 激活 | `[N, 128]` | — |
| Linear | 128→32 | `[N, 32]` | 输出 scandots 维度特征 |
| 输出激活 | Tanh / ELU | `[N, 32]` | 由 `output_activation` 参数控制 |

**关键点**：CNN 输出 32 维深度图空间特征向量，与教师 `scan_encoder` 输出维度对齐。

### 2. RecurrentDepthBackbone —— 时序融合与输出

`depth_backbone.py` L6-L42

| 层 | 操作 | 输出形状 | 说明 |
|----|------|----------|------|
| 输入 | depth_image + proprioception | `[B,1,58,87]` + `[B,53]` | 深度图 + 本体感知 |
| base_backbone | CNN 提取 | `[B, 32]` | 深度图空间特征 |
| combination_mlp | Linear(32+53→128) → ELU → Linear(128→32) | `[B, 32]` | 与本体感知融合 |
| GRU | input=32, hidden=512, batch_first | `[B, 1, 512]` | 时序记忆（hidden_states 跨步保持） |
| output_mlp | Linear(512→34) → Tanh | `[B, 34]` | 输出 = 32维深度特征 + 2维偏航预测 |

**关键点**：
- GRU 隐状态在每轮训练后通过 `detach_hidden_states` 分离，实现跨时序步的短期记忆；
- 输出 34 维：前 32 维为蒸馏用 `depth_latent`，后 2 维为 `yaw`（偏航角预测，乘 1.5 缩放）。

---

## 三、训练流程（训练分析的焦点）

训练分为两个阶段，由 `if_depth` 标志控制。

### 阶段一：教师策略 RL 训练（`learn_RL`）

`on_policy_runner.py` L121-L219

- 使用 PPO 训练 `ActorCriticRMA`（含特权信息编码器 + 历史编码器）；
- 不使用深度图，通过本体感知 + 高度扫描点（scan dots, 132 维）+ 特权信息训练；
- 同时训练 `Estimator`（从本体感知预测特权状态）；
- 周期性 DAgger 更新（`update_dagger`）让历史编码器蒸馏特权信息。

### 阶段二：学生策略视觉蒸馏（`learn_vision`）

`on_policy_runner.py` L221-L323

每轮迭代流程：

1. **Rollout 采集**（`num_steps_per_env = update_interval × 24 = 120` 步）：
   - 教师策略 `actor_critic.act_inference` 产生 `actions_teacher`（用特权信息+历史编码，无梯度）；
   - 深度编码器 `depth_encoder(depth_image, prop)` 产生 `depth_latent`（32 维）+ `yaw`（2 维）；
   - 学生策略 `depth_actor(obs_student, scandots_latent=depth_latent)` 产生 `actions_student`；
   - `obs_student` 的偏航位 [6:8] 被替换为深度网络预测的 yaw（仅当 `delta_yaw_ok` 为真时）。
2. **损失计算**（`ppo.py` L325-L336）：
   - `depth_actor_loss` = `‖actions_teacher - actions_student‖₂`（动作模仿 L2 损失）；
   - `yaw_loss` = `‖yaw_teacher - yaw_student‖₂`（偏航预测 L2 损失）；
   - 总损失 = depth_actor_loss + yaw_loss；
   - 优化器 `depth_actor_optimizer` 同时优化 depth_actor 与 depth_encoder 参数。
3. **GRU 隐状态分离**：每轮结束后调用 `detach_hidden_states()`。

**重要事实**：代码中还存在 `update_depth_encoder`（纯特征蒸馏，L2 损失到 scandots_latent）和 `update_depth_both`（联合优化）两个备选方法，但在 `learn_vision` 中**被注释掉，实际只使用 `update_depth_actor`**。

---

## 四、输入输出规格与预处理管线

### 深度图采集与预处理（`legged_robot.py` L161-L203）

| 阶段 | 规格 | 说明 |
|------|------|------|
| 原始相机图像 | `(106, 60)` | Isaac Gym 深度相机原始分辨率 |
| 裁剪 | `(58, 102)` → 实际取 `[:-2, 4:-4]` | 去除底部2行、左右各4列 |
| 加噪 + 裁剪 | clip 到 `[-far_clip, -near_clip]` = `[-2, 0]` | `dis_noise=0.0` |
| Resize | `(87, 58)` 即宽87高58 | 双三次插值，`resized=(87,58)` |
| 归一化 | `img = img * -1; (img - near)/(far-near) - 0.5` | 映射到 `[-0.5, 0.5]` |
| Buffer | `[num_envs, buffer_len=2, 58, 87]` | 滑动窗口缓存2帧 |

### 输入输出汇总

| 数据 | 形状 | 类型 | 说明 |
|------|------|------|------|
| **depth_image 输入** | `[B, 1, 58, 87]` | `torch.float32` | 单帧深度图（CNN 输入，unsqueeze 通道维） |
| **proprioception 输入** | `[B, 53]` | `torch.float32` | 本体感知（n_proprio=53），其中 [6:8] 偏航被置零 |
| **depth_latent 输出** | `[B, 32]` | `torch.float32` | 蒸馏特征，替代 scandots_latent |
| **yaw 输出** | `[B, 2]` | `torch.float32` | 偏航预测（×1.5缩放） |
| **GRU hidden_states** | `[1, B, 512]` | `torch.float32` | 跨步时序记忆 |

### 配置参数（`legged_robot_config.py`）

| 参数 | 值 | 说明 |
|------|-----|------|
| `depth.resized` | `(87, 58)` | 深度图输入分辨率 |
| `depth.buffer_len` | 2 | 历史帧数 |
| `depth.update_interval` | 5 | 每5个控制步更新一次深度图 |
| `depth.near_clip / far_clip` | 0 / 2 | 深度裁剪范围 [m] |
| `depth_encoder.learning_rate` | 1e-3 | 深度编码器学习率 |
| `depth_encoder.num_steps_per_env` | 120 | 每轮 rollout 步数 |
| `policy.scan_encoder_dims` | `[128, 64, 32]` | 教师扫描点编码器，输出32维（与 depth_latent 对齐） |
| `env.n_proprio` | 53 | 本体感知维度 |

---

## 五、训练设计的核心经验（可复用要点）

1. **两阶段蒸馏顺序是前提**：先让教师（特权观测）收敛，再启动学生视觉蒸馏；学生阶段教师完全冻结，目标静止，纯回归收敛快；
2. **"特征对齐"= 注入点对齐，而非显式 latent 损失**：深度编码器输出 32 维，与教师 `scan_encoder` 输出维度一致（`scan_encoder_dims[-1]=32`），学生策略通过 `scandots_latent` 参数注入点无缝替换——对齐由架构保证，不需要额外的 latent MSE；
3. **实际训练只用了动作级 BC**：`update_depth_actor` 是唯一启用的损失；纯特征蒸馏（`update_depth_encoder`）虽然存在但被注释——说明在 Parkour 实践中，动作监督足够驱动深度编码器学到有用特征；
4. **时序训练有配套机制**：GRU 隐状态跨步保持 + 每轮 `detach_hidden_states` 截断；低帧率（`update_interval=5`）下 rollout 步数取 `120 = 5×24`，保证每轮至少覆盖多次深度更新周期；
5. **yaw 辅助任务解决观测缺口**：深度图无法直接观测全局朝向，额外预测 2 维偏航并用 `delta_yaw_ok` 门控决定是否信任预测值；
6. **输入预处理规范化**：裁剪无用边缘 → 深度归一化到 `[-0.5, 0.5]` → 双帧缓冲，固定且良态的输入空间是稳定训练的基础。

---

## 六、对本项目的启示（对照 GRU 失败分析）

本项目 GRU 方案失败的核心原因之一，正是**没有照搬 Parkour 的真实训练方式**：

| Parkour 实际做法 | 本项目 GRU 方案（失败实现） |
|------------------|------------------------------|
| 注入点对齐：depth_latent 顶替 scan_encoder 槽位 | 设计了显式 latent 损失，但从未实现 |
| 只训练 `update_depth_actor`（BC） | 无辅助监督，且取最新本体帧逻辑错误 |
| yaw 预测头 + `delta_yaw_ok` 门控 | 未实现 |
| GRU 配合 `detach_hidden_states` + 足够 rollout 长度 | `gradient_length=1` 截断时序梯度 |
| 低帧率（update_interval=5）配套设计 | 10Hz 深度但 `depth_image_age` 被丢弃（后已补上） |

若后续仍需时序能力，应优先按 Parkour 的**注入点对齐 + BC** 路线完整实现，而不是补做未兑现的显式 latent 损失（详见 `失败分析-Go2-Mid360Depth-10Hz-GRU-时序融合.md`）。

---

*相关参考：`docs/go2_analysis_docs/失败分析-Go2-Mid360Depth-10Hz-GRU-时序融合.md`、`docs/go2_analysis_docs/优势分析-Go2-Mid360Depth-10Hz-CNN特征提取.md`*
