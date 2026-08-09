# 失败分析：Go2-Mid360Depth-10Hz-GRU 学生策略时序融合方案

> **日期**: 2026-08-09（重塑自 2026-08-04 的 `go2_gru_student_policy_architecture_redesign.md` 设计文档）
> **状态**: **已定论失败**——GRU 学生最终地形可达等级约 **3.0**，显著低于 base 学生（4.5 → 补 `depth_image_age` 后 4.7）与教师（5.3/有效上限 5.0）
> **一句话结论**: GRU 方案失败**不是"时序建模思路"本身的问题**，而是实现与设计严重脱节：核心的特征对齐损失从未实现、取最新本体帧的逻辑错误、`gradient_length=1` 截断时序梯度、32 维瓶颈无辅助监督、10Hz 时钟信号（`depth_image_age`）被丢弃。base 方案以更简单的结构达到更好效果，恰恰证明了"正确归纳偏置 + 直接梯度"比"更复杂的架构"更重要。

---

## 一、背景与目标

### 1.1 动机

base 学生（`StudentTeacherDepthImage`，192 维深度特征直连 MLP）Stage4 地形等级止步 4.5。分析认为瓶颈在单帧 10Hz 深度感知的时序补偿，因此参照 extreme Parkour 设计 GRU 学生：

- 深度图 → CNN → 32 维 latent（紧凑瓶颈）；
- GRU 时序融合本体 + 深度特征，补偿 10Hz/50Hz 频率差；
- 教师侧引入 `HeightScanEncoder`/`PrivilegeEncoder`，输出 32 维 latent 与学生 GRU 输出做"特征对齐"。

### 1.2 设计方案要点（原文档 v0.1 存档摘要）

```
学生: depth(180×32) → StudentDepthCNN → depth_latent(32)
      prop_latest(47) + depth_latent(32) → input_mlp → GRU(256) → output_mlp → hs(32)+priv(32)
      student_MLP(282+32+32=346) → 动作
教师: mapScans(187) → HeightScanEncoder → 32 维
      privileged(54) → PrivilegeEncoder → 32 维
      teacher_MLP(235+187+54=476) → 动作
损失: BC + α·MSE(hs, teacher_hs.detach()) + β·MSE(priv, teacher_priv.detach())
```

设计参数：CNN ~30K、GRU 64→256、学生 MLP 346→512→256→128→12。

---

## 二、设计与实现的偏差对照（失败的直接线索）

| # | 设计文档要求 | 代码实际实现 | 后果 |
|---|-------------|--------------|------|
| 1 | 总损失 = BC + α·hs_loss + β·priv_loss（特征对齐） | `distillation.py` 只有 BC（action MSE） | 教师编码器是"死重"，latent 无监督目标 |
| 2 | 教师编码器预训练后接入蒸馏损失 | 仅加载+冻结，从未参与前向/损失 | 预训练完全无效 |
| 3 | GRU 应做时序融合（隐含 BPTT） | `gradient_length=1` | 时序梯度被截断到 1 步 |
| 4 | `prop_latest` = 最新本体帧（47 维） | term 主序布局下 `[:, -47:]` 取到错位切片 | GRU 输入损坏 |
| 5 | `depth_image_age` 作为 10Hz 时钟信号 | 被切片丢弃（后已补上） | GRU 感知不到深度帧陈旧度 |
| 6 | `teacher_driving=True`，约 2500 iter 切换 | `teacher_driving=False` | 无 DAgger 早期引导 |
| 7 | Parkour 参考含 yaw 预测头 | 未实现 | 深度无法直接观测全局朝向 |
| 8 | 文档标注 privileged 60 维/教师 482 维 | 实际 54 维/476 维 | 文档与实现脱节（代码动态算维度，未崩但说明失准） |

---

## 三、实验结果

| 方案 | 地形可达等级 | 说明 |
|------|-------------|------|
| 教师（Stage3，特权观测） | ~5.3 | 有效上限 5.0，超出为边缘平坦地形拓展 |
| base 学生（原始） | ~4.5 | 192 维深度特征直连 MLP |
| base 学生（补 `depth_image_age`） | ~4.7 | 已接近有效上限 |
| **GRU 学生** | **~3.0** | 显著低于 base，方案失败 |

---

## 四、失败根因（按影响排序）

### 1. 特征对齐损失从未实现（方案的核心价值缺失）

全代码库检索确认：`Distillation.update()` 只计算行为克隆损失；`HeightScanEncoder`/`PrivilegeEncoder` 仅在 `__init__` 创建、`train()` 置 eval、`DistillationRunner.load()` 加载冻结，**从未在损失中出现**。学生 GRU 的 `hs_latent`/`priv_latent` 没有任何监督目标，属于无约束内部特征——"Parkour 式特征对齐"从设计到实现从未成立。

### 2. 取最新本体感知帧的逻辑错误

Isaac Lab 历史观测组默认是 **term 主序**拼接（每个 term 独立 `CircularBuffer`，展平后按 term 顺序拼接），不是帧主序。`prop_all[:, -proprio_per_frame:]` 实际取到的是**最后一个 term（actions）历史块的错位尾部**（第 2 帧 11 个元素 + 第 3/4/5 帧各 12 个），完全不是"最新帧"。GRU 的时序输入从设计到实现从未正确过。详见 `重要经验-IsaacLab-观测历史拼接.md`。

### 3. `gradient_length=1` 截断时序梯度

GRU 需要 BPTT 才能学习"识别重复帧/感知陈旧 → 保持或外推状态"；`gradient_length=1` 使每次 backward 只回传 1 步，隐状态仅以值前传、无时间维度梯度。10Hz 深度每帧重复 5 次的时序模式完全无法被学习（rsl_rl 循环蒸馏常规取 10–24）。

### 4. 32 维瓶颈 + 长特征路径 + 无辅助监督

深度信息需穿过 32 维瓶颈 → input_mlp → GRU → output_mlp 才到达策略；对比 base 的 192 维富池化特征直连 MLP，路径更长、瓶颈更窄，且没有 latent 监督补偿。BC 梯度经长路径回传至 CNN 大幅衰减，学生更容易"忽略深度、只靠本体历史"——在陡峭地形上表现为等级骤降。

### 5. `depth_image_age` 被丢弃

专为 10Hz/50Hz 同步设计的帧龄信号在两个网络中都因 `depth_flat[:, :5760]` 切片被丢弃（GRU 侧后已补上）。GRU 想用时序补偿感知陈旧性，却连"这帧多旧"的时钟信号都没有。

### 6. 教师编码器预训练方式的缺陷（次要，但在接入损失前必须解决）

- AE 重建最大化"输入信息量"而非"任务相关信息量"；
- 数据来自 `curriculum=False`、`max_init_terrain_level=None` 的采集环境，不覆盖高难度地形；
- 文档设计的推力扰动（`--push_interval`）在实际脚本中完全未实现。

### 7. 前提问题：教师上限未先行验证

蒸馏上限 = 教师上限。若 Stage3 教师自身在 terrain 5.0 成功率不高，则学生无论怎么改都难以突破——应先验证教师再改学生架构。

---

## 五、经验教训（可复用原则）

1. **复杂架构只有在核心机制"真正实现"时才值得引入**：方案文档写了特征对齐损失，代码里却没有——复杂度全加了，收益一点没兑现；
2. **时序模型必须先验证时序输入的正确性**：取帧、布局、BPTT 长度是 GRU 成立的三个前提，任何一个出错都会让时序模块退化为噪声通路；
3. **瓶颈必须有监督**：压缩维度（32 维）本身不是问题，但没有对齐/重建/任务监督的瓶颈只会丢信息；
4. **小参数 + 强归纳偏置 + 直接梯度（base 路线）在同类任务上更稳健**：先榨干简单结构的上限（如补 `depth_image_age`、双帧拼接），再考虑引入时序复杂度；
5. **设计文档与实现要保持同步**：维度、参数、损失公式任何一处脱节，都可能是失败的前兆（本方案中 privileged 60 vs 54、teacher_driving、gradient_length 均与文档不符）。

---

## 六、修正方向（若仍需要时序能力）

按 Parkour 的真实做法（`docs/ref_analysis_docs/参考-Parkour-深度特征提取训练.md`）完整实现，而不是照搬未实现的文档设计：

1. **特征对齐二选一**：
   - 推荐：Parkour 原版注入点对齐——学生 MLP 输入 = 本体历史 + `depth_latent`，纯 BC 训练（参考代码中纯特征蒸馏 `update_depth_encoder` 实际被注释，仅用 `update_depth_actor`）；
   - 或补显式 latent 损失：`Distillation.update()` 中用冻结教师编码器输出 `teacher_hs/priv`（detach），学生 `hs/priv` 回归，损失 = BC + α·hs + β·priv；优化器改为只含学生参数；
2. **`gradient_length` 改为 10–24**（GRU 成立的前提）；
3. **`depth_image_age` 已接入 GRU 输入**（本次会话已修复）；
4. **取最新本体帧逻辑已修复**：`flatten_history_dim=False` 帧主序 + 时间主序展平（本次会话已修复）；
5. **保留 32 维 latent**，不回退 192 维（瓶颈不是问题，缺监督才是）；
6. 教师 driving 早期引导（`teacher_driving=True`，约 2500 iter 切换）；
7. 预训练改为任务对齐（回归教师内部特征）并覆盖 terrain 3–5 + 实现推力扰动；
8. 可选：补 Parkour 的 yaw 预测头。

### 更保守的替代路线（先做）

在 base 架构上验证低成本改进，确认确实需要时序后再启动 GRU：

- 共享权重的双帧拼接（同一 CNN 处理最近两帧 + 各自 age，特征 192+192 进 MLP）；
- 适度增大 CNN 输出（192→256）；
- 蒸馏侧 `teacher_driving=True` + huber 损失。

---

## 七、已修复项清单（供后续复用）

| 修复项 | 文件 | 状态 |
|--------|------|------|
| `depth_image_age` 接入学生输入 | `rsl_rl/modules/student_teacher_depth_image.py`、`student_teacher_depth_image_recurrent.py` | 已实施 |
| 本体感知组帧主序（`flatten_history_dim=False`） | `go2_loco_skill_walk_mid360_depth_10hz_gru_cfg.py` | 已实施 |
| GRU 取最新帧逻辑（时间主序展平 + 强制 3D 断言） | `student_teacher_depth_image_recurrent.py` | 已实施 |

---

*原设计文档全文（1065 行）可通过 git 历史恢复；本文件为其失败分析的结论性替代。*
