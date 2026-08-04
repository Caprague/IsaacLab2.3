# Go2 Mid360 阶段训练方案分析

## 背景

`go2_loco_skill_walk_mid360_depth_10hz_cfg.py` 是搭载 Mid360 激光雷达 + Orin NX 的拟真 Go2 训练配置。为提高训练效果，将其与验证有效的基础版 `go2_loco_skill_walk_cfg.py`（简单 Go2 模型）进行了对齐改造。

## 最终架构

```
Stage1: 简单Go2模型 + 严格奖励 → Stage2: Mid360模型 + 宽松奖励 → Stage3: Mid360模型 + 宽松奖励 + 深度传感器蒸馏
```

## 优点

### 1. Stage1 模型简化减少了初期干扰

基础 Go2 模型没有 `orin_nx_loader`、`head_mid360_loader`、`head_mid360` 等额外刚体。Stage1 阶段专注于学习基本行走步态，质量随机化只涉及 base / hip / thigh / calf，动力学更干净。这避免了在策略尚未学会站立时就因为头部碰地（`undesired_contacts_head` 惩罚）或背上负载偏移导致频繁摔倒，从而加速收敛。这与原始简单版训练良好的经验一致。

### 2. 惩罚函数三阶段递进合理

| 惩罚项 | Stage1 | Stage2 / 3 | 逻辑 |
|--------|--------|------------|------|
| `feet_slide` | **-0.15** | -0.075 | 简单模型上严格要求足端不打滑，培养干净步态 |
| `feet_stumble` | **-0.5** | -0.05 | 简单模型上严格要求不绊倒，形成稳健抬腿习惯 |
| `undesired_contacts_head` | **-5.0** (Head.*) | -0.05 (head_mid360_loader) | Stage1 严厉惩罚头部触地；Stage2/3 放宽，因为 mid360 保护罩合理接触不可避免 |

这个设计符合课程学习（Curriculum Learning）原则：早期用强约束塑形，后期放宽以适应真实硬件。

### 3. Head 惩罚差异化的语义正确性

- **Stage1** 的 `Head.*` 对应原始 Go2 头部外壳——一旦触地说明姿态严重失衡，`-5.0` 重罚正确。
- **Stage2/3** 的 `head_mid360_loader` 是 mid360 安装基座（带保护罩），其位置凸出，正常穿越狭窄空间时轻微剐蹭是预期行为，`-0.05` 微罚避免策略过度保守。

两者语义不同，分开设定是合理的。

### 4. 终止条件的差异化

`orin_nx_loader_contact` 终止只在 Stage2/3 生效。这是 Orin NX 计算模块，碰地意味着机器人严重倾覆（超过了 mid360 保护罩的防护范围）——应该终止。Stage1 没有这个刚体，设为 None 避免了引用不存在 body 的运行时错误。

### 5. Stage2 和 Stage3 奖励一致，仅传感器不同

Stage3 相对于 Stage2，唯一变化是：
- 用 Mid360 深度图替代 `head_proximity_scanner`
- 启用带噪声的学生本体感知（`proprioception_noised`）

奖励函数完全相同。这意味着 Stage3 只做蒸馏/迁移，不改变行为目标，逻辑正确。

## 潜在风险

### 1. Stage1→Stage2 的模型切换可能导致策略退化（Catastrophic Forgetting）

这是**最大的风险点**。Stage1 训练好的策略运行在简单 Go2 模型上，切换到 Stage2 时：

- 机器人质量分布变了（背上多了 Orin NX + mid360 及安装座）
- 质量随机化范围变了（base mass 从 (-1.0, 3.0) 变为 (-1.0, 1.0)，但新增了 mid360 设备的随机质量）
- 质心范围变了（z 从 0.12 缩小到 0.06，但增加了 loader 的独立质心随机化）

这意味着 Stage1 学到的 value function 和 policy 在 Stage2 初始时会面临显著的 sim-to-sim gap。

**缓解建议**：
- 在 Stage2 初期监控 reward 曲线是否出现断崖式下跌
- 如果退化严重，可考虑 Stage1 后期逐步引入少量 mid360 质量（渐进式课程）
- 或者增大 Stage2 的 entropy coefficient 以鼓励探索

### 2. `trot_gait` 参数细微差异

`trot_gait` 在 mid360 配置中使用 `body_names=["FL_foot", "FR_foot", "RL_foot", "RR_foot"]`（显式列表），而简单版用的是 `body_names=".*_foot"`（正则匹配）。功能上等价，不构成实际差异。

### 3. Stage2/3 代码重复

Stage2 和 Stage3 的 mid360 配置恢复代码约 50 行完全一致。如果后续需要调整 mid360 奖励参数，需要改两处。

**改进建议**：可抽取为 `_restore_mid360_config()` 公共方法复用，但不是当前优先事项。

### 4. Stage1 中 `push_jump` 合理性

简单版始终有 `push_jump`，Stage1 也有。Stage2/3 没有——因为全向运动时卡住概率低，且随机推动可能干扰 mid360 质量负载下的平衡。设计合理。

## 综合评分

| 维度 | 评价 | 说明 |
|------|------|------|
| 课程递进逻辑 | ✅ 合理 | 简单模型+严格惩罚 → 复杂模型+宽松惩罚 → 加传感器蒸馏 |
| 模型差异化处理 | ✅ 语义正确 | Head.* vs head_mid360_loader 含义不同，分别处理 |
| 奖励对齐 | ✅ 正确 | Stage1 与简单版等价，Stage2/3 保留原有设定 |
| 事件/终止差分 | ✅ 正确 | mid360 特有刚体仅在对应阶段激活 |
| 主要风险 | ⚠️ 需关注 | Stage1→Stage2 模型切换可能导致策略退化，需监控训练曲线 |
| 代码质量 | ⚠️ 可改进 | Stage2/3 重复代码，后续可考虑抽取公共方法 |

## 结论

**方案合理。** Stage1 先用简单模型打好基础，Stage2/3 再适配复杂模型和传感器的思路是正确的。核心监控指标是 Stage1→Stage2 过渡时的 reward / success rate 曲线，如出现显著退化，考虑用渐进式质量引入平滑过渡。

---

*分析日期：2026-07-25*
*涉及文件：`go2_loco_skill_walk_mid360_depth_10hz_cfg.py`*
