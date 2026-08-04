# Go2 后腿拖步问题 — 非惩罚视角的优化方案

## 问题复述

Stage2 训练后前腿摆荡良好、后腿拖地垫步、周期不稳。已有的足端碰撞/力类惩罚（stumble/slide/air_time）调大系数会在训练初期抑制运动学习，不愿走这条路。

## 换个视角：问题本质是"不对称"，而非"不够用力"

| 视角 | 思路 | 问题 |
|------|------|------|
| 惩罚类 | 拖地就打，打重一点 | 初期抑制学习，且不区分前腿后腿 |
| **对称类** | 前腿做得好，拉后腿一把 | 不增加总惩罚量，只要求前后协调 |

前腿摆荡已经干净利落，这说明策略**有能力**做好摆荡——只是后腿没有动力跟上。与其加惩罚强迫后腿，不如用前腿当"老师"，奖励对角线足对的摆荡一致性。

---

## 方案一（推荐）：对角线摆荡对称奖励 `diagonal_swing_symmetry`

### 物理直觉

Trot 步态中，**对侧足对**（FL+RR 一组，FR+RL 一组）同时进入摆荡相。理想情况下，FL 和 RR 的摆荡轨迹应该完全镜像。当前前腿（FL/FR）质量好，后腿（RL/RR）拖地——对角线对内出现了高度不对称。如果能奖励"FL 和 RR 摆荡高度一致"，后腿自然会向前腿看齐。

### 数学设计

```
phase: 0.0–0.5 → FR+RL 摆荡,  0.5–1.0 → FL+RR 摆荡

diag1 (FL+RR) 摆荡期间:  reward1 = exp(-|h_FL - h_RR| / σ)
diag2 (FR+RL) 摆荡期间:  reward2 = exp(-|h_FR - h_RL| / σ)

total = (reward1 + reward2) / 2
```

其中 h 是各足端高度（已有 FL/FR/RL/RR_foot_height_scanner 提供），σ 控制对称容忍度（建议 0.02m）。

### 特点

- **不加罚总力度**：只奖励协调性，不额外惩罚任何单足行为
- **自标定**：前腿高度天然是后腿的参考线，不需要人工调 target
- **初期安全**：Stage1 早期前腿也不高 → 对称奖励低但不产生惩罚，不会抑制探索
- **直接对症**：后腿拖地的本质是对角线不对称，这个奖励精准命中

### 配置建议

```python
# Stage2/3 新增
diagonal_swing_symmetry = RewTerm(
    func=mdp.diagonal_swing_symmetry,
    weight=1.5,  # 略高于 trot_gait(1.0)，强调对称性
    params={
        "command_name": "base_velocity",
        "cycle_period": 0.7,
        "std": 0.02,
        "FL_foot_sensor_cfg": SceneEntityCfg("FL_foot_height_scanner"),
        "FR_foot_sensor_cfg": SceneEntityCfg("FR_foot_height_scanner"),
        "RL_foot_sensor_cfg": SceneEntityCfg("RL_foot_height_scanner"),
        "RR_foot_sensor_cfg": SceneEntityCfg("RR_foot_height_scanner"),
    }
)
```

---

## 方案二（备选）：摆动相最低离地间隙 `swing_min_clearance`

### 物理直觉

不要求达到多高（那是 `feet_swing` 的事），只要求"摆动中的脚必须离开地面至少 2cm"——低于此线就给温和惩罚。这是一个"地板"而非"天花板"。

### 数学设计

```
for each foot in swing phase:
    clearance = foot_height - ground_height
    if clearance < threshold (e.g., 0.02m):
        penalty += (threshold - clearance)  # 线性惩罚
    else:
        reward += 0  # 达标不给奖励，避免策略刷分

total = penalty × weight  # weight 为负值
```

### 特点

- **语义精确**："拖地"就是 clearance ≈ 0，该函数直接测量这个量
- **不干扰正常步态**：一旦离地 > threshold，函数完全静默，不影响前腿已有的好行为
- **可渐进收紧**：threshold 从 0.01→0.02→0.03 逐步提升

### 与 `feet_swing` 的区别

| | feet_swing | swing_min_clearance |
|---|---|---|
| 目标 | 达到 5cm 峰值高度 | 不低于 2cm 最低高度 |
| 奖励曲线 | 高斯，偏离即衰减 | 阈值以下才激活 |
| 作用 | 鼓励高抬腿 | 制止拖地 |
| 初期影响 | 全阶段生效 | 仅在拖地时生效 |

两者互补：`feet_swing` 拉高上限，`swing_min_clearance` 守住下限。

---

## 方案三（长远考虑）：接触时序一致性 `contact_consistency`

### 物理直觉

"小垫步"的本质是——后足在一次名义上的摆荡周期内，发生了 >1 次的接触-离地-接触切换。正常摆荡应该只有一次触地（摆荡结束时）。如果检测到多余的接触状态切换，说明有拖地/弹跳/垫步。

### 设计思路

```
统计每个 contact_sensor 的 recent_contacts（过去 N 帧）:
    if transitions > 1 in last swing phase:
        penalty
```

### 特点

- **精准打击垫步**：正常触地永远只有 1 次切换，垫步产生 ≥2 次
- **实现复杂度高**：需要在 reward function 中维护 per-foot 的状态机或滚动窗口

这个方案实现成本较高，可作为如果前两个方案不够时的 Plan C。

---

## 综合推荐

| 优先级 | 方案 | 改动量 | 风险 | 预期效果 |
|--------|------|--------|------|----------|
| 🥇 | 对角线对称奖励 | 新增一个 mdp 函数 + 配置项 | 低 | 后腿向前腿对齐 |
| 🥈 | 最低离地间隙 | 新增一个 mdp 函数 + 配置项 | 极低 | 直接消除拖地 |
| 🥉 | 接触一致性 | 新增函数 + 状态维护 | 中 | 精准消除垫步 |

**建议落地顺序**：先上方案一（对角线对称），因为它在不增加惩罚量的前提下，利用了前腿作为天然教师信号，设计最优雅。如果对称奖励还不够，再叠方案二作为保险。

两个方案都可以只在 Stage2/3 启用，Stage1 保持不变，不影响早期探索。

---

*分析日期：2026-07-26*
