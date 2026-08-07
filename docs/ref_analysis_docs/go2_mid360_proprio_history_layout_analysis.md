# Go2 Mid360 学生本体感知历史拼接顺序分析与修复验证

> **日期**: 2026-08-08
> **范围**: `StudentTeacherDepthImageRecurrent`（GRU 学生）中 `prop_latest = prop_all[:, -proprio_per_frame:]` 取最新本体感知帧的逻辑正确性
> **结论**: 原切片在 Isaac Lab 默认布局下**不正确**；已通过 `flatten_history_dim=False` + 时间主序展平修复，并经端到端模拟验证。

---

## 一、问题

GRU 学生模块 `rsl_rl/modules/student_teacher_depth_image_recurrent.py` 中：

```python
prop_all   = torch.cat(prop_list, dim=-1)  # [B, 282]
prop_latest = prop_all[:, -self.proprio_per_frame:]  # [B, 47]
```

需要确认：本体感知组是否一定按"时间帧"拼接，使得末尾 47 维就是最新帧？如果按"观测条目（term）"拼接，这个切片就是错的。

---

## 二、Isaac Lab 实际拼接机制（代码依据）

1. **每个观测 term 各自拥有独立的 CircularBuffer**（`source/isaaclab/isaaclab/managers/observation_manager.py` 中 `group_entry_history_buffer[term_name] = CircularBuffer(...)`）；
2. `CircularBuffer.buffer` 返回 `(batch, history, feat)`，**时间维在 term 内部**：最旧在前、最新在后（`source/isaaclab/isaaclab/utils/buffers/circular_buffer.py` 的 `roll + transpose`）；
3. `flatten_history_dim=True`（`ObservationGroupCfg` 默认）时，每个 term 单独 `reshape(B, H*d)`；
4. 组内所有 term 按**定义顺序**沿最后一维拼接：`torch.cat(list(group_obs.values()), dim=-1)`。

因此组的真实布局是 **term 主序 × 时间次序**，而不是"帧主序"：

```
[phase_t0..t5(6)] [vel_cmd_t0..t5(24)] [ang_vel_t0..t5(18)] [grav_t0..t5(18)]
[jpos_t0..t5(72)] [jvel_t0..t5(72)] [act_t0..t5(72)]
```

（以 `ProprioceptionNoised` 为例：phase 1、vel_cmd 4、ang_vel 3、grav 3、jpos 12、jvel 12、act 12，每帧共 47 维，历史 6 帧 → 282。）

---

## 三、原切片的错误（复现实测）

用纯 Python 忠实复刻上述代码路径（编码 `term*1000 + frame*100 + j` 追溯元素归属）实测：

```
Group layout (term-major):
  phase[0,5] vel_cmd[6,29] ang_vel[30,47] grav[48,65] jpos[66,137] jvel[138,209] act[210,281]

[BUG] prop_all[:, -47:] 实际内容:
  {(act, frame2): 11, (act, frame3): 12, (act, frame4): 12, (act, frame5): 12}

[TRUE] 正确最新帧（每个 term 块取最后 d 个元素）:
  {(phase,5):1, (vel_cmd,5):4, (ang_vel,5):3, (grav,5):3, (jpos,5):12, (jvel,5):12, (act,5):12}

切片 == 正确最新帧 ?  False
```

结论：末尾 47 维只落在**最后一个 term（actions）的历史块内**，且还是错位的（第 2 帧 11 个元素 + 第 3/4/5 帧各 12 个），**完全不是最新本体帧**。`proprio_per_frame = 282 // 6 = 47` 数值恰好正确（每帧总维数确实是 47），但拼接布局不支持直接尾部切片。

---

## 四、影响范围

| 模块 | 是否受影响 | 原因 |
|------|-----------|------|
| GRU `StudentTeacherDepthImageRecurrent` | **严重受影响** | GRU 输入中的 `prop_latest` 是 action 历史尾部错位切片，缺少最新帧的角速度、重力投影、关节位置/速度、速度指令等关键信息 |
| base `StudentTeacherDepthImage` | 不受影响 | 282 维全量历史直接喂给 MLP，MLP 对布局不敏感 |
| 教师侧 | 不受影响 | 全量喂给教师 MLP，无切片 |
| `depth_image_age` 接入（方案A改动） | 不受影响 | `mid360_depth` 组 history_length=1，尾部 1 维就是 age |

该 bug 是 GRU 方案地形等级降至 3.0 的又一个直接根因：GRU 路径从设计到实现从未拿到过正确的最新本体帧。

---

## 五、修复方案（已实施）

### 5.1 环境侧：本体感知组改为帧主序

`go2_loco_skill_walk_mid360_depth_10hz_gru_cfg.py` 的 `ProprioceptionNoised.__post_init__` 增加：

```python
self.flatten_history_dim = False
```

该组由 `(B, 282)` 变为 `(B, 6, 47)`（帧主序，时间维在中间维）。

### 5.2 网络侧：帧主序 → 时间主序展平

`student_teacher_depth_image_recurrent.py`：

```python
# __init__：允许 2D/3D 观测组，记录 prop_ndim，维度按 numel//B 计算
self.prop_ndim = len(obs[g].shape)  # 3 = 帧主序 (B, H, D)
num_student_basic_obs += obs[g].numel() // obs[g].shape[0]

# get_student_obs：3D 组先时间主序展平，再取尾部
if self.prop_ndim == 3:
    prop_all = torch.cat(prop_list, dim=-1).reshape(prop_list[0].shape[0], -1)
else:
    prop_all = torch.cat(prop_list, dim=-1)
prop_latest = prop_all[:, -self.proprio_per_frame:]
```

`(B, H, D)` 展平为 `(B, H*D)` 后布局为 `[t0(47) t1(47) ... t5(47)]`，末尾 47 维即最新帧。

### 5.3 维度变化

| 项 | 修改前 | 修改后 |
|----|--------|--------|
| `proprioception_noised` 组形状 | `(B, 282)` term 主序 | `(B, 6, 47)` 帧主序 |
| `prop_latest` | 错位 action 历史切片（47） | **真正最新帧（47）** |
| GRU 输入 | 80（47+1+32） | 80（不变） |
| 学生策略输入 | 346 | 346（不变） |

---

## 六、重新验证结果

用纯 Python 复刻修复后的完整管线（真实 `CircularBuffer` 语义 + `flatten_history_dim=False` 帧主序组 + 网络时间主序展平 + 尾部切片），批量 4 环境、多组填充场景：

```
[full-fill]    append_count=6  num_basic=282  per_frame=47  tail==latest: True
[wrap-around]  append_count=9  num_basic=282  per_frame=47  tail==latest: True
[wrap-around-2] append_count=8 num_basic=282  per_frame=47  tail==latest: True
All checks passed: tail slice == true latest frame under the fixed pipeline.
gru_input_mlp input dim = 47 + 1 + 32 = 80
student MLP input dim   = 282 + 32 + 32 = 346
```

验证要点：
- 满 6 帧填充：尾部切片 == 正确最新帧；
- 超过 6 帧（环形回绕）：最新帧仍是末尾 47 维（时间主序展平不受回绕影响，因为 CircularBuffer 已按最旧→最新重排）；
- 网络输入维度全部与设计一致。

另：两个修改文件均通过 `py_compile` 语法校验。

---

## 七、注意事项

- 学生侧 checkpoint **不兼容**：`proprioception_noised` 组形状变化 + 学生输入层维度变化，Stage4 需重新训练；
- 教师侧 checkpoint 完全兼容（教师观测组未改）；
- 模块**强制要求 prop 观测组为帧主序 3D**：若某环境未设 `flatten_history_dim=False`，`__init__` 断言会直接报错（提示设置 `flatten_history_dim=False`），避免旧布局下静默产生错误 `prop_latest`；
- base 环境未改（保持 `flatten_history_dim=True` 与全量 MLP 输入），不受影响。
