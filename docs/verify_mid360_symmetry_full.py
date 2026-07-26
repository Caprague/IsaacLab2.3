"""
go2_mid360_teacher_walk.py 全套左右对称验证脚本

生成模拟观测数据 → 调用真实对称函数 → 逐项对比原始/镜像输出

用法: cd D:\GitSyncVaults\IsaacLab2.3 && python docs/verify_mid360_symmetry_full.py
"""

import torch
import sys
import os
import math
from typing import Optional
from dataclasses import dataclass, field
from unittest.mock import MagicMock

# ---------------------------------------------------------------------------
# 0. 构建最小化 mock 环境，让对称函数能正常运行
# ---------------------------------------------------------------------------

class MockCommandManager:
    """模拟 command_manager，返回可控的速度指令"""
    def __init__(self, num_envs, device):
        self._cmd = torch.zeros(num_envs, 3, device=device)  # [vx, vy, wz]
        self._cmd[:, 0] = 1.0  # 默认向前走 1m/s

    def get_command(self, name):
        return self._cmd


class MockSensorData:
    """模拟传感器数据"""
    def __init__(self, data: dict):
        self.__dict__.update(data)


class MockContactSensor:
    """模拟接触力传感器"""
    def __init__(self, name, body_ids, num_envs, device):
        self.name = name
        self.cfg = MagicMock()
        self.cfg.name = name
        n_bodies = len(body_ids)
        # net_forces_w_history: (envs, history, bodies, 3)
        self.data = MockSensorData({
            'net_forces_w_history': torch.zeros(num_envs, 1, n_bodies, 3, device=device),
            'body_ids': body_ids,
        })


class MockRayCaster:
    """模拟射线传感器"""
    def __init__(self, name, n_rays, num_envs, device, is_height_scanner=True):
        self.name = name
        self.cfg = MagicMock()
        self.cfg.name = name
        self.cfg.pattern_cfg = MagicMock()
        if is_height_scanner:
            self.cfg.pattern_cfg.resolution = 0.1
            self.cfg.pattern_cfg.size = (1.6, 1.0)
        # pos_w: (envs, 3), ray_hits_w: (envs, n_rays, 3)
        self.data = MockSensorData({
            'pos_w': torch.zeros(num_envs, 3, device=device),
            'ray_hits_w': torch.zeros(num_envs, n_rays, 3, device=device),
        })


class MockImu:
    def __init__(self, name, num_envs, device):
        self.name = name
        self.cfg = MagicMock()
        self.cfg.name = name


class MockScene:
    """模拟场景，存放传感器"""
    def __init__(self):
        self.sensors = {}
        self._entities = {}

    def __getitem__(self, name):
        return self._entities.get(name)


class MockEnv:
    """最小化 mock 环境"""
    def __init__(self, num_envs=1, device='cpu'):
        self.num_envs = num_envs
        self.device = device
        self.step_dt = 0.02  # 50Hz
        self.episode_length_buf = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.command_manager = MockCommandManager(num_envs, device)
        self.scene = MockScene()

    @property
    def unwrapped(self):
        return self


# ---------------------------------------------------------------------------
# 1. 生成模拟观测数据
# ---------------------------------------------------------------------------

def make_mock_obs(env: MockEnv, history_proprio=5, history_priv=3):
    """生成各观测组的模拟数据，每个位置用唯一可辨识的值填充。"""
    device = env.device
    B = env.num_envs

    # -- Proprioception: 47 dims/step × history=5 = 235 --
    # Per-step: phase(2)+vel_cmd(3)+imu(3)+grav(3)+joints(12)+joint_vel(12)+actions(12)
    steps = history_proprio
    obs_proprio = torch.zeros(B, 235, device=device)
    for s in range(steps):
        base = s * 47
        # phase: sin,cos — 用不同小数值标记
        obs_proprio[0, base + 0] = math.sin(s * 0.5) * 2 + 0.1   # sin
        obs_proprio[0, base + 1] = math.cos(s * 0.5) * 2 + 0.2   # cos
        # vel_cmd: vx, vy, wz
        obs_proprio[0, base + 2] = 1.0 + s * 0.03
        obs_proprio[0, base + 3] = 0.2 + s * 0.03
        obs_proprio[0, base + 4] = 0.1 + s * 0.03
        # imu: wx, wy, wz
        obs_proprio[0, base + 5] = 0.3 + s * 0.03
        obs_proprio[0, base + 6] = -0.1 + s * 0.03
        obs_proprio[0, base + 7] = 0.15 + s * 0.03
        # grav: gx, gy, gz
        obs_proprio[0, base + 8] = 0.05
        obs_proprio[0, base + 9] = -0.02
        obs_proprio[0, base + 10] = -9.8  # mostly pointing down
        # joints: 12 values — 编码为 100*joint_idx + s
        for j in range(12):
            obs_proprio[0, base + 11 + j] = 100.0 + j * 10.0 + s
        # joint_vel: 12 values — 编码为 200+j*10+s
        for j in range(12):
            obs_proprio[0, base + 23 + j] = 200.0 + j * 10.0 + s
        # actions: 12 values — 编码为 300+j*10+s
        for j in range(12):
            obs_proprio[0, base + 35 + j] = 300.0 + j * 10.0 + s

    # -- MapScans: 187 dims (11×17 grid) --
    # 编码: row*100 + col (row=y/width, col=x/length)
    obs_map = torch.zeros(B, 187, device=device)
    for r in range(11):
        for c in range(17):
            obs_map[0, r * 17 + c] = r * 100.0 + c

    # -- Privileged: 18 dims/step × history=3 = 54 --
    # Per-step: gait(2)+contact(4)+lin_vel(3)+feet_dist(4)+base_h(1)+foot_h(4)
    steps = history_priv
    obs_priv = torch.zeros(B, 54, device=device)
    for s in range(steps):
        base = s * 18
        # gait_trot_mask: [pair0, pair1]
        obs_priv[0, base + 0] = 1.0  # pair0 active (FL+RR stance)
        obs_priv[0, base + 1] = 0.0  # pair1 inactive
        # feet_contact: [FL, FR, RL, RR]
        obs_priv[0, base + 2] = 1.0  # FL on ground
        obs_priv[0, base + 3] = 0.0  # FR in air
        obs_priv[0, base + 4] = 0.0  # RL in air
        obs_priv[0, base + 5] = 1.0  # RR on ground
        # base_lin_vel: vx, vy, vz
        obs_priv[0, base + 6] = 1.0
        obs_priv[0, base + 7] = 0.2
        obs_priv[0, base + 8] = -0.05
        # feet_distance: [FL, FR, RL, RR]
        obs_priv[0, base + 9] = 0.25
        obs_priv[0, base + 10] = 0.25
        obs_priv[0, base + 11] = 0.25
        obs_priv[0, base + 12] = 0.25
        # base_height
        obs_priv[0, base + 13] = 0.30
        # foot heights: [FL, FR, RL, RR]
        obs_priv[0, base + 14] = 0.05  # FL
        obs_priv[0, base + 15] = 0.04  # FR
        obs_priv[0, base + 16] = 0.06  # RL
        obs_priv[0, base + 17] = 0.03  # RR

    # -- HeadProximity: 32 dims (4 zenith rows × 8 azimuth cols) --
    # 编码: row*10 + col
    obs_head = torch.zeros(B, 32, device=device)
    for r in range(4):
        for c in range(8):
            obs_head[0, r * 8 + c] = r * 10.0 + c

    # -- Actions: 12 dims (joint position targets) --
    actions = torch.zeros(B, 12, device=device)
    for j in range(12):
        actions[0, j] = 400.0 + j * 10.0

    from tensordict import TensorDict
    obs_dict = TensorDict({
        "proprioception": obs_proprio,
        "mapScans": obs_map,
        "privileged": obs_priv,
        "headProximity": obs_head,
    }, batch_size=[B])

    return obs_dict, actions


# ---------------------------------------------------------------------------
# 2. 可视化对比
# ---------------------------------------------------------------------------

def print_header(title):
    print("\n" + "=" * 100)
    print(f"  {title}")
    print("=" * 100)


def compare_tensors(name, original, mirrored, per_step=None, step_names=None,
                    joint_labels=False, feet_labels=False, gait_labels=False,
                    grid_shape=None, head_prox=False):
    """对比 original 和 mirrored 张量，逐项对齐输出。

    Args:
        per_step: 每步维度数
        step_names: 每步内各子项名称列表
        joint_labels: 是否标注关节名
        feet_labels: 是否标注足端名
        gait_labels: 是否标注步态掩码名
        grid_shape: (rows, cols) 用于 2D 数据
        head_prox: 是否用 head proximity 标注
    """
    B = original.shape[0]
    total = original.shape[1]
    device = original.device

    orig_1d = original[0].cpu().tolist()
    mirr_1d = mirrored[0].cpu().tolist()

    if per_step:
        n_steps = total // per_step
        for s in range(n_steps):
            base = s * per_step
            print(f"\n  --- history step {s} (offset {base}:{base+per_step}) ---")

            if step_names:
                col = 0
                for seg_name, seg_len in step_names:
                    vals_orig = orig_1d[base + col:base + col + seg_len]
                    vals_mirr = mirr_1d[base + col:base + col + seg_len]
                    label = _make_label(seg_name, seg_len, col,
                                        joint_labels and seg_name == "joint_pos",
                                        joint_labels and seg_name == "joint_vel",
                                        joint_labels and seg_name == "actions",
                                        feet_labels and seg_name in ("contact", "feet_dist", "foot_h"),
                                        gait_labels and seg_name == "gait")
                    _print_row(vals_orig, vals_mirr, label, seg_len)
                    col += seg_len
            else:
                _print_row(orig_1d[base:base+per_step], mirr_1d[base:base+per_step],
                           "", per_step)
    elif grid_shape:
        nr, nc = grid_shape
        print(f"  Grid ({nr}×{nc}), row=y/width, col=x/length:")
        # 逐行对比
        for r in range(nr):
            orig_row = orig_1d[r*nc:(r+1)*nc]
            mirr_row = mirr_1d[r*nc:(r+1)*nc]
            # 打印值摘要: 只打印前后各 3 个元素
            o_str = "[" + " ".join(f"{v:6.1f}" for v in orig_row[:4]) + " ... " + " ".join(f"{v:6.1f}" for v in orig_row[-3:]) + "]"
            m_str = "[" + " ".join(f"{v:6.1f}" for v in mirr_row[:4]) + " ... " + " ".join(f"{v:6.1f}" for v in mirr_row[-3:]) + "]"
            print(f"  row {r:2d}: orig {o_str}")
            print(f"         mirr {m_str}")
    elif head_prox:
        nr, nc = 4, 8  # zenith=4, azimuth=8
        print(f"  HeadProximity ({nr} zenith × {nc} azimuth):")
        for r in range(nr):
            orig_row = orig_1d[r*nc:(r+1)*nc]
            mirr_row = mirr_1d[r*nc:(r+1)*nc]
            o_str = "[" + " ".join(f"{v:5.1f}" for v in orig_row) + "]"
            m_str = "[" + " ".join(f"{v:5.1f}" for v in mirr_row) + "]"
            print(f"  ze {r}: orig {o_str}")
            print(f"         mirr {m_str}")
    else:
        _print_row(orig_1d, mirr_1d, "", total)


JOINT_NAMES = ["FL_hip","FR_hip","RL_hip","RR_hip",
               "FL_thigh","FR_thigh","RL_thigh","RR_thigh",
               "FL_calf","FR_calf","RL_calf","RR_calf"]

FOOT_NAMES = ["FL","FR","RL","RR"]
GAIT_NAMES = ["pair0(FL+RR)","pair1(FR+RL)"]

def _make_label(seg_name, seg_len, offset, is_joint, is_joint_vel, is_action, is_feet, is_gait):
    details = []
    if is_joint or is_joint_vel or is_action:
        for j in range(seg_len):
            details.append(JOINT_NAMES[j % 12])
    elif is_feet:
        for f in range(seg_len):
            details.append(FOOT_NAMES[f % 4])
    elif is_gait:
        for g in range(seg_len):
            details.append(GAIT_NAMES[g % 2])
    return (seg_name, seg_len, details)


def _print_row(orig, mirr, label, n):
    """打印一行对比数据"""
    n_show = min(n, 24)
    # 取每个 history step 的第一个值（省略重复模式）
    if n > 24 and n % n_show != 0:
        step = n // 6  # 大约打印6组
        indices = list(range(0, n, step))[:12]
    else:
        indices = list(range(n_show))

    o_vals = [f"{orig[i]:8.3f}" if i < len(orig) else "?" for i in indices]
    m_vals = [f"{mirr[i]:8.3f}" if i < len(mirr) else "?" for i in indices]

    o_str = " ".join(o_vals)
    m_str = " ".join(m_vals)
    diff_flags = []
    for i in indices:
        if i < len(orig) and i < len(mirr):
            diff_flags.append(" ≠" if abs(orig[i] - mirr[i]) > 0.001 else "  ")
        else:
            diff_flags.append(" ?")
    d_str = " ".join(f"{f:^8}" for f in diff_flags)

    print(f"  orig [{indices[0]:3d}..{indices[-1]:3d}]: {o_str}")
    print(f"  mirr [{indices[0]:3d}..{indices[-1]:3d}]: {m_str}")
    print(f"  diff [{indices[0]:3d}..{indices[-1]:3d}]: {d_str}")


# ---------------------------------------------------------------------------
# 3. 主验证流程
# ---------------------------------------------------------------------------

def main():
    print("go2_mid360_teacher_walk.py — 全套左右对称验证")
    print(f"PyTorch {torch.__version__}")

    # 导入真实对称模块
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..',
        'source', 'isaaclab_tasks', 'isaaclab_tasks', 'manager_based',
        'locomotion', 'velocity', 'mdp', 'symmetry'))
    import go2_mid360_teacher_walk as sym

    # 构建 mock 环境
    env = MockEnv(num_envs=4, device='cpu')

    # 生成模拟观测
    obs, actions = make_mock_obs(env)

    # 注意: 新版本各 transform 函数不再依赖 env，但 compute_symmetric_states 签名保留 env 参数
    obs_aug, actions_aug = sym.compute_symmetric_states(env, obs, actions)

    # 提取原始和镜像部分
    B = obs.batch_size[0]
    orig_proprio = obs_aug["proprioception"][:B]
    mirr_proprio = obs_aug["proprioception"][B:]
    orig_map = obs_aug["mapScans"][:B]
    mirr_map = obs_aug["mapScans"][B:]
    orig_priv = obs_aug["privileged"][:B]
    mirr_priv = obs_aug["privileged"][B:]
    orig_head = obs_aug["headProximity"][:B]
    mirr_head = obs_aug["headProximity"][B:]
    orig_act = actions_aug[:B]
    mirr_act = actions_aug[B:]

    # =======================================================================
    # 逐步对比每一项
    # =======================================================================

    # -- Proprioception --
    print_header("1. Proprioception (235 dims, history=5)")

    # 逐 history step 检查关键变换
    per_step = 47
    step_names = [
        ("phase", 2),
        ("vel_cmd", 3),
        ("imu", 3),
        ("grav", 3),
        ("joint_pos", 12),
        ("joint_vel", 12),
        ("actions", 12),
    ]

    print("\n  仅打印 history step 0 的详细对比（其余步模式相同）:")
    for s in [0]:
        base = s * per_step
        print(f"\n  --- history step {s} (offset {base}:{base+per_step}) ---")
        col = 0
        for seg_name, seg_len in step_names:
            o = orig_proprio[0, base+col:base+col+seg_len].cpu().tolist()
            m = mirr_proprio[0, base+col:base+col+seg_len].cpu().tolist()
            print(f"\n  [{seg_name}] ({seg_len} dims):")
            if seg_name == "phase":
                print(f"    orig: {[f'{v:.3f}' for v in o]}")
                print(f"    mirr: {[f'{v:.3f}' for v in m]}")
                print(f"           sin→-sin, cos→-cos  (phase +0.5)")
            elif seg_name in ("vel_cmd", "imu", "grav"):
                print(f"    orig: {[f'{v:.3f}' for v in o]}")
                print(f"    mirr: {[f'{v:.3f}' for v in m]}")
                if seg_name == "vel_cmd":
                    print(f"           [vx, vy, ωz] → [vx, -vy, -ωz]")
                elif seg_name == "imu":
                    print(f"           [ωx, ωy, ωz] → [-ωx, ωy, -ωz]")
                else:
                    print(f"           [gx, gy, gz] → [gx, -gy, gz]")
            elif seg_name in ("joint_pos", "joint_vel", "actions"):
                print(f"    orig: {[f'{v:.1f}' for v in o]}")
                print(f"    mirr: {[f'{v:.1f}' for v in m]}")
                print(f"           joints: FL↔FR, RL↔RR, hip×(-1)")
                # 详细关节标注
                for j in range(12):
                    jn = JOINT_NAMES[j]
                    diff_mark = " ≠" if abs(o[j] - m[j]) > 0.5 else "  "
                    print(f"    {jn:10s}: orig {o[j]:8.1f}  mirr {m[j]:8.1f} {diff_mark}")
            col += seg_len

    # -- MapScans --
    print_header("2. MapScans (187 dims, 11×17 grid)")

    print("\n  Grid: 11 rows (y/w) × 17 cols (x/l)")
    print("  y 值沿行变化, 左右镜像应翻转行序 (row i → row 10-i)")
    for r in range(11):
        o_row = orig_map[0, r*17:(r+1)*17].cpu()
        m_row = mirr_map[0, r*17:(r+1)*17].cpu()
        o_min, o_max = o_row.min().item(), o_row.max().item()
        m_min, m_max = m_row.min().item(), m_row.max().item()
        print(f"  y[{r:2d}]: orig [{o_min:6.0f}, {o_max:6.0f}]   mirr [{m_min:6.0f}, {m_max:6.0f}]")

    print("\n  验证: y=0 (最左) 应与 y=10 (最右) 互换:")
    row0_orig = orig_map[0, 0:17]
    row10_mirr = mirr_map[0, 10*17:11*17]
    match = torch.allclose(row0_orig, row10_mirr)
    print(f"  orig[y=0] == mirr[y=10]: {match}")
    row10_orig = orig_map[0, 10*17:11*17]
    row0_mirr = mirr_map[0, 0:17]
    match = torch.allclose(row10_orig, row0_mirr)
    print(f"  orig[y=10] == mirr[y=0]: {match}")

    # -- Privileged --
    print_header("3. Privileged (54 dims, history=3)")

    per_step = 18
    priv_names = [
        ("gait_trot_mask", 2),
        ("feet_contact", 4),
        ("base_lin_vel", 3),
        ("feet_distance", 4),
        ("base_height", 1),
        ("foot_heights", 4),
    ]

    for s in [0]:
        base = s * per_step
        print(f"\n  --- history step {s} ---")
        col = 0
        for seg_name, seg_len in priv_names:
            o = orig_priv[0, base+col:base+col+seg_len].cpu().tolist()
            m = mirr_priv[0, base+col:base+col+seg_len].cpu().tolist()
            print(f"\n  [{seg_name}] ({seg_len} dims):")
            print(f"    orig: {[f'{v:.3f}' for v in o]}")
            print(f"    mirr: {[f'{v:.3f}' for v in m]}")
            if seg_name == "gait_trot_mask":
                print(f"           pair0↔pair1")
            elif seg_name in ("feet_contact", "feet_distance", "foot_heights"):
                print(f"           FL↔FR, RL↔RR")
            elif seg_name == "base_lin_vel":
                print(f"           [vx, vy, vz] → [vx, -vy, vz]")
            elif seg_name == "base_height":
                print(f"           不变（高度对称）")
            col += seg_len

    # -- HeadProximity --
    print_header("4. HeadProximity (32 dims, 4 zenith × 8 azimuth)")

    print("\n  Grid: 4 rows (zenith) × 8 cols (azimuth)")
    print("  左右镜像: azimuth φ → -φ, 即 col [0,1,2,3,4,5,6,7] → [7,6,5,4,3,2,1,0]")
    for r in range(4):
        o_row = orig_head[0, r*8:(r+1)*8].cpu().tolist()
        m_row = mirr_head[0, r*8:(r+1)*8].cpu().tolist()
        print(f"  ze[{r}]: orig {[f'{v:5.1f}' for v in o_row]}")
        print(f"         mirr {[f'{v:5.1f}' for v in m_row]}")

    # 验证 azimuth 翻转
    print("\n  验证 azimuth 翻转:")
    for r in range(4):
        orig_row = orig_head[0, r*8:(r+1)*8]
        mirr_row = mirr_head[0, r*8:(r+1)*8]
        # mirr[col c] 应等于 orig[col 7-c] (简单翻转)
        flip_ok = True
        for c in range(8):
            if not torch.allclose(mirr_row[c], orig_row[7-c]):
                flip_ok = False
                break
        print(f"  ze[{r}]: mirr[col] == orig[col_reversed]: {flip_ok}")

    # -- Actions --
    print_header("5. Actions (12 dims, joint position targets)")

    o = orig_act[0].cpu().tolist()
    m = mirr_act[0].cpu().tolist()
    print(f"\n  orig: {[f'{v:.1f}' for v in o]}")
    print(f"  mirr: {[f'{v:.1f}' for v in m]}")
    print(f"  joints: FL↔FR, RL↔RR, hip×(-1)")
    for j in range(12):
        jn = JOINT_NAMES[j]
        diff_mark = " ≠" if abs(o[j] - m[j]) > 0.5 else "  "
        print(f"  {jn:10s}: orig {o[j]:8.1f}  mirr {m[j]:8.1f} {diff_mark}")

    print_header("验证完成")
    print()


if __name__ == "__main__":
    main()
