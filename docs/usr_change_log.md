# 用户变更日志
<!-- 按时间倒序排列，最新修改在最顶部 -->

## v0.1.3 (2026-07-17)

### 新增功能

- **蒸馏训练配置完善**：添加 `RslRlDistillationStudentTeacherDepthImageCfg` 配置类，支持学生网络融入深度图 CNN 特征提取模块
- **新环境注册**：注册 `Go2-Loco-Skill-Walk-Mid360Depth-10Hz` 和 `Go2-Loco-Skill-Walk-Mid360Depth-10Hz-GRU` 环境及 Play 版本
  - `Go2-Loco-Skill-Walk-Mid360Depth-10Hz`：采用**原始深度图+帧标记**方式处理深度图10Hz与Policy 50Hz的频率兼容性，通过 `depth_image_age` 观测函数标记深度图存留帧数，CNN提取空间特征后拼接时序标记位输入策略网络
  - `Go2-Loco-Skill-Walk-Mid360Depth-10Hz-GRU`：预期使用**GRU时序网络**提取本体信息和深度图信息的时序特征来解决频率同步问题，环境配置已创建，GRU网络实现及配置正确性有待后期检查验证
- **PPO 深度图配置**：添加 `RslRlPpoActorCriticDepthImageCfg` 配置类，支持深度图特征提取网络

### Bug修复

- **深度信息截断问题**：修复 `mid360_depth` 观测 `clip=(0.0, 1.0)` 导致大部分深度值被截断为1的问题，改为 `clip=(0.0, 2.5)`
- **可视化色条标注**：修复 `mid360_visualizer.py` 色条标注格式错误（`.0f`→`.1f`），正确显示2.5m

### 修改文件

- `User/Projs/02_mid360_to_depth_image/mid360_visualizer.py` - 色条标注格式修正，包围球半径统一为2.5m
- `source/isaaclab_assets/isaaclab_assets/robots/unitree.py` - 更新 Go2 Mid360 和 D435X2 机器人 USD 文件路径
- `source/isaaclab_rl/isaaclab_rl/rsl_rl/distillation_cfg.py` - 添加 `RslRlDistillationStudentTeacherDepthImageCfg` 配置类
- `source/isaaclab_rl/isaaclab_rl/rsl_rl/rl_cfg.py` - 添加 `RslRlPpoActorCriticDepthImageCfg` 配置类
- `source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/go2/__init__.py` - 注册新环境
- `source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/go2/go2_loco_skill_walk_mid360_depth_10hz_cfg.py` - 深度图观测参数优化（scale: 3.0→2.0, clip: (0.0,1.0)→(0.0,2.5), dropout_prob: 0.1→0.05），更新 raycast target 和随机化配置
- `source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/go2/go2_loco_skill_walk_mid360_depth_10hz_gru_cfg.py` - 同上修改

### 新增文件

- `source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/go2/agents/rsl_rl_distillation_cfg_walk_mid360_depth_10hz.py` - 蒸馏训练配置
- `source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/go2/agents/rsl_rl_distillation_cfg_walk_mid360_depth_10hz_gru.py` - GRU 蒸馏训练配置

---

## v0.1.2 (2026-07-07)

### 新增功能

- **雷达噪声系统**：实现完整的雷达测量噪声模型
  - 距离噪声：基于距离的高斯噪声（σ = 0.005 + 0.0015×d），在接收端施加
  - 角度噪声：统一角度噪声参数（0.15°），在发射端施加，更符合物理现实
- **深度图dropout**：`mid360_structured_depth_image` 添加 `dropout_prob` 参数（默认5%），随机置零像素模拟信号丢失
- **性能优化**：从 Warp kernel 直接获取 `ray_distance`，避免二次计算 `torch.norm`

### Bug修复

- **静态模式角度噪声累积**：保存原始射线方向副本，每帧从原始方向加噪
- **动态模式噪声共享**：先 repeat 到所有环境再独立加噪，确保每个环境噪声独立
- **RayCasterBoxData 缺少 ray_distance**：添加字段并正确初始化

### 修改文件

- `source/isaaclab/isaaclab/sensors/ray_caster/ray_caster_lidar.py` - 噪声注入逻辑、性能优化、bug修复
- `source/isaaclab/isaaclab/sensors/ray_caster/ray_caster_lidar_cfg.py` - 新增噪声配置类
- `source/isaaclab/isaaclab/sensors/ray_caster/multi_mesh_ray_caster.py` - 添加 `return_distance` 支持
- `source/isaaclab/isaaclab/sensors/ray_caster/multi_mesh_ray_caster_cfg.py` - 添加 `return_distance` 配置项
- `source/isaaclab/isaaclab/sensors/ray_caster/multi_mesh_ray_caster_data.py` - 添加 `ray_distance` 字段
- `source/isaaclab/isaaclab/sensors/ray_caster/ray_caster_box_data.py` - 添加 `ray_distance` 字段
- `source/isaaclab/isaaclab/envs/mdp/observations.py` - 添加 `dropout_prob` 参数
- `source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/go2/go2_loco_skill_walk_mid360_depth_10hz_cfg.py` - 启用噪声配置
- `docs/usr_todo_list.md` - 添加循环填充方案

### 新增文件

- `source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/go2/go2_loco_skill_walk_mid360_depth_10hz_cfg.py` - 10Hz深度图配置
- `source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/go2/go2_loco_skill_walk_mid360_depth_10hz_gru_cfg.py` - GRU时序版本配置
- `source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/go2/agents/rsl_rl_ppo_cfg_walk_mid360_depth_10hz.py` - PPO代理配置
- `source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/go2/agents/rsl_rl_ppo_cfg_walk_mid360_depth_10hz_gru.py` - GRU代理配置

---

## v0.1.1 (2026-07-06)

### 新增功能

- **深度图时间标记**：新增 `depth_image_age` 观测函数，用于标记转换后深度图的存留帧数，支持 50Hz policy 兼容 10Hz 深度图输入
- **RayCasterLidar frame_id 计数**：参考 `RayCasterCamera` 实现帧号自动递增，支持传感器数据更新检测
- **Mid360PatternCfg 参数重构**：将 `points_per_scan` 改为 property 自动计算，新增 `points_per_second`（默认 200000）和 `update_frequency_hz`（默认 10.0）参数

### 修改文件

- `source/isaaclab/isaaclab/envs/mdp/observations.py` - 新增 `depth_image_age` 函数，`mid360_structured_depth_image` 添加距离截断和对数映射参数
- `source/isaaclab/isaaclab/sensors/ray_caster/ray_caster_lidar.py` - 添加 `_frame` 计数器、`frame` 属性、`reset` 重置逻辑
- `source/isaaclab/isaaclab/sensors/ray_caster/ray_caster_box_data.py` - 添加 `frame_id` 字段
- `source/isaaclab/isaaclab/sensors/ray_caster/patterns/patterns_cfg.py` - `Mid360PatternCfg` 参数重构，支持频率配置
- `source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/go2/go2_test.py` - 更新雷达配置
- `docs/usr_todo_list.md` - 添加深度图频率方案分析，标记方案一（50Hz）为已验证不可行

---

## v0.1.1 (2026-07-03)

### 新增功能

- **Mid360 点云转深度图**：实现从原始点云到 180×32 结构化深度图的转换，支持对数映射增强近距离区分度
- **对数距离映射**：引入 `log_k` 参数，通过 `log(1+k*d)/log(1+k*max)` 压缩远距离、拉伸近距离
- **可视化工具**：创建完整的点云和深度图可视化脚本，支持绿→蓝→红配色、线框包围球、颜色对照条

### 修改文件

- `source/isaaclab/isaaclab/envs/mdp/observations.py` - `mid360_structured_depth_image` 添加 `log_k` 参数，实现向量化对数映射和距离截断
- `source/isaaclab/isaaclab/sensors/ray_caster/ray_caster_box_data.py` - 更新配置
- `source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/go2/go2_test.py` - 更新配置

### 新增文件

- `User/Projs/02_mid360_to_depth_image/mid360_visualizer.py` - 完整可视化脚本，包含点云着色、深度图显示、包围球、色带
- `User/Projs/02_mid360_to_depth_image/mid360_simple_vis.py` - 简化版可视化脚本

### 项目重命名

- `User/Projs/01_simulation_data_collector/` → `User/Projs/01_lidar_data_collector/`

---

## v0.1.0 (2026-07-03)

### 新增功能

- **雷达多 Mesh 检测支持**：将 `RayCasterLidar` 继承自 `MultiMeshRayCaster`，支持同时对多个 mesh 进行光线投射检测
- **Go2 头部 Mid360 雷达测试**：完成 Go2 机器人头部 Mid360 激光雷达的配置和测试

### 修改文件

- `source/isaaclab/isaaclab/sensors/ray_caster/ray_caster_lidar.py` - 改为继承 `MultiMeshRayCaster`，重写 `_update_buffers_impl` 追加 LiDAR 后处理逻辑
- `source/isaaclab/isaaclab/sensors/ray_caster/ray_caster_lidar_cfg.py` - 改为继承 `MultiMeshRayCasterCfg`，保留 LiDAR 特有配置字段

---

## v0.1.0 (2026-07-01)

### 修复与完善

- **修正迁移 bug**：修复项目从 IsaacLab 2.2 迁移到 2.3 过程中出现的兼容性问题
- **补充 RSL-RL Attention 模块**：迁移并完善 rsl_rl attention 模块相关代码

---

## v0.1.0 (2026-07-01)

### 初始迁移

- **项目迁移**：初步完成 IsaacLab 2.2 项目内容向 IsaacLab 2.3 的迁移
- **依赖修复**：修复 `flatdict` 安装冲突问题
