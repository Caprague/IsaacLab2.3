# 用户变更日志

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