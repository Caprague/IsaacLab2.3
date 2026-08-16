# 用户变更日志
<!-- 按时间倒序排列，最新修改在最顶部 -->

## v0.1.7 (2026-08-10 ~ 08-13)

### 新增功能

- **Parkour 式教师策略网络**：新增 `ActorCriticScan` 网络与 `ScanEncoder` 扫描编码器，教师策略在 PPO 阶段端到端联合训练高度扫描编码（`mapScans`→32维 `scan_latent`）与特权编码（`privileged`→32维 `privilege_latent`），训练出的教师 checkpoint 原生包含编码器权重，蒸馏时直接加载并冻结
- **GRU 深度图学生-教师网络完整实现**：`StudentTeacherDepthImageRecurrent` 完成深度图 CNN（180×32）→ 32维 `depth_latent` → 与最新本体帧、`depth_image_age` 帧龄信号拼接 → GRU 时序融合 → 输出 `depth_latent`/`privilege_latent` 双 32维 latent → 与本体历史拼接进学生 MLP 的完整管线，支持 `teacher_recurrent` 选项
- **潜空间对齐蒸馏算法 DistillationAlign**：新增蒸馏算法，在行为克隆损失基础上叠加深度/特权 latent 对齐损失（MSE，教师 latent 冻结 detach），通过 `align_weight_depth`/`align_weight_privilege` 控制权重，配套 `RslRlDistillationAlignAlgorithmCfg` 配置类
- **深度图循环策略专用导出**：`exporter.py` 新增 JIT/ONNX 专用导出类，完整复刻 CNN+GRU 融合前向，按网络类型自动派发，解决标准导出器结构不匹配问题
- **训练日志增强**：`train.py` 新增 `--startup_log_seconds` 启动日志捕获（stdout/stderr 同时写入 `<log_dir>/train_startup.log`）；蒸馏 runner 打印观测组形状与周期训练损失
- **文档整理**：`docs` 目录重构为 `Go2相关`（Go2参考改进/Go2训练总结/Go2训练管线），新增 Parkour 参考与训练管线总结文档

### Bug修复

- **Play 导出 AttributeError**：学生记忆模块按 rsl_rl 惯例命名为 `memory_s`（原为 `memory`），修复 exporter 访问 `policy.memory_s.rnn` 崩溃；`load_state_dict` 增加旧 checkpoint `memory.*` 键兼容映射，已有模型无需重训
- **ONNX 导出形状不匹配**：标准导出器将 GRU 输出直接喂给学生 MLP（`1x256 vs 346x512`）导致导出失败，新增专用导出器按真实前向导出 `policy.pt/policy.onnx`
- **train.py resume 失败**：在新建 `log_dir` 之前解析 checkpoint 路径，避免 `get_checkpoint_path` 默认匹配到空 run 目录导致 "No checkpoints" 错误；`DistillationAlign` 纳入断点续训加载条件（含 `train_attention.py`）
- **左右对称变换修正**：`compute_symmetric_states` 改为直接在 term-major 扁平观测布局上操作（原实现按 frame 假设，与实际观测布局不符），同步修正教师阶段观测处理
- **教师编码器加载机制重构**：删除"预训练编码器单独加载"流程，改为从教师 PPO checkpoint 直接加载 `scan_encoder.*`/`privilege_encoder.*` 权重并冻结

### 修改文件

- `rsl_rl/modules/student_teacher_depth_image_recurrent.py` - GRU 深度图学生网络完整实现、`memory_s` 命名、旧 checkpoint 兼容加载
- `rsl_rl/algorithms/distillation_align.py` - 新增潜空间对齐蒸馏算法（DistillationAlign）
- `rsl_rl/modules/actor_critic_scan.py` - 新增 Parkour 式教师策略网络（ActorCriticScan）
- `rsl_rl/networks/scan_encoder.py` - 新增高度扫描编码器（ScanEncoder）
- `rsl_rl/networks/teacher_encoders.py` - `PrivilegeEncoder` 改为仅编码（移除 decoder）
- `rsl_rl/runners/distillation_runner.py` - 教师编码器从 checkpoint 加载、观测组形状与训练损失打印
- `rsl_rl/runners/on_policy_runner.py` - 支持 `ActorCriticScan` 网络
- `scripts/reinforcement_learning/rsl_rl/train.py` - 启动日志捕获、resume 修复、`DistillationAlign` 断点续训支持
- `scripts/reinforcement_learning/rsl_rl/train_attention.py` - `DistillationAlign` 断点续训支持
- `source/isaaclab_rl/isaaclab_rl/rsl_rl/exporter.py` - 深度图循环策略专用 JIT/ONNX 导出器
- `source/isaaclab_rl/isaaclab_rl/rsl_rl/distillation_cfg.py` - 新增 `RslRlDistillationAlignAlgorithmCfg`
- `source/isaaclab_rl/isaaclab_rl/rsl_rl/rl_cfg.py` - 新增 `RslRlPpoActorCriticScanCfg`
- `source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/mdp/symmetry/go2_mid360_teacher_walk.py` - 左右对称变换修正
- `source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/mdp/symmetry/go2_skill_walk.py` - 左右对称变换修正
- `source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/go2/agents/rsl_rl_distillation_cfg_walk_mid360_depth_10hz_gru.py` - 改用 `DistillationAlign` 配置
- `source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/go2/agents/rsl_rl_ppo_cfg_walk_mid360_depth_10hz_gru.py` - 改用 `RslRlPpoActorCriticScanCfg`
- `source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/go2/go2_loco_skill_walk_mid360_depth_10hz_gru_cfg.py` - 训练命令与默认 stage 更新

### 新增文件

- `rsl_rl/algorithms/distillation_align.py` - 潜空间对齐蒸馏算法
- `rsl_rl/modules/actor_critic_scan.py` - Parkour 式教师策略网络
- `rsl_rl/networks/scan_encoder.py` - 高度扫描编码器
- `docs/Go2相关/Go2参考改进/参考-Parkour-GRU方案训练改进.md` - Parkour GRU 方案训练改进参考
- `docs/Go2相关/Go2训练管线/总结-Go2-Mid360Depth-10Hz-GRU-训练管线.md` - GRU 训练管线总结
- `docs/Go2相关/Go2训练管线/总结-Go2-Mid360Depth-10Hz-训练管线.md` - 训练管线总结
- `docs/Go2相关/` - Go2参考改进/Go2训练总结/Go2训练管线 目录重构（原 `docs/ref_analysis_docs`、`docs/go2_analysis_docs` 文档迁移）

### 删除文件

- `scripts/tools/encoder_pretrain/pretrain_teacher_encoders.py` - 编码器预训练脚本（改为教师策略 PPO 联合训练）
- `scripts/tools/encoder_pretrain/visualize_encoder_reconstruction.py` - 编码器重建可视化脚本（同上）
- `docs/verify_mid360_symmetry_full.py` - 对称性验证脚本
- `docs/verify_symmetry.py` - 对称性验证脚本

### 配置优化

- **教师 PPO 配置**：改用 `RslRlPpoActorCriticScanCfg`，启用 `scan_obs_group=mapScans`、`privilege_obs_group=privileged`（latent 32维）
- **蒸馏训练配置**：改用 `RslRlDistillationAlignAlgorithmCfg`，启用 latent 对齐损失（`align_weight_depth=1.0`、`align_weight_privilege=1.0`），`num_steps_per_env` 60→120
- **环境默认阶段**：`Go2LocomotionSkillEnvCfg` 默认 `stage` 由 `stage4` 改为 `stage1`；训练命令示例更新为从 `model_6000` 断点加载

---

## v0.1.4 (2026-07-20 ~ 07-25)

### 新增功能

- **规则网格射线模式**：新增 `Mid360GridPatternCfg` 配置类和 `mid360_grid_pattern` 函数，生成 180×32 规则网格射线（5760条），替代动态扫描模式提升训练性能
- **mid360_grid_depth_image 观测函数**：针对规则网格模式优化的深度图转换函数，直接reshape无需坐标转换
- **LiDAR模式切换**：添加 `use_simple_lidar` 参数，训练时启用简化模式（规则网格+mid360_grid_depth_image），测试/部署时使用原动态模式（CSV扫描+mid360_structured_depth_image）
- **邻域填充功能**：`mid360_structured_depth_image` 和 `mid360_grid_depth_image` 均支持 `fill_invalid` 参数，对无效像素进行1轮填充（周围≥2个有效点时取邻域均值）

### Bug修复

- **深度图上下颠倒**：修复 `mid360_grid_pattern` 函数中 `torch.meshgrid` 参数顺序错误，改为 `torch.meshgrid(zenith, azimuth, indexing="xy")` 确保输出格式与真实模式一致（行优先）
- **深度图截断逻辑**：统一 `mid360_structured_depth_image` 的距离截断规则，超出max_range_m和低于min_range_m的点均设为0
- **inf值处理**：聚合后inf值截断到max_range_m
- **IsaacSim 5.1 暂停/恢复后机器人可视化冻结**（重要修复）：

  **问题现象**：在 IsaacSim 5.1 + IsaacLab 2.3 环境下，运行 `play.py` 时通过 GUI 暂停仿真后恢复，机器人模型的可视化网格冻结在原地不再更新，但物理仿真仍在正常运行。控制台持续输出 `FabricManager::initializePointInstancer mismatched prototypes on point instancer: /Visuals/Command/velocity_current` 和 `/Visuals/Command/velocity_goal` 警告。

  **根因分析**：PhysX fabric 107.3.21+（随 Isaac Sim 5.1 发布）存在回归 bug。FabricManager 在首次播放时会触发 `FabricManager::resume:rigidBodyInitialization:writeToFabric` 将关节体（articulation）的初始位姿写入 fabric 同步层。但在后续的暂停/恢复周期中，FabricManager 跳过了此写入步骤，导致 fabric 中存储的变换数据过期，Hydra 渲染器无法获取最新的关节体位姿，表现为机器人网格视觉冻结。此问题在 Isaac Sim 5.0 中不存在，属于 5.1 版本回归。

  **修复机理**：在 `SimulationContext.step()` 的暂停等待循环退出后（即用户点击恢复时），调用新增的 `_re_sync_fabric()` 方法。该方法通过 `fabric_iface.detach_stage()` 分离当前 USD stage，强制 FabricManager 丢弃内部过期状态；随后通过 `fabric_iface.attach_stage(stage_id)` 重新附加 stage，迫使 FabricManager 执行完整重新初始化流程——包括重新写入所有变换数据到 fabric，使 Hydra 渲染器通过 IFabricHierarchy 缓存变换管线获取正确的位姿。

  **修复代码**（`source/isaaclab/isaaclab/sim/simulation_context.py`）：
  1. `step()` 方法暂停循环后新增调用：
  ```python
  if not self.is_stopped():
      self._re_sync_fabric()
  ```
  2. 新增 `_re_sync_fabric()` 方法：
  ```python
  def _re_sync_fabric(self):
      if self._fabric_iface is None:
          return
      stage = self.stage
      if stage is None:
          return
      stage_id = UsdUtils.StageCache.Get().GetId(stage).ToLongInt()
      if stage_id <= 0:
          return
      try:
          self._fabric_iface.detach_stage()
      except Exception:
          logger.warning("Failed to detach fabric stage during re-sync.")
          return
      try:
          self._fabric_iface.attach_stage(stage_id)
          self._update_fabric(0.0, 0.0)
      except Exception:
          logger.warning("Failed to re-attach fabric stage after pause/resume.")
  ```

  **上游参考**：[IsaacLab #4279](https://github.com/isaac-sim/IsaacLab/issues/4279)、[PR #5178](https://github.com/isaac-sim/IsaacLab/pull/5178)

### 修改文件

- `source/isaaclab/isaaclab/sim/simulation_context.py` - 新增 `_re_sync_fabric()` 方法，在暂停恢复时 detach/attach USD stage 强制 FabricManager 重新初始化，修复机器人可视化冻结问题
- `source/isaaclab/isaaclab/envs/mdp/observations.py` - 添加 `mid360_grid_depth_image` 函数，完善 `mid360_structured_depth_image` 的截断、填充逻辑，使用预计算ray_distance
- `source/isaaclab/isaaclab/sensors/ray_caster/patterns/patterns.py` - 添加 `mid360_grid_pattern` 函数，修复meshgrid参数顺序
- `source/isaaclab/isaaclab/sensors/ray_caster/patterns/patterns_cfg.py` - 添加 `Mid360GridPatternCfg` 配置类
- `source/isaaclab/isaaclab/sensors/ray_caster/ray_caster_lidar.py` - 修复 `_original_ray_directions` 初始化问题
- `source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/go2/go2_loco_skill_walk_mid360_depth_10hz_cfg.py` - 添加 `use_simple_lidar` 参数，完善stage配置
- `source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/go2/go2_loco_skill_walk_mid360_depth_10hz_gru_cfg.py` - 同上修改
- `User/Projs/02_mid360_to_depth_image/mid360_visualizer.py` - 深度图处理逻辑与observations.py完全对齐，添加fill_invalid和dropout_prob参数

### 新增文件

- `User/Projs/02_mid360_to_depth_image/visualize_depth_comparison.py` - 深度图对比可视化工具

### 配置优化

- **环境切换机制**：`unitree.py` 和 `go2_loco_skill_walk_mid360_depth_10hz_cfg.py` 添加 `environment` 参数，通过 `"local"`/`"server"` 自动切换所有硬编码路径
- **训练阶段化配置**：`stage` 参数控制 base_velocity 命令范围、传感器配置、mid360_depth 观测组和更新周期
- **PPO训练配置自动同步**：`max_iterations` 根据stage值自动设置（stage1=4001, stage2=6001）
- **事件统一管理**：`push_jump` 事件从 EventsCfg 中移除，改为在 `_apply_stage1_config()` 中动态创建，stage2/stage3 自动禁用
- **stage配置集中化**：mid360 传感器更新频率配置从独立判断移到 `_apply_stage3_config()` 中统一管理
- **深度图帧标记**：`Mid360Depth` 和 `Mid360DepthGrid` 观测组添加 `depth_image_age` 观测项，配置 `max_age=5` + `scale=5.0`，实现10Hz深度图与50Hz策略的频率同步

---

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

### 配置优化

- **环境切换机制**：在 `unitree.py` 和 `go2_loco_skill_walk_mid360_depth_10hz_cfg.py` 中添加 `environment` 参数，通过修改 `"local"` 或 `"server"` 自动切换所有硬编码路径
- **训练阶段化配置**：`Go2LocomotionSkillEnvCfg` 添加 `stage` 参数，实现三个阶段自动配置切换（stage1: 低速向前/S1地形，stage2: 全向移动/S2地形，stage3: 启用mid360传感器）
- **PPO训练配置自动同步**：`UnitreeGo2LocoSkillPPORunnerCfg` 自动读取环境配置的 `stage` 值，stage1→max_iterations=4001，stage2→max_iterations=6001
- **地形配置重命名**：`SKILL_WALK_PLUS_TERRAINS_S1_CFG`→`SKILL_WALK_PLUS_TERRAINS_EASY_CFG`，`SKILL_WALK_PLUS_TERRAINS_S2_CFG`→`SKILL_WALK_PLUS_TERRAINS_HARD_CFG`

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
