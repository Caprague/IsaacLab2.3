"""配置射线投射激光雷达传感器。"""

from dataclasses import MISSING

from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.markers.config import RAY_CASTER_LIDAR_MARKER_CFG
from isaaclab.utils import configclass

from .multi_mesh_ray_caster_cfg import MultiMeshRayCasterCfg
from .ray_caster_lidar import RayCasterLidar


@configclass
class RayCasterLidarCfg(MultiMeshRayCasterCfg):
    """射线投射激光雷达传感器的配置类。

    继承自 :class:`MultiMeshRayCasterCfg`，支持多 mesh 光线检测，
    并扩展了激光雷达特有的功能（动态扫描模式、包围盒裁剪、数据保存等）。
    """

    @configclass
    class DataSaverCfg:
        """数据采集保存器的配置参数。"""

        data_type: str = "pcd"
        """保存的数据格式类型。默认为 'pcd'。"""
        sub_dir_name: str = "partial"
        """保存数据的子目录名称。默认为 'partial'。"""
        max_sequence: int = 100
        """每个环境保存的最大序列数。默认为 100。"""
        T_max: int = 2
        """每个序列的最大时间步数。默认为 2。"""

    class_type: type = RayCasterLidar
    """对应的传感器类类型。"""

    visualizer_cfg: VisualizationMarkersCfg = RAY_CASTER_LIDAR_MARKER_CFG.replace(prim_path="/Visuals/RayCaster")
    """可视化标记的配置对象。默认为 RAY_CASTER_LIDAR_MARKER_CFG。

    注意：
        仅在启用调试可视化时使用此属性。
    """

    dynamic_pattern: bool = False
    """是否在每次传感器更新时动态更新射线模式。

    启用后，模式生成函数会在每次更新时被调用以生成新的扫描模式。
    这对于需要通过随时间循环不同扫描模式来模拟连续扫描的传感器（如 Livox Mid-360）非常有用。

    默认为 False，这意味着射线模式仅在初始化时计算一次，并在整个仿真过程中保持静态。
    这是大多数射线投射传感器的默认行为。
    """

    data_box_clip: tuple[float, float, float] | None = None
    """非 None 时，将按照给定的包围盒尺寸，对数据进行裁剪。"""

    data_normalization: bool = False
    """仅当时 data_box_clip 有效时可用，控制是否对包围盒裁剪后的数据进行归一化处理，范围 (-0.5, +0.5)"""

    data_collection: bool = False
    """是否启用数据收集。"""

    data_save_path: str | None = None
    """数据保存路径。"""

    pc_data_saver_cfg: DataSaverCfg = DataSaverCfg(data_type='pcd', sub_dir_name='partial', max_sequence=100, T_max=2)
    """点云数据保存器配置。默认保存为 pcd 格式到 partial 子目录。"""

    pose_data_saver_cfg: DataSaverCfg = DataSaverCfg(data_type='npz', sub_dir_name='transform', max_sequence=100, T_max=2)
    """位姿数据保存器配置。默认保存为 npz 格式到 transform 子目录。"""

    yaw_inv: bool = False
    """是否将传感器的 yaw 角额外旋转 180°。默认为 False。

    当雷达倒装（朝下扫描）时，传感器坐标系的前向轴与机器人本体前向轴相反，
    导致逆变换后点云的 yaw 方向与直觉期望相反。启用此参数后，将在 yaw 逆旋转
    之前额外叠加一个绕 Z 轴 180° 的修正旋转，使点云朝向恢复为机器人前向。

    仅在 ``ray_alignment`` 为 ``"yaw"`` 时生效。
    """
