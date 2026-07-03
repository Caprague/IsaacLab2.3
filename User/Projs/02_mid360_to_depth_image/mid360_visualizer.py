import os
import sys
import numpy as np
from pathlib import Path
import argparse
import time

try:
    import open3d as o3d
    OPEN3D_AVAILABLE = True
except ImportError:
    OPEN3D_AVAILABLE = False
    print("错误: 未找到Open3D库，请运行: pip install open3d")
    sys.exit(1)

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("错误: 未找到PyTorch库，请运行: pip install torch")
    sys.exit(1)

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    print("错误: 未找到OpenCV库，请运行: pip install opencv-python")
    sys.exit(1)


def mid360_pointcloud_to_depth_image(
    points_np,
    width=180,
    height=32,
    min_range_m=0.1,
    max_range_m=2.5,
    min_elevation_deg=-7.0,
    max_elevation_deg=52.0,
    aggregation_method="mean",
    log_k=10.0,
):
    if points_np.ndim == 2:
        points_np = points_np[np.newaxis, ...]
    
    device = torch.device("cpu")
    points = torch.from_numpy(points_np).to(device, dtype=torch.float32)
    N = points.shape[0]
    
    x, y, z = points[..., 0], points[..., 1], points[..., 2]
    ranges = torch.sqrt(x**2 + y**2 + z**2)
    azimuths = torch.atan2(y, x)
    elevations = torch.asin(z / (ranges + 1e-8))
    
    min_elev_rad = torch.tensor(min_elevation_deg * torch.pi / 180.0, device=device)
    max_elev_rad = torch.tensor(max_elevation_deg * torch.pi / 180.0, device=device)
    azimuth_res_rad = 2.0 * torch.pi / width
    elevation_res_rad = (max_elev_rad - min_elev_rad) / (height - 1)
    
    u_idx = ((azimuths + torch.pi) / azimuth_res_rad).long()
    v_idx = ((elevations - min_elev_rad) / elevation_res_rad).long()
    u_idx = torch.clamp(u_idx, 0, width - 1)
    v_idx = torch.clamp(v_idx, 0, height - 1)
    
    ranges = torch.where(ranges > max_range_m, torch.tensor(max_range_m, device=device), ranges)
    ranges = torch.where(ranges < min_range_m, torch.tensor(0.0, device=device), ranges)
    valid_mask = ranges > 0
    
    if log_k > 0:
        log_denominator = np.log(1 + log_k * max_range_m)
        ranges = torch.where(
            valid_mask,
            torch.log(1 + log_k * ranges) / log_denominator * max_range_m,
            ranges
        )
    depth_map = torch.full((N, height * width), float("inf"), device=device)
    flat_idx = v_idx * width + u_idx
    
    if aggregation_method == "min":
        filtered_ranges = torch.where(valid_mask, ranges, float("inf"))
        depth_map.scatter_reduce_(1, flat_idx, filtered_ranges, reduce="min", include_self=False)
    elif aggregation_method == "max":
        filtered_ranges = torch.where(valid_mask, ranges, -float("inf"))
        depth_map.scatter_reduce_(1, flat_idx, filtered_ranges, reduce="max", include_self=False)
    elif aggregation_method == "mean":
        count_map = torch.zeros((N, height * width), device=device)
        count_map.scatter_add_(1, flat_idx, valid_mask.float())
        sum_map = torch.zeros((N, height * width), device=device)
        sum_map.scatter_add_(1, flat_idx, ranges * valid_mask.float())
        depth_map = torch.where(count_map > 0, sum_map / count_map, float("inf"))
    else:
        raise ValueError(f"Unknown aggregation method: {aggregation_method}")
    
    depth_map = depth_map.view(N, height, width)
    depth_map = torch.where(depth_map >= float("inf"), torch.tensor(0.0, device=device), depth_map)
    
    return depth_map[0].cpu().numpy()


def depth_image_to_color_map(depth_map, max_range=10.0):
    normalized = np.clip(depth_map / max_range, 0.0, 1.0)
    
    color_map = np.zeros((depth_map.shape[0], depth_map.shape[1], 3), dtype=np.uint8)
    
    mask_valid = depth_map > 0
    t = normalized
    
    r = np.zeros_like(t)
    g = np.zeros_like(t)
    b = np.zeros_like(t)
    
    mask_near = t < 0.5
    mask_far = t >= 0.5
    
    g[mask_near] = 1.0 - 2 * t[mask_near]
    b[mask_near] = 2 * t[mask_near]
    
    r[mask_far] = 2 * (t[mask_far] - 0.5)
    b[mask_far] = 1.0 - 2 * (t[mask_far] - 0.5)
    
    r = np.power(r, 0.2)
    g = np.power(g, 0.2)
    b = np.power(b, 0.2)
    
    color_map[mask_valid, 0] = (255 * b[mask_valid]).astype(np.uint8)
    color_map[mask_valid, 1] = (255 * g[mask_valid]).astype(np.uint8)
    color_map[mask_valid, 2] = (255 * r[mask_valid]).astype(np.uint8)
    
    color_map[~mask_valid] = [30, 30, 30]
    
    return cv2.resize(color_map, (960, 540), interpolation=cv2.INTER_NEAREST)


def create_color_bar(max_range=10.0, bar_height=540, bar_width=60):
    bar = np.zeros((bar_height, bar_width, 3), dtype=np.uint8)
    
    for y in range(bar_height):
        t = 1.0 - y / (bar_height - 1)
        
        if t < 0.5:
            r, g, b = 0.0, 1.0 - 2 * t, 2 * t
        else:
            r, g, b = 2 * (t - 0.5), 0.0, 1.0 - 2 * (t - 0.5)
        
        r = np.power(r, 0.2)
        g = np.power(g, 0.2)
        b = np.power(b, 0.2)
        
        bar[y, :] = [int(255 * b), int(255 * g), int(255 * r)]
    
    cv2.putText(bar, f"{max_range:.0f}m", (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cv2.putText(bar, "0m", (15, bar_height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    return bar


class Mid360Visualizer:
    def __init__(self, root_path, play_delay=0.1):
        self.root_path = Path(root_path)
        self.playback_delay = play_delay
        self.last_next_time = 0
        self.pcd_files = []
        self._find_pcd_files()
    
    def _find_pcd_files(self):
        if not self.root_path.exists():
            print(f"错误: 路径不存在 - {self.root_path}")
            return
        
        direct_pcds = list(self.root_path.glob("*.pcd"))
        
        if direct_pcds:
            self.pcd_files.extend(direct_pcds)
            print(f"检测到 {len(direct_pcds)} 个PCD文件")
        else:
            for pcd_file in self.root_path.rglob("*.pcd"):
                self.pcd_files.append(pcd_file)
            print(f"递归检测到 {len(self.pcd_files)} 个PCD文件")
        
        self.pcd_files.sort(key=lambda x: (x.parent.name, x.stem))
    
    def visualize(self, start_idx=0):
        if len(self.pcd_files) == 0:
            print("错误: 未找到任何PCD文件")
            return
        
        vis_pcd = o3d.visualization.VisualizerWithKeyCallback()
        vis_pcd.create_window(window_name="点云视图", width=960, height=540, left=0, top=0)
        
        pcd = o3d.geometry.PointCloud()
        vis_pcd.add_geometry(pcd)
        
        coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0, 0, 0])
        vis_pcd.add_geometry(coordinate_frame)
        
        render_option = vis_pcd.get_render_option()
        render_option.point_size = 8.0
        render_option.background_color = np.asarray([0.35, 0.35, 0.35])
        
        max_range_m = 2.0
        sphere_points = []
        sphere_lines = []
        
        num_lat = 8
        num_lon = 16
        
        for i in range(num_lat + 1):
            theta = np.pi * i / num_lat
            for j in range(num_lon + 1):
                phi = 2 * np.pi * j / num_lon
                x = max_range_m * np.sin(theta) * np.cos(phi)
                y = max_range_m * np.sin(theta) * np.sin(phi)
                z = max_range_m * np.cos(theta)
                sphere_points.append([x, y, z])
        
        for i in range(num_lat + 1):
            for j in range(num_lon):
                idx1 = i * (num_lon + 1) + j
                idx2 = i * (num_lon + 1) + (j + 1)
                sphere_lines.append([idx1, idx2])
        
        for j in range(num_lon + 1):
            for i in range(num_lat):
                idx1 = i * (num_lon + 1) + j
                idx2 = (i + 1) * (num_lon + 1) + j
                sphere_lines.append([idx1, idx2])
        
        bounding_sphere = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(sphere_points),
            lines=o3d.utility.Vector2iVector(sphere_lines),
        )
        bounding_sphere.paint_uniform_color([1.0, 0.6, 0.0])
        vis_pcd.add_geometry(bounding_sphere)
        
        cv2.namedWindow("深度图视图", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("深度图视图", 960, 540)
        cv2.moveWindow("深度图视图", 960, 0)
        
        current_idx = start_idx
        
        def update_visualization():
            nonlocal current_idx
            
            if current_idx >= len(self.pcd_files):
                return
            
            file_path = self.pcd_files[current_idx]
            print(f"加载: {file_path.name}")
            
            try:
                loaded_pcd = o3d.io.read_point_cloud(str(file_path))
                if len(loaded_pcd.points) == 0:
                    return
                
                points_np = np.asarray(loaded_pcd.points)
                ranges = np.linalg.norm(points_np, axis=1)
                max_range = 2.5
                print(f"  点云: {len(ranges)}点, 距离范围: [{ranges.min():.4f}, {ranges.max():.4f}]m")
                
                colors = np.zeros((len(ranges), 3))
                mask_valid = (ranges >= 0.1) & (ranges <= max_range)
                t = ranges[mask_valid] / max_range
                
                r = np.zeros_like(t)
                g = np.zeros_like(t)
                b = np.zeros_like(t)
                
                mask_near = t < 0.5
                mask_far = t >= 0.5
                
                g[mask_near] = 1.0 - 2 * t[mask_near]
                b[mask_near] = 2 * t[mask_near]
                
                r[mask_far] = 2 * (t[mask_far] - 0.5)
                b[mask_far] = 1.0 - 2 * (t[mask_far] - 0.5)
                
                r = np.power(r, 0.2)
                g = np.power(g, 0.2)
                b = np.power(b, 0.2)
                
                colors[mask_valid, 0] = r
                colors[mask_valid, 1] = g
                colors[mask_valid, 2] = b
                colors[ranges < 0.1] = [0.1, 0.1, 0.1]
                
                loaded_pcd.colors = o3d.utility.Vector3dVector(colors)
                
                pcd.points = loaded_pcd.points
                pcd.colors = loaded_pcd.colors
                vis_pcd.update_geometry(pcd)
                vis_pcd.poll_events()
                vis_pcd.update_renderer()
                
                depth_map = mid360_pointcloud_to_depth_image(
                    points_np,
                    width=180,
                    height=32,
                    min_range_m=0.1,
                    max_range_m=2.5,
                    min_elevation_deg=-7.0,
                    max_elevation_deg=52.0,
                    aggregation_method="mean",
                    log_k=10.0,
                )
                
                print(f"  深度图: {depth_map.shape}, 有效像素: {(depth_map > 0).sum()}, max深度: {depth_map.max():.2f}m")
                
                color_map = depth_image_to_color_map(depth_map, max_range=2.5)
                color_bar = create_color_bar(max_range=2.5, bar_height=color_map.shape[0], bar_width=60)
                combined = np.concatenate([color_map, color_bar], axis=1)
                cv2.imshow("深度图视图", combined)
                
            except Exception as e:
                import traceback
                print(f"加载失败: {e}")
                traceback.print_exc()
        
        def next_callback(vis):
            nonlocal current_idx
            
            current_time = time.time()
            if current_time - self.last_next_time < self.playback_delay:
                return
            
            self.last_next_time = current_time
            
            update_visualization()
            current_idx += 1
        
        def prev_callback(vis):
            nonlocal current_idx
            current_idx -= 1
            if current_idx < 0:
                current_idx = 0
            update_visualization()
        
        def restart_callback(vis):
            nonlocal current_idx
            current_idx = 0
            update_visualization()
        
        vis_pcd.register_key_callback(ord("N"), next_callback)
        vis_pcd.register_key_callback(ord("P"), prev_callback)
        vis_pcd.register_key_callback(ord("R"), restart_callback)
        
        print("\n" + "=" * 40)
        print("Mid360点云与深度图可视化已启动")
        print("按 'N' 下一帧 | 按 'P' 上一帧 | 按 'R' 重置 | 按 'Q' 退出")
        print("=" * 40)
        
        update_visualization()
        
        ctr = vis_pcd.get_view_control()
        ctr.set_lookat([0, 0, 0])
        ctr.set_up([0, 0, 1])
        ctr.set_front([0, -1, 0])
        
        while True:
            vis_pcd.poll_events()
            vis_pcd.update_renderer()
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == ord('Q'):
                break
            elif key == ord('n') or key == ord('N'):
                current_time = time.time()
                if current_time - self.last_next_time >= self.playback_delay:
                    self.last_next_time = current_time
                    update_visualization()
                    current_idx += 1
            elif key == ord('p') or key == ord('P'):
                current_idx -= 1
                if current_idx < 0:
                    current_idx = 0
                update_visualization()
            elif key == ord('r') or key == ord('R'):
                current_idx = 0
                update_visualization()
        
        vis_pcd.destroy_window()
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="Mid360点云与深度图可视化工具")
    parser.add_argument("--root_path", type=str, required=True, help="PCD文件根目录")
    parser.add_argument("--start_idx", type=int, default=0, help="起始索引")
    parser.add_argument("--play_delay", type=float, default=0.1, help="播放延迟(秒)")
    args = parser.parse_args()
    
    try:
        visualizer = Mid360Visualizer(args.root_path, args.play_delay)
        visualizer.visualize(args.start_idx)
    except Exception as e:
        import traceback
        print(f"错误: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  Mid360点云与深度图可视化工具")
    print("=" * 60)
    
    if len(sys.argv) == 1:
        print("用法示例:")
        print("  python3 mid360_visualizer.py --root_path /path/to/pcd/files")
        print("  python3 mid360_visualizer.py --root_path /path/to/pcd/files --start_idx 100")
        print("\n快捷键说明:")
        print("  [N] 下一帧 | [P] 上一帧 | [R] 重置 | [Q] 退出")
        print("=" * 60 + "\n")
    else:
        main()