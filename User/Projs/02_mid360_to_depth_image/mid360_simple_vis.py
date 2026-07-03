#!/usr/bin/env python3
import open3d as o3d
import numpy as np
import torch
import glob
import sys

def mid360_pointcloud_to_depth_image(points, width=180, height=32, min_range_m=0.1, max_range_m=10.0, min_elevation_deg=-7.0, max_elevation_deg=52.0):
    N = 1
    x, y, z = points[..., 0], points[..., 1], points[..., 2]
    ranges = torch.sqrt(x**2 + y**2 + z**2)
    azimuths = torch.atan2(y, x)
    elevations = torch.asin(z / (ranges + 1e-8))
    
    min_elev_rad = torch.tensor(min_elevation_deg * torch.pi / 180.0)
    max_elev_rad = torch.tensor(max_elevation_deg * torch.pi / 180.0)
    azimuth_res_rad = 2.0 * torch.pi / width
    elevation_res_rad = (max_elev_rad - min_elev_rad) / (height - 1)
    
    u_idx = ((azimuths + torch.pi) / azimuth_res_rad).long()
    v_idx = ((elevations - min_elev_rad) / elevation_res_rad).long()
    u_idx = torch.clamp(u_idx, 0, width - 1)
    v_idx = torch.clamp(v_idx, 0, height - 1)
    
    valid_mask = (ranges >= min_range_m) & (ranges <= max_range_m)
    depth_map = torch.full((N, height * width), float("inf"))
    flat_idx = v_idx * width + u_idx
    
    filtered_ranges = torch.where(valid_mask, ranges, float("inf"))
    depth_map.scatter_reduce_(1, flat_idx.unsqueeze(0), filtered_ranges.unsqueeze(0), reduce="min", include_self=False)
    
    depth_map = depth_map.view(N, height, width).squeeze(0)
    depth_map = torch.where(depth_map >= float("inf"), torch.tensor(0.0), depth_map)
    return depth_map

def depth_image_to_color_map(depth_map, max_range=10.0):
    depth_np = depth_map.numpy()
    color = np.zeros((depth_np.shape[0], depth_np.shape[1], 3), dtype=np.float32)
    valid = depth_np > 0
    normalized = np.clip(depth_np[valid] / max_range, 0, 1)
    color[valid] = np.stack([1 - normalized, normalized * 0.5, normalized], axis=-1)
    return color

def create_depth_mesh(depth_color, width=180, height=32):
    vertices = []
    triangles = []
    for v in range(height):
        for u in range(width):
            x = (u / width - 0.5) * 10
            y = (v / height - 0.5) * 2
            z = 0
            vertices.append([x, y, z])
            
            if u < width - 1 and v < height - 1:
                idx = v * width + u
                triangles.append([idx, idx + 1, idx + width])
                triangles.append([idx + 1, idx + width + 1, idx + width])
    
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(triangles)
    
    colors = depth_color.reshape(-1, 3)
    mesh.vertex_colors = o3d.utility.Vector3dVector(colors)
    
    return mesh

def main():
    if len(sys.argv) < 2:
        print("Usage: python mid360_simple_vis.py <pcd_file>")
        sys.exit(1)
    
    pcd_path = sys.argv[1]
    print(f"Loading: {pcd_path}")
    
    pcd = o3d.io.read_point_cloud(pcd_path)
    points_np = np.asarray(pcd.points)
    print(f"Points: {points_np.shape[0]}")
    
    points_t = torch.from_numpy(points_np).float()
    depth_map = mid360_pointcloud_to_depth_image(points_t)
    print(f"Depth image: {depth_map.shape}, valid: {(depth_map > 0).sum().item()}")
    
    depth_color = depth_image_to_color_map(depth_map)
    depth_mesh = create_depth_mesh(depth_color)
    
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=1280, height=720)
    
    # 添加点云
    pcd.paint_uniform_color([0.2, 0.8, 0.2])
    vis.add_geometry(pcd)
    
    # 添加深度图网格（放在点云前面）
    depth_mesh.translate([0, 0, 2])
    vis.add_geometry(depth_mesh)
    
    # 设置视角
    ctr = vis.get_view_control()
    ctr.set_lookat([0, 0, 0])
    ctr.set_up([0, 0, 1])
    ctr.set_front([0, -1, 0])
    
    vis.run()
    vis.destroy_window()

if __name__ == "__main__":
    main()
