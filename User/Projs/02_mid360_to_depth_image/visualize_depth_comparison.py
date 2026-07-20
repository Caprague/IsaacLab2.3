import os
import sys
import numpy as np
from pathlib import Path
import time

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


def depth_image_to_color_map(depth_map, max_range=2.5):
    depth_map = np.squeeze(depth_map)
    
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
    
    return cv2.resize(color_map, (960, 160), interpolation=cv2.INTER_NEAREST)


def create_color_bar(max_range=2.5, bar_height=160, bar_width=60):
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
    
    cv2.putText(bar, f"{max_range:.1f}m", (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(bar, "0m", (12, bar_height - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    return bar


class DepthComparisonVisualizer:
    def __init__(self, grid_path, structured_path, max_range=2.5, play_delay=0.1):
        self.grid_path = Path(grid_path)
        self.structured_path = Path(structured_path)
        self.max_range = max_range
        self.playback_delay = play_delay
        self.last_next_time = 0
        self.grid_files = []
        self.structured_files = []
        self._find_files()
    
    def _find_files(self):
        if not self.grid_path.exists():
            print(f"错误: Grid路径不存在 - {self.grid_path}")
            return
        if not self.structured_path.exists():
            print(f"错误: Structured路径不存在 - {self.structured_path}")
            return
        
        self.grid_files = sorted(list(self.grid_path.glob("*.pt")), key=lambda x: int(x.stem.split("_")[1]))
        self.structured_files = sorted(list(self.structured_path.glob("*.pt")), key=lambda x: int(x.stem.split("_")[1]))
        
        print(f"找到 {len(self.grid_files)} 个Grid深度图帧")
        print(f"找到 {len(self.structured_files)} 个Structured深度图帧")
    
    def analyze_stats(self):
        min_count = min(len(self.grid_files), len(self.structured_files))
        if min_count == 0:
            print("没有找到深度图文件")
            return
        
        print("\n深度图统计分析:")
        print("=" * 70)
        
        for i in range(min(min_count, 5)):
            grid_depth = torch.load(self.grid_files[i])
            structured_depth = torch.load(self.structured_files[i])
            
            grid_valid = grid_depth[grid_depth > 0]
            struct_valid = structured_depth[structured_depth > 0]
            
            diff = torch.abs(grid_depth - structured_depth)
            diff_valid = diff[grid_depth > 0]
            
            print(f"\nFrame {i}:")
            print(f"  Grid       - 有效像素: {grid_valid.numel()}, 均值: {grid_valid.mean():.3f}, "
                  f"最大: {grid_valid.max():.3f}, 最小: {grid_valid.min():.3f}")
            print(f"  Structured - 有效像素: {struct_valid.numel()}, 均值: {struct_valid.mean():.3f}, "
                  f"最大: {struct_valid.max():.3f}, 最小: {struct_valid.min():.3f}")
            print(f"  差异       - 均值: {diff_valid.mean():.4f}, 最大: {diff_valid.max():.4f}")
        
        if min_count > 5:
            print("\n... (仅显示前5帧)")
    
    def visualize(self, start_idx=0):
        min_count = min(len(self.grid_files), len(self.structured_files))
        if min_count == 0:
            print("没有找到深度图文件")
            return
        
        cv2.namedWindow("深度图对比", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("深度图对比", 1020, 400)
        
        current_idx = start_idx
        
        def update_visualization():
            nonlocal current_idx
            
            if current_idx >= min_count:
                return
            
            grid_depth = torch.load(self.grid_files[current_idx])
            structured_depth = torch.load(self.structured_files[current_idx])
            
            grid_color = depth_image_to_color_map(grid_depth, max_range=self.max_range)
            struct_color = depth_image_to_color_map(structured_depth, max_range=self.max_range)
            
            color_bar = create_color_bar(max_range=self.max_range, bar_height=grid_color.shape[0], bar_width=60)
            
            grid_with_bar = np.concatenate([grid_color, color_bar], axis=1)
            struct_with_bar = np.concatenate([struct_color, color_bar], axis=1)
            
            grid_valid = grid_depth[grid_depth > 0]
            struct_valid = structured_depth[structured_depth > 0]
            diff_valid = torch.abs(grid_depth - structured_depth)[grid_depth > 0]
            
            info_bar = np.zeros((40, grid_with_bar.shape[1], 3), dtype=np.uint8)
            cv2.putText(info_bar, 
                        f"Frame {current_idx} / {min_count-1} | Grid: {grid_valid.numel()}px | Structured: {struct_valid.numel()}px | Diff: {diff_valid.mean():.4f}",
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            title_grid = np.zeros((20, grid_with_bar.shape[1], 3), dtype=np.uint8)
            cv2.putText(title_grid, "Grid Depth Map (Top)", (10, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            
            title_struct = np.zeros((20, struct_with_bar.shape[1], 3), dtype=np.uint8)
            cv2.putText(title_struct, "Structured Depth Map (Bottom)", (10, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            
            combined = np.vstack([title_grid, grid_with_bar, title_struct, struct_with_bar, info_bar])
            
            cv2.imshow("深度图对比", combined)
        
        print("\n" + "=" * 60)
        print("深度图对比可视化已启动")
        print("按 'N' 下一帧 | 按 'P' 上一帧 | 按 'R' 重置 | 按 'Q' 退出")
        print("=" * 60)
        
        update_visualization()
        
        while True:
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q') or key == ord('Q'):
                break
            elif key == ord('n') or key == ord('N'):
                current_time = time.time()
                if current_time - self.last_next_time >= self.playback_delay:
                    self.last_next_time = current_time
                    current_idx += 1
                    if current_idx >= min_count:
                        current_idx = min_count - 1
                    update_visualization()
            elif key == ord('p') or key == ord('P'):
                current_idx -= 1
                if current_idx < 0:
                    current_idx = 0
                update_visualization()
            elif key == ord('r') or key == ord('R'):
                current_idx = 0
                update_visualization()
        
        cv2.destroyAllWindows()


if __name__ == "__main__":
    grid_path = "/home/gms/Isaac/IsaacLab2.3/DataCollection/GridDepthMap"
    structured_path = "/home/gms/Isaac/IsaacLab2.3/DataCollection/StructedDepthMap"
    
    print("\n" + "=" * 60)
    print("  深度图对比可视化工具")
    print("=" * 60)
    
    visualizer = DepthComparisonVisualizer(grid_path, structured_path, max_range=2.5)
    visualizer.analyze_stats()
    
    print("\n按任意键开始可视化...")
    cv2.waitKey(0)
    
    visualizer.visualize()