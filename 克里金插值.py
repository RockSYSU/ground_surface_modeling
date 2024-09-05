import open3d as o3d
import numpy as np
from pykrige.ok import OrdinaryKriging
from tqdm import tqdm
from scipy.spatial import cKDTree

# 读取原始点云文件
pcd = o3d.io.read_point_cloud("ground.pcd")

# 显示原始点云
print("原始点云:")
o3d.visualization.draw_geometries([pcd], window_name="原始点云")

# 下采样点云
voxel_size = 0.25  # 你需要的采样大小
down_pcd = pcd.voxel_down_sample(voxel_size)

# 显示下采样后的点云
print("下采样后的点云:")
o3d.visualization.draw_geometries([down_pcd], window_name="下采样后的点云")

# 转换为NumPy数组
points = np.asarray(down_pcd.points)

# 获取 x, y, z 轴的最小值和最大值
x_min, y_min, z_min = points.min(axis=0)
x_max, y_max, z_max = points.max(axis=0)

# 构建网格点
x_range = np.arange(x_min, x_max, voxel_size)
y_range = np.arange(y_min, y_max, voxel_size)
x_grid, y_grid = np.meshgrid(x_range, y_range)

# 准备克里金插值
x_points = points[:, 0]
y_points = points[:, 1]
z_points = points[:, 2]

# 分块处理数据，减小内存消耗
block_size = 1000  # 每块处理的网格大小

z_grid = np.empty_like(x_grid)
num_blocks = (x_grid.size + block_size - 1) // block_size

with tqdm(total=num_blocks, desc="Kriging interpolation") as pbar:
    for i in range(0, x_grid.size, block_size):
        x_block = x_grid.ravel()[i:i + block_size]
        y_block = y_grid.ravel()[i:i + block_size]

        OK = OrdinaryKriging(x_points, y_points, z_points, variogram_model='linear', verbose=False, enable_plotting=False)
        z_block, _ = OK.execute('points', x_block, y_block)
        z_grid.ravel()[i:i + block_size] = z_block

        pbar.update(1)

# 构建新的点云
new_points = np.c_[x_grid.ravel(), y_grid.ravel(), z_grid.ravel()]
new_pcd = o3d.geometry.PointCloud()
new_pcd.points = o3d.utility.Vector3dVector(new_points)

# 使用 cKDTree 查找并去除与原始点云重合的点
original_tree = cKDTree(points)
distances, _ = original_tree.query(new_points, k=1)
threshold = voxel_size / 2  # 距离阈值，小于这个距离的点认为是重合的
filtered_points = new_points[distances > threshold]

# 构建过滤后的点云
filtered_pcd = o3d.geometry.PointCloud()
filtered_pcd.points = o3d.utility.Vector3dVector(filtered_points)

# 保存过滤后的点云文件
o3d.io.write_point_cloud("filtered_resampled_ground.pcd", filtered_pcd)

# 显示过滤后的点云
print("去除重合点后的均匀采样点云:")
o3d.visualization.draw_geometries([filtered_pcd], window_name="去除重合点后的均匀采样点云")
