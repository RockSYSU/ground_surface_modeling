import open3d as o3d
import numpy as np
from tqdm import tqdm


def compute_point_density(pcd, search_radius):
    densities = []
    pcd_tree = o3d.geometry.KDTreeFlann(pcd)
    for point in tqdm(pcd.points, desc="Computing Densities"):
        [_, idx, _] = pcd_tree.search_radius_vector_3d(point, search_radius)
        densities.append(len(idx))
    return np.array(densities)


def find_missing_regions(pcd, densities, density_threshold, search_radius_factor):
    missing_points_indices = []
    mean_density = np.mean(densities)
    std_density = np.std(densities)
    pcd_tree = o3d.geometry.KDTreeFlann(pcd)

    for i, density in tqdm(enumerate(densities), desc="Finding Missing Regions", total=len(densities)):
        # 动态调整搜索半径
        dynamic_radius = search_radius_factor * (1 / (density + 1e-6)) * mean_density / (std_density + 1e-6)
        [_, idx, _] = pcd_tree.search_radius_vector_3d(pcd.points[i], dynamic_radius)
        if len(idx) < density_threshold:
            missing_points_indices.append(i)

    return missing_points_indices


# 加载点云数据
pcd = o3d.io.read_point_cloud("ground.pcd")

# 计算点云密度
search_radius = 0.05  # 初始搜索半径
densities = compute_point_density(pcd, search_radius)
print(f"平均最近邻距离: {average_nn_distance}")

# 找出缺失区域点
density_threshold = 7  # 邻域点数阈值
search_radius_factor = 1.1  # 动态调整因子
missing_points_indices = find_missing_regions(pcd, densities, density_threshold, search_radius_factor)
print(len(missing_points_indices))

# 提取缺失区域点 466
missing_points = pcd.select_by_index(missing_points_indices)

# 可视化缺失区域点
o3d.visualization.draw_geometries([missing_points])

# 保存缺失区域点云为PCD格式文件
# o3d.io.write_point_cloud("missing_points.pcd", missing_points)
print("Missing points saved to missing_points.pcd")