# Calculate the signed distance field of the road mask
import os
import cv2
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
import sys
print(sys.getrecursionlimit())
sys.setrecursionlimit(5000)

def generate_sdf(binary_mask):
    """
    生成 SDF 掩码，要求在道路内部为正值，外部为负值，道路边缘为0。
    Args:
        binary_mask (np.ndarray): 二值化的道路掩码，1 表示道路，0 表示背景。
    Returns:
        sdf (np.ndarray): SDF 掩码，内部为正值，外部为负值，边缘为 0。
    """

    # 输入应该是二值化的 mask，1 表示道路区域，0 表示背景区域
    assert binary_mask.dtype == np.uint8, "输入掩码应为 uint8 类型"

    # 计算道路内部到边缘的距离（正值）
    dist_to_road = cv2.distanceTransform(binary_mask, distanceType=cv2.DIST_L2, maskSize=5)

    # 计算背景到边缘的距离（负值）
    dist_to_background = cv2.distanceTransform(255 - binary_mask, distanceType=cv2.DIST_L2, maskSize=5)

    # SDF 掩码：道路区域为正值，背景为负值，边缘为 0
    sdf = dist_to_road - dist_to_background

    return sdf


path_list = ["/home/fk/python_code/datasets/dataset_sz_grid/train_val/mask",
             "/home/fk/python_code/datasets/dataset_sz_grid/test/mask",
             "/home/fk/python_code/datasets/dataset_bj_time/train_val/mask",
             "/home/fk/python_code/datasets/dataset_bj_time/test/mask"]

delta = 20

for path in path_list:
    save_path = os.path.join(path[:path.rfind('/')] , f"{path.split('/')[-1]}_sdf_clamp")
    os.makedirs(save_path, exist_ok=True)
    jet_save_path = os.path.join(path[:path.rfind('/')] , f"{path.split('/')[-1]}_jet")
    os.makedirs(jet_save_path, exist_ok=True)
    for file_name in tqdm(os.listdir(path)):
        road_mask = cv2.imread(os.path.join(path, file_name), 0)  # 0表示灰度图读取

        # 计算符号距离函数
        sdf_mask = generate_sdf(road_mask)
        sdf_clamped = np.clip(sdf_mask, -delta, delta) / delta
        np.save(os.path.join(save_path, file_name.split(".")[0]), sdf_clamped)

        # # 可视化SDF图像
        # sdf_min = np.min(sdf_mask)
        # sdf_max = np.max(sdf_mask)
        # sdf_normalized = 2 * (sdf_mask - sdf_min) / (sdf_max - sdf_min) - 1

        plt.clf()
        plt.imshow(sdf_clamped, cmap='seismic', origin='upper')  # 使用'seismic'色彩图，正值为红色，负值为蓝色
        plt.colorbar(label='SDF Values')  # 添加色条表示不同数值
        plt.title('SDF Visualization')

        # 保存图像
        plt.savefig(os.path.join(jet_save_path, file_name))
        # plt.show()
