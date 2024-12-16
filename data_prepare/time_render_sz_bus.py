"""GPS数据按照时间进行渲染,并且渲染出数量信息"""

import pandas as pd
import numpy as np
import cv2
import pickle
import os
import sys
from PIL import Image
from tqdm import tqdm
sys.path.append('..')


def GPS_sort_by_time(df):
    """按车牌号筛选，按时间信息排序"""

    # Get the unique carid values
    unique_carids = df['carID'].unique()

    # Initialize an empty DataFrame to store the sorted results
    sorted_dataframes = []

    iterater = tqdm(unique_carids)
    # Loop through each unique carid
    for carid in iterater:
    # for carid in unique_carids:
        # Filter the DataFrame by the current carid
        filtered_df = df[df['carID'] == carid]

        # Sort the filtered DataFrame by a specific column (e.g., 'brand')
        sorted_df = filtered_df.sort_values(by='systime')

        # Append the sorted DataFrame to the list
        sorted_dataframes.append(sorted_df)
        iterater.set_description_str(f"Sort GPS by time {len(sorted_dataframes)}")

    # print(sorted_dataframes[0].shape)

    return sorted_dataframes


def split_by_time(sorted_dfs):
    """按照30minates的时间间隔对排序后的数据进行切分"""
    # Initialize a list to hold the split DataFrames
    split_dataframes = []
    # Initialize variables to keep track of the time interval
    time_interval = pd.Timedelta(minutes=30)

    # Iterate through the sorted DataFrame
    iterer = tqdm(sorted_dfs)
    for sorted_df in iterer:
    # for sorted_df in sorted_dfs:
        # current_df = pd.DataFrame()
        current_df = []
        previous_time = None
        for index, row in sorted_df.iterrows():
            time = row['systime']
            if previous_time is None:
                previous_time = time
            if time - previous_time > time_interval:
                split_dataframes.append(current_df)
                current_df = []
            current_df.append(tuple(row))
            previous_time = time
        # Append the last DataFrame
        split_dataframes.append(current_df)
        iterer.set_description_str(f"Split DataFrame {len(split_dataframes)}")

    # Display the split DataFrames
    # for i, split_df in enumerate(split_dataframes):
    #     print(f"Split DataFrame {i}:\n{split_df}")
    return split_dataframes


def render_by_time(dfs):
    """按照时间来渲染数据"""
    gps_image = np.zeros((1024, 1024, 4))
    height, width, _ = gps_image.shape

    iterater = tqdm(dfs) #dfs是一个列表，里面存储了按时间切分的一条条轨迹
    for df in iterater:
        iterater.set_description_str(f"Render DataFrame")

        # 过滤掉包含相同时间戳的GPS点的轨迹
        unique_times = set() #用来存储时间戳
        for l in df:
            unique_times.add(l[0])
        if len(df)!= len(unique_times):
            print(f'Trip has duplicated timestamps, skipping...')
            continue

    # for df in dfs:
        #过滤一下数据，去除异常值
        # dflong, dflat = zip(*[
        #     (t[3], t[4]) for t in df if 0 <= t[3] < 1024 and 0 <= t[4] < 1024] )

        dflong = []
        dflat = []
        dspeed = []
        for l in df:
            if 0 <= l[3] < 1024 and 0 <= l[4] < 1024: #TODO:sz_1024size数据集的经纬度和速度在dataframe中的位置变了
                # print(t[3], t[4])
                dflong.append(int(l[3]))
                dflat.append(int(l[4]))
                dspeed.append(l[5])

        n = len(dflong)
        # n = eachID_num[carid]
        # t = np.arange(n) / n * 2 * np.pi
        t = np.arange(n) / n * np.pi

        t1, t2 = (np.sin(t) + 1) / 2, (np.cos(t) + 1) / 2
        # lats = np.array(lat_carid + 0.5, np.int_)
        # lons = np.array(lon_carid + 0.5, np.int_)
        v = np.stack([t1, t2, dspeed, np.ones_like(t1)], axis=-1)

        # 定义高斯核
        kernel_size = 3
        sigma = 1.0
        gaussian_kernel = cv2.getGaussianKernel(kernel_size, sigma)
        gaussian_kernel_2d = np.outer(gaussian_kernel, gaussian_kernel.T)
        #复制四份
        gaussian_kernel_2d = np.repeat(gaussian_kernel_2d[..., None], 4, axis=-1)
        margin = 1 #图像边缘宽度

        #乘以高斯核添加到图像上
        for i in range(n):
            x = dflat[i]
            y = dflong[i]

            at_edge = (x <= margin or x >= width - margin or y <= margin or y >= height - margin)
            #在图像边缘的点不进行高斯核的计算
            if at_edge:
                gps_image[x, y] += v[i]
            else:
                #Define the size of the region of interest (ROI)
                roi_size = 3  # You can adjust this based on your requirements
                # Extract the ROI around the specified pixel
                roi = gps_image[x-int(roi_size//2): x+roi_size//2+1, y-int(roi_size//2): y+roi_size//2+1]
                roi += gaussian_kernel_2d * v[i]

        gps_image[dflat, dflong] += v

    return gps_image

"""
-------------------------------------分割线-------------------------------------
"""


def _direct_render(patchedGPS, length=1024):
    """direct render GPS data to image"""
    gps = np.zeros((length, length, 1), np.uint8)
    ratio = length / 1024.
    patchedGPS = patchedGPS[(0 <= patchedGPS['lat']) & (patchedGPS['lat'] < 1024) &
                            (0 <= patchedGPS['lon']) & (patchedGPS['lon'] < 1024)]
    y = np.array(patchedGPS['lon'] * ratio, np.int_)
    x = np.array(patchedGPS['lat'] * ratio, np.int_)

    gps[x, y] = 255

    gps = cv2.dilate(gps, np.ones((3, 3))) # 膨胀
    gps = gps[..., None] # 增加一个维度
    return gps


def _time_render(patchedGPS):
    """只渲染GPS数据的时间信息"""

    patchedGPS['time'] = pd.to_datetime(patchedGPS['time'])

    sorted_dfs = GPS_sort_by_time(patchedGPS)
    split_dfs = split_by_time(sorted_dfs)
    gps_image_array = render_by_time(split_dfs)

    freq = gps_image_array[..., 3]
    #给二维alpha通道添加一个维度成为（1024，1024，1）并且复制两份，变成（1024，1024，3）
    freq = np.repeat(freq[..., None], 3, axis=-1)
    gps_image_array2 = gps_image_array.copy()
    gps_image_array2[...,2] = 0 #将R通道置为0,去掉speed信息
    #对每个点的R、G、B通道进行归一化，这个点的像素累加和除以这个点经过车辆的总数
    gps_image_array2[..., :3] /= freq.clip(1)

    # #持久化存储…………
    # with open('output.pkl', 'wb') as f:
    #     pickle.dump(gps_image_array2, f)

    gps_image_array2[..., 3] = 1 #将alpha通道置为1

    #将RGBA四个通道映射到255的范围内
    gps_image_array2 = gps_image_array2 * 255 + 0.5

    return gps_image_array2


def _quantity_render(patchedGPS, length=1024):
    """取log渲染出GPS点的数量信息"""

    ratio = length / 1024.
    patchedGPS = patchedGPS[(0 <= patchedGPS['lat']) & (patchedGPS['lat'] < 1024) &
                            (0 <= patchedGPS['lon']) & (patchedGPS['lon'] < 1024)]
    y = np.array(patchedGPS['lon'] * ratio, np.int_)
    x = np.array(patchedGPS['lat'] * ratio, np.int_)

    #统计每个像素点上的GPS数量
    gps_counts = np.bincount(x * length + y, minlength=length*length)

    # 根据GPS数量设置像素的亮度
    max_count = np.max(gps_counts)
    print("单个像素点上最大GPS数量：", max_count)
    #加1是为了防止出现log(0)的情况
    gps = (np.log(gps_counts + 1) / np.log(max_count + 1) * 255).reshape((length, length)).astype(np.uint8)

    # gps = cv2.dilate(gps, np.ones((3, 3))) # 膨胀
    gps = gps[..., None] # 增加一个维度
    return gps


def _speed_render(patchedGPS, length=1024):
    """只渲染GPS数据的速度信息"""

    patchedGPS['time'] = pd.to_datetime(patchedGPS['time'])

    sorted_dfs = GPS_sort_by_time(patchedGPS)
    split_dfs = split_by_time(sorted_dfs)
    gps_image_array = render_by_time(split_dfs)

    freq = gps_image_array[..., 3]
    #给二维alpha通道添加一个维度成为（1024，1024，1）并且复制两份，变成（1024，1024，3）
    freq = np.repeat(freq[..., None], 3, axis=-1)
    gps_image_array2 = gps_image_array.copy()
    gps_image_array2[...,:2] = 0 #将B、G通道置为0,去掉time信息
    #对每个点的B、G、R通道进行归一化，这个点的像素累加和除以这个点经过车辆的总数
    gps_image_array2[..., :3] /= freq.clip(1)

    # #持久化存储…………
    # with open('output.pkl', 'wb') as f:
    #     pickle.dump(gps_image_array2, f)

    gps_image_array2[..., 3] = 1 #将alpha通道置为1
    # print(gps_image_array2.max((0, 1))) #不懂
    #将RGBA四个通道映射到255的范围内
    gps_image_array2 = gps_image_array2 * 255 + 0.5
    return gps_image_array2


def _time_quantity_speed_render(patchedGPS):
    """渲染GPS数据的时间、数量、速度信息"""

    patchedGPS['systime'] = pd.to_datetime(patchedGPS['systime'])

    sorted_dfs = GPS_sort_by_time(patchedGPS)
    split_dfs = split_by_time(sorted_dfs)
    gps_image_array = render_by_time(split_dfs)

    freq = gps_image_array[..., 3]
    #给二维alpha通道添加一个维度成为（1024，1024，1）并且复制两份，变成（1024，1024，3）
    freq = np.repeat(freq[..., None], 3, axis=-1)
    gps_image_array2 = gps_image_array.copy()
    #对每个点的R、G、B通道进行归一化，这个点的像素累加和除以这个点经过车辆的总数
    gps_image_array2[..., :3] /= freq.clip(1)

    # #持久化存储…………
    # with open('output.pkl', 'wb') as f:
    #     pickle.dump(gps_image_array2, f)

    #对alpha通道取对数，加一防止出现负数，这里后续可以继续进行扩展
    gps_image_array2[..., 3] = np.log(gps_image_array2[..., 3] + 1)
    #对alpha通道进行归一化
    if gps_image_array2[..., 3].max() != 0:
        gps_image_array2[..., 3] /= gps_image_array2[..., 3].max()
    # print(gps_image_array2.max((0, 1))) #不懂
    #将RGBA四个通道映射到255的范围内
    gps_image_array2 = gps_image_array2 * 255 + 0.5
    return gps_image_array2


if __name__ == "__main__":
    """render sz GPS data to image, taxi data"""
    # path = "Datasets/dataset_sz_grid/GPS/taxi/"
    # path = "/home/fk/python_code/datasets/dataset_sz_grid/GPS/taxi"
    path = "/home/fk/python_code/datasets/dataset_sz_grid/GPS/bus"
    file_list = os.listdir(path)
    iterater = tqdm(file_list)
    for file_name in iterater:
        iterater.set_description_str(f"Processing {file_name}...")
        with open(os.path.join(path, file_name), 'rb') as f:
            patchedGPS = pickle.load(f)
        print(patchedGPS.shape)

        # file_name = "1_4_gps.pkl"
        # with open(os.path.join(path, file_name), 'rb') as f:
        #     patchedGPS = pickle.load(f)

        # #direct render
        # gps_image_array2 = _direct_render(patchedGPS)
        # save_path = "datasets/dataset_sz_1024size/GPS/bus_direct_patch"
        # if not os.path.isdir(save_path):
        #     os.makedirs(save_path)
        # cv2.imwrite(os.path.join(save_path, f"{file_name[:-8]}_gps.png"), gps_image_array2)

        # #只渲染时间信息
        # gps_image_array2 = _time_render(patchedGPS)
        # save_path = "Datasets/dataset_mask7/GPS/taxi_time_patch"
        # if not os.path.isdir(save_path):
        #     os.makedirs(save_path)
        # cv2.imwrite(os.path.join(save_path, f"{file_name[:-8]}_gps.png"), gps_image_array2)

        # #只渲染数量信息
        # gps_image_array2 = _quantity_render(patchedGPS)
        # save_path = "Datasets/dataset_mask7/GPS/taxi_quantity_patch"
        # if not os.path.isdir(save_path):
        #     os.makedirs(save_path)
        # cv2.imwrite(os.path.join(save_path, f"{file_name[:-8]}_gps.png"), gps_image_array2)

        # #只渲染速度信息
        # gps_image_array2 = _speed_render(patchedGPS)
        # save_path = "Datasets/dataset_mask7/GPS/taxi_speed_patch"
        # if not os.path.isdir(save_path):
        #     os.makedirs(save_path)
        # cv2.imwrite(os.path.join(save_path, f"{file_name[:-8]}_gps.png"), gps_image_array2)

        # #渲染时间、数量、速度信息
        # gps_image_array2 = _time_quantity_speed_render(patchedGPS)
        # save_path = "Datasets/dataset_sz_grid/GPS/taxi_time_quantity_speed_patch"
        # if not os.path.isdir(save_path):
        #     os.makedirs(save_path)
        # cv2.imwrite(os.path.join(save_path, f"{file_name[:-8]}_gps.png"), gps_image_array2)

        #渲染时间、数量、速度信息
        gps_image_array2 = _time_quantity_speed_render(patchedGPS)
        save_path = "/home/fk/python_code/datasets/dataset_sz_grid/GPS/bus_filtered_time_quantity_speed_gaussian_patch"
        os.makedirs(save_path, exist_ok=True)
        cv2.imwrite(os.path.join(save_path, f"{file_name[:-8]}_gps.png"), gps_image_array2)


    print("Done!")
