#generate shenzhen trajectory data from GPS data

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
    unique_carids = df['id'].unique()

    # Initialize an empty DataFrame to store the sorted results
    sorted_dataframes = []

    iterater = tqdm(unique_carids)
    # Loop through each unique carid
    for carid in iterater:
    # for carid in unique_carids:
        # Filter the DataFrame by the current carid
        filtered_df = df[df['id'] == carid]

        # Sort the filtered DataFrame by a specific column (e.g., 'brand')
        sorted_df = filtered_df.sort_values(by='time')

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
            time = row['time']
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


def gps2traj(patchedGPS):
    """Convert GPS data to trajectory data."""
    patchedGPS['time'] = pd.to_datetime(patchedGPS['time'])

    sorted_dfs = GPS_sort_by_time(patchedGPS)
    split_dfs = split_by_time(sorted_dfs)
    traj_dfs = []
    iter = tqdm(enumerate(split_dfs))
    for index, split_df in iter:
        iter.set_description_str(f"Add trajectory number for {index}...")
        #fileter out the trajectory with less than 100 points
        if len(split_df) < 20 : #filter the trajectory with less than 20 points
            continue
        #add a column to the DataFrame to store the trajectory ID
        indexed_data = [(index, *value) for value in split_df]
        #add the traj list to the list
        traj_dfs = traj_dfs + indexed_data
    #merge each split_df into a single DataFrame
    #taxi TODO
    #bus
    traj_df = pd.DataFrame(traj_dfs, columns=['traj', 'id', 'time', 'lon', 'lat', 'angle', 'speed', 'device', 'dspeed', 'mileage'])
    #reset the index of the DataFrame
    # traj_df = traj_df.reset_index(drop=True)
    return traj_df


if __name__ == "__main__":
    #classify GPS point into different trajectories
    # path = "datasets/dataset_sz_1024size/GPS/bus_pixel"
    # path = "datasets/dataset_sz_1024size/GPS/bus_wgs84_coor"
    path = "/home/fk/python_code/datasets/dataset_bj_time/GPS/patch_geo_coor"
    file_list = os.listdir(path)
    iterater = tqdm(file_list)
    for file_name in iterater:
        iterater.set_description_str(f"Processing {file_name}...")
        with open(os.path.join(path, file_name), 'rb') as f:
            patchedGPS = pickle.load(f)
        # print(patchedGPS.shape)
        #Need these two columns to construct trajectory
        if not all(col in patchedGPS.columns for col in ["time", "id"]):
            continue

        traj_df = gps2traj(patchedGPS)
        if traj_df.empty:
            continue
        # save_path = "datasets/dataset_sz_1024size/GPS/bus_long50_traj_pixel"
        # save_path = "datasets/dataset_sz_1024size/GPS/bus_long20_traj_wgs84_coor"
        save_path = "datasets/dataset_sz_1024size/GPS/bus_long20_traj_wgs84_coor"
        if not os.path.isdir(save_path):
            os.makedirs(save_path)
        with open(os.path.join(save_path, file_name), 'wb') as f:
            pickle.dump(traj_df, f)


    print("Done!")
