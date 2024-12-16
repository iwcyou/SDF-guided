#Generate dataset by spliting the satellite image using ImageSampler

from PIL import Image, ImageFile
import cv2
import numpy as np
import pickle
from tqdm import tqdm
import sys
sys.path.append("../")
from image_sampler import ImageSampler
Image.MAX_IMAGE_PIXELS = None # Disable DecompressionBombError
ImageFile.LOAD_TRUNCATED_IMAGES = True # Disable OSError: image file is truncated


#Nanshan district
train_path = "datasets/dataset_original/train.png"
test_path = "datasets/dataset_original/test.png"
mask_path = "datasets/ns_road_nofootpath_T.png"
taxi_gps_path = "datasets/GPS_data/taxi/GPS_taxi_2.0.pkl"
bus_gps_path = "datasets/GPS_data/bus/GPS_bus_2.0.pkl"

# train_path = "../Datasets/test_6/22.520000-113.920000.png"
# test_path = "../Datasets/test_6/22.520000-113.920000.png"
# mask_path = "../Datasets/test_6/22.520000-113.920000.png"
# taxi_gps_path = "../Datasets/GPS_data/test.pkl"
# bus_gps_path = "../Datasets/GPS_data/test.pkl"


print("Loading satellite image data...")
train_sat_img = Image.open(train_path)
train_sat_img = cv2.cvtColor(np.array(train_sat_img), cv2.COLOR_RGBA2BGRA)
test_sat_img = Image.open(test_path)
test_sat_img = cv2.cvtColor(np.array(test_sat_img), cv2.COLOR_RGBA2BGRA)
mask_img = Image.open(mask_path)
mask_img = cv2.cvtColor(np.array(mask_img), cv2.COLOR_RGBA2BGRA)
print("Loading GPS data...")
with open(taxi_gps_path, 'rb') as f:
        taxi_gps = pickle.load(f, encoding='bytes')
with open(bus_gps_path, 'rb') as f:
        bus_gps = pickle.load(f, encoding='bytes')
print("Have loaded.")

def _read_gps_data (big_gps, coordinate_ranges):
        """
        return GPS data by coordinate ranges,
        return geographphical coordinate GPS and pixel coordinate GPS.
        """
        gps_data = []
        gps_data_pixel = []
        for coordinate_range in coordinate_ranges:
            #坐标筛选
            selected_rows = big_gps[(big_gps['lon'].between(coordinate_range[0][0], coordinate_range[1][0])) & (big_gps['lat'].between(coordinate_range[1][1], coordinate_range[0][1]))]
            gps_data.append(selected_rows)
            #转换为图片上的像素坐标,"+0.5"做到四舍五入
            #对GPS数据的经纬度坐标进行了筛选和转换，使其能够对应到图片上的像素坐标
            pixel_rows = selected_rows.copy()
            pixel_rows['lon'] = ((selected_rows['lon'] - coordinate_range[0][0]) / (coordinate_range[1][0] - coordinate_range[0][0]) * 1024 + 0.5).astype(int)
            pixel_rows['lat'] = ((coordinate_range[0][1] - selected_rows['lat']) / (coordinate_range[0][1] - coordinate_range[1][1]) * 1024 + 0.5).astype(int)
            gps_data_pixel.append(pixel_rows)
        return gps_data, gps_data_pixel


import pandas as pd
pd.options.mode.chained_assignment = None  # Suppress the warning

train_sam = ImageSampler(train_sat_img, mask_img, "train")
train_sat_imgs, train_mask_imgs, train_coors, train_indexes = train_sam.images_sample()

test_sam = ImageSampler(test_sat_img, mask_img, "test")
test_sat_imgs, test_mask_imgs, test_coors, test_indexes = test_sam.images_sample()


#geographical coordinate and pixel coordinate
train_gps_data, train_gps_data_pixel = _read_gps_data(taxi_gps, train_coors)
test_gps_data, test_gps_data_pixel = _read_gps_data(taxi_gps, test_coors)
train_gps_data_bus, train_gps_data_bus_pixel = _read_gps_data(bus_gps, train_coors)
test_gps_data_bus, test_gps_data_bus_pixel = _read_gps_data(bus_gps, test_coors)


#Save data. The image name is row_col_sat.png, row_col_mask.png, row_col_gps.txt, row_col_gps.pkl
iterater = tqdm(range(len(train_sat_imgs)))
for i in iterater:
    iterater.set_description_str("saving train dataset")
    row = train_indexes[i][0]
    col = train_indexes[i][1]
    train_coor_range = {
        "west": train_coors[i][0][0],
        "east": train_coors[i][1][0],
        "north": train_coors[i][0][1],
        "south": train_coors[i][1][1]
    }
    #卫星图像数据
    cv2.imwrite(f"datasets/dataset_template_copy/train_val/image/{row}_{col}_sat.png", train_sat_imgs[i])
    #路网数据
    cv2.imwrite(f"datasets/dataset_template_copy/train_val/road_network/{row}_{col}_road.png", train_mask_imgs[i])
    #坐标数据
    with open(f'datasets/dataset_template_copy/coordinates/{row}_{col}_gps.txt', 'w') as f:
        f.write(f"{train_coor_range}")
    #taxi数据
    with open(f'datasets/dataset_template_copy/GPS/taxi/{row}_{col}_gps.pkl', 'wb') as f:
        pickle.dump(train_gps_data[i], f)
    #bus数据
    with open(f'datasets/dataset_template_copy/GPS/bus/{row}_{col}_gps.pkl', 'wb') as f:
        pickle.dump(train_gps_data_bus[i], f)
    #pixel coordinate version
    with open(f'datasets/dataset_template_copy/GPS/taxi_pixel/{row}_{col}_gps.pkl', 'wb') as f:
        pickle.dump(train_gps_data_pixel[i], f)
    with open(f'datasets/dataset_template_copy/GPS/bus_pixel/{row}_{col}_gps.pkl', 'wb') as f:
        pickle.dump(train_gps_data_bus_pixel[i], f)

iter = tqdm(range(len(test_sat_imgs)))
for i in iter:
    iter.set_description_str("saving test dataset")
    #because the train sat and test sat are two images, so the row and col number maybe repetition
    row = test_indexes[i][0] + 2000
    col = test_indexes[i][1] + 2000
    test_coor_range = {
        "west": test_coors[i][0][0],
        "east": test_coors[i][1][0],
        "north": test_coors[i][0][1],
        "south": test_coors[i][1][1]
    }
    cv2.imwrite(f"datasets/dataset_template_copy/test/image_test/{row}_{col}_sat.png", test_sat_imgs[i])
    cv2.imwrite(f"datasets/dataset_template_copy/test/road_network/{row}_{col}_road.png", test_mask_imgs[i])
    #坐标数据
    with open(f'datasets/dataset_template_copy/coordinates/{row}_{col}_gps.txt', 'w') as f:
        f.write(f"{test_coor_range}")
    #taxi数据
    with open(f'datasets/dataset_template_copy/GPS/taxi/{row}_{col}_gps.pkl', 'wb') as f:
        pickle.dump(test_gps_data[i], f)
    #bus数据
    with open(f'datasets/dataset_template_copy/GPS/bus/{row}_{col}_gps.pkl', 'wb') as f:
        pickle.dump(test_gps_data_bus[i], f)
    #pixel coordinate version
    with open(f'datasets/dataset_template_copy/GPS/taxi_pixel/{row}_{col}_gps.pkl', 'wb') as f:
        pickle.dump(test_gps_data_pixel[i], f)
    #bus数据
    with open(f'datasets/dataset_template_copy/GPS/bus_pixel/{row}_{col}_gps.pkl', 'wb') as f:
        pickle.dump(test_gps_data_bus_pixel[i], f)


print("Done!")
