"""
图片拼接
"""
import numpy as np
import os
import cv2
from tqdm import tqdm

def merge_iamge(path, save_path, num_columns):
    imagepath_list = []
    image_list = os.listdir(path)
    for i in image_list:
        imagepath_list.append(os.path.join(path, i))

    #给图片列表排序；按照纬度从大到小，经度从小到大的规则
    imagepath_list = sorted(imagepath_list, key=lambda x: (-float(x.split("-")[0].split("/")[-1]), float(x.split("-")[1].split(".")[0]), float(x.split("-")[1].split(".")[1])))

    num_images = len(imagepath_list)
    print(imagepath_list,'\n',num_images)

    # Calculate the number of rows required
    num_rows = int(np.ceil(num_images / num_columns))

    # Create an empty canvas to compose the images
    example_image = cv2.imread(imagepath_list[0], cv2.IMREAD_UNCHANGED) #拿第一个图片的大小创建画布,IMREAD_UNCHANGED保留alpha通道

    canvas_height = example_image.shape[0] * num_rows #高
    canvas_width = example_image.shape[1] * num_columns #宽
    canvas = np.zeros((canvas_height, canvas_width, example_image.shape[2]), dtype=example_image.dtype)
    print(canvas.shape)

    # Compose the images on the canvas
    for i, imagepath in tqdm(enumerate(imagepath_list)):
        row = i // num_columns
        col = i % num_columns
        image = cv2.imread(imagepath,cv2.IMREAD_UNCHANGED) #读取图片

        start_row = row * example_image.shape[0]
        start_col = col * example_image.shape[1]
        canvas[start_row:start_row+image.shape[0], start_col:start_col+image.shape[1]] = image

    # Save the composed image
    cv2.imwrite(save_path, canvas)
    print("Have a good day!")


if __name__ == "__main__":
    path = 'Datasets/mask_all/road_class4' #图片文件夹
    save_path = 'Datasets/mask_all/road_class4.png'
    num_columns = 39 #每行图片数量
    merge_iamge(path, save_path, num_columns)