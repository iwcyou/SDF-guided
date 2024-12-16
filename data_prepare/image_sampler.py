from PIL import Image, ImageFile
Image.MAX_IMAGE_PIXELS = None # Disable DecompressionBombError
ImageFile.LOAD_TRUNCATED_IMAGES = True # Disable OSError: image file is truncated
from multiprocessing import Lock
import cv2
import numpy as np
import random


IMAGE_SIZE = (1024, 1024) # Set the size of the small images
# IMAGE_SIZE = (4096, 4096) # Set the size of the small images
TRAIN_SAT_COORDINATE_RANGE = [(12665921.334966816, 2590450.8885846175), (12692827.1689232, 2574934.17184272595)]
TEST_SAT_COORDINATE_RANGE = [(12665921.334966816, 2574934.17184272595), (12692827.1689232, 2567175.813471780175)]
ROAD_COORDINATE_RANGE = [(12660417.89499784, 2592493.9833760057), (12698658.366469823, 2553607.944832368)]


class ImageSampler:
    """Satellite images and road images sampler"""

    def __init__(self, sat_image, road_image, image_type):

        self.sat_large_image = sat_image # 大卫星图片
        self.road_large_image = road_image #路网图
        self.image_type = image_type
        # self.lock = Lock()


    def sample_images_from_large_image(self, image_size):
        """从大图中采样小图片并返回像素范围,将大图切分为网格"""

        large_image_height, large_image_width,  _ = self.sat_large_image.shape

        # Calculate the number of rows and columns in the grid
        num_rows = large_image_height // image_size[0]
        num_cols = large_image_width // image_size[1]

        sum = num_rows * num_cols

        images = []
        pixel_ranges = []
        indexes = []

        # Sample n patches
        for i in range(sum):
            # Calculate the row and column indices for the patch
            row_index = i // num_cols
            col_index = i % num_cols

            # Calculate the top-left corner coordinates of the patch
            top_left_x = col_index * image_size[1]
            top_left_y = row_index * image_size[0]

            # Extract the patch
            cropped_image = self.sat_large_image[top_left_y:top_left_y + image_size[1], top_left_x:top_left_x + image_size[0]]

            images.append(cropped_image)
            pixel_ranges.append(((top_left_x, top_left_y), (top_left_x + image_size[0], top_left_y + image_size[1])))
            indexes.append((row_index, col_index))

        return images, pixel_ranges, indexes


    def sample_images(self, large_image, pixel_ranges):
        """sample images from large image based on pixel ranges"""
        # mutex = mp.Lock() #创建锁
        # try:
        #     mutex.acquire()

        images = []
        for pixel_range in pixel_ranges:

            # Extract the region of interest from the large image based on the top-left position and right-bottom position
            # with self.lock:
            cropped_image = large_image[pixel_range[0][1]:pixel_range[1][1],  pixel_range[0][0]:pixel_range[1][0]]

            images.append(cropped_image)

        return images
        # except Exception as e:
        #     print(e)
        # finally:
        #     mutex.release()


    def get_resolution_ratio(self, image_shape, coordinate_range):
        """获取分辨率比例"""
        pixel_x_ratio = (coordinate_range[1][0] - coordinate_range[0][0]) / image_shape[1]
        pixel_y_ratio = (coordinate_range[1][1] - coordinate_range[0][1]) / image_shape[0]
        return pixel_x_ratio, pixel_y_ratio


    def pixel_to_coordinate(self, x, y , pixel_x_ratio, pixel_y_ratio , coordinate_range):
        """像素坐标转换为地理坐标"""
        lon = coordinate_range[0][0] + x * pixel_x_ratio
        lat = coordinate_range[0][1] + y * pixel_y_ratio
        return  lon, lat


    def coordinate_to_pixel(self, lon, lat, pixel_x_ratio, pixel_y_ratio, coordinate_range):
        """地理坐标转换为像素坐标"""
        x = (lon - coordinate_range[0][0]) / pixel_x_ratio
        y = (lat - coordinate_range[0][1]) / pixel_y_ratio
        return int(x+0.5), int(y+0.5) # 四舍五入,确保不会出现半个像素的情况


    def reszie_images(self, images, image_size):
        """对图像进行批量resize"""
        new_images = []
        for image in images:
            new_images.append(cv2.resize(image, image_size))
        return new_images


    def images_sample(self):
        """
        sample satellite images and road images and return coordinate range list of lon and lat,
        indexes is (row number, column number) of the small images in the large image.
        """

        image_size = IMAGE_SIZE # Set the size of the small images

        #采样卫星图片和像素范围
        sat_images, sat_pixel_ranges, indexes = self.sample_images_from_large_image(image_size)

        #卫星图坐标范围
        if self.image_type == "train":
            sat_coordinate_range = TRAIN_SAT_COORDINATE_RANGE
        elif self.image_type == "test":
            sat_coordinate_range = TEST_SAT_COORDINATE_RANGE

        #获取卫星图分辨率比例
        sat_pixel_x_ratio, sat_pixel_y_ratio = self.get_resolution_ratio(self.sat_large_image.shape, sat_coordinate_range)

        coordinate_ranges = []
        for sat_pixel_range in sat_pixel_ranges:
            #卫星图像素坐标转换为地理坐标
            l_x, l_y = sat_pixel_range[0][0], sat_pixel_range[0][1]
            r_x, r_y = sat_pixel_range[1][0], sat_pixel_range[1][1]
            #获取图像左上角和右下角经纬度
            l_lon, l_lat = self.pixel_to_coordinate(l_x, l_y, sat_pixel_x_ratio, sat_pixel_y_ratio, sat_coordinate_range)
            r_lon, r_lat = self.pixel_to_coordinate(r_x, r_y, sat_pixel_x_ratio, sat_pixel_y_ratio, sat_coordinate_range)
            coordinate_ranges.append(((l_lon, l_lat), (r_lon, r_lat)))

        #路网图坐标范围
        #road_coordinate_range = [(12660417.89499784  , 2592578.6326045664), (12698827.572095804, 2553607.944832368)]
        road_coordinate_range = ROAD_COORDINATE_RANGE
        #获取路网图分辨率比例
        road_pixel_x_ratio, road_pixel_y_ratio = self.get_resolution_ratio(self.road_large_image.shape, road_coordinate_range)

        road_pixel_ranges = []
        for coordinate_range in coordinate_ranges:
            #地理坐标转换为路网图像素坐标
            l_lon, l_lat = coordinate_range[0][0], coordinate_range[0][1]
            r_lon, r_lat = coordinate_range[1][0], coordinate_range[1][1]
            #获取路网左上角和右下角像素坐标
            l_x, l_y = self.coordinate_to_pixel(l_lon, l_lat, road_pixel_x_ratio, road_pixel_y_ratio, road_coordinate_range)
            r_x, r_y = self.coordinate_to_pixel(r_lon, r_lat, road_pixel_x_ratio, road_pixel_y_ratio, road_coordinate_range)
            road_pixel_ranges.append(((l_x, l_y), (r_x, r_y)))

        road_images = self.sample_images(self.road_large_image, road_pixel_ranges) #采样路网图片

        #对路网图片进行resize
        road_images = self.reszie_images(road_images, image_size)

        return sat_images, road_images, coordinate_ranges, indexes

if __name__ == "__main__":

    sat = Image.open("dataset/train.png")
    # sat = Image.open("test/custom.png")

    sat_cv =cv2.cvtColor(np.array(sat), cv2.COLOR_RGBA2BGRA)
    road = Image.open("dataset/custom_ns_road_T.png")
    # road = Image.open("test/custom.png")

    road_cv = cv2.cvtColor(np.array(road), cv2.COLOR_RGBA2BGRA)
    train_sampler = ImageSampler(sat_cv, road_cv, "train")
    sats, roads, coors = train_sampler.images_sample()
    cv2.imwrite("test/sat.png", sats[0])
    cv2.imwrite("test/road.png", roads[0])
    print("done")
