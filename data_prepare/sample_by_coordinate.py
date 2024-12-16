from PIL import Image, ImageFile
import cv2
import numpy as np
import pickle
import sys
sys.path.append("../")
from image_sampler import ImageSampler
Image.MAX_IMAGE_PIXELS = None # Disable DecompressionBombError
ImageFile.LOAD_TRUNCATED_IMAGES = True # Disable OSError: image file is truncated


class SampleByCoordinate:
    """通过坐标范围返回卫星图、路网图或者GPS数据"""

    def __init__(self):
        pass

    def sample_images(self, large_image, pixel_ranges):
        """根据像素坐标从图像中采样"""
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



    def sat_by_coor(self, sat_large_image, image_type, coordinate_range):
        """通过坐标范围返回卫星图"""

        image_size = (1024, 1024) # Set the size of the small images

        #卫星图坐标范围
        if image_type == "train":
            sat_coordinate_range = [(12665921.334966816, 2590450.8885846175), (12692827.1689232, 2574934.17184272595)]
        elif image_type == "test":
            sat_coordinate_range = [(12665921.334966816, 2574934.17184272595), (12692827.1689232, 2567175.813471780175)]

        #获取卫星图分辨率比例
        sat_pixel_x_ratio, sat_pixel_y_ratio = self.get_resolution_ratio(sat_large_image.shape, sat_coordinate_range)

        road_pixel_ranges = []

        #地理坐标转换为路网图像素坐标
        l_lon, l_lat = coordinate_range[0][0], coordinate_range[0][1]
        r_lon, r_lat = coordinate_range[1][0], coordinate_range[1][1]
        #获取路网左上角和右下角像素坐标
        l_x, l_y = self.coordinate_to_pixel(l_lon, l_lat, sat_pixel_x_ratio, sat_pixel_y_ratio, sat_coordinate_range)
        r_x, r_y = self.coordinate_to_pixel(r_lon, r_lat, sat_pixel_x_ratio, sat_pixel_y_ratio, sat_coordinate_range)
        road_pixel_ranges.append(((l_x, l_y), (r_x, r_y)))

        road_images = self.sample_images(sat_large_image, road_pixel_ranges) #采样路网图片

        #对路网图片进行resize
        road_images = self.reszie_images(road_images, image_size)

        return road_images[0]


    def mask_by_coor(self, road_large_image, coordinate_range):
        """通过坐标范围返回路网图"""

        image_size = (1024, 1024) # Set the size of the small images

        #路网图坐标范围
        #road_coordinate_range = [(12660417.89499784  , 2592578.6326045664), (12698827.572095804, 2553607.944832368)]
        road_coordinate_range = [(12660417.89499784  , 2592493.9833760057), (12698658.366469823, 2553607.944832368)]
        #获取路网图分辨率比例
        road_pixel_x_ratio, road_pixel_y_ratio = self.get_resolution_ratio(road_large_image.shape, road_coordinate_range)

        road_pixel_ranges = []

        #地理坐标转换为路网图像素坐标
        l_lon, l_lat = coordinate_range[0][0], coordinate_range[0][1]
        r_lon, r_lat = coordinate_range[1][0], coordinate_range[1][1]
        #获取路网左上角和右下角像素坐标
        l_x, l_y = self.coordinate_to_pixel(l_lon, l_lat, road_pixel_x_ratio, road_pixel_y_ratio, road_coordinate_range)
        r_x, r_y = self.coordinate_to_pixel(r_lon, r_lat, road_pixel_x_ratio, road_pixel_y_ratio, road_coordinate_range)
        road_pixel_ranges.append(((l_x, l_y), (r_x, r_y)))

        road_images = self.sample_images(road_large_image, road_pixel_ranges) #采样路网图片

        #对路网图片进行resize
        road_images = self.reszie_images(road_images, image_size)

        return road_images[0]


    def gps_by_coor(self,  big_gps, coordinate_range, coordinate_type):
        """通过坐标范围返回gps数据，对GPS数据的经纬度坐标进行了筛选和转换，使其能够对应到图片上的像素坐标"""

        gps_data = []
        #坐标筛选
        selected_rows = big_gps[(big_gps['lon'].between(coordinate_range[0][0], coordinate_range[1][0])) & (big_gps['lat'].between(coordinate_range[1][1], coordinate_range[0][1]))]
        if coordinate_type == "geographical":
            pass
        elif coordinate_type == "pixel":
            #转换为图片上的像素坐标,"+0.5"做到四舍五入
            selected_rows['lon'] = ((selected_rows['lon'] - coordinate_range[0][0]) / (coordinate_range[1][0] - coordinate_range[0][0]) * 1024 + 0.5).astype(int)
            selected_rows['lat'] = (1023 - (selected_rows['lat'] - coordinate_range[1][1]) / (coordinate_range[0][1] - coordinate_range[1][1]) * 1024 - 0.5).astype(int)
        gps_data.append(selected_rows)

        return gps_data[0]


if __name__ == "__main__":

    print("done")
