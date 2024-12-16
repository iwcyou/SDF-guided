import numpy as np
import cv2
import pickle
from PIL import Image

class CountRender:
    """gps数据数量信息渲染"""
    def __init__(self, patchedGPS, render_type) -> Image:
        self.patchedGPS = patchedGPS
        self.render_type = render_type


    def _direct_render(self, length=1024):
        """直接渲染GPS点"""
        gps = np.zeros((length, length, 1), np.uint8)
        ratio = length / 1024.
        patchedGPS = self.patchedGPS
        patchedGPS = patchedGPS[(0 <= patchedGPS['lat']) & (patchedGPS['lat'] < 1024) &
                                (0 <= patchedGPS['lon']) & (patchedGPS['lon'] < 1024)]
        y = np.array(patchedGPS['lon'] * ratio, np.int_)
        x = np.array(patchedGPS['lat'] * ratio, np.int_)

        gps[x, y] = 255

        gps = cv2.dilate(gps, np.ones((3, 3))) # 膨胀
        gps = gps[..., None] # 增加一个维度
        return gps
    

    def _linear_render(self, length=1024):
        """线性渲染出GPS点的数量信息"""
        ratio = length / 1024.
        patchedGPS = self.patchedGPS
        patchedGPS = patchedGPS[(0 <= patchedGPS['lat']) & (patchedGPS['lat'] < 1024) &
                                (0 <= patchedGPS['lon']) & (patchedGPS['lon'] < 1024)]
        y = np.array(patchedGPS['lon'] * ratio, np.int_)
        x = np.array(patchedGPS['lat'] * ratio, np.int_)

        #统计每个像素点上的GPS数量
        gps_counts = np.bincount(x * length + y, minlength=length*length)

        # 根据GPS数量设置像素的亮度
        max_count = np.max(gps_counts)
        gps = (gps_counts * 255 / max_count).reshape((length, length)).astype(np.uint8)

        gps = cv2.dilate(gps, np.ones((3, 3))) # 膨胀
        gps = gps[..., None] # 增加一个维度
        return gps


    def _log_render(self, length=1024):
        """取log渲染出GPS点的数量信息"""
        ratio = length / 1024.
        patchedGPS = self.patchedGPS
        patchedGPS = patchedGPS[(0 <= patchedGPS['lat']) & (patchedGPS['lat'] < 1024) &
                                (0 <= patchedGPS['lon']) & (patchedGPS['lon'] < 1024)]
        y = np.array(patchedGPS['lon'] * ratio, np.int_)
        x = np.array(patchedGPS['lat'] * ratio, np.int_)

        #统计每个像素点上的GPS数量
        gps_counts = np.bincount(x * length + y, minlength=length*length)

        # 根据GPS数量设置像素的亮度
        max_count = np.max(gps_counts)
        # print("单个像素点上最大GPS数量：", max_count)
        #加1是为了防止出现log(0)的情况
        gps = (np.log(gps_counts + 1) / np.log(max_count + 1) * 255).reshape((length, length)).astype(np.uint8)

        gps = cv2.dilate(gps, np.ones((3, 3))) # 膨胀
        gps = gps[..., None] # 增加一个维度
        return gps
    

    def _render_gps_to_image(self):
        """将GPS点渲染为图像"""
        if self.render_type == "direct":
            gps_image = self._direct_render()
        elif self.render_type == "linear":
            gps_image = self._linear_render()
        elif self.render_type == "log":
            gps_image = self._log_render()
        else:
            print("[ERROR] Unknown render type.")
            exit(1)
        return gps_image