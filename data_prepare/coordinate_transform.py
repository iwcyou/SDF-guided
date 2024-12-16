"""
wgs84坐标转换为gcj02坐标系下的web墨卡托坐标
"""

from PIL import Image
import math
import os
Image.MAX_IMAGE_PIXELS = None # Disable DecompressionBombError，解决PIL的像素限制问题

def wgs_to_mercator(x, y):
    """经纬度转web墨卡托"""
    y = 85.0511287798 if y > 85.0511287798 else y
    y = -85.0511287798 if y < -85.0511287798 else y

    x2 = x * 20037508.34 / 180
    y2 = math.log(math.tan((90 + y) * math.pi / 360)) / (math.pi / 180)
    y2 = y2 * 20037508.34 / 180
    return x2, y2

def mercator_to_wgs(x, y):
    """web墨卡托转经纬度"""
    x2 = x / 20037508.34 * 180
    y2 = y / 20037508.34 * 180
    y2 = 180 / math.pi * \
        (2 * math.atan(math.exp(y2 * math.pi / 180)) - math.pi / 2)
    return x2, y2

"""---wgs84和gcj02的纠偏与互转---"""
def transformLat(x, y):
    ret = -100.0 + 2.0 * x + 3.0 * y + 0.2 * y * \
        y + 0.1 * x * y + 0.2 * math.sqrt(abs(x))
    ret += (20.0 * math.sin(6.0 * x * math.pi) + 20.0 *
            math.sin(2.0 * x * math.pi)) * 2.0 / 3.0
    ret += (20.0 * math.sin(y * math.pi) + 40.0 *
            math.sin(y / 3.0 * math.pi)) * 2.0 / 3.0
    ret += (160.0 * math.sin(y / 12.0 * math.pi) + 320 *
            math.sin(y * math.pi / 30.0)) * 2.0 / 3.0
    return ret

def transformLon(x, y):
    ret = 300.0 + x + 2.0 * y + 0.1 * x * x + \
        0.1 * x * y + 0.1 * math.sqrt(abs(x))
    ret += (20.0 * math.sin(6.0 * x * math.pi) + 20.0 *
            math.sin(2.0 * x * math.pi)) * 2.0 / 3.0
    ret += (20.0 * math.sin(x * math.pi) + 40.0 *
            math.sin(x / 3.0 * math.pi)) * 2.0 / 3.0
    ret += (150.0 * math.sin(x / 12.0 * math.pi) + 300.0 *
            math.sin(x / 30.0 * math.pi)) * 2.0 / 3.0
    return ret

def delta(lat, lon):
    '''
    Krasovsky 1940
    //
    // a = 6378245.0, 1/f = 298.3
    // b = a * (1 - f)
    // ee = (a^2 - b^2) / a^2;
    '''
    a = 6378245.0  # a: 卫星椭球坐标投影到平面地图坐标系的投影因子。
    ee = 0.00669342162296594323  # ee: 椭球的偏心率。
    dLat = transformLat(lon - 105.0, lat - 35.0)
    dLon = transformLon(lon - 105.0, lat - 35.0)
    radLat = lat / 180.0 * math.pi
    magic = math.sin(radLat)
    magic = 1 - ee * magic * magic
    sqrtMagic = math.sqrt(magic)
    dLat = (dLat * 180.0) / ((a * (1 - ee)) / (magic * sqrtMagic) * math.pi)
    dLon = (dLon * 180.0) / (a / sqrtMagic * math.cos(radLat) * math.pi)
    return {'lat': dLat, 'lon': dLon}

def outOfChina(lat, lon):
    if (lon < 72.004 or lon > 137.8347):
        return True
    if (lat < 0.8293 or lat > 55.8271):
        return True
    return False

def gcj_to_wgs(gcjLon, gcjLat):
    """gcj02转wgs84"""
    if outOfChina(gcjLat, gcjLon):
        return (gcjLon, gcjLat)
    d = delta(gcjLat, gcjLon)
    return (gcjLon - d["lon"], gcjLat - d["lat"])

def wgs_to_gcj(wgsLon, wgsLat):
    """wgs84转gcj02"""
    if outOfChina(wgsLat, wgsLon):
        return wgsLon, wgsLat
    d = delta(wgsLat, wgsLon)
    return wgsLon + d["lon"], wgsLat + d["lat"]

"""---wgs84和gcj02的纠偏与互转---"""

def wgs_to_gcj02_to_mercator(x, y):
    """wgs84转gcj02再转web墨卡托,x:lon, y:lat"""
    x, y = wgs_to_gcj(x, y)
    x, y = wgs_to_mercator(x, y)
    return x, y

def mercator_to_gcj02_to_wgs(x, y):
    """web墨卡托转gcj02再转wgs84"""
    x, y = mercator_to_wgs(x, y)
    x, y = gcj_to_wgs(x, y)
    return x, y



#print(wgs_to_gcj02_to_mercator(114.01656780474369, 22.5299198686470205))

#print(mercator_to_gcj02_to_wgs(12692827.1689232    , 2567175.813471780175))

#print(wgs_to_mercator(113.875749, 22.511775))
