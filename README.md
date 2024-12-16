# SDF-Guided Multi-modal Big Data Road Extraction

The source code of eccv 2024 paper *"LTQS:Improving Road Extraction from Satellite Images with Rich GPS Feature Augmentations"*

## Usage

```bash
pip install -r
```
```bash
source tr_sz_sdf.sh
```
## rendering of GPS data

*"./data_prepare/image_sampler.py"* split the satellite, road network, GPS and coordinate to grid.

*"./data_prepare/time_render_sun.py"* render the Beijing gps data to PNG. And the other render ours.

## Beijing dataset

As we do not have the permission to publish the Beijing dataset, you will need to request the dataset from [here](https://github.com/suniique/Leveraging-Crowdsourced-GPS-Data-for-Road-Extraction-from-Aerial-Imagery).

## Shenzhen dataset

In our dataset, you can get the satellite images of every city by [Amap API ](https://github.com/myBestLove/googleMapDownloader), and get the road network(mask) by our crawler *"road_tile_downloader.html"* .

And our GPS is available upon request.

### Dataset discription

- **train_val/**
  - **image/**: contains 624 satellite images (`x_y_sat.png `)
  - **mask/**: contains 624 mask images (`x_y_mask.png `)
  - **road_network/** :contains 624 road network images (`x_y_mask.png `)
- **test/**
  - **image/**: contains 172 satellite images (`x_y_sat.png `)
  - **mask/**: contains 172 mask images (`x_y_mask.png `)
  - **road_network/** :contains 172 road network images (`x_y_mask.png `)
- **GPS/**
  - **taxi/**: contains 796 GPS patch files (`x_y_gps.pkl`). Each stores the GPS records located in the area of input image `x_y_sat.png`
  - **taxi_gaussian_ltqs_patch/**:contains 796 rendered GPS data (`x_y_gps.png`) with ltqs and gaussian rendering method.
- **coordinate/**: contains `x_y_coor.txt`  (web mercator GCJ02 format) files, (left up corner, right down corner) <- format

Each input image `image/x_y_sat.png ` is a RGB satellite image of 1024 $\times$ 1024 pixel size. Its corresponding GPS data is stored in file  `/GPS/patch/x_y_gps.pkl`, and corresponding mask image is   `mask/x_y_mask.png`.

Unfortunately, we haven't got the permission to publish the satellite images due to the license of the data provider. However, we provide all the GPS coordinates of each satellite image (GCJ format) in `/coordinate/`. You might apply for the access and download these images from Amap (高德地图) or DigitalGlobe referencing the coordinates.

### GPS Data

To save the loading time, we publish the dataset in Python's Pickle format, which can be directly loaded like:

```python
import pandas
import pickle
gps_data = pickle.load(open('sz_gps_dir_speed_interval_sorted.pkl', 'rb'))
```

**Definition of columns**:

- `id`: Vehical ID (integer)
- `time`: Timestamp (UNIX format, in second)
- `lat`: Latitude (in pixel coordinate)
- `lon`: Lontitude (in pixel coordiante)
- `direction`: Heading (in degree, 0 means the vehical is heading north or isn't moving)
- `speed`: Speed (in meter per minute)
- `time`: The time stamp.

The `lat`/`lon` are in the gcj02 System.

**Range of sampling**

Range of satellite images's coordinates in Nanshan district

> wgs84 format：\
> Top left corner：113.77477269727868, 22.658708423462986 \
> Lower right corner：114.01655951201688, 22.401131313831055

> web mercator on gcj02 format：\
> TLC：12665921.334966816,2590450.8885846175\
> LRC：12692827.1689232    ,2559417.4551008344

Range of road network images's coordinates in Nanshan district

> wgs84 format：\
> TLC：113.72531536623958, 22.676333371889751640059225977739\
> LRC：114.07037282840729, 22.352754460489630359940774022261

> web mercator on gcj02 format：\
> TLC：12660417.89499784  , 2592578.6326045664\
> LRC：12698827.572095804, 2553607.944832368

train(satellite)coordinate range：4.956456081279460437192118226601e-6

> wgs84 format：\
> TLC：113.77477269727868, 22.658708423462986\
> LRC：114.01655951201688, 22.52994959856712

> web mercator on gcj02 format：\
> TLC：12665921.334966816, 2590450.8885846175\
> LRC：12692827.1689232    , 2574934.17184272595

test(satellite)coordiante range：

> wgs84 format：\
> TLC：113.77477269727868, 22.52994959856712\
> LRC：114.01655951201688, 22.465558186041523

> web mercator on gcj02 format：\
> TLC：12665921.334966816, 2574934.17184272595\
> LRC：12692827.1689232    , 2567175.813471780175

Coordinat of Nanshan road network (overbold version)：

> web mercator on gcj02 format：\
> TLC：12660417.89499784  , 2592493.9833760057\
> LRC：12698658.366469823, 2553607.944832368

All of our *Shenzhen* dataset is based on the web mercator projection in the gcj02 coordinate system. *Beijing* dataset is based on WGS-84 coordinate system.

**Minor**

The difference in `datasets/gps_data/taxi/GPS_taxi.pkl` and `datasets/gps_data/taxi/GPS_taxi_2.0.pkl` mainly is the first one`s coordinate is WGS84 and the last one is Web Mercator based on GCJ-02 coordinate.


### License

![img](https://licensebuttons.net/l/by-nc-sa/3.0/88x31.png)

This dataset is published under [**CC BY-NC-SA**](https://creativecommons.org/licenses/by-nc-sa/4.0/) (Attribution-NonCommercial-ShareAlike) License . Please note that it can be **ONLY** used for academic or scientific purpose.
