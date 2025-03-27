# SDF-Guided Multi-modal Big Data Road Extraction

The source code about special session of PAKDD 2025 paper *"SDF-Guided Multi-modal Big Data Road Extraction"*

## Usage

```bash
pip install -r requirement.txt # install the dependency library
```
```bash
source tr_sz_sdf.sh # run code
```
## Shenzhen dataset

All of our *Shenzhen* dataset is based on the web Mercator projection in the GCJ-02 coordinate system. 

### Dataset description

- **train_val/**
  - **image/**: contains 614 satellite images (`x_y_sat.png `)
  - **mask/**: contains 614 binary mask images (`x_y_mask.png `)
  - **mask_sdf_T/**: contains 614 SDF mask images (`x_y_mask.npy `)
  - **road_network/** :contains 614 road network images (`x_y_mask.png `)
- **test/**
  - **image/**: contains 152 satellite images (`x_y_sat.png `)
  - **mask/**: contains 152 mask images (`x_y_mask.png `)
  - **mask_sdf_T/**: contains 152 SDF mask images (`x_y_mask.npy `)
  - **road_network/** :contains 152 road network images (`x_y_mask.png `)
- **GPS/**
  - **taxi/**: contains 766 GPS patch files (`x_y_gps.pkl`). Each stores the GPS records located in the area of input image `x_y_sat.png`
- **coordinates/**: contains `x_y_gps.txt`  (web Mercator GCJ-02 format) files, (left up corner, right down corner) <- format

Each input image `image/x_y_sat.png ` is a RGB satellite image of 1024 $\times$ 1024 pixel size. Its corresponding GPS data is stored in file  `/GPS/patch/x_y_gps.pkl`, and corresponding mask image is   `mask/x_y_mask.png`.

### GPS Data

To save the loading time, we publish the dataset in Python's Pickle format, which can be directly loaded like:

```python
import pandas
import pickle
gps_data = pickle.load(open('dataset_sz_sdf/GPS/taxi/0_6_gps.pkl', 'rb'))
```

**Definition of columns**:

- `id`: Vehicle ID (integer)
- `time`: Timestamp (UNIX format, in second)
- `lat`: Latitude (in pixel coordinate)
- `lon`: Longitude (in pixel coordinate)
- `direction`: Heading (in degree, 0 means the vehicle is heading north or isn't moving)
- `speed`: Speed (in meter per minute)
- `time`: The time stamp.

The `lat`/`lon` are in the gcj02 System.

**Range of sampling**

Coordinate Range of satellite images in Nanshan district

> wgs84 format：\
> Top left corner：113.77477269727868, 22.658708423462986 \
> Lower right corner：114.01655951201688, 22.401131313831055

> web Mercator on GCJ-02 format：\
> TLC：12665921.334966816,2590450.8885846175\
> LRC：12692827.1689232    ,2559417.4551008344

Coordinate Range of road networks in Nanshan district

> wgs84 format：\
> TLC：113.72531536623958, 22.676333371889751640059225977739\
> LRC：114.07037282840729, 22.352754460489630359940774022261

> web Mercator on GCJ-02 format：\
> TLC：12660417.89499784  , 2592578.6326045664\
> LRC：12698827.572095804, 2553607.944832368

Coordinate range of train(satellite) ：

> wgs84 format:\
> TLC：113.77477269727868, 22.658708423462986\
> LRC：114.01655951201688, 22.52994959856712

> web Mercator on GCJ-02 format：\
> TLC：12665921.334966816, 2590450.8885846175\
> LRC：12692827.1689232    , 2574934.17184272595

Coordinate range of test(satellite) ：

> wgs84 format：\
> TLC：113.77477269727868, 22.52994959856712\
> LRC：114.01655951201688, 22.465558186041523

> web Mercator on GCJ-02 format：\
> TLC：12665921.334966816, 2574934.17184272595\
> LRC：12692827.1689232    , 2567175.813471780175

Coordinate range of Nanshan road network (overbold version)：

> web Mercator on GCJ-02 format：\
> TLC：12660417.89499784  , 2592493.9833760057\
> LRC：12698658.366469823, 2553607.944832368

## License

![img](https://licensebuttons.net/l/by-nc-sa/3.0/88x31.png)

This dataset is published under [**CC BY-NC-SA**](https://creativecommons.org/licenses/by-nc-sa/4.0/) (Attribution-NonCommercial-ShareAlike) License . Please note that it can be **ONLY** used for academic or scientific purpose.
