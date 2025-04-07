from datasets import Dataset, DatasetDict, Features, Image as HFImage, Array2D, Sequence, Value
import os
import numpy as np
import pickle
from PIL import Image
from tqdm import tqdm
import json
import pandas as pd


# 数据集路径
DATASET_PATH = "../datasets/Shenzhen_SDF"

features = Features({
    "image": HFImage(),
    "mask": HFImage(),
    "sdf_mask": Array2D(dtype="float32", shape=(1024, 1024)),
    "road_network": HFImage(),
    "gps": Sequence({
        "id": Value("string"),
        "longitude": Value("float32"),
        "latitude": Value("float32"),
        "timestamp": Value("string"),  # ISO 格式时间戳
        "speed": Value("float32"),
        "direction": Value("float32"),
    }),
    "filename": Value("string"),
})

# 加载图像数据
def load_image(image_path):
    return Image.open(image_path).convert("RGB")

# 加载 mask 数据
def load_mask(mask_path):
    return Image.open(mask_path).convert("L")

# 加载 SDF mask
def load_sdf_mask(sdf_path):
    return np.load(sdf_path).astype(np.float32)  # 转换为 float32

# 加载路网数据
def load_road_network(road_path):
    return Image.open(road_path).convert("RGBA")

def load_gps(gps_path):
    with open(gps_path, "rb") as f:
        gps_data = pickle.load(f)  # 读取 DataFrame
    if isinstance(gps_data, pd.DataFrame):
        return gps_data.to_dict(orient="records")  # 转换为 [{"longitude": x, "latitude": y, "timestamp": t}, ...]
    return []

# 加载数据
def load_data(split):
    image_dir = os.path.join(DATASET_PATH, split, "image")
    mask_dir = os.path.join(DATASET_PATH, split, "mask")
    sdf_dir = os.path.join(DATASET_PATH, split, "mask_sdf_T")
    road_dir = os.path.join(DATASET_PATH, split, "road_network")
    gps_dir = os.path.join(DATASET_PATH, "GPS/taxi")

    data = []
    for mask_name in tqdm(os.listdir(mask_dir)):
        base_name = mask_name.replace("_mask.png", "")
        mask_path = os.path.join(mask_dir, mask_name)
        img_path = os.path.join(image_dir, f"{base_name}_sat.png")
        sdf_path = os.path.join(sdf_dir, f"{base_name}_mask.npy")
        road_path = os.path.join(road_dir, f"{base_name}_mask.png")
        gps_path = os.path.join(gps_dir, f"{base_name}_gps.pkl")

        if not os.path.exists(gps_path):
            continue  # 跳过没有 GPS 数据的样本

        data.append({
            "image": load_image(img_path),
            "mask": load_mask(mask_path),
            "sdf_mask": load_sdf_mask(sdf_path),
            "road_network": load_road_network(road_path),
            "gps": load_gps(gps_path),
            "filename": base_name,
        })
    print(f"Loaded {len(data)} samples from {split} split.")
    return Dataset.from_list(data, features=features)

# 构建数据集
dataset = DatasetDict({
    "train": load_data("train_val"),
    "test": load_data("test"),
})

# 保存数据集到本地
# dataset.save_to_disk("./Shenzhen_SDF_dataset")

# 推送到 Hugging Face
# from huggingface_hub import notebook_login
# notebook_login()  # 需要登录 Hugging Face 账号
dataset.push_to_hub("iwcyou/Shenzhen_SDF")
