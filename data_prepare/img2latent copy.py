import torch
from diffusers.models import AutoencoderKL
from PIL import Image
from torchvision import transforms
import numpy as np
import os
from tqdm import tqdm

# 加载经过微调的 VAE 模型
vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse")

# 将模型移动到 GPU（如果可用）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vae.to(device)
vae.eval()  # 设置为评估模式

# 定义图像的预处理步骤
preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),  # 假设 VAE 期望输入归一化到 [-1, 1]
])

def process_image(image):
    """
    将图像拆分为 16 个 256x256 的小块，生成潜在空间表示，并将其拼接回原始大小。
    """
    patches = []
    latent_spaces = []

    # 将图像拆分为 16 个 256x256 的小块
    for i in range(4):
        for j in range(4):
            patch = image.crop((j * 256, i * 256, (j + 1) * 256, (i + 1) * 256))
            patches.append(patch)

    # 遍历每个小块，生成潜在空间表示
    for patch in patches:
        patch_tensor = preprocess(patch).unsqueeze(0).to(device)
        with torch.no_grad():  # 禁用梯度计算，因为这是推理过程
            latent_distribution = vae.encode(patch_tensor)[0]  # 获取潜在向量
            latent_representation = latent_distribution.sample().cpu().numpy()
            latent_spaces.append(latent_representation.squeeze(0))

    # 检查 latent_spaces 的形状
    latent_spaces = np.array(latent_spaces)  # 转换为 NumPy 数组
    # print("latent_spaces shape:", latent_spaces.shape)  # 打印形状进行调试

    # 将潜在空间表示拼接回原始大小
    # latent_spaces 应该有形状 (16, 4, 32, 32)
    latent_spaces = latent_spaces.reshape(4, 4, *latent_spaces.shape[1:])  # 形状为 (4, 4, 4, 32, 32)

    # 重新排列维度并调整为 (1, C, H, W)
    latent_grid = latent_spaces.transpose(0, 2, 1, 3, 4).reshape(1, 4, 128, 128)  # (1, C, H, W)

    return latent_grid

# 加载并预处理图像
if __name__ == '__main__':
    image_paths = [
        "/home/fk/python_code/datasets/dataset_sz_grid/train_val/image",
        "/home/fk/python_code/datasets/dataset_sz_grid/test/image_test",
        "/home/fk/python_code/datasets/dataset_sz_grid/GPS/taxi_filtered_time_quantity_speed_patch",
        "/home/fk/python_code/datasets/dataset_sz_grid/GPS/taxi_filtered_time_quantity_speed_gaussian_patch"
    ]

    for path in image_paths:
        save_path = os.path.join(path[:path.rfind('/')], f"{os.path.basename(path)}_latent")
        os.makedirs(save_path, exist_ok=True)
        image_list = os.listdir(path)

        for image_name in tqdm(image_list):
            # 打开 RGB 图像
            image = Image.open(os.path.join(path, image_name)).convert("RGB")
            assert image.size == (1024, 1024), f"Image {image_name} size is not (1024, 1024)."

            # 处理图像并获取潜在空间表示
            latent_grid = process_image(image)

            # 保存为 numpy 文件
            np.save(os.path.join(save_path, image_name.split(".")[0]), latent_grid)
