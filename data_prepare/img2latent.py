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
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),  # 假设 VAE 期望输入归一化到 [-1, 1]
])

# 加载并预处理图像
if __name__ == '__main__':
    image_path = ["/home/fk/python_code/datasets/dataset_sz_grid/train_val/image",
                  "/home/fk/python_code/datasets/dataset_sz_grid/test/image_test",
                  "/home/fk/python_code/datasets/dataset_sz_grid/GPS/taxi_filtered_time_quantity_speed_patch",
                  "/home/fk/python_code/datasets/dataset_sz_grid/GPS/taxi_filtered_time_quantity_speed_gaussian_patch"]
    for path in image_path:
        save_path = os.path.join(path[:path.rfind('/')] , f"{path.split('/')[-1]}_latent")
        os.makedirs(save_path, exist_ok=True)
        image_list = os.listdir(path)
        for image_name in tqdm(image_list):
            image = Image.open(os.path.join(path, image_name)).convert("RGB")
            input_tensor = preprocess(image).unsqueeze(0).to(device)  # 添加批量维度并移动到 GPU

            with torch.no_grad():  # 禁用梯度计算，因为这是推理过程
                latent_distribution = vae.encode(input_tensor)[0]  # 获取潜在向量
                if isinstance(latent_distribution, torch.Tensor):
                    latent_representation = latent_distribution.cpu().numpy()
                else:
                    latent_representation = latent_distribution.mean.cpu().numpy()  # Assuming mean is a tensor
            # 保存为 numpy 文件
            np.save(os.path.join(save_path, image_name.split(".")[0]), latent_representation)

# loaded_latent_representation = np.load("latent_representation.npy")
