'''
单张png图片切割成100张小图片
'''
from PIL import Image
import os
from tqdm import tqdm

Image.MAX_IMAGE_PIXELS = None # Disable DecompressionBombError，解决PIL的像素限制问题


save_dir = 'save_image'

os.makedirs(save_dir, exist_ok=True)

image = Image.open('Datasets/overbold_ns_road.png')

width, height = image.size
sample_size = 0.1  # 10% of the original image

sample_width = int(width * sample_size)
sample_height = int(height * sample_size)

n = int(1//sample_size)
for i in tqdm(range(n)):
    for j in range(n):
        start_x = i * int(width * sample_size)
        start_y = j * int(height * sample_size)
        end_x = min(start_x + sample_width, width)
        end_y = min(start_y + sample_height, height)

        sample = image.crop((start_x, start_y, end_x, end_y))
        sample.save(os.path.join(save_dir,'{}_{}.png'.format(i,j)))
