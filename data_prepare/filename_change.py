#将“axb.png”更改成“bxa.png”

import os
import re
from tqdm import tqdm

# Directory containing the files to be renamed
directory = r"D:\remote\cache1\web_tile"

# Loop through all the files in the directory
for filename in tqdm(os.listdir(directory)):
    if filename.endswith(".png"): # Check if the file is a text file
        # Split the file name into words using spaces as separators
        words = re.split("[- .]",filename)
        if len(words) > 1: # Check if there are at least two words in the file name
            # Swap the first two words in the list of words
            words[0], words[1],words[2], words[3] = words[2], words[3],words[0], words[1]
            # Join the words back into a file name with spaces as separators
            new_filename = words[0]+"."+words[1]+"-"+words[2]+"."+words[3]+"."+words[4]
            # Rename the file
            os.rename(os.path.join(directory, filename), os.path.join(directory, new_filename))


def swap_words_in_filename(filename):
    """交换文件名中的前两个词"""
    # 获取文件名和扩展名
    name, ext = os.path.splitext(filename)
    # 将文件名拆分为单词列表
    words = name.split()
    # 如果单词数小于 2，则返回原始文件名
    if len(words) < 2:
        return filename
    # 交换前两个单词
    words[0], words[1] = words[1], words[0]
    # 将单词列表重新组合为文件名，并添加扩展名
    new_name = ' '.join(words) + ext
    return new_name
