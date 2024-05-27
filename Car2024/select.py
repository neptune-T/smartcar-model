import os
import random
import shutil

# 设置图片的源文件夹和目标文件夹路径
images_path = 'E:\\car\\Car2024\\images'
output_folder = 'E:\\car\\Car2024\\test_img'
keywords = ['bomb', 'bridge', 'safety', 'cone', 'crosswalk', 'danger', 'evil', 'block', 'patient', 'prop', 'spy', 'thief', 'tumble']

# 创建一个输出文件夹，如果不存在的话
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 创建一个字典来存储每个关键词对应的图片列表
keyword_images = {key: [] for key in keywords}

# 遍历图片文件夹中的所有图片文件
for image_file in os.listdir(images_path):
    for keyword in keywords:
        if keyword in image_file.lower():  # 确保关键词匹配时不区分大小写
            keyword_images[keyword].append(image_file)

# 从每个关键词对应的列表中随机选择10张图片
selected_images = {key: random.sample(images, 10) if len(images) >= 10 else images for key, images in keyword_images.items()}

# 复制选中的图片到新的文件夹
for key, images in selected_images.items():
    for image in images:
        source_path = os.path.join(images_path, image)
        destination_path = os.path.join(output_folder, image)
        shutil.copy(source_path, destination_path)
    print(f"Keyword: {key}, Selected Images: {len(images)} have been copied to {output_folder}")
