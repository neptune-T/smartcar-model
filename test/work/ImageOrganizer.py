import os
import shutil

def move_images(txt_file, source_folder_img, target_folder_img, source_folder_xml):
    with open(txt_file, 'r') as file:
        lines = file.readlines()
        for line in lines:
            img_name, xml_name = line.strip().split()  
            img_path = os.path.join(source_folder_img, img_name)  
            xml_path = os.path.join(source_folder_xml, xml_name)  

            if os.path.exists(img_path) and os.path.exists(xml_path):
                target_img_path = os.path.join(target_folder_img, os.path.basename(img_path))
                shutil.move(img_path, target_img_path)
                print(f"Moved {img_path} to {target_img_path}")
            else:
                if not os.path.exists(img_path):
                    print(f"Image file {img_path} not found")
                if not os.path.exists(xml_path):
                    print(f"XML file {xml_path} not found")

# 指定文件和目录路径
train_txt_path = 'E:\\CAR\\test\\train.txt'
val_txt_path = 'E:\\CAR\\test\\val.txt'
img_dir = 'E:\\CAR\\test\\images\\img'  # 图片源目录
train_img_dir = 'E:\\CAR\\test\\images\\train_img'  # 训练集目标目录
val_img_dir = 'E:\\CAR\\test\\images\\val_img'  # 验证集目标目录
xml_dir = 'E:\\CAR\\test\\annotations(voc)'  # XML文件目录

# 确保目标目录存在
os.makedirs(train_img_dir, exist_ok=True)
os.makedirs(val_img_dir, exist_ok=True)

# 移动训练集和验证集的图片
move_images(train_txt_path, img_dir, train_img_dir, xml_dir)
move_images(val_txt_path, img_dir, val_img_dir, xml_dir)
