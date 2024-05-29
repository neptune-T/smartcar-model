import os
import shutil

# 指定文件和目录路径
train_txt_path = 'E:\\CAR\\test\\train.txt'
val_txt_path = 'E:\\CAR\\test\\val.txt'
train_xml_dir = 'E:\\CAR\\test\\annotations\\train'  # 目标训练集XML目录
val_xml_dir = 'E:\\CAR\\test\\annotations\\val'  # 目标验证集XML目录

# 确保目标目录存在
os.makedirs(train_xml_dir, exist_ok=True)
os.makedirs(val_xml_dir, exist_ok=True)

def move_xml_files(txt_path, xml_target_dir):
    # 读取文本文件中的每一行，每行包含图片和XML文件的完整路径
    with open(txt_path, 'r') as file:
        lines = file.read().splitlines()
    
    # 移动XML文件
    for line in lines:
        parts = line.split()
        if len(parts) < 2:
            continue  # 如果分割后少于两个部分，跳过这行
        xml_source_path = parts[1]  # 第二个元素是XML文件的路径
        file_name = os.path.basename(xml_source_path)  # 获取文件名
        xml_target_path = os.path.join(xml_target_dir, file_name)
        
        # 如果源文件存在，则移动它
        if os.path.exists(xml_source_path):
            shutil.move(xml_source_path, xml_target_path)
        else:
            print(f"Warning: '{xml_source_path}' does not exist and cannot be moved.")

# 移动训练集XML文件
move_xml_files(train_txt_path, train_xml_dir)
# 移动验证集XML文件
move_xml_files(val_txt_path, val_xml_dir)
