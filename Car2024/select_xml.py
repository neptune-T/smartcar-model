import os
import shutil

# 设置源文件夹和目标文件夹的路径
images_path = 'E:\\car\\test\\test_img'
original_annotations_path = 'E:\\car\\Car2024\\annotations'
new_annotations_path = 'E:\\car\\test\\test_annotations'

# 如果新的annotations文件夹不存在，创建它
if not os.path.exists(new_annotations_path):
    os.makedirs(new_annotations_path)

# 遍历test_img文件夹中的所有图片
for image_filename in os.listdir(images_path):
    # 构建原始XML文件的完整路径
    xml_filename = image_filename.replace('.jpg', '.xml')  # 假设图片和XML文件的前缀相同，只是扩展名不同
    original_xml_path = os.path.join(original_annotations_path, xml_filename)
    
    # 检查这个XML文件是否存在
    if os.path.exists(original_xml_path):
        # 构建新的XML文件路径
        new_xml_path = os.path.join(new_annotations_path, xml_filename)
        
        # 复制文件
        shutil.copy(original_xml_path, new_xml_path)
        print(f"Copied {xml_filename} to {new_annotations_path}")
    else:
        print(f"XML file {xml_filename} not found for image {image_filename}")

print("Operation completed.")
