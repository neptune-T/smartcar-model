import os

# 生成 train.txt、val.txt
xml_dir = 'E:\\CAR\\test\\annotations(voc)'  # XML 文件的实际位置
img_dir = 'E:\\CAR\\test\\images\\img'  # 图片文件的实际位置

# 检查目录存在
if not os.path.exists(xml_dir):
    print(f"XML directory {xml_dir} does not exist.")
    exit(1)
if not os.path.exists(img_dir):
    print(f"Image directory {img_dir} does not exist.")
    exit(1)

path_list = []
for img in os.listdir(img_dir):
    if img.endswith('.jpg'):  # 确保处理的是图像文件
        img_path = os.path.join(img_dir, img)
        xml_file = img.replace('.jpg', '.xml')  # 确保正确地替换扩展名
        xml_path = os.path.join(xml_dir, xml_file)
        if os.path.exists(xml_path):  # 确保对应的XML文件存在
            path_list.append((img_path, xml_path))
        else:
            print(f"Warning: No XML file for {img_path}")

train_f = open('E:\\CAR\\test\\train.txt', 'w') 
val_f = open('E:\\CAR\\test\\val.txt', 'w') 

# 分配图像到训练集或验证集
for i, content in enumerate(path_list):
    img, xml = content
    text = f'{img} {xml}\n'
    if i % 5 == 0:
        val_f.write(text)
    else:
        train_f.write(text)

train_f.close()
val_f.close()
