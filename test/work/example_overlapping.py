# 检查 train.txt 和 val.txt 是否有重叠

def read_file_list(file_path):
    with open(file_path, 'r') as file:
        return set(file.read().splitlines())

# 替换文件路径
train_file_path = 'E:/car/test/train.txt'
val_file_path = 'E:/car/test/val.txt'

train_files = read_file_list(train_file_path)
val_files = read_file_list(val_file_path)

# 查找重叠的文件
overlap_files = train_files.intersection(val_files)
if overlap_files:
    print("有重叠的文件:", overlap_files)
else:
    print("没有重叠的文件.")
