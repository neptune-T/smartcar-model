import os

# 读取文件列表
def read_file_list(file_path):
    with open(file_path, 'r') as file:
        return file.read().splitlines()


def check_files_exist(file_list, directory_path):
    missing_files = []
    for file_name in file_list:
        if not os.path.exists(os.path.join(directory_path, file_name)):
            missing_files.append(file_name)
    return missing_files


file_list_path = 'E:/car/test/train.txt'  # 可以改为 val.txt 来检查验证集
directory_path = 'E:/car/test/test_annotations/'

file_list = read_file_list(file_list_path)
missing_files = check_files_exist(file_list, directory_path)

if missing_files:
    print("以下文件不存在于指定目录:", missing_files)
else:
    print("所有列出的文件都存在于目录中.")
