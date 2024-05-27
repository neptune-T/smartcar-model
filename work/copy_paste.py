# -*- coding: utf-8 -*- 
# @File             : copy_paste.py 
# @Author           : hanjiangli
# @Contact          : 809691105@qq.com
# @Time Create First: 2022/4/16 10:25 
# @Contributor      : hanjiangli
# @Time Modify Last : 2022/5/2 19:46
'''
@File Description:
# 运行之前需要安装必要的库：
!pip install pycocotools 
!pip install tqdm --upgrade
# 对特定类别的目标进行copy-paste数据增强
!python /home/aistudio/work/copy_paste.py \
    --input_dir=/home/aistudio/data/detection/JPEGImages \
    --json_path=/home/aistudio/data/detection/train_final.json \
    --output_json=/home/aistudio/data/detection/train.json \
    --muti_obj \
    --category_id=5 \
    --copypaste_ratio=0.45
'''

import os
import tqdm
import json
import random
import argparse
import numpy as np
from PIL import Image
from pycocotools.coco import COCO 

def get_bboxes(anns):
    '''
    从annotations获得所有的bbox
    :param anns: list，包含所有的标注信息，格式为[{ },{ },……]
    :return: list, 包含所有的bbox，格式为[[ ],[ ],……]
    '''
    bbox = []
    for i in range(len(anns)):
        bbox.append(anns[i]['bbox'])
    return bbox
 
def is_coincide(bbox_1, bbox_2):
    '''
    判断2个矩形框是否有重叠部分
    :param bbox_1: [x1, y1, w1, h1]
    :param bbox_2: [x2, y2, w2, h2]
    :return:  bool，True表示有重叠
    '''
    if (bbox_1[0]>=bbox_2[0]+bbox_2[2]) or (bbox_1[0]+bbox_1[2]<=bbox_2[0]) or (bbox_1[1]+bbox_1[3]<=bbox_2[1]) or (bbox_1[1]>=bbox_2[1]+bbox_2[3]):
        return False
    else:
        return True

def img_add(img_src_path, img_main_path, bbox_src):
    '''
    将src中的部分区域加到main图像中，结果图还是main图像的大小
    :param img_src_path：src图片路径
    :param img_main_path：main图片路径
    :param bbox_src：需要裁剪区域的bbox，格式为[[ ],[ ],……]
    '''
    img_src = Image.open(img_src_path)
    for bboxi in bbox_src:
        img_main = Image.open(img_main_path)
        coordinate = (bboxi[0],bboxi[1],bboxi[0]+bboxi[2],bboxi[1]+bboxi[3]) # 转成Image.crop函数需要的参数格式
        clipped_area = img_src.crop(coordinate) # 裁剪后的区域为clipped_area(这里img_src保持不变)
        img_main.paste(clipped_area, (int(bboxi[0]), int(bboxi[1]))) # 将裁剪的区域粘贴到img_main中(按原坐标位置粘贴)
        # 裁剪后保存并覆盖原来的main图像
        img_main.save(img_main_path)

def copy_paste(img_main_path, img_src_path, main_anns, src_anns, coincide=False, muti_obj=False):
    """
    整个复制粘贴操作
    :param coincide表示是否允许融合区域有重叠
    :param muti_obj表示是否复制粘贴src中的所有目标物
    1. 传入随机选择的一张src图像和一张main图像的img路径和anns信息；(src_anns只包含特定类别的标注信息)
    2. 从anns中获得bbox信息；
    3. 将从src裁剪的区域加到main中,将修改后的main保存并覆盖原图像
    4. 返回是否粘贴成功的标志位flag和裁剪区域的bbox
    """
    # 获得图像中的所有bbox
    bbox_main= get_bboxes(main_anns)
    bbox_src= get_bboxes(src_anns)
    
    # 只复制粘贴一个目标
    if not muti_obj or len(bbox_src)==1: 
        id = random.randint(0, len(bbox_src)-1)
        bbox_src = bbox_src[id]
        bbox_src = [bbox_src] # 再加一对中括号，为了与粘贴多个目标时维数匹配
    # 判断融合后的区域是否有重叠
    if not coincide:
        for point_main in bbox_main:
            for point_src in bbox_src:
                if is_coincide(point_main, point_src):
                    return False, None

    # 将从src裁剪的区域加到main中
    img_add(img_src_path, img_main_path, bbox_src)
    # 粘贴成功标志位
    flag = True
    return flag, bbox_src
 
def main(args):
    print('Copy paste'.center(100,'-'))
    print()
    # 主字典
    data = {'images': [], 'annotations': [], "categories": []}
    # 需要进行copy-patse的特定类别，(本次比赛共7类：1~7)
    # 比赛链接https://aistudio.baidu.com/aistudio/competition/detail/132/0/introduction
    category_id = args.category_id
    # 图像的输入路径
    JPEGs = args.input_dir
    # 构建coco对象
    coco = COCO(args.json_path)
    # 获取所有的图像ID,用于数据增强
    list_imgIds = coco.getImgIds() 
    # 获取所有的类别ID(list)(本次比赛共7类：1~7)
    list_catIds = coco.getCatIds()
    # 获取所有的标注ID
    list_annIds = coco.getAnnIds()
    # 获取所有标注信息
    list_anns = coco.loadAnns(list_annIds)
    # 存放含有特定类别的图片ID
    imgIds_specificCat = []
    # 存放copypaste后的图片ID
    imgIds_copypaste = []
    for ann in list_anns:
        if ann['category_id'] != category_id:
            continue
        imgIds_specificCat.append(ann['image_id'])
    # 去掉重复的图片ID
    imgIds_specificCat = list(set(imgIds_specificCat))
    with tqdm.tqdm(list_imgIds, colour='red') as tbar: # 设置进度条
        for imgId in tbar:
            # main图像
            # 从list_imgIds中依次取得一个图像元素
            main_img = coco.loadImgs(imgId)[0] 
            # 获取该图片的路径
            img_main_path = os.path.join(JPEGs,main_img['file_name'])
            # 获取该图片中含有的标注信息的ID
            main_annIds = coco.getAnnIds(imgIds=main_img['id'], catIds=list_catIds, iscrowd=None)
            # 获取该图像中含有的标注信息
            main_anns = coco.loadAnns(main_annIds)
            # 将main图像中原有的标注信息写入主字典中
            for i in range(len(main_anns)):
                tmp1 = {'image_id': imgId,
                        'id': int(len(data['annotations']) + 1), # 重新编号
                        'category_id': main_anns[i]['category_id'],
                        'bbox': main_anns[i]['bbox'],
                        'area': main_anns[i]['area'],
                        'iscrowd': 0,
                        'segmentation': []
                        }
                data['annotations'].append(tmp1)
            # 按照一定概率，让该图片不进行copy-paste
            if random.random() > args.copypaste_ratio:
                continue
            # 从含有特定类别的图片中随机选择一张，作为src图像
            src_imgId = np.random.choice(imgIds_specificCat)
            src_img = coco.loadImgs(int(src_imgId))[0] # 不取int会报错，因为原类型为'numpy.int32'
            img_src_path = os.path.join(JPEGs,src_img['file_name'])
            src_annIds = coco.getAnnIds(imgIds=src_img['id'], catIds=list_catIds, iscrowd=None)
            src_anns = coco.loadAnns(src_annIds)
            src_anns_specificCat = []
            for i in range(len(src_anns)):
                if src_anns[i]['category_id'] != category_id:
                    continue
                src_anns_specificCat.append(src_anns[i])

            # 数据增强（copy-paste）
            flag, bbox_src= copy_paste(img_main_path, img_src_path, main_anns,
                                                src_anns_specificCat,args.coincide, args.muti_obj)
            # 如果不进行copy-paste则跳到下一张图片(flag=False表示图像中目标出现重叠)
            if not flag:
                continue
            else:
                imgIds_copypaste.append(imgId)
            # copy-paste后的标注信息写入主字典中
            for bboxi in bbox_src:
                tmp2 = {'image_id': imgId,
                        'id': int(len(data['annotations']) + 1),
                        'category_id': category_id,
                        'bbox': bboxi,
                        'area': bboxi[2]*bboxi[3],
                        'iscrowd': 0,
                        'segmentation': []
                        }
                data['annotations'].append(tmp2)
    # 由于整个过程不增加图片数量，所以images字段用原来的
    data['images'] = coco.loadImgs(list_imgIds)
    # categories字段也是
    data['categories'] = coco.loadCats(list_catIds)
    # 保存json文件
    json.dump(data, open(args.output_json, 'w'))
    # 将copypaste后的图片ID写进txt，以便后续可视化的时候读取
    f = open('/home/aistudio/imgIds_copypaste.txt','w')
    for id in imgIds_copypaste:
        f.write(str(id) + '\n') 
 
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", default="./input_dir", type=str,
                        help="要进行数据增强的图像路径")
    parser.add_argument("--json_path", default="./train_final.json", type=str,
                        help="要进行数据增强的图像对应的总的json文件")
    parser.add_argument("--output_json", default="./train.json", type=str,
                        help="保存数据增强结果的json路径")
    parser.add_argument("--coincide", action='store_true', default=False,
                        help="在命令行执行py文件时，加--coincide表示传入的coincide参数为True,True表示允许数据增强后的图像目标出现重叠，默认不允许有重叠区域")
    parser.add_argument("--muti_obj", action='store_true', default=False,
                        help="在命令行执行py文件时，加--muti_obj表示传入的muti_obj参数为True,True表示将src图片上的所有的特定目标都复制粘贴，默认只随机粘贴一个目标,")
    parser.add_argument("--category_id", default=5, type=int,
                        help="需要进行copypaste的类别号，默认5，也就是黄灯")
    parser.add_argument("--copypaste_ratio", default=0.5, type=float,
                        help="需要进行copypaste的概率（对所有图片来说）")
    parser.add_argument('-Args_show', '--Args_show', type=bool, default=True,
                        help='Args_show(default: True), if True, show args info')
    args = parser.parse_args()
    if args.Args_show:
        print('Args'.center(100,'-'))
        for k, v in vars(args).items():
            print('%s = %s' % (k, v))
        print()
    return args
 
 
if __name__ == "__main__":
    args = get_args()
    main(args)
    print('Successfully copy-paste!')