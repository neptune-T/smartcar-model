# 十九届完全模型深度学习目标检测线下炼丹工程

感谢百度开源，工程使用paddle框架

## 数据增强

### 预数据增强（训练前）

`Car2024`是文件数据集和主要的目标检测文件夹，在`work/work/DataAnalyze`文件里面生成针对这个赛区官方数据的详细解析，在`work`里面是官方提供的工作代码。
随后使用`copy-paste`等技术将数据增强，得到相对满意的数据，train_x.json文件是经过数据增强得出的数据，在`ipynb`打印出在模型数据，数据不出问题后开始训练只需要重命名最后一个`json`文件。

### 动态增强（训练时）

在训练阶段，将常规的数据增强技术（如翻转、旋转、颜色调整等）应用于加载的图像。这些操作是动态进行的，只影响当前训练周期内的图像，不会更改磁盘上的图像或JSON文件。

*在paddle2.3.2版本之中，官方paddledetection添加了动态增强，数据只能进行翻转旋转，我尝试增加了颜色变化和其他动态增强方案，但是因为版本号不支持不能使用。*

## 模型训练

在此项目之中，`ipynb`文件里面设置的关于文件配置的位置没有规定好，根据自己的项目文件位置选择性的改正读取图像信息json文件等一系列文件读取路径问题。

## 数据蒸馏

项目建立了一个小模型来进行数据蒸馏`test`，可以体量更小更加轻量的训练模型来改正真正数据的模型参数问题。

在`test/work`文件里面设置了分出数据代码、按比列区分验证集和数据集代码、生成`txt`文件代码、查看图片是否重复代码，代码可用但是需要自己改正项目位置。

建议数据蒸馏时候记录下全部参数，将所选模型的`yaml`文件的`epoch`记录下来，按照比例慢慢来去扩批次。

---
Plote：

胜负非取胜者之全部,攻城掠地也绝非目的。真谛所在,乃是在征战的硝烟中,历练锻造自我,淬火重塑心智。唯有胸怀崭新气象,毅然矢志高阶,方能在赫赫征途中乘风破浪、扬帆远航,成就非凡。

青春少年,气盖风云,理应奋然踌躇满腔,超然物外勠力追求。赛场非单凭一朝一夕而斯人独得,实需泰然自若历万重践,孜孜矻矻而勇攀巅锥。

万丈赓续,孳孳而新篇,惟有傲然凌云自许,方能于艰辛求索中领略人生至理。非一蹴而就,斯道必由曲折蜿蜒而至;非坎坷跫音,殆无蹉跎岁月而奋飞。​​​​​​​​​​​​​​​​

初心高于胜负，成长胜于输赢，祝各位十九届全国智能车比赛取得好成绩！

---
## 项目文件树：
```
├─Car2024
│  ├─annotations
│  └─images
├─output
├─PaddleDetection
│  ├─benchmark
│  │  └─configs
│  ├─build
│  │  ├─bdist.win-amd64
│  │  └─lib
│  │      └─ppdet
│  │          ├─core
│  │          │  └─config
│  │          ├─data
│  │          │  ├─crop_utils
│  │          │  ├─source
│  │          │  └─transform
│  │          ├─engine
│  │          ├─metrics
│  │          ├─modeling
│  │          │  ├─architectures
│  │          │  ├─assigners
│  │          │  ├─backbones
│  │          │  ├─coders
│  │          │  ├─heads
│  │          │  ├─losses
│  │          │  ├─mot
│  │          │  │  ├─matching
│  │          │  │  ├─motion
│  │          │  │  └─tracker
│  │          │  ├─necks
│  │          │  ├─proposal_generator
│  │          │  ├─reid
│  │          │  ├─tests
│  │          │  └─transformers
│  │          ├─model_zoo
│  │          │  └─tests
│  │          ├─slim
│  │          └─utils
│  ├─configs
│  │  ├─cascade_rcnn
│  │  │  └─_base_
│  │  ├─centernet
│  │  │  └─_base_
│  │  ├─datasets
│  │  ├─dcn
│  │  ├─deformable_detr
│  │  │  └─_base_
│  │  ├─detr
│  │  │  └─_base_
│  │  ├─dota
│  │  │  └─_base_
│  │  ├─face_detection
│  │  │  └─_base_
│  │  ├─faster_rcnn
│  │  │  └─_base_
│  │  ├─fcos
│  │  │  └─_base_
│  │  ├─gfl
│  │  │  └─_base_
│  │  ├─gn
│  │  ├─hrnet
│  │  │  └─_base_
│  │  ├─keypoint
│  │  │  ├─higherhrnet
│  │  │  ├─hrnet
│  │  │  ├─lite_hrnet
│  │  │  └─tiny_pose
│  │  ├─mask_rcnn
│  │  │  └─_base_
│  │  ├─mot
│  │  │  ├─bytetrack
│  │  │  │  ├─detector
│  │  │  │  └─_base_
│  │  │  ├─deepsort
│  │  │  │  ├─detector
│  │  │  │  ├─reid
│  │  │  │  └─_base_
│  │  │  ├─fairmot
│  │  │  │  └─_base_
│  │  │  ├─headtracking21
│  │  │  ├─jde
│  │  │  │  └─_base_
│  │  │  ├─mcfairmot
│  │  │  ├─mtmct
│  │  │  ├─pedestrian
│  │  │  │  └─tools
│  │  │  │      └─visdrone
│  │  │  └─vehicle
│  │  │      └─tools
│  │  │          ├─bdd100kmot
│  │  │          └─visdrone
│  │  ├─pedestrian
│  │  │  └─demo
│  │  ├─picodet
│  │  │  ├─legacy_model
│  │  │  │  ├─application
│  │  │  │  │  ├─mainbody_detection
│  │  │  │  │  └─pedestrian_detection
│  │  │  │  ├─more_config
│  │  │  │  ├─pruner
│  │  │  │  └─_base_
│  │  │  └─_base_
│  │  ├─ppyolo
│  │  │  └─_base_
│  │  ├─ppyoloe
│  │  │  └─_base_
│  │  ├─rcnn_enhance
│  │  │  └─_base_
│  │  ├─res2net
│  │  ├─retinanet
│  │  │  └─_base_
│  │  ├─slim
│  │  │  ├─distill
│  │  │  ├─extensions
│  │  │  ├─ofa
│  │  │  ├─post_quant
│  │  │  ├─prune
│  │  │  └─quant
│  │  ├─sniper
│  │  │  └─_base_
│  │  ├─solov2
│  │  │  └─_base_
│  │  ├─sparse_rcnn
│  │  │  └─_base_
│  │  ├─ssd
│  │  │  └─_base_
│  │  ├─tood
│  │  │  └─_base_
│  │  ├─ttfnet
│  │  │  └─_base_
│  │  ├─vehicle
│  │  │  └─demo
│  │  └─yolov3
│  │      └─_base_
│  ├─dataset
│  │  ├─coco
│  │  ├─dota_coco
│  │  ├─mot
│  │  ├─roadsign_voc
│  │  ├─spine_coco
│  │  ├─voc
│  │  └─wider_face
│  ├─demo
│  ├─deploy
│  │  ├─benchmark
│  │  ├─cpp
│  │  │  ├─cmake
│  │  │  ├─docs
│  │  │  ├─include
│  │  │  ├─scripts
│  │  │  └─src
│  │  ├─lite
│  │  │  ├─include
│  │  │  └─src
│  │  ├─pphuman
│  │  │  ├─config
│  │  │  └─docs
│  │  │      └─images
│  │  ├─pptracking
│  │  │  ├─cpp
│  │  │  │  ├─cmake
│  │  │  │  ├─include
│  │  │  │  ├─scripts
│  │  │  │  └─src
│  │  │  └─python
│  │  │      └─mot
│  │  │          ├─matching
│  │  │          ├─motion
│  │  │          ├─mtmct
│  │  │          └─tracker
│  │  ├─python
│  │  ├─serving
│  │  └─third_engine
│  │      ├─demo_mnn
│  │      │  └─python
│  │      ├─demo_mnn_kpts
│  │      ├─demo_ncnn
│  │      │  └─python
│  │      ├─demo_openvino
│  │      │  └─python
│  │      └─demo_openvino_kpts
│  ├─dist
│  ├─docs
│  │  ├─advanced_tutorials
│  │  │  └─openvino_inference
│  │  ├─feature_models
│  │  ├─images
│  │  └─tutorials
│  │      ├─config_annotation
│  │      └─FAQ
│  ├─infer_output
│  ├─output
│  │  ├─picodet_m_320_coco_lcnet
│  │  └─test
│  ├─output_inference
│  │  └─picodet_m_320_coco_lcnet
│  ├─paddledet.egg-info
│  ├─ppdet
│  │  ├─core
│  │  │  ├─config
│  │  │  │  └─__pycache__
│  │  │  └─__pycache__
│  │  ├─data
│  │  │  ├─crop_utils
│  │  │  │  └─__pycache__
│  │  │  ├─source
│  │  │  │  └─__pycache__
│  │  │  ├─transform
│  │  │  │  └─__pycache__
│  │  │  └─__pycache__
│  │  ├─engine
│  │  ├─ext_op
│  │  ├─metrics
│  │  │  └─__pycache__
│  │  ├─modeling
│  │  │  ├─architectures
│  │  │  │  └─__pycache__
│  │  │  ├─assigners
│  │  │  │  └─__pycache__
│  │  │  ├─backbones
│  │  │  │  └─__pycache__
│  │  │  ├─coders
│  │  │  │  └─__pycache__
│  │  │  ├─heads
│  │  │  │  └─__pycache__
│  │  │  ├─losses
│  │  │  │  └─__pycache__
│  │  │  ├─mot
│  │  │  │  ├─matching
│  │  │  │  │  └─__pycache__
│  │  │  │  ├─motion
│  │  │  │  │  └─__pycache__
│  │  │  │  ├─tracker
│  │  │  │  │  └─__pycache__
│  │  │  │  └─__pycache__
│  │  │  ├─necks
│  │  │  │  └─__pycache__
│  │  │  ├─proposal_generator
│  │  │  │  └─__pycache__
│  │  │  ├─reid
│  │  │  │  └─__pycache__
│  │  │  ├─tests
│  │  │  │  └─imgs
│  │  │  ├─transformers
│  │  │  │  └─__pycache__
│  │  │  └─__pycache__
│  │  ├─model_zoo
│  │  │  ├─tests
│  │  │  └─__pycache__
│  │  ├─slim
│  │  │  └─__pycache__
│  │  ├─utils
│  │  │  └─__pycache__
│  │  └─__pycache__
│  ├─scripts
│  ├─static
│  │  ├─application
│  │  │  └─christmas
│  │  │      ├─blazeface
│  │  │      ├─demo_images
│  │  │      ├─element_source
│  │  │      │  ├─background
│  │  │      │  ├─beard
│  │  │      │  ├─glasses
│  │  │      │  └─hat
│  │  │      ├─solov2
│  │  │      └─solov2_blazeface
│  │  ├─configs
│  │  │  ├─acfpn
│  │  │  ├─anchor_free
│  │  │  ├─autoaugment
│  │  │  ├─dcn
│  │  │  ├─face_detection
│  │  │  ├─gcnet
│  │  │  ├─gn
│  │  │  ├─gridmask
│  │  │  ├─hrnet
│  │  │  ├─htc
│  │  │  ├─iou_loss
│  │  │  ├─libra_rcnn
│  │  │  ├─mobile
│  │  │  ├─obj365
│  │  │  ├─oidv5
│  │  │  ├─ppyolo
│  │  │  ├─random_erasing
│  │  │  ├─rcnn_enhance
│  │  │  │  └─generic
│  │  │  ├─res2net
│  │  │  ├─solov2
│  │  │  ├─ssd
│  │  │  └─yolov4
│  │  ├─contrib
│  │  │  ├─PedestrianDetection
│  │  │  │  └─demo
│  │  │  └─VehicleDetection
│  │  │      └─demo
│  │  ├─dataset
│  │  │  ├─coco
│  │  │  ├─fddb
│  │  │  ├─fruit
│  │  │  ├─roadsign_voc
│  │  │  ├─voc
│  │  │  └─wider_face
│  │  ├─demo
│  │  ├─deploy
│  │  │  ├─android_demo
│  │  │  │  ├─app
│  │  │  │  │  └─src
│  │  │  │  │      ├─androidTest
│  │  │  │  │      │  └─java
│  │  │  │  │      │      └─com
│  │  │  │  │      │          └─baidu
│  │  │  │  │      │              └─paddledetection
│  │  │  │  │      │                  └─detection
│  │  │  │  │      ├─main
│  │  │  │  │      │  ├─assets
│  │  │  │  │      │  │  ├─images
│  │  │  │  │      │  │  └─labels
│  │  │  │  │      │  ├─cpp
│  │  │  │  │      │  ├─java
│  │  │  │  │      │  │  └─com
│  │  │  │  │      │  │      └─baidu
│  │  │  │  │      │  │          └─paddledetection
│  │  │  │  │      │  │              ├─common
│  │  │  │  │      │  │              └─detection
│  │  │  │  │      │  └─res
│  │  │  │  │      │      ├─drawable
│  │  │  │  │      │      ├─drawable-v24
│  │  │  │  │      │      ├─drawable-xxhdpi-v4
│  │  │  │  │      │      ├─layout
│  │  │  │  │      │      ├─layout-land
│  │  │  │  │      │      ├─menu
│  │  │  │  │      │      ├─mipmap-anydpi-v26
│  │  │  │  │      │      ├─mipmap-hdpi
│  │  │  │  │      │      ├─mipmap-mdpi
│  │  │  │  │      │      ├─mipmap-xhdpi
│  │  │  │  │      │      ├─mipmap-xxhdpi
│  │  │  │  │      │      ├─mipmap-xxxhdpi
│  │  │  │  │      │      ├─navigation
│  │  │  │  │      │      ├─values
│  │  │  │  │      │      └─xml
│  │  │  │  │      └─test
│  │  │  │  │          └─java
│  │  │  │  │              └─com
│  │  │  │  │                  └─baidu
│  │  │  │  │                      └─paddledetection
│  │  │  │  │                          └─detection
│  │  │  │  ├─demo
│  │  │  │  └─gradle
│  │  │  │      └─wrapper
│  │  │  ├─cpp
│  │  │  │  ├─cmake
│  │  │  │  ├─docs
│  │  │  │  ├─include
│  │  │  │  ├─scripts
│  │  │  │  └─src
│  │  │  ├─lite
│  │  │  ├─python
│  │  │  └─serving
│  │  ├─docs
│  │  │  ├─advanced_tutorials
│  │  │  │  ├─config_doc
│  │  │  │  ├─deploy
│  │  │  │  │  └─docs
│  │  │  │  └─slim
│  │  │  │      ├─distillation
│  │  │  │      ├─nas
│  │  │  │      ├─prune
│  │  │  │      └─quantization
│  │  │  ├─featured_model
│  │  │  │  └─champion_model
│  │  │  ├─images
│  │  │  │  └─models
│  │  │  └─tutorials
│  │  ├─ppdet
│  │  │  ├─core
│  │  │  │  └─config
│  │  │  ├─data
│  │  │  │  ├─shared_queue
│  │  │  │  ├─source
│  │  │  │  ├─tests
│  │  │  │  └─transform
│  │  │  ├─experimental
│  │  │  ├─ext_op
│  │  │  │  ├─src
│  │  │  │  └─test
│  │  │  ├─modeling
│  │  │  │  ├─anchor_heads
│  │  │  │  ├─architectures
│  │  │  │  ├─backbones
│  │  │  │  ├─losses
│  │  │  │  ├─mask_head
│  │  │  │  ├─roi_extractors
│  │  │  │  ├─roi_heads
│  │  │  │  └─tests
│  │  │  └─utils
│  │  ├─slim
│  │  │  ├─distillation
│  │  │  ├─extensions
│  │  │  │  └─distill_pruned_model
│  │  │  ├─nas
│  │  │  │  └─search_space
│  │  │  ├─prune
│  │  │  ├─quantization
│  │  │  │  └─images
│  │  │  └─sensitive
│  │  │      └─images
│  │  └─tools
│  ├─test_tipc
│  │  ├─configs
│  │  │  ├─cascade_rcnn
│  │  │  ├─deformable_detr
│  │  │  ├─dota
│  │  │  ├─face_detection
│  │  │  ├─faster_rcnn
│  │  │  ├─fcos
│  │  │  ├─gfl
│  │  │  ├─keypoint
│  │  │  ├─mask_rcnn
│  │  │  ├─mot
│  │  │  ├─picodet
│  │  │  ├─ppyolo
│  │  │  ├─solov2
│  │  │  ├─ssd
│  │  │  ├─ttfnet
│  │  │  └─yolov3
│  │  ├─docs
│  │  └─static
│  │      ├─mask_rcnn_r50_1x_coco
│  │      │  ├─benchmark_common
│  │      │  ├─N1C1
│  │      │  └─N1C8
│  │      ├─mask_rcnn_r50_fpn_1x_coco
│  │      │  ├─benchmark_common
│  │      │  ├─N1C1
│  │      │  └─N1C8
│  │      └─yolov3_darknet53_270e_coco
│  │          ├─benchmark_common
│  │          ├─N1C1
│  │          └─N1C8
│  └─tools
├─test
│  ├─annotations
│  ├─annotations(voc)
│  ├─images
│  │  ├─img
│  │  ├─train_img
│  │  └─val_img
│  └─work
└─work
    └─work
        └─DataAnalyze
            ├─out
            │  └─img
            │      └─EachCategoryBboxWH
            ├─utils
            │  └─__pycache__
            └─__pycache__
```