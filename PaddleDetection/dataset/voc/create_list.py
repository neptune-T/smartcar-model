# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import sys
import os.path as osp
import logging
# add python path of PadleDetection to sys.path
# parent_path = osp.abspath(osp.join(__file__, *(['..'] * 3)))
#if parent_path not in sys.path:
#    sys.path.append(parent_path)

# from ppdet.utils.download import create_voc_list

# logging.basicConfig(level=logging.INFO)

# voc_path = osp.split(osp.realpath(sys.argv[0]))[0]
# create_voc_list(voc_path)

# 设定日志级别
logging.basicConfig(level=logging.INFO)

car2024_path = 'Car2024'  # 这里使用相对路径，根据实际情况可能需要调整

from ppdet.utils.download import create_voc_list

# 生成Car2024数据列表
create_voc_list(car2024_path)
