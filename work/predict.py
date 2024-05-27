import os
import sys
import json
import yaml
import numpy as np
import paddle
from paddle.inference import Config, create_predictor
import cv2

from deploy.python.preprocess import preprocess, Resize, NormalizeImage, Permute, PadStride
from deploy.python.utils import argsparser, Timer, get_current_memory_mb

class PredictConfig():
    def __init__(self, model_dir):
        deploy_file = os.path.join(model_dir, 'infer_cfg.yml')
        try:
            with open(deploy_file) as f:
                yml_conf = yaml.safe_load(f)
            self.arch = yml_conf['arch']
            self.preprocess_infos = yml_conf['Preprocess']
            self.min_subgraph_size = yml_conf['min_subgraph_size']
            self.labels = yml_conf['label_list']
            self.mask = yml_conf.get('mask', False)
            self.use_dynamic_shape = yml_conf['use_dynamic_shape']
            self.tracker = yml_conf.get('tracker', None)
            self.nms = yml_conf.get('NMS', None)
            self.fpn_stride = yml_conf.get('fpn_stride', None)
            self.print_config()
        except FileNotFoundError:
            raise ValueError(f"Config file {deploy_file} not found.")
        except yaml.YAMLError:
            raise ValueError(f"Error parsing config file {deploy_file}.")


    def print_config(self):
        print('%s: %s' % ('Model Arch', self.arch))
        for op_info in self.preprocess_infos:
            print('--%s: %s' % ('transform op', op_info['type']))

def get_test_images(infer_file):
    try:
        with open(infer_file, 'r') as f:
            dirs = f.readlines()
        images = [dir.strip().replace('\\', '/') for dir in dirs]
        assert len(images) > 0, "no image found in {}".format(infer_file)
        return images
    except Exception as e:
        raise ValueError(f"Error reading test images from {infer_file}: {e}")

def load_images_to_memory(image_list):
    images = []
    for image_path in image_list:
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Error loading image {image_path}")
        images.append(img)
      
    return images



def load_predictor(model_dir, use_tensorrt=True, use_fp16=False):
    try:
        config = Config(
            os.path.join(model_dir, 'model.pdmodel'),
            os.path.join(model_dir, 'model.pdiparams'))
        config.enable_use_gpu(2000, 0)
        config.switch_ir_optim(True)
        config.disable_glog_info()
        config.enable_memory_optim()
        config.switch_use_feed_fetch_ops(False)

        if use_tensorrt:
            config.enable_tensorrt_engine(workspace_size=1 << 30, 
                                          max_batch_size=1, 
                                          min_subgraph_size=3, 
                                          precision_mode=paddle.inference.PrecisionType.Half if use_fp16 else paddle.inference.PrecisionType.Float32)
        predictor = create_predictor(config)
        return predictor, config
    except Exception as e:
        raise ValueError(f"Error creating predictor: {e}")

def create_inputs(imgs, im_info):
    inputs = {}
    try:
        im_shape = []
        scale_factor = []
        for e in im_info:
            im_shape.append(np.array((e['im_shape'], )).astype('float32'))
            scale_factor.append(np.array((e['scale_factor'], )).astype('float32'))

        origin_scale_factor = np.concatenate(scale_factor, axis=0)

        imgs_shape = [[e.shape[1], e.shape[2]] for e in imgs]
        max_shape_h = max([e[0] for e in imgs_shape])
        max_shape_w = max([e[1] for e in imgs_shape])
        padding_imgs = []
        padding_imgs_shape = []
        padding_imgs_scale = []
        for img in imgs:
            im_c, im_h, im_w = img.shape[:]
            padding_im = np.zeros((im_c, max_shape_h, max_shape_w), dtype=np.float32)
            padding_im[:, :im_h, :im_w] = np.array(img, dtype=np.float32)
            padding_imgs.append(padding_im)
            padding_imgs_shape.append(np.array([max_shape_h, max_shape_w]).astype('float32'))
            rescale = [float(max_shape_h) / float(im_h), float(max_shape_w) / float(im_w)]
            padding_imgs_scale.append(np.array(rescale).astype('float32'))
        inputs['image'] = np.stack(padding_imgs, axis=0)
        inputs['im_shape'] = np.stack(padding_imgs_shape, axis=0)
        inputs['scale_factor'] = origin_scale_factor
        return inputs
    except Exception as e:
        raise ValueError(f"Error creating inputs: {e}")

class Detector(object):

    def __init__(self,
                 pred_config,
                 model_dir,
                 device='GPU',
                 run_mode='paddle',
                 batch_size=1,
                 trt_min_shape=1,
                 trt_max_shape=1280,
                 trt_opt_shape=640,
                 trt_calib_mode=False,
                 cpu_threads=1,
                 enable_mkldnn=False):
        self.pred_config = pred_config
        self.predictor, self.config = load_predictor(model_dir, use_tensorrt=True, use_fp16=True)
        self.det_times = Timer()
        self.cpu_mem, self.gpu_mem, self.gpu_util = 0, 0, 0
        self.preprocess_ops = self.get_ops()

    def get_ops(self):
        preprocess_ops = []
        for op_info in self.pred_config.preprocess_infos:
            new_op_info = op_info.copy()
            op_type = new_op_info.pop('type')
            preprocess_ops.append(eval(op_type)(**new_op_info))
        return preprocess_ops

    def predict(self, inputs):
        try:
            # 预处理
            input_names = self.predictor.get_input_names()
            for i in range(len(input_names)):
                input_tensor = self.predictor.get_input_handle(input_names[i])
                input_tensor.copy_from_cpu(inputs[input_names[i]])

            np_boxes, np_boxes_num = [], []

            # 模型预测
            self.predictor.run()
            np_boxes.clear()
            np_boxes_num.clear()
            output_names = self.predictor.get_output_names()
            num_outs = int(len(output_names) / 2)

            for out_idx in range(num_outs):
                np_boxes.append(
                    self.predictor.get_output_handle(output_names[out_idx])
                    .copy_to_cpu())
                np_boxes_num.append(
                    self.predictor.get_output_handle(output_names[
                        out_idx + num_outs]).copy_to_cpu())

            np_boxes, np_boxes_num = np.array(np_boxes[0]), np.array(np_boxes_num[0])
            return dict(boxes=np_boxes, boxes_num=np_boxes_num)
        except Exception as e:
            raise ValueError(f"Error during prediction: {e}")

def predict_image(detector, image_list, result_path, threshold, batch_size=1):
    c_results = {"result": []}

    for batch_start in range(0, len(image_list), batch_size):
        batch_end = min(batch_start + batch_size, len(image_list))
        batch_images = image_list[batch_start:batch_end]
        
        input_im_lst = []
        input_im_info_lst = []

        for im_path in batch_images:
            im, im_info = preprocess(im_path, detector.preprocess_ops)
            input_im_lst.append(im)
            input_im_info_lst.append(im_info)

        inputs = create_inputs(input_im_lst, input_im_info_lst)

        det_results = detector.predict(inputs)

        for index in range(len(batch_images)):
            im_path = batch_images[index]
            image_id = os.path.basename(im_path).split('.')[0]

            im_bboxes_num = det_results['boxes_num'][index]

            if im_bboxes_num > 0:
                bbox_results = det_results['boxes'][index][0:im_bboxes_num, 2:]
                id_results = det_results['boxes'][index][0:im_bboxes_num, 0]
                score_results = det_results['boxes'][index][0:im_bboxes_num, 1]

                for idx in range(im_bboxes_num):
                    if float(score_results[idx]) >= threshold:
                        c_results["result"].append({"image_id": image_id,
                                                    "type": int(id_results[idx]) + 1,
                                                    "x": float(bbox_results[idx][0]),
                                                    "y": float(bbox_results[idx][1]),
                                                    "width": float(bbox_results[idx][2]) - float(bbox_results[idx][0]),
                                                    "height": float(bbox_results[idx][3]) - float(bbox_results[idx][1]),
                                                    "segmentation": []})

    with open(result_path, 'w') as ft:
        json.dump(c_results, ft)



def main(infer_txt, result_path, det_model_path, threshold, batch_size=1):
    pred_config = PredictConfig(det_model_path)
    detector = Detector(pred_config, det_model_path, batch_size=batch_size)

    # 从图像进行预测
    img_list = get_test_images(infer_txt)
    predict_image(detector, img_list, result_path, threshold, batch_size=batch_size)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Paddle Inference")
    parser.add_argument('--infer_txt', type=str, required=True, help="Path to the image list file.")
    parser.add_argument('--result_path', type=str, required=True, help="Path to save the result json.")
    parser.add_argument('--det_model_path', type=str, default="model/", help="Path to the model directory.")
    parser.add_argument('--threshold', type=float, default=0.3, help="Threshold for detection.")
    parser.add_argument('--batch_size', type=int, default=1, help="Batch size for prediction.")

    args = parser.parse_args()
    
    start_time = time.time()
    paddle.enable_static()
    main(args.infer_txt, args.result_path, args.det_model_path, args.threshold, args.batch_size)
    print('total time:', time.time() - start_time)
