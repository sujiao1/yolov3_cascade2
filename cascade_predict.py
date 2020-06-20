#coding:utf-8
#-*-coding:utf-8-*-
# date:2019-08
# Author: X.li
# function: cascade predict

import os
import torch
from utils.datasets import *
from utils.utils import *
from utils.parse_config import parse_data_cfg
from yolov3 import Yolov3, Yolov3Tiny
import time
from common_models import *
import torch.nn as nn
import math
import cv2
import torch.nn.functional as F
from utils.torch_utils import select_device

# os.environ['CUDA_VISIBLE_DEVICES'] = "0"
# cuda = torch.cuda.is_available()
# device = torch.device('cuda:0' if cuda else 'cpu')
import sys
print(sys.path)
sys.path.append('/lib/python3.7/site-packages')


def Create_Classify_Model(device,model_dir,label_path):
    print('/**************** Create Classify Model **************/')
    train_path = label_path
    labels_list = []
    for label_ in os.listdir(train_path):
        labels_list.append(label_.split('-')[1])

    print('datasets label : %s'%(labels_list))
    output_node = len(os.listdir(train_path))
    print('output_node : %s'%(output_node))
    model_name = "resnet"#squeezenet resnet
    print('use model : %s'%(model_name))
    # Number of classes in the dataset
    num_classes = output_node

    feature_extract = False
    model_, _ = initialize_model(model_name, num_classes, feature_extract, use_pretrained=False)
    print('model_ out put %s'%model_.fc)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_ = model_.to(device)

    img_size = (224,224)

    print('image size    :',img_size)
    if os.access(model_dir,os.F_OK):
        my_model_path = model_dir
        chkpt = torch.load(my_model_path, map_location=device)
        print('device:',device)
        # print('chkpt:\n',chkpt)
        model_.load_state_dict(chkpt['model'])
    else:
        print('error no classify_model')

    model_.eval()

    return model_,labels_list

def prewhiten(x):
    mean = np.mean(x)
    std = np.std(x)
    std_adj = np.maximum(std, 1.0 / np.sqrt(x.size))
    y = np.multiply(np.subtract(x, mean), 1 / std_adj)
    return y

def process_data(img, img_size=416):# 图像预处理
    img, _, _, _ = letterbox(img, height=img_size)
    # Normalize RGB
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB
    img = np.ascontiguousarray(img, dtype=np.float32)  # uint8 to float32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    # cv2.imwrite('letterbox.jpg', 255 * img.transpose((1, 2, 0))[:, :, ::-1])  # save letterbox image
    return img

def show_model_param(model):
    params = list(model.parameters())
    k = 0
    for i in params:
        l = 1
        for j in i.size():
            l *= j
        print("该层的结构: {}, 参数和: {}".format(str(list(i.size())), str(l)))
        k = k + l
    print("----------------------")
    print("总参数数量和: " + str(k))

def detect(
        model_path,
        classify_model_path,
        label_path,
        root_path,
        cfg,
        data_cfg,
        img_size=416,
        conf_thres=0.5,
        nms_thres=0.5,
):
    classes = load_classes(parse_data_cfg(data_cfg)['names'])
    num_classes = len(classes)
    # Initialize model
    if "-tiny" in cfg:
        model = Yolov3Tiny(num_classes)
        weights = model_path
    else:
        model = Yolov3(num_classes)
        weights = model_path

    show_model_param(model)# 显示模型参数

    device = select_device(False)# 运行硬件选择

    classify_model,labels_dogs_list = Create_Classify_Model(device,classify_model_path,label_path)

    # Load weights
    if os.access(weights,os.F_OK):# 判断模型文件是否存在
        model.load_state_dict(torch.load(weights, map_location=device)['model'])
    else:
        print('error model not exists')
        return False
    model.to(device).eval()# 设置 模型 eval

    colors = [(v // 32 * 64 + 64, (v // 8) % 4 * 64, v % 8 * 32) for v in range(1, num_classes + 1)][::-1]
    use_cuda = torch.cuda.is_available()
    for img_name in os.listdir(root_path):
        img_path  = root_path + img_name
        im0 = cv2.imread(img_path)
        im_c = cv2.imread(img_path)
        print("---------------------")

        t = time.time()
        img = process_data(im0, img_size)
        if use_cuda:
            torch.cuda.synchronize()
        t1 = time.time()
        print("process time:", t1-t)
        img = torch.from_numpy(img).unsqueeze(0).to(device)

        pred, _ = model(img)
        if use_cuda:
            torch.cuda.synchronize()
        t2 = time.time()
        print("inference time:", t2-t1)
        detections = non_max_suppression(pred, conf_thres, nms_thres)[0]
        if use_cuda:
            torch.cuda.synchronize()
        t3 = time.time()
        print("get res time:", t3-t2)
        if detections is None or len(detections) == 0:
            continue
        # Rescale boxes from 416 to true image size
        detections[:, :4] = scale_coords(img_size, detections[:, :4], im0.shape).round()
        result = []
        for res in detections:
            result.append((classes[int(res[-1])], float(res[4]), [int(res[0]), int(res[1]), int(res[2]), int(res[3])]))
        if use_cuda:
            torch.cuda.synchronize()
        s2 = time.time()
        print("detect time:", s2 - t)
        print(result)

        # Draw bounding boxes and labels of detections
        for *xyxy, conf, cls_conf, cls in detections:
            label = '%s %.2f' % (classes[int(cls)], conf)

            #-------------------------------------------------------------------
            plot_one_box(xyxy, im0, label=label, color=colors[int(cls)])

            x_1 = int(xyxy[0])
            y_1 = int(xyxy[1])
            x_2 = int(xyxy[2])
            y_2 = int(xyxy[3])
            #--------------------
            img_crop_ = cv2.resize(im_c[y_1:y_2,x_1:x_2,:], (224,224), interpolation = cv2.INTER_CUBIC)
            img_crop_ = img_crop_.astype(np.float32)
            img_crop_ = prewhiten(img_crop_)

            img_crop_ = torch.from_numpy(img_crop_)
            img_crop_ = img_crop_.unsqueeze_(0)
            img_crop_ = img_crop_.permute(0,3,1,2)

            if use_cuda:#
                img_crop_ = img_crop_.cuda()  # (bs, 3, h, w)

            outputs = F.softmax(classify_model(img_crop_.float()),dim = 1)

            outputs = outputs[0]
            outputx = outputs.cpu().detach().numpy()
            # print('output: ',output)
            max_index = np.argmax(outputx)

            scorex_ = outputx[max_index]
            label_dog_ = labels_dogs_list[max_index]

            print('label_dog_ : ',label_dog_)

            plot_one_box((x_1,y_1+20,x_2,y_2), im0, label=label_dog_+'_'+'%.2f'%(scorex_), color=colors[int(cls)])
            #-----------------------
            cv2.namedWindow('crop',0)
            cv2.imshow('crop',im_c[y_1:y_2,x_1:x_2,:])

        cv2.namedWindow('result',0)
        cv2.imshow("result", im0)
        key = cv2.waitKey(0)
        if key == 27:
            break

if __name__ == '__main__':

    model_path = "./weights-yolov3-tiny/latest.pt" # 检测模型路径
    classify_model_path = './model_dir/latest.pt' # 分类模型路径
    label_path = './crop_train/'# 指向分类训练文件夹
    root_path = './datasets_dogs/anno/images/'# 测试图片路径
    model_cfg = 'yolov3-tiny'#指定检测模型类型 需要与 模型匹配
    voc_config = 'cfg/voc_dog.data'#指定模型相关配置文件
    img_size = 416 # 图片输入尺寸
    conf_thres = 0.03 # 检测 置信度阈值
    nms_thres = 0.5# nms 阈值

    with torch.no_grad():#设置无梯度运行
        detect(
            model_path = model_path,
            classify_model_path = classify_model_path,
            label_path = label_path,
            root_path = root_path,
            cfg = model_cfg,
            data_cfg = voc_config,
            img_size=img_size,
            conf_thres=conf_thres,
            nms_thres=nms_thres,
        )
