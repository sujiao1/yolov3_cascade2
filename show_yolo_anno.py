#coding:utf-8
# date:2019-12
# Author: jiao su
# function: show yolo anno

import cv2
import os
import numpy as np

if __name__ == "__main__":

    Flag_Predict_Dogs = True # 显示选择

    if Flag_Predict_Dogs == True:
        path = './datasets_dogs/anno/train.txt'
        path_voc_names = './cfg/voc_dog.names'
    elif Flag_Predict_Dogs == False:
        path ='./datasets/anno/train.txt'
        path_voc_names = './cfg/voc_coco.names'

    with open(path_voc_names, 'r') as f:
        label_map = f.readlines()
    
    # 移除字符串头尾指定的字符（默认为空格或换行符）或字符序列
    # ./cfg/voc_dog.names  仅有dog，框出来图片名为dog
    for i in range(len(label_map)):
        label_map[i] = label_map[i].strip()
        print(i, ') ', label_map[i].strip())

    # 去除train.txt路径中的空格，筛选出路径大于0的图片路径
    with open(path, 'r') as file:
        img_files = file.read().splitlines()
        print('img_files')
        print(img_files)
        img_files = list(filter(lambda x: len(x) > 0, img_files))
        print('img_files_filter')
        print(img_files)

    # 设置labels,将图片路径中的字符进行替换 img_files = train.txt,相当于读取datasets_dogs/anno/labels下的txt内容
    label_files = [
        x.replace('images', 'labels').replace("JPEGImages", 'labels').replace('.bmp', '.txt').replace('.jpg', '.txt').replace('.png', '.txt')
        for x in img_files]
    print('label_files')
    print(label_files)

    # print('img_files   : ',img_files[1])
    # print('label_files : ',label_files[1])
    # 读取train.txt中每一张图片路径，由cv读取，获取图片的宽和高
    for i in range(len(img_files)):
        # 从0开始，读取每一张图片
        img = cv2.imread(img_files[i])
        print('img')
        print(img)
        # 每张图片为三维数组，w指三维数组中二维数组的个数，表明图片的水平像素
        w = img.shape[1]
        print('水平像素')
        print(w)
        # h指每一个二维数组中二维数组的行数，表面图片的垂直像素
        h = img.shape[0]
        print('垂直像素')
        print(h)
        # 指每一个二维数组的列数，表明图像的通道数
        print('图片通道数')
        print(img.shape[2])

        # 读取每一个标签。读取datasets_dogs/anno/labels下的每一个图片对应的txt内容
        label_path = label_files[i]
        if os.path.isfile(label_path):
            with open(label_path, 'r') as file:
                lines = file.read().splitlines()
                print('txt内容')
                print(lines)

            # 空格为分隔符，分割路径，用numpy生成数组
            x = np.array([x.split() for x in lines], dtype=np.float32)
            print('x')
            print(x)
        for k in range(len(x)):
            anno = x[k]
            print('anno')
            print(anno)
            label = int(anno[0])
            print('label')
            print(label)

            # 设置识别框，相当于gird truth
            x1 = int((float(anno[1])-float(anno[3])/2)*w)
            print('x1')
            print(x1)
            y1 = int((float(anno[2])-float(anno[4])/2)*h)
            print('y1')
            print(y1)

            x2 = int((float(anno[1])+float(anno[3])/2)*w)
            print('x2')
            print(x2)
            y2 = int((float(anno[2])+float(anno[4])/2)*h)
            print('y2')
            print(y2)


            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 30, 30), 2)

            cv2.putText(img, ("%s" % (str(label_map[label]))), (x1, y1),\
            cv2.FONT_HERSHEY_PLAIN, 2.5, (0, 255, 55), 6)
            cv2.putText(img, ("%s" % (str(label_map[label]))), (x1, y1),\
            cv2.FONT_HERSHEY_PLAIN, 2.5, (0, 55, 255), 2)
            # cv2.circle(img, (x1,y1), 4, (0,255,225), 6)

        cv2.namedWindow('image', 0)
        cv2.imshow('image', img)
        cv2.waitKey(0)
    cv2.destroyAllWindows()
