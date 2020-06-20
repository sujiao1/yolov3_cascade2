#-*-coding:utf-8-*-
# date:2019-05-20
# Author: X.li
# function: imgagu example
import numpy as np
np.random.bit_generator = np.random._bit_generator
import cv2
import math
import os
import random
import json
import imgaug
from imgaug import augmenters as iaa

random.seed(6666)
def M_rotate_image(image , angle , cx , cy):
    '''
    图像旋转
    :param image:
    :param angle:
    :return: 返回旋转后的图像以及旋转矩阵
    '''
    (h , w) = image.shape[:2]
    # (cx , cy) = (int(0.5 * w) , int(0.5 * h))
    M = cv2.getRotationMatrix2D((cx , cy) , -angle , 1.0)
    cos = np.abs(M[0 , 0])
    sin = np.abs(M[0 , 1])

    # 计算新图像的bounding
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    M[0 , 2] += int(0.5 * nW) - cx
    M[1 , 2] += int(0.5 * nH) - cy
    return cv2.warpAffine(image , M , (nW , nH)) , M

def image_aug_fun(imgn,idx=0):
    img_aug_list=[]
    img_aug_list.append(imgn.copy())

    if idx == 0:#单一方式增强
        seq = iaa.Sequential([iaa.Sharpen(alpha=(0.0, 0.45), lightness=(0.65, 1.35))])
        print('-------------------->>> imgaug 0 : Sharpen--锐化')
    elif idx == 1:
        seq = iaa.Sequential([iaa.AverageBlur(k=(2))])# blur image using local means with kernel sizes between 2 and 4
        print('-------------------->>> imgaug 1 : AverageBlur--均值滤波')
    elif idx == 2:
        seq = iaa.Sequential([iaa.MedianBlur(k=(3))])# blur image using local means with kernel sizes between 3 and 5
        print('-------------------->>> imgaug 2 : MedianBlur--中值滤波')
    elif idx == 3:
        seq = iaa.Sequential([iaa.GaussianBlur((0.0, 0.55))])
        print('-------------------->>> imgaug 3 : GaussianBlur--高斯滤波')
    elif idx == 4:
        seq = iaa.Sequential([iaa.ContrastNormalization((0.90, 1.10))])
        print('-------------------->>> imgaug 4 : ContrastNormalization--对比度') #  对比度
    elif idx == 5:
        seq = iaa.Sequential([iaa.Add((-55, 55))])
        print('-------------------->>> imgaug 5 : Add')
    elif idx == 6:
        seq = iaa.Sequential([iaa.AddToHueAndSaturation((-10, 10), per_channel=True)])
        print('-------------------->>> imgaug 6 : AddToHueAndSaturation--添加椒盐')
    elif idx == 7:
        # seq = iaa.Sequential([iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.02*255), per_channel=False, name=None, deterministic=False, random_state=None)])
        seq = iaa.Sequential([iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.1*255), per_channel=False, name=None, deterministic=False, random_state=None)])
        print('-------------------->>> imgaug 7 : AdditiveGaussianNoise--添加高斯噪声')
    elif idx == 8:#复合增强方式
        print(' *** 复合增强方式')
        # print('-------------------->>> 复合增强')
        seq = iaa.Sequential([
            iaa.Sharpen(alpha=(0.0, 0.05), lightness=(0.9, 1.1)),
            iaa.GaussianBlur((0, 0.8)),
            iaa.ContrastNormalization((0.9, 1.1)),
            iaa.Add((-5, 5)),
            iaa.AddToHueAndSaturation((-5, 5)),
        ])
    images_aug = seq.augment_images(img_aug_list)
    return images_aug[0].copy()
def random_image_color_channel(img_):#颜色通道变换
    print('----->> random_image_color_channel')
    (B,G,R) = cv2.split(img_)

    id_C = random.randint(0,5)

    if id_C ==0:
        img_ = cv2.merge([B,G,R])
    elif id_C ==1:
        img_ = cv2.merge([B,R,G])
    elif id_C ==2:
        img_ = cv2.merge([R,G,B])
    elif id_C ==3:
        img_ = cv2.merge([R,B,G])
    elif id_C ==4:
        img_ = cv2.merge([G,B,R])
    elif id_C ==5:
        img_ = cv2.merge([G,R,B])
    return img_
if __name__ == "__main__":

    path_ = './n02089867_1243.jpg'
    img_ = cv2.imread(path_)

    for i in range(13):
        cx = img_.shape[1]/2+random.randint(-10, 10)
        cy = img_.shape[0]/2+random.randint(-10, 10)
        angle = random.randint(-45, 45)
        if i <= 8:
            img_agu = image_aug_fun(img_, i)
        else:
            if i == 9:
                img_agu,_ = M_rotate_image(img_ , angle , int(cx) , int(cy))
                print('----->> random offset rotation --随机旋转')
            elif i == 10:
                img_agu = cv2.flip(img_, random.randint(0,1))
                print('----->> flip--翻转')
            elif i == 11:
                img_agu = img_[45:img_.shape[0]-10,30:img_.shape[1]-20]
                print('----->> random crop')
            elif i ==12 :
                img_agu = random_image_color_channel(img_)

        cv2.namedWindow('img',0)
        cv2.imshow('img',img_)
        cv2.namedWindow('img_agu',0)
        cv2.imshow('img_agu',img_agu)
        cv2.waitKey(0)

    cv2.destroyAllWindows()
