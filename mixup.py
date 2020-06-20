#coding:utf-8
# date:2019-10
# Author: X.li
# function: mixup image
import os
import cv2
import numpy as np
import time
import random

if __name__ == "__main__":
    path_a = './mixup_data/A/'
    path_b = './mixup_data/B/'

    for file_a in os.listdir(path_a):
        img_a = cv2.imread(path_a + file_a)
        idx = 0
        for file_b in os.listdir(path_b):
            if random.randint(0,3) !=0:
                continue
            idx += 1
            img_b = cv2.imread(path_b + file_b)
            # INTER_AREA  INTER_CUBIC INTER_LINEAR
            img_b = cv2.resize(img_b, (img_a.shape[1],img_a.shape[0]), interpolation=cv2.INTER_LINEAR)

            alfa = 1.- float(random.randint(250,360))/1000.
            img_agu = cv2.addWeighted(img_a, alfa, img_b, (1.-alfa), 0)

            cv2.namedWindow('img_a',0)
            cv2.imshow('img_a',img_a)
            cv2.namedWindow('img_mixup',0)
            cv2.imshow('img_mixup',img_agu)
            key_id = cv2.waitKey(0)

            if idx >=1:
                break

    cv2.destroyAllWindows()
