#-*-coding:utf-8-*-
# date:2019-08
# Author: jiao.su
# function: make yolo train datasets

import os
import xml.etree.cElementTree as et
import cv2
import numpy as np
import os.path
import shutil

if __name__ == "__main__":
    Flag_Predict_Dogs = True # 是否制作 dogs 检测数据集

    if Flag_Predict_Dogs ==False:
        choose_list = ['person', 'car', 'motorcycle', 'dog', 'bus', 'truck']
    # 狗的品种
    elif Flag_Predict_Dogs == True:
        choose_list = ['otterhound', 'French_bulldog', 'English_foxhound', 'dhole', 'African_hunting_dog', 'Leonberg', 'Newfoundland', 'Samoyed', 'Great_Dane', 'affenpinscher', 'basenji', 'Doberman', 'EntleBucher', 'malinois', 'komondor', 'Bouvier_des_Flandres', 'Sussex_spaniel', 'Irish_water_spaniel', 'Australian_terrier', 'standard_schnauzer', 'Kerry_blue_terrier', 'Norfolk_terrier', 'Ibizan_hound', 'Scottish_deerhound', 'bloodhound', 'Irish_wolfhound', 'Japanese_spaniel', 'Blenheim_spaniel', 'papillon', 'toy_terrier', 'Rhodesian_ridgeback', 'dingo', 'Mexican_hairless', 'standard_poodle', 'miniature_poodle', 'toy_poodle', 'Cardigan', 'Pembroke', 'Brabancon_griffon', 'keeshond', 'chow', 'Pomeranian', 'Great_Pyrenees', 'pug', 'Siberian_husky', 'malamute', 'Eskimo_dog', 'Saint_Bernard', 'Tibetan_mastiff', 'bull_mastiff', 'boxer', 'Appenzeller', 'Bernese_mountain_dog', 'Greater_Swiss_Mountain_dog', 'miniature_pinscher', 'German_shepherd', 'Rottweiler', 'Border_collie', 'collie', 'Old_English_sheepdog', 'kelpie', 'briard', 'groenendael', 'schipperke', 'kuvasz', 'cocker_spaniel', 'Welsh_springer_spaniel', 'English_springer', 'clumber', 'Brittany_spaniel', 'Gordon_setter', 'Irish_setter', 'English_setter', 'vizsla', 'German_short-haired_pointer', 'Chesapeake_Bay_retriever', 'Labrador_retriever', 'golden_retriever', 'curly-coated_retriever', 'flat-coated_retriever', 'Lhasa', 'West_Highland_white_terrier', 'soft-coated_wheaten_terrier', 'silky_terrier', 'Tibetan_terrier', 'Scotch_terrier', 'giant_schnauzer', 'miniature_schnauzer', 'Boston_bull', 'Dandie_Dinmont', 'cairn', 'Airedale', 'Sealyham_terrier', 'Lakeland_terrier', 'wire-haired_fox_terrier', 'Yorkshire_terrier', 'Norwich_terrier', 'Irish_terrier', 'Border_terrier', 'Bedlington_terrier', 'American_Staffordshire_terrier', 'Staffordshire_bullterrier', 'Weimaraner', 'Saluki', 'Norwegian_elkhound', 'whippet', 'Italian_greyhound', 'borzoi', 'redbone', 'Walker_hound', 'black-and-tan_coonhound', 'bluetick', 'beagle', 'basset', 'Afghan_hound', 'Shih-Tzu', 'Pekinese', 'Maltese_dog', 'Chihuahua', 'Shetland_sheepdog']

    label_dict_ = {}
    label_cnt_dict_ = {}

    for i in range(len(choose_list)):
        if Flag_Predict_Dogs == False:
            label_dict_[choose_list[i]] = i
        elif Flag_Predict_Dogs == True:
            label_dict_[choose_list[i]] = 0

        # 初始化计数（物种数量或品类数量）
        label_cnt_dict_[choose_list[i]] = 0

    print('label_dict_', label_dict_)
    path_root = os.getcwd()
    path_root = path_root.replace('\\', '/')

    print(path_root)

    if Flag_Predict_Dogs == True:
        path_data = './dogs_origin/'
        path_datasets_ = 'datasets_dogs/'
    elif Flag_Predict_Dogs == False:
        path_data = './datasets_origin/'
        path_datasets_ = 'datasets/'

    if not os.path.exists(path_datasets_):
        os.mkdir(path_datasets_)

    if not os.path.exists(path_datasets_+'anno/'):
        os.mkdir(path_datasets_ + 'anno/')

    save_images_path = path_datasets_ + 'anno/images/'
    save_labels_path = path_datasets_ + 'anno/labels/'

    if not os.path.exists(save_images_path):
        os.mkdir(save_images_path)
    if not os.path.exists(save_labels_path):
        os.mkdir(save_labels_path)

    idx = 0

    train_ = open(path_datasets_ + 'anno/train.txt', 'w')

    for file in os.listdir(path_data):
        if os.path.splitext(file)[1] == ".jpg" or os.path.splitext(file)[1] == ".png":
            # 在image_path_o中存入dogs_origin/datasets_origin中的图片路径
            image_path_o = path_root + '/' + path_data + file
            # 将存入的图片路径后缀全部替换为.xml，不会改变原来图片的后缀，只是添加了相同图片文件的后缀为xml的文件
            # 自动转码为xml文件，bndbox为目标框
            xml_path_o = image_path_o.replace('.jpg', '.xml').replace('.png', '.xml')
            # 在save_image_path中存入dogs_origin/datasets_origin中的图片,并将图片路径保存在image_path(anno/image)中
            image_path = path_root + '/' + save_images_path + file
            # 将image_path中的图片的路径为images的全部替换为labels，将.jpg、.png全部替换为.txt
            # 此时只是新建labels文件夹，并未将图片文件生成txt文件，会有txt文件名
            txt_path = image_path.replace('images', 'labels').replace('.jpg', '.txt').replace('.png', '.txt')

            if not os.path.exists(xml_path_o):
                continue

            # 将image_path_o(dogs_origin/datasets_origin)中的xml文件路径复制到image_path(anno/image)中
            shutil.copy(image_path_o, image_path)
            # 计数，记录存入多少个图片文件
            idx += 1
            print(idx, ') ', txt_path)

            flag_txt = False
            #---------------------------------------------------------

            # 读取xml_path_o中的每一个xml文件
            tree = et.parse(xml_path_o)
            # 获取第一标签
            root = tree.getroot()
            # print(filename)
            # 读取含有中文路径的图片，读取图片，读取形式为数组
            img = cv2.imread(image_path)
            try:
                height = img.shape[0]
                width = img.shape[1]
            except:
                print('-------->>> image error ')
                continue
            # 根据图片xml文件的x、y坐标求出中心点
            for Object in root.findall('object'):
                # 狗的品类名称
                name = Object.find('name').text
                print('name')
                print(name)

                bndbox = Object.find('bndbox')
                xmin= np.float32((bndbox.find('xmin').text))
                ymin= np.float32((bndbox.find('ymin').text))
                xmax= np.float32((bndbox.find('xmax').text))
                ymax= np.float32((bndbox.find('ymax').text))

                # 像素点LoadImages
                x_mid = (xmax + xmin)/2./float(width)
                y_mid = (ymax + ymin)/2./float(height)

                w_box = (xmax-xmin)/float(width)
                h_box = (ymax-ymin)/float(height)

                # 标签均为0？
                label_xx = label_dict_[name]
                print('labelxx')
                print(label_xx)
                # 对查找到的品类进行计数
                label_cnt_dict_[name] += 1
                if flag_txt == False:
                    flag_txt = True
                    # 写入后生成txt文件
                    anno_txt = open(txt_path, 'w')
                    print('-- label ', name)
                # 标签名和内容
                anno_txt.write(str(label_xx)+' '+str(x_mid)+' '+str(y_mid)+' '+str(w_box)+' '+str(h_box) + '\n')
                # anno_txt.write('0'+' '+str(x_mid)+' '+str(y_mid)+' '+str(w_box)+' '+str(h_box)+ '\n')

            #---------------------------------------------------------
            if flag_txt == True:
                anno_txt.close()
                # 在train.txt文件中写入图片路径
                train_.write(image_path + '\n')
    train_.close()

    # 输出每种狗的品类的数量，dictionary结构
    for key in label_cnt_dict_.keys():
        print('%s : %s' % (key, label_cnt_dict_[key]))
