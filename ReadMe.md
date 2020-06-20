
step 1 :coco相关训练集图片和标注文件 xml在 datasets_origin文件夹下，而且数据是完整的,stanford dogs相关训练图片和标注文件 xml 在 dogs_origin 下，但是这个数据集原本有120类狗的种类，这里只用了10种。因为考虑到同学的设备性能等问题，提供10类数据是保证同学们先可以把代码跑起来，如果有同学有服务器或者GPU资源的话，可以直接用coco数据集。

step 2 ：所有脚本都有 Flag_Predict_Dogs : 相应的选择配置标志位，Flag_Predict_Dogs == True 运行 stanford dogs 相关数据制作、 训练 预测，为False 为 coco mini

step 3 : make_data_yolo_train_datasets.py 
 
step 4 : show_yolo_anno.py

step 5 : train.py

step 6 : predict.py / cascade_predict.py(cascade无Flag_Predict_Dogs标志位设置，只对stanford dogs使用。cascade是分类+检测的一个预测，其中分类的模型是我们提前训练好的)


img_agu.py  使用 imgagu的 扩增样例

mixup.py    mixup image 案例
