#-*-coding:utf-8-*-
# date:2019-05-20
# Author: xiang li
# function:

import torch
torch.manual_seed(1)
from data_iterator.data_iterator import *
# from models.mobilenetv2 import MobileNetV2
from common_models import *
import torch.nn as nn
import torch.optim as optim
import time
import datetime
from tensorboardX import SummaryWriter
import os
import math
from logger import logger
from datetime import datetime
import cv2
from torch.autograd import Variable
import torch.nn.functional as F
from soft_label import LabelSmoothing,FocalLoss
# if os.access('./logs',os.F_OK):
#     os.remove('./logs')
def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)
def mixup_data(x, y, alpha=1.0, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
        # print('mixup : ',index.cpu().numpy())
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def get_acc(output, label):
    total = output.shape[0]
    _, pred_label = output.max(1)
    num_correct = (pred_label == label).sum().item()
    return num_correct / float(total)

def set_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
if __name__ == "__main__":
    save_model_dir= './model_dir/'
    model_dir = save_model_dir
    if not os.path.exists(save_model_dir):
        os.mkdir(save_model_dir)

    train_path = './datasets_new/train_new/'
    logger.info('datasets label : %s'%(os.listdir(train_path)))
    output_node = len(os.listdir(train_path))
    logger.info('output_node : %s'%(output_node))
    model_name = "resnet"#squeezenet resnet
    logger.info('use model : %s'%(model_name))
    # Number of classes in the dataset
    num_classes = output_node
    # Flag for feature extracting. When False, we finetune the whole model,
    #   when True we only update the reshaped layer params
    feature_extract = False
    model_, _ = initialize_model(model_name, num_classes, feature_extract, use_pretrained=True)

    writer = SummaryWriter(logdir='./logs_train', comment='resnet51')
    input_data = Variable(torch.rand(8, 3, 224, 224))
    writer.add_graph(model_, (input_data,))
    logger.info('model_ out put %s'%model_.fc)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_ = model_.to(device)


    # criterion = define_my_loss(num_classes)
    init_lr = 0.001

    batch_size = 32
    start_epoch = 0
    epochs = 100
    num_workers = 8
    img_size = (224,224)
    lr_decay_setp = 1
    is_train = True
    print('image size    :',img_size)
    print('batch_size    : ',batch_size)
    print('num_workers   : ',num_workers)
    print('init_lr       : ',init_lr)
    print('epochs        : ',epochs)
    # Dataset
    dataset = LoadImagesAndLabels(path = train_path,img_size=img_size,is_train=is_train, augment=True)
    logger.info('len train datasets : %s'%(dataset.__len__()))
    # print('------> dataset len : ',dataset.__len__())
    # # Dataloader
    dataloader = DataLoader(dataset,
                            batch_size=batch_size,
                            num_workers=num_workers,
                            shuffle=True,
                            pin_memory=False,
                            drop_last=True)

    # optimizer_Adam = torch.optim.Adam(model_.parameters(), lr=init_lr, betas=(0.9, 0.99),weight_decay=1e-5)
    optimizer_SGD = optim.SGD(model_.parameters(), lr=init_lr, momentum=0.9, weight_decay=1e-4)
    optimizer = optimizer_SGD
    criterion = nn.CrossEntropyLoss()#CrossEntropyLoss() 是 softmax 和 负对数损失的结合
    if os.access(model_dir+'latest.pt',os.F_OK):
        my_model_path = model_dir+'latest.pt'
        chkpt = torch.load(my_model_path, map_location=device)
        print('device:',device)
        # print('chkpt:\n',chkpt)
        model_.load_state_dict(chkpt['model'])
        optimizer.load_state_dict(chkpt['optimizer'])
        # print('optimizer:',optimizer)
        start_epoch = chkpt['epoch']
        init_lr = chkpt['init_lr']
        print('load model : ',model_dir+'latest.pt')
    #
    # for param in model.parameters():  # freeze layers
    #     param.requires_grad = False
    #


    print('/**********************************************/')
    # for param in optimizer.param_groups:
    #     print("weight_decay:", param['weight_decay'])
    #     # print("momentum:", param['momentum'])
    #     print("init lr:", param['lr'])
    #
    # summary_writter = SummaryWriter("./logs")
    #
    # Flag_soft_label = True
    # if Flag_soft_label:
    #     crit = LabelSmoothing(size=num_classes,smoothing= 0.1)
    # criterion = nn.MSELoss(reduce=True, reduction='mean')
    #
    loss_define = 'focal_loss'
    if 'mixup' == loss_define:
        criterion = nn.CrossEntropyLoss()# mixup
    elif 'focal_loss' == loss_define:
        criterion = FocalLoss(num_class = num_classes)

    step = 0
    idx = 0
    use_cuda = torch.cuda.is_available()
    test_moment = 20
    for epoch in range(start_epoch, epochs):
        print('\nepoch %d ------>>>'%epoch)
        model_.train()
        if epoch % lr_decay_setp == 0 and epoch != 0:
            init_lr = init_lr*0.9
            # init_lr = 0.0001
            set_learning_rate(optimizer, init_lr)

        for i, (imgs_, labels_) in enumerate(dataloader):

            # print(i,') --> ',imgs_.size(),labels_.size(),end ='\r')
    #         # imgs_crop = imgs_crop.unsqueeze_(1)
    #         # print('1)imgs_crop',imgs_crop.size())
            imgs_ = imgs_.permute(0,3,1,2)
    #         # print('2)imgs_crop',imgs_crop.size())
            if use_cuda:
                imgs_ = imgs_.cuda()  # (bs, 3, h, w)
                labels_ = labels_.cuda()

            if 'mixup' == loss_define and i%test_moment != 0:
                if use_cuda:
                    imgs_, targets_a, targets_b, lam = mixup_data(imgs_, labels_,alpha = 1,use_cuda = use_cuda)
                    imgs_, targets_a, targets_b = map(Variable, (imgs_,targets_a, targets_b))



            try:
                output = model_(imgs_.float())
            except RuntimeError as exception:
                if "out of memory" in str(exception):
                    print("WARNING: out of memory")
                    if hasattr(torch.cuda, 'empty_cache'):
                        torch.cuda.empty_cache()
                    else:
                        raise exception


            # print('output  : ',output.size(),output[0],labels_)
            if 'mixup' == loss_define and i%test_moment != 0:
                loss = mixup_criterion(criterion, output, targets_a, targets_b, lam)

            else:
                loss = criterion(output, labels_)
            if i%test_moment == 0:
                acc = get_acc(output, labels_)
                if 'mixup' == loss_define:
                    print('mixup -       epoch : ',epoch,' loss : ',loss.item(),' acc : ',acc,' lr : ',init_lr)
                else:
                    print('       epoch : ',epoch,' loss : ',loss.item(),' acc : ',acc,' lr : ',init_lr)

                writer.add_scalar('data/loss', loss, step)
                writer.add_scalars('data/scalar_group', {'acc':acc,'lr':init_lr,'baseline':0.}, step)
            # backward
            if i%test_moment != 0:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            step += 1

            if i%100 == 0 and i > 0:
                print('           save model ~')
                # Create checkpoint
                chkpt = {'epoch': epoch,'init_lr':init_lr,
                    'model': model.module.state_dict() if type(
                    model_) is nn.parallel.DistributedDataParallel else model_.state_dict(),
                    'optimizer': optimizer.state_dict()}

                # Save latest checkpoint
                torch.save(chkpt, save_model_dir + 'latest.pt')
                torch.save(chkpt, save_model_dir + 'model_epoch'+str(epoch)+'.pt')
                print('           save model done !!!')

    writer.close()
    #         # loss = torch.mean(landmark_abs,dim=1,keepdim=True)
    #         # loss = torch.mean(loss)
    #
    #
    #         #
    #         landmark_abs = torch.abs(output-crop_landmarks.float())
    #         loss_faceedge = torch.abs(output[:,0:17]-crop_landmarks[:,0:17].float())
    #         loss_eyebrow = torch.abs(output[:,17:27]-crop_landmarks[:,17:27].float())
    #         loss_nose = torch.abs(output[:,27:36]- crop_landmarks[:,27:36].float())
    #         loss_eye = torch.abs(output[:,36:48]- crop_landmarks[:,36:48].float())
    #         loss_mouse = torch.abs(output[:,48:66]- crop_landmarks[:,48:66].float())
    #
    #         loss = torch.mean(landmark_abs)+torch.mean(loss_faceedge)*0.55+torch.mean(loss_eyebrow)*0.5+\
    #         torch.mean(loss_nose)*0.5+torch.mean(loss_eye)*0.55+torch.mean(loss_mouse)*0.5
    #
    #         #
    #         # loss = gost_total_wing_loss(output,crop_landmarks)
    #         # print('--------->>> loss mean size : ',loss.size())
    #
    #         # backward
    #         optimizer.zero_grad()
    #         loss.backward()
    #         optimizer.step()
    #         print("Epoch: %d, - %d -, lr: %f, Train Loss: %f" % (epoch, i, optimizer.param_groups[0]['lr'], loss.item()),end = '\r')
    #
    #
    #
    #         if i%50 == 0:
    #             summary_writter.add_scalars("loss", {"train loss": loss.item()}, idx)
    #             idx += 1
    #
    #         # summary_writter.add_scalars("loss", {"train": train_loss / len(train_data), "val": valid_loss / len(valid_data)}, epoch)
    #         if i%500 == 0 and i>1:
    #             year = datetime.datetime.now().year
    #             month = datetime.datetime.now().month
    #             day = datetime.datetime.now().day
    #             hour = datetime.datetime.now().hour
    #             minute = datetime.datetime.now().minute
    #             # now_time = time.strftime("%Y_%m_%d_%H_%M_%p", time.localtime())
    #             now_time = str(year)+'_' + str(month)+'_'+str(day)+'_'+str(hour)+'_'+str(minute)
    #             torch.save(model.state_dict(), (save_model_dir + 'model_'+now_time+'-'+str(loss.item())+'.pt'))
    #
    #
    #             # Create checkpoint
    #             chkpt = {'epoch': epoch,'init_lr':init_lr,
    #                      'model': model.module.state_dict() if type(
    #                      model) is nn.parallel.DistributedDataParallel else model.state_dict(),
    #                      'optimizer': optimizer.state_dict()}
    #
    #             # Save latest checkpoint
    #             torch.save(chkpt, save_model_dir + 'latest.pt')
    #
    #             print()
    #
    #     # Create checkpoint
    #     chkpt = {'epoch': epoch,'init_lr':init_lr,
    #              'model': model.module.state_dict() if type(
    #              model) is nn.parallel.DistributedDataParallel else model.state_dict(),
    #              'optimizer': optimizer.state_dict()}
    #
    #     # Save latest checkpoint
    #     torch.save(chkpt, save_model_dir + 'model_epoch'+str(epoch)+'.pt')


    print('well done ')
