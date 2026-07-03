import torch
from torch.utils.data import DataLoader
import timm
import transforms as T
# from datasets.dataset import NPY_datasets
from datasets.datasets1 import DriveDataset, Chasedb1Datasets, STAREDataset, HRFDataset#, FIVESDataset
# from tensorboardX import SummaryWriter
from thop import profile
from torch.utils.tensorboard import SummaryWriter
# writer = SummaryWriter()
import sys
sys.path.append('/home/YDM/EGE-UNet-eye/models/DCNv4_op/DCNv4')  # 添加 DCNv4 根目录

from models.egeunet import EGEUNet

from engine import *
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn

from utils import *
from configs.config_setting import setting_config

import warnings
warnings.filterwarnings("ignore")

from torchvision import models
# from torchsummary import summary
# from torchinfo import summary
from torch.optim.lr_scheduler import CosineAnnealingLR

class SegmentationPresetTrain:
    # 用于图像分割任务训练阶段的数据预处理和增强操作。
    def __init__(self, crop_size, hflip_prob=0.5, vflip_prob=0.5,
                 mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        trans = []
        if hflip_prob > 0:
            trans.append(T.RandomHorizontalFlip(hflip_prob))
        if vflip_prob > 0:
            trans.append(T.RandomVerticalFlip(vflip_prob))
        trans.extend([
            T.RandomCrop(crop_size),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])
        self.transforms = T.Compose(trans)

    def __call__(self, img, target):
        return self.transforms(img, target)


class SegmentationPresetEval:
    def __init__(self, crop_size, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.transforms = T.Compose([
            # T.RandomCrop(crop_size),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])

    def __call__(self, img, target):
        return self.transforms(img, target)



def get_transform(train, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    crop_size = 400 #stare
    # crop_size = 500 #chase
    # crop_size1 = (998,960)#chase
    # crop_size = 400 #chase
    # crop_size = 512  # fives

    if train:
        return SegmentationPresetTrain(crop_size, mean=mean, std=std)
    else:
        # return SegmentationPresetEval(mean=mean, std=std)
        return SegmentationPresetEval(crop_size, mean=mean, std=std)


def main(config):

    print('#----------Creating logger----------#')
    sys.path.append(config.work_dir + '/')
    log_dir = os.path.join(config.work_dir, 'log')
    checkpoint_dir = os.path.join(config.work_dir, 'checkpoints')
    resume_model = os.path.join(checkpoint_dir, 'latest.pth')

    # # 恢复模型的路径
    # resume_model = '/home/ydm/EGE-UNet-eye/results/egeunet_stare_Monday_10_March_2025_16h_20m_24s/checkpoints/latest.pth'
    
    outputs = os.path.join(config.work_dir, 'outputs')
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    if not os.path.exists(outputs):
        os.makedirs(outputs)

    global logger
    logger = get_logger('train', log_dir)
    global writer
    writer = SummaryWriter(config.work_dir + 'summary')

    log_config_info(config, logger)



    print('#----------GPU init----------#')
    os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu_id
    set_seed(config.seed)
    torch.cuda.empty_cache()



    print('#----------Preparing dataset----------#')
    # # stare
    mean = (0.79472653, 0.42516014, 0.1087752)
    std = (0.17206669, 0.11271852, 0.06384674)
    mean1 = (0.80197152, 0.44979242, 0.13198617)
    std1 = (0.16273552, 0.10759267, 0.04646922)

    # # chase
    # mean = (0.44191112, 0.1608891,  0.02802536)
    # std = (0.33401432, 0.13775645, 0.03543717)
    # mean1 = (0.48036714, 0.17151246, 0.02771437)
    # std1 = (0.35466849, 0.14648287, 0.03462432)

#fives
    # mean = (0.33796545, 0.15286063, 0.06382662)
    # std = (0.21406654, 0.10442075, 0.04361498)
    # mean1 = (0.32846952, 0.15863456, 0.07484341)
    # std1 = (0.21214407, 0.10909516, 0.05035124)


    train_dataset = STAREDataset(config.data_path,  train=True,
                                 transforms=get_transform(train=True, mean=mean, std=std)) #+
    val_dataset = STAREDataset(config.data_path,  train=False,
                               transforms=get_transform(train=False, mean=mean1, std=std1))

    # train_dataset = Chasedb1Datasets(config.data_path,  train=True,
    #                              transforms=get_transform(train=True, mean=mean, std=std)) #+
    #
    # val_dataset = Chasedb1Datasets(config.data_path,  train=False,
    #                            transforms=get_transform(train=False, mean=mean1, std=std1))

    # train_dataset = FIVESDataset(config.data_path,  train=True,
    #                              transforms=get_transform(train=True, mean=mean, std=std))
    # val_dataset = FIVESDataset(config.data_path,  train=False,
    #                            transforms=get_transform(train=False, mean=mean1, std=std1))

    train_loader = DataLoader(train_dataset,
                                batch_size=config.batch_size,
                                shuffle=True,
                                pin_memory=True,
                                num_workers=config.num_workers,
                                # collate_fn=train_dataset.collate_fn #+
                              )


    val_loader = DataLoader(val_dataset,
                                batch_size=1,
                                shuffle=False,
                                pin_memory=True,
                                num_workers=config.num_workers,
                                drop_last=True,
                                # collate_fn=val_dataset.collate_fn
                            )

    # 打印数据集的大小
    print(f'#----------Training dataset size: {len(train_dataset)}----------#')
    print(f'#----------Validation dataset size: {len(val_dataset)}----------#')

    # 打印训练集图片的大小
    print('#----------Training dataset image sizes----------#')
    for i, (images, _) in enumerate(train_loader):
        print(f"Batch {i}: {images.shape}")
        if i == 0:  # 只打印第一个批次的大小，如果需要可以取消这个条件
            break

    # 打印测试集图片的大小
    print('#----------Validation dataset image sizes----------#')
    for i, (images, _) in enumerate(val_loader):
        print(f"Batch {i}: {images.shape}")
        if i == 0:  # 只打印第一个批次的大小，如果需要可以取消这个条件
            break


    print('#----------Prepareing Model----------#')
    model_cfg = config.model_config
    if config.network == 'egeunet':
        model = EGEUNet(num_classes=model_cfg['num_classes'],
                        input_channels=model_cfg['input_channels'],
                        c_list=model_cfg['c_list'],

                        )
        model = nn.parallel.DataParallel(model)
        model = model.cuda()
    else: raise Exception('network in not right!')

    # model.module = model.module.cuda()
    # input = torch.randn(4, 3, 400, 400).cuda()
    # flops, params = profile(model.module, inputs=(input,))
    # print(f'FLOPs = {flops / 1e9:.2f} G')
    # print(f'Params = {params / 1e6:.2f} M')

    # summary(model, input_size=(1,3, 400, 400))  # 输入图像尺寸


    print('#----------Prepareing loss, opt, sch and amp----------#')

    optimizer = make_optimizer(model.parameters(), {'name': 'adam','args': {'lr': config.lr}})

    # Configure learning rate scheduler (Remove warmup phase)
    scheduler = CosineAnnealingLR(optimizer, T_max=config.epochs)

    # optimizer = get_optimizer(config, model)
    # scheduler = get_scheduler(config, optimizer)

    # scheduler = get_scheduler(config, optimizer, len(train_loader), config.epochs)



    print('#----------Set other params----------#')
    min_loss = 999
    start_epoch = 1
    min_epoch = 1



    if os.path.exists(resume_model):
        print('#----------Resume Model and Other params----------#')
        checkpoint = torch.load(resume_model, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        saved_epoch = checkpoint['epoch']
        start_epoch += saved_epoch
        min_loss, min_epoch, loss = checkpoint['min_loss'], checkpoint['min_epoch'], checkpoint['loss']

        log_info = f'resuming model from {resume_model}. resume_epoch: {saved_epoch}, min_loss: {min_loss:.4f}, min_epoch: {min_epoch}, loss: {loss:.4f}'
        logger.info(log_info)




    step = 0

    # 初始化用于记录训练和验证损失的列表
    train_losses = []
    val_losses = []

    print('#----------Training----------#')
    for epoch in range(start_epoch, config.epochs + 1):

        torch.cuda.empty_cache()

        step, train_loss = train_one_epoch(
            train_loader,
            model,
            optimizer,
            scheduler,
            epoch,
            step,
            logger,
            config,
            writer
        )
        train_losses.append(train_loss)


        # loss = val_one_epoch(
        #         val_loader,
        #         model,
        #         criterion,
        #         epoch,
        #         logger,
        #         config
        #     )

        # 添加条件判断，只有当 epoch 是 50 的倍数时才进行验证
        # if epoch % 10 == 0:
        loss = val_one_epoch(
                val_loader,
                model,
                criterion,
                epoch,
                logger,
                config
            )

        val_losses.append(loss)

        if loss < min_loss:
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, 'best.pth'))
            min_loss = loss
            min_epoch = epoch

        torch.save(
            {
                'epoch': epoch,
                'min_loss': min_loss,
                'min_epoch': min_epoch,
                'loss': loss,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
            }, os.path.join(checkpoint_dir, 'latest.pth'))

    # # 绘制损失变化图
    # plt.figure(figsize=(10, 8))  # 调整图形大小

    # # 绘制训练损失子图
    # plt.subplot(2, 1, 1)
    # plt.plot(train_losses, label='Training Loss')
    # plt.xlabel('Epoch')
    # plt.ylabel('Loss')
    # plt.title('Training Loss')
    # plt.legend()

    # # 绘制验证损失子图
    # plt.subplot(2, 1, 2)
    # plt.plot(val_losses, label='Validation Loss')
    # plt.xlabel('Epoch')
    # plt.ylabel('Loss')
    # plt.title('Validation Loss')
    # plt.legend()

    # # 调整子图布局，防止重叠
    # plt.tight_layout()

    # # 创建保存图像的文件夹
    # if not os.path.exists('loss-result'):
    #     os.makedirs('loss-result')

    # # 保存图像
    # plt.savefig(os.path.join('loss-result', 'loss_plot.png'))
    # plt.close()

    # 绘制损失变化图
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()

    # 创建保存图像的文件夹
    if not os.path.exists('loss-result'):
        os.makedirs('loss-result')

    # 保存图像
    plt.savefig(os.path.join('loss-result', 'loss_plot.png'))
    plt.close()

    if os.path.exists(os.path.join(checkpoint_dir, 'best.pth')):
        print('#----------Testing----------#')
        best_weight = torch.load(config.work_dir + 'checkpoints/best.pth', map_location=torch.device('cpu'))
        model.load_state_dict(best_weight)
        loss = test_one_epoch(
                val_loader,
                model,
                criterion,
                logger,
                config,
            )
        os.rename(
            os.path.join(checkpoint_dir, 'best.pth'),
            os.path.join(checkpoint_dir, f'best-epoch{min_epoch}-loss{min_loss:.4f}.pth')
        )


if __name__ == '__main__':
    config = setting_config
    main(config)