import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision.transforms.functional as TF
import torchvision.transforms as transforms
from torch.optim import SGD
import numpy as np
import os
import math
from torch.optim.lr_scheduler import LambdaLR
import random
import logging
import logging.handlers
from matplotlib import pyplot as plt
import cv2
from torchvision.transforms.functional import to_pil_image, to_tensor
from Adam import Adam

def set_seed(seed):
    # for hash
    os.environ['PYTHONHASHSEED'] = str(seed)
    # for python and numpy
    random.seed(seed)
    np.random.seed(seed)
    # for cpu gpu
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # for cudnn
    cudnn.benchmark = False
    cudnn.deterministic = True


def get_logger(name, log_dir):
    '''
    Args:
        name(str): name of logger
        log_dir(str): path of log
    '''

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    info_name = os.path.join(log_dir, '{}.info.log'.format(name))
    info_handler = logging.handlers.TimedRotatingFileHandler(info_name,
                                                             when='D',
                                                             encoding='utf-8')
    info_handler.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(message)s',
                                  datefmt='%Y-%m-%d %H:%M:%S')

    info_handler.setFormatter(formatter)

    logger.addHandler(info_handler)

    return logger


def log_config_info(config, logger):
    config_dict = config.__dict__
    log_info = f'#----------Config info----------#'
    logger.info(log_info)
    for k, v in config_dict.items():
        if k[0] == '_':
            continue
        else:
            log_info = f'{k}: {v},'
            logger.info(log_info)


def get_optimizer(config, model):
    assert config.opt in ['Adadelta', 'Adagrad', 'Adam', 'AdamW', 'Adamax', 'ASGD', 'RMSprop', 'Rprop',
                          'SGD'], 'Unsupported optimizer!'

    if config.opt == 'Adadelta':
        return torch.optim.Adadelta(
            model.parameters(),
            lr=config.lr,
            rho=config.rho,
            eps=config.eps,
            weight_decay=config.weight_decay
        )
    elif config.opt == 'Adagrad':
        return torch.optim.Adagrad(
            model.parameters(),
            lr=config.lr,
            lr_decay=config.lr_decay,
            eps=config.eps,
            weight_decay=config.weight_decay
        )
    elif config.opt == 'Adam':
        return torch.optim.Adam(
            model.parameters(),
            lr=config.lr,
            betas=config.betas,
            eps=config.eps,
            weight_decay=config.weight_decay,
            amsgrad=config.amsgrad
        )
    elif config.opt == 'AdamW':
        return torch.optim.AdamW(
            model.parameters(),
            lr=config.lr,
            betas=config.betas,
            eps=config.eps,
            weight_decay=config.weight_decay,
            amsgrad=config.amsgrad
        )
    elif config.opt == 'Adamax':
        return torch.optim.Adamax(
            model.parameters(),
            lr=config.lr,
            betas=config.betas,
            eps=config.eps,
            weight_decay=config.weight_decay
        )

    else:  # default opt is SGD
        return torch.optim.SGD(
            model.parameters(),
            lr=0.01,
            momentum=0.9,
            weight_decay=0.05,
        )

def create_lr_scheduler(optimizer, num_step: int, epochs: int, warmup=True, warmup_epochs=50, warmup_factor=1e-3):
    assert num_step > 0 and epochs > 0
    if warmup is False:
        warmup_epochs = 0

    def f(x):
        """根据step数返回一个学习率倍率因子，注意在训练开始之前，pytorch会提前调用一次lr_scheduler.step()方法 """
        if warmup is True and x <= (warmup_epochs * num_step):
            alpha = float(x) / (warmup_epochs * num_step)
            # warmup过程中lr倍率因子从warmup_factor -> 1
            return warmup_factor * (1 - alpha) + alpha
        else:
            # warmup后lr倍率因子从1 -> 0
            # 参考deeplab_v2: Learning rate policy
            return (1 - (x - warmup_epochs * num_step) / ((epochs - warmup_epochs) * num_step)) #** 0.9
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=f)


def get_scheduler(config, optimizer):
    assert config.sch in ['StepLR', 'MultiStepLR', 'ExponentialLR', 'CosineAnnealingLR', 'ReduceLROnPlateau',
                          'CosineAnnealingWarmRestarts', 'WP_MultiStepLR', 'WP_CosineLR'], 'Unsupported scheduler!'
    if config.sch == 'StepLR':
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=config.step_size,
            gamma=config.gamma,
            last_epoch=config.last_epoch
        )
    elif config.sch == 'MultiStepLR':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=config.milestones,
            gamma=config.gamma,
            last_epoch=config.last_epoch
        )
    elif config.sch == 'ExponentialLR':
        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer,
            gamma=config.gamma,
            last_epoch=config.last_epoch
        )
    elif config.sch == 'CosineAnnealingLR':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config.T_max,
            eta_min=config.eta_min,
            last_epoch=config.last_epoch
        )

    elif config.sch == 'WP_CosineLR':
        lr_func = lambda epoch: epoch / config.warm_up_epochs if epoch <= config.warm_up_epochs else 0.5 * (
                math.cos((epoch - config.warm_up_epochs) / (config.epochs - config.warm_up_epochs) * math.pi) + 1)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_func)

    return scheduler


# def get_scheduler(config, optimizer, num_step, epochs):
#     assert config.sch in ['StepLR', 'MultiStepLR', 'ExponentialLR', 'CosineAnnealingLR', 'ReduceLROnPlateau',
#                           'CosineAnnealingWarmRestarts', 'WP_MultiStepLR', 'WP_CosineLR','CustomLR'], 'Unsupported scheduler!'
#     if config.sch == 'CustomLR':
#         return create_lr_scheduler(optimizer, num_step, epochs, warmup=True, warmup_epochs=1, warmup_factor=1e-3)


def save_imgs(img, msk, msk_pred, i, save_path, datasets, threshold=0.5, test_data_name=None):
    img = img.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
    img = img / 255. if img.max() > 1.1 else img
    if datasets == 'retinal':
        msk = np.squeeze(msk, axis=0)
        msk_pred = np.squeeze(msk_pred, axis=0)
    else:
        msk = np.where(np.squeeze(msk, axis=0) > 0.5, 1, 0)
        msk_pred = np.where(np.squeeze(msk_pred, axis=0) > threshold, 1, 0)

    plt.figure(figsize=(7, 15))

    plt.subplot(3, 1, 1)
    plt.imshow(img)
    plt.axis('off')

    plt.subplot(3, 1, 2)
    plt.imshow(msk, cmap='gray')
    plt.axis('off')

    plt.subplot(3, 1, 3)
    plt.imshow(msk_pred, cmap='gray')
    plt.axis('off')

    if test_data_name is not None:
        save_path = save_path + test_data_name + '_'
    plt.savefig(save_path + str(i) + '.png')
    plt.close()


class BCELoss(nn.Module):
    def __init__(self):
        super(BCELoss, self).__init__()
        self.bceloss = nn.BCELoss()

    def forward(self, pred, target):

        # print(f"Prediction size: {pred.size()}")
        # print(f"Target size: {target.size()}")

        size = pred.size(0)
        pred_ = pred.view(size, -1)
        # target_ = target.view(size, -1)
        target_ = target.reshape(size, -1)

        return self.bceloss(pred_, target_)

def make_optimizer(param_list, optimizer_spec, load_sd=False):
    Optimizer = {
        'sgd': SGD,
        'adam': Adam
    }[optimizer_spec['name']]
    optimizer = Optimizer(param_list, **optimizer_spec['args'])
    if load_sd:
        optimizer.load_state_dict(optimizer_spec['sd'])
    return optimizer

class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, pred, target):
        smooth = 1
        size = pred.size(0)

        pred_ = pred.view(size, -1)
        # target_ = target.view(size, -1)
        target_ = target.reshape(size, -1)

        intersection = pred_ * target_
        dice_score = (2 * intersection.sum(1) + smooth) / (pred_.sum(1) + target_.sum(1) + smooth)
        dice_loss = 1 - dice_score.sum() / size

        return dice_loss


class BceDiceLoss(nn.Module):
    def __init__(self, wb=1, wd=1):
        super(BceDiceLoss, self).__init__()
        self.bce = BCELoss()
        self.dice = DiceLoss()
        self.wb = wb
        self.wd = wd

    def forward(self, pred, target):
        bceloss = self.bce(pred, target)
        diceloss = self.dice(pred, target)

        loss = self.wd * diceloss + self.wb * bceloss
        return loss

def dice_loss(x: torch.Tensor, target: torch.Tensor, epsilon=1e-6):
    # Average of Dice coefficient for all batches, or for a single mask
    # 计算一个batch中所有图片某个类别的dice_coefficient
    d = 0.
    batch_size = x.shape[0]
    for i in range(batch_size):
        x_i = x[i].reshape(-1)
        t_i = target[i].reshape(-1).float()
        inter = torch.dot(x_i, t_i)
        sets_sum = torch.sum(x_i) + torch.sum(t_i)
        if sets_sum == 0:
            sets_sum = 2 * inter
        d += (2 * inter + epsilon) / (sets_sum + epsilon)
    return 1 - d / batch_size

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        # 计算交叉熵损失
        ce_loss = F.binary_cross_entropy(inputs, targets, reduction='none')

        # 计算Focal Loss
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        return torch.mean(focal_loss)

def criterion(inputs, target, dice: bool = True, bce: bool = True, focal: bool = True):
    loss1 = 0
    if dice:
        loss1 = dice_loss(inputs, target)
    loss2 = 0
    target = target.unsqueeze(1).float()
    if focal:
        loss_fn = FocalLoss(alpha=0.25, gamma=2)
        loss2 = loss_fn(inputs, target)
    loss3 = 0
    if bce:
        loss3 = nn.BCELoss()(inputs, target)  # 二元交叉熵
    return 0.2 * loss1 + loss2 + loss3


class myToTensor:
    def __init__(self):
        pass

    def __call__(self, data):
        image, mask = data
        return torch.tensor(image).permute(2, 0, 1), torch.tensor(mask).permute(2, 0, 1)


class myResize:
    def __init__(self, size_h=256, size_w=256):
        self.size_h = size_h
        self.size_w = size_w

    def __call__(self, data):
        image, mask = data
        return TF.resize(image, [self.size_h, self.size_w]), TF.resize(mask, [self.size_h, self.size_w])


class myRandomHorizontalFlip:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, data):
        image, mask = data
        if random.random() < self.p:
            return TF.hflip(image), TF.hflip(mask)
        else:
            return image, mask


class myRandomVerticalFlip:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, data):
        image, mask = data
        if random.random() < self.p:
            return TF.vflip(image), TF.vflip(mask)
        else:
            return image, mask

class myRandomRotation:
    def __init__(self, p=0.5, degree=[0, 360]):
        self.angle = random.uniform(degree[0], degree[1])
        self.p = p

    def __call__(self, data):
        image, mask = data
        if random.random() < self.p:
            return TF.rotate(image, self.angle), TF.rotate(mask, self.angle)
        else:
            return image, mask

# class myNormalize:
#     def __init__(self, data_name, train=True):
#         if data_name == 'isic18':
#             if train:
#                 self.mean = 157.561
#                 self.std = 26.706
#         elif data_name == 'isic18_82':
#             if train:
#                 self.mean = 156.2899
#                 self.std = 26.5457
#         # else :drive
#         #         self.mean = 237.2739
#         #         self.std = 164.7371
#
#         else :#stare
#             self.mean = 261.3473
#             self.std = 163.5453

        # else:
        # #     chase
        #     self.mean = 160.8604
        #     self.std = 135.457

        # else:
        # # hrf
        #     self.mean = 233.2453
        #     self.std = 116.9573
        #
        # print('mean=', self.mean)
        # print('std=', self.std)

    # def __call__(self, data):
    #     img, msk = data
    #     img_normalized = (img - self.mean) / self.std
    #     img_normalized = ((img_normalized - np.min(img_normalized))
    #                       / (np.max(img_normalized) - np.min(img_normalized))) * 255.
    #     return img_normalized, msk


class myNormalize:
    def __init__(self, data_name, train=True):
        if data_name == 'stare':
            if train:
                self.mean = [0.79472653, 0.42516014, 0.1087752 ]
                self.std = [0.17206669, 0.11271852, 0.06384674]
            else:
                self.mean = [0.80197152, 0.44979242, 0.13198617]
                self.std = [0.16273552, 0.10759267, 0.04646922]
        else:
            raise ValueError(f"Unsupported data name: {data_name}")

    def __call__(self, data):
        image, target = data
        # print(f"image 的类型: {type(image)}")
        # print(f"Before normalization, image shape: {image.shape}")

        if isinstance(image, torch.Tensor):
            image = image.float()
        else:
            try:
                image = torch.from_numpy(image).float()
            except TypeError:
                print(f"Error: Expected numpy.ndarray or torch.Tensor, got {type(image)}")
                raise

        if image.ndim == 3:
            if image.shape[0] not in [1, 3, 4]:  # 常见通道数为 1、3、4
                image = image.permute(2, 0, 1)

        # print(f"After channel permutation, image shape: {image.shape}")
        try:
            image = TF.normalize(image, mean=self.mean, std=self.std)
        except RuntimeError as e:
            print(f"Normalization error: {e}")
            print(f"Image shape: {image.shape}, Mean: {self.mean}, Std: {self.std}")
            raise

        # 检查 target 的类型
        if isinstance(target, torch.Tensor):
            target = target.clone().detach()  # 克隆一份，避免修改原始数据
        else:
            try:
                target = torch.from_numpy(target)
            except TypeError:
                print(f"Error: Expected numpy.ndarray for target, got {type(target)}")
                raise
        target = (target / 255).long()

        # print(f"After normalization, image shape: {image.shape}")
        return image, target

# 直方图校正和对比度调整模块对数据进行增强
class myHistogramEqualization:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, data):
        image, mask = data
        if random.random() < self.p:
            # Convert image to PIL Image for OpenCV compatibility
            pil_image = to_pil_image(image)
            cv_image = np.array(pil_image)

            # Apply histogram equalization
            cv_image_yuv = cv2.cvtColor(cv_image, cv2.COLOR_RGB2YUV)
            cv_image_yuv[:, :, 0] = cv2.equalizeHist(cv_image_yuv[:, :, 0])
            cv_image_equalized = cv2.cvtColor(cv_image_yuv, cv2.COLOR_YUV2RGB)

            # Convert back to tensor
            image_equalized = to_tensor(cv_image_equalized)
            return image_equalized, mask
        else:
            return image, mask


class myContrastAdjustment:
    def __init__(self, p=0.5, alpha=1.5, beta=0):
        self.p = p
        self.alpha = alpha  # Contrast control (1.0-3.0)
        self.beta = beta  # Brightness control (0-100)

    def __call__(self, data):
        image, mask = data
        if random.random() < self.p:
            # Convert image to PIL Image for OpenCV compatibility
            pil_image = to_pil_image(image)
            cv_image = np.array(pil_image)

            # Apply contrast adjustment
            cv_image_contrasted = cv2.convertScaleAbs(cv_image, alpha=self.alpha, beta=self.beta)

            # Convert back to tensor
            image_contrasted = to_tensor(cv_image_contrasted)
            return image_contrasted, mask
        else:
            return image, mask
