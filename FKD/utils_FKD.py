import os
import torch
import torch.distributed
import torch.nn as nn
import torchvision
from torchvision.transforms import functional as t_F
from torch.nn import functional as F
from torchvision.datasets.folder import ImageFolder
from torch.nn.modules import loss
from torchvision.transforms import InterpolationMode
import random
import numpy as np


class Soft_CrossEntropy(loss._Loss):
    def forward(self, model_output, soft_output):

        size_average = True

        model_output_log_prob = F.log_softmax(model_output, dim=1)

        soft_output = soft_output.unsqueeze(1)
        model_output_log_prob = model_output_log_prob.unsqueeze(2)

        cross_entropy_loss = -torch.bmm(soft_output, model_output_log_prob)
        if size_average:
             cross_entropy_loss = cross_entropy_loss.mean()
        else:
             cross_entropy_loss = cross_entropy_loss.sum()

        return cross_entropy_loss


class RandomResizedCrop_FKD(torchvision.transforms.RandomResizedCrop):
    def __init__(self, **kwargs):
        super(RandomResizedCrop_FKD, self).__init__(**kwargs)

    def __call__(self, img, coords, status):
        i = coords[0].item() * img.size[1]
        j = coords[1].item() * img.size[0]
        h = coords[2].item() * img.size[1]
        w = coords[3].item() * img.size[0]

        if self.interpolation == 'bilinear':
            inter = InterpolationMode.BILINEAR
        elif self.interpolation == 'bicubic':
            inter = InterpolationMode.BICUBIC
        return t_F.resized_crop(img, i, j, h, w, self.size, inter)


class RandomHorizontalFlip_FKD(torch.nn.Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, img, coords, status):
    
        if status == True:
            return t_F.hflip(img)
        else:
            return img

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)


class Compose_FKD(torchvision.transforms.Compose):
    def __init__(self, **kwargs):
        super(Compose_FKD, self).__init__(**kwargs)

    def __call__(self, img, coords, status):
        for t in self.transforms:
            if type(t).__name__ == 'RandomResizedCrop_FKD':
                img = t(img, coords, status)
            elif type(t).__name__ == 'RandomCrop_FKD':
                img, coords = t(img)
            elif type(t).__name__ == 'RandomHorizontalFlip_FKD':
                img = t(img, coords, status)
            else:
                img = t(img)
        return img


class ImageFolder_FKD(torchvision.datasets.ImageFolder):
    def __init__(self, **kwargs):
        self.num_crops = kwargs['num_crops']
        self.softlabel_path = kwargs['softlabel_path']
        kwargs.pop('num_crops')
        kwargs.pop('softlabel_path')
        super(ImageFolder_FKD, self).__init__(**kwargs)

    def __getitem__(self, index):
            path, target = self.samples[index]

            label_path = os.path.join(self.softlabel_path, '/'.join(path.split('/')[-3:]).split('.')[0] + '.tar')

            label = torch.load(label_path, map_location=torch.device('cpu'))

            coords, flip_status, output = label

            rand_index = torch.randperm(len(output))
            soft_target = []

            sample = self.loader(path)
            sample_all = [] 
            hard_target = []

            for i in range(self.num_crops):
                if self.transform is not None:
                    soft_target.append(output[rand_index[i]])
                    sample_trans = self.transform(sample, coords[rand_index[i]], flip_status[rand_index[i]])
                    sample_all.append(sample_trans)
                    hard_target.append(target)
                else:
                    coords = None
                    flip_status = None
                if self.target_transform is not None:
                    target = self.target_transform(target)

            return sample_all, hard_target, soft_target


def Recover_soft_label(label, label_type, n_classes):
    # recover quantized soft label to n_classes dimension.
    if label_type == 'hard':

        return torch.zeros(label.size(0), n_classes).scatter_(1, label.view(-1, 1), 1)

    elif label_type == 'smoothing':
        index = label[:,0].to(dtype=int)
        value = label[:,1]
        minor_value = (torch.ones_like(value) - value)/(n_classes-1)
        minor_value = minor_value.reshape(-1,1).repeat_interleave(n_classes, dim=1)
        soft_label = (minor_value * torch.ones(index.size(0), n_classes)).scatter_(1, index.view(-1, 1), value.view(-1, 1))

        return soft_label

    elif label_type == 'marginal_smoothing_k5':
        index = label[:,0,:].to(dtype=int)
        value = label[:,1,:]
        minor_value = (torch.ones(label.size(0),1) - torch.sum(value, dim=1, keepdim=True))/(n_classes-5)
        minor_value = minor_value.reshape(-1,1).repeat_interleave(n_classes, dim=1)
        soft_label = (minor_value * torch.ones(index.size(0), n_classes)).scatter_(1, index, value)

        return soft_label

    elif label_type == 'marginal_renorm':
        index = label[:,0,:].to(dtype=int)
        value = label[:,1,:]
        soft_label = torch.zeros(index.size(0), n_classes).scatter_(1, index, value)
        soft_label = F.normalize(soft_label, p=1.0, dim=1, eps=1e-12)

        return soft_label

    elif label_type == 'marginal_smoothing_k10':
        index = label[:,0,:].to(dtype=int)
        value = label[:,1,:]
        minor_value = (torch.ones(label.size(0),1) - torch.sum(value, dim=1, keepdim=True))/(n_classes-10)
        minor_value = minor_value.reshape(-1,1).repeat_interleave(n_classes, dim=1)
        soft_label = (minor_value * torch.ones(index.size(0), n_classes)).scatter_(1, index, value)

        return soft_label


def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


def mixup_cutmix(images, soft_label, args):
    enable_p = np.random.rand(1)
    if enable_p < args.mixup_cutmix_prob:
        switch_p = np.random.rand(1)
        if switch_p < args.mixup_switch_prob:
            lam = np.random.beta(args.mixup, args.mixup)
            rand_index = torch.randperm(images.size()[0]).cuda()
            target_a = soft_label
            target_b = soft_label[rand_index]
            mixed_x = lam * images + (1 - lam) * images[rand_index]
            target_mix = target_a * lam + target_b * (1 - lam)
            return mixed_x, target_mix
        else:
            lam = np.random.beta(args.cutmix, args.cutmix)
            rand_index = torch.randperm(images.size()[0]).cuda()
            target_a = soft_label
            target_b = soft_label[rand_index]
            bbx1, bby1, bbx2, bby2 = rand_bbox(images.size(), lam)
            images[:, :, bbx1:bbx2, bby1:bby2] = images[rand_index, :, bbx1:bbx2, bby1:bby2]
            # adjust lambda to exactly match pixel ratio
            lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (images.size()[-1] * images.size()[-2]))
            target_mix = target_a * lam + target_b * (1 - lam)
    else:
        target_mix = soft_label

    return images, target_mix
