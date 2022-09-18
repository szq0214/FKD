import os

import torch
import torch.distributed
import torch.nn as nn
import torchvision
from torchvision.transforms import functional as t_F
from torchvision.datasets.folder import ImageFolder


class RandomResizedCropWithCoords(torchvision.transforms.RandomResizedCrop):
    def __init__(self, **kwargs):
        super(RandomResizedCropWithCoords, self).__init__(**kwargs)

    def __call__(self, img, coords):
        try:
            reference = (coords.any())
        except:
            reference = False
        if not reference:
            i, j, h, w = self.get_params(img, self.scale, self.ratio)
            coords = (i / img.size[1],
                      j / img.size[0],
                      h / img.size[1],
                      w / img.size[0])
            coords = torch.FloatTensor(coords)
        else:
            i = coords[0].item() * img.size[1]
            j = coords[1].item() * img.size[0]
            h = coords[2].item() * img.size[1]
            w = coords[3].item() * img.size[0]
        return t_F.resized_crop(img, i, j, h, w, self.size,
                                 self.interpolation), coords


class ComposeWithCoords(torchvision.transforms.Compose):
    def __init__(self, **kwargs):
        super(ComposeWithCoords, self).__init__(**kwargs)

    def __call__(self, img, coords, status):
        for t in self.transforms:
            if type(t).__name__ == 'RandomResizedCropWithCoords':
                img, coords = t(img, coords)
            elif type(t).__name__ == 'RandomCropWithCoords':
                img, coords = t(img, coords)
            elif type(t).__name__ == 'RandomHorizontalFlipWithRes':
                img, status = t(img, status)
            else:
                img = t(img)
        return img, status, coords


class RandomHorizontalFlipWithRes(torch.nn.Module):
    """Horizontally flip the given image randomly with a given probability.
    If the image is torch Tensor, it is expected
    to have [..., H, W] shape, where ... means an arbitrary number of leading
    dimensions

    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """

    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, img, status):
        """
        Args:
            img (PIL Image or Tensor): Image to be flipped.

        Returns:
            PIL Image or Tensor: Randomly flipped image.
        """

        if status is not None:
            if status == True:
                return t_F.hflip(img), status
            else:
                return img, status
        else:
            status = False
            if torch.rand(1) < self.p:
                status = True
                return t_F.hflip(img), status
            return img, status


    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)


class ImageFolder_FKD_GSL(torchvision.datasets.ImageFolder):
    def __init__(self, **kwargs):
        self.num_crops = kwargs['num_crops']
        self.save_path = kwargs['save_path']
        self.reference_path = kwargs['reference_path']
        kwargs.pop('num_crops')
        kwargs.pop('save_path')
        kwargs.pop('reference_path')
        super(ImageFolder_FKD_GSL, self).__init__(**kwargs)

    def __getitem__(self, index):
        path, target = self.samples[index]

        if self.reference_path is not None:
            ref_path = os.path.join(self.reference_path,'/'.join(path.split('/')[-4:-1]))
            ref_filename = os.path.join(ref_path,'/'.join(path.split('/')[-1:]).split('.')[0] + '.tar')
            label = torch.load(ref_filename, map_location=torch.device('cpu'))
            coords_ref, flip_ref, _ = label
        else:
            coords_ref = None

        sample = self.loader(path)
        sample_all = [] 
        flip_status_all = []
        coords_all = []
        for i in range(self.num_crops):
            if self.transform is not None:
                if coords_ref is not None:
                    coords_ = coords_ref[i]
                    flip_ = flip_ref[i]
                else:
                    coords_ = None
                    flip_ = None
                sample_new, flip_status, coords_single = self.transform(sample, coords_, flip_)
                sample_all.append(sample_new)
                flip_status_all.append(flip_status)
                coords_all.append(coords_single)
            else:
                coords = None
                flip_status = None
            if self.target_transform is not None:
                target = self.target_transform(target)

        return sample_all, target, flip_status_all, coords_all, path