from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import os
from PIL import Image
import cv2
import os.path as osp


import torch
from torch.utils.data import Dataset


def read_image(img_path, box):
    """Keep reading image until succeed.
    This can avoid IOError incurred by heavy IO process."""
    got_img = False
    if not osp.exists(img_path):
        raise IOError("{} does not exist".format(img_path))
    while not got_img:
        try:
            img = Image.open(img_path).convert('RGB')
            img_crop = img.crop(box)
            got_img = True
        except IOError:
            print("IOError incurred when reading '{}'. Will redo. Don't worry. Just chill.".format(img_path))
            pass
    return img_crop


class ImageDataset(Dataset):
    """Image Person ReID Dataset"""
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img_path , position1, position2, gender, staff, customer, stand, sit, play_with_phone = self.dataset[index]
        box = (position1[0], position1[1], position2[0], position2[1])
        img = read_image(img_path, box)
        # cv2.imshow(img)
        if self.transform is not None:
            img = self.transform(img)

        return img, gender, staff, customer, stand, sit, play_with_phone


class ImageDataset_demo(Dataset):
    """Image Person ReID Dataset"""

    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return 1

    def __getitem__(self, index):
        img_path = self.img_dir
        img = read_image(img_path)
        if self.transform is not None:
            img = self.transform(img)

        return img

