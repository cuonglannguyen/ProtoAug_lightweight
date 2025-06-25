# -*- coding: utf-8 -*-

import os
import cv2
import copy
import numpy as np
import torch
from torch.autograd import Variable
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from data_loader.data_processor import DataProcessor
from PIL import Image
import random
from PIL import ImageFilter, ImageOps
import torchvision.transforms.v2 as transforms_v2
from torchvision.transforms import AugMix

CLIP_NORM_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_NORM_STD = (0.26862954, 0.26130258, 0.27577711)
mean=(0.485, 0.456, 0.406)
std= (0.229, 0.224, 0.225)
class PyTorchDataset(Dataset):
    def __init__(self, txt, config, transform=None, loader = None,
                 target_transform=None,  is_train_set=True):
        self.config = config
        imgs = []
        with open(txt,'r') as f:
            for line in f:
                line = line.strip('\n\r').strip('\n').strip('\r')
                words = line.split(self.config['file_label_separator'])
                # single label here so we use int(words[1])
                imgs.append((words[0], int(words[1])))

        self.DataProcessor = DataProcessor(self.config)
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        self.is_train_set = is_train_set


    def __getitem__(self, index):
        fn, label = self.imgs[index]
        _root_dir = self.config['train_data_root_dir'] if self.is_train_set else self.config['val_data_root_dir']
        #print(os.path.join(_root_dir, fn))
        image = self.self_defined_loader(os.path.join(_root_dir, fn))
        #image = Image.fromarray(image.astype(np.float32))

        if self.transform is not None:
            image = self.transform(image)

        return image, label


    def __len__(self):
        return len(self.imgs)


    def self_defined_loader(self, filename):
        image = self.DataProcessor.image_loader(filename)
        #image = self.DataProcessor.image_resize(image)
        #if self.is_train_set and self.config['data_aug']:
        #    image = self.DataProcessor.data_aug(image)
        #image = self.DataProcessor.input_norm(image)
        #image = image.astype(np.float32)
        return image


def get_data_loader(config):
    """
    
    :param config: 
    :return: 
    """
    train_data_file = config['train_data_file']
    test_data_file = config['val_data_file']
    batch_size = config['batch_size']
    num_workers =config['dataloader_workers']
    shuffle = config['shuffle']
    augmix = config['augmix']
    clip = config['clip']
    module = config['model_module_name']
    augmix = config['augmix']
    cutmix = config['cutmix']
    mixup = config['mixup']
    num_classes = config['num_classes']
    if not os.path.isfile(train_data_file):
        raise ValueError('train_data_file is not existed')
    if not os.path.isfile(test_data_file):
        raise ValueError('val_data_file is not existed')
    MEAN = [0.49139968, 0.48215841, 0.44653091]
    STD  = [0.24703223, 0.24348513, 0.26158784]
    aux_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply(
            [
                transforms.ColorJitter(
                    brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1
                )
            ],
            p=0.8,
        ),
        transforms.RandomGrayscale(p=0.2),
        GaussianBlur(0.2),
        Solarization(0.2),
    ])
    if clip:
        transform_train = transforms.Compose([
        transforms.Lambda(lambda x: x.convert("RGB")),
        transforms.RandAugment(),
        transforms.RandomResizedCrop(
            224, 
            scale=(0.25, 1.0), 
            interpolation=transforms.InterpolationMode.BICUBIC,
            antialias=None,
        ),
        aux_transform,
        transforms.ToTensor(),
        transforms.Normalize(CLIP_NORM_MEAN, CLIP_NORM_STD)
    ])
    elif module == 'resnet_module':
        transform_train = transforms.Compose([
                                            transforms.RandomResizedCrop(size=224, scale=(0.5, 1), interpolation=transforms.InterpolationMode.BICUBIC),
                                            transforms.RandomHorizontalFlip(p=0.5),
                                            transforms.ToTensor(),
                                            transforms.Normalize(mean=mean, std=std)
                                        ])
    else:
        transform_train = transforms.Compose([

                                transforms.ToTensor(),
                                transforms.Normalize(MEAN, STD),
                                transforms.RandomCrop(32, padding=4),
                                transforms.RandomHorizontalFlip(), # FLips the image w.r.t horizontal axis

                        ])
    if clip:
        transform_test = transforms.Compose([
        transforms.Lambda(lambda x: x.convert("RGB")),
        transforms.Resize(
            224, 
            interpolation=transforms.functional.InterpolationMode.BICUBIC
        ),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(CLIP_NORM_MEAN, CLIP_NORM_STD)
    ])
    elif module == 'resnet_module':
        transform_test = transforms.Compose([
                                        transforms.Resize((224, 224)),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=mean, std=std)
                                        ])

    else:
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ])
    train_data = PyTorchDataset(txt=train_data_file,config=config,
                           transform=transform_train, is_train_set=True)
    test_data = PyTorchDataset(txt=test_data_file,config=config,
                                transform=transform_test, is_train_set=False)
    train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=shuffle,
                             num_workers=num_workers, pin_memory = True)
    test_loader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory = True)

    return train_loader, test_loader

class GaussianBlur(object):
    def __init__(self, p=0.5, radius_min=0.1, radius_max=2.0):
        self.p = p
        self.radius_min = radius_min
        self.radius_max = radius_max

    def __repr__(self):
        return "{}(p={}, radius_min={}, radius_max={})".format(
            self.__class__.__name__, self.p, self.radius_min, self.radius_max
        )

    def __call__(self, img):
        if random.random() <= self.p:
            radius = random.uniform(self.radius_min, self.radius_max)
            return img.filter(ImageFilter.GaussianBlur(radius=radius))
        else:
            return img


class Solarization(object):
    def __init__(self, p):
        self.p = p

    def __repr__(self):
        return "{}(p={})".format(self.__class__.__name__, self.p)

    def __call__(self, img):
        if random.random() < self.p:
            return ImageOps.solarize(img)
        else:
            return img

