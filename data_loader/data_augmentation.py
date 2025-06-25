# coding=utf-8

import numpy as np

##########################################################
# name:     DataAugmenters
# breif:
#
# usage:
##########################################################
class DataAugmenters:
    def __init__(self, config):
        self.config = config
        self.augmentation =  self._aug_albumentations()

    def _example(self,image,**kwargs):
        """
        example
        :param image: 
        :param kwargs: 
        :return: 
        """
        return image


    def run(self,image,**kwargs):
        """
        augment your image data
        :param image: 
        :param kwargs: 
        :return: 
        """
        data = {'image': image}
        augmented = self.augmentation(**data)
        return augmented['image']


    def _aug_albumentations(self):
        p = 0.9
        from albumentations import (
            HorizontalFlip, ShiftScaleRotate, CLAHE, RandomRotate90,
            Transpose, ShiftScaleRotate, Blur, OpticalDistortion, GridDistortion, HueSaturationValue,
            GaussNoise, MotionBlur, MedianBlur, PiecewiseAffine,
            Sharpen, Emboss, RandomBrightnessContrast, Flip, OneOf, Compose, ImageCompression,
            DualTransform)
        return Compose([
            RandomRotate90(),
            Flip(),
            Transpose(),
            OneOf([
                GaussNoise(),
            ], p=0.2),
            OneOf([
                MotionBlur(p=0.2),
                MedianBlur(blur_limit=3, p=0.1),
                Blur(blur_limit=3, p=0.1),
            ], p=0.2),
            ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, p=0.2),
            OneOf([
                OpticalDistortion(p=0.3),
                GridDistortion(p=0.1),
                PiecewiseAffine(p=0.3),
            ], p=0.2),
            OneOf([
                CLAHE(clip_limit=2),
                Sharpen(),
                Emboss(),
                RandomBrightnessContrast(),
            ], p=0.3),
            HueSaturationValue(p=0.3),
        ], p=p)


if __name__ == '__main__':
    print('done!')
