import torch
from PIL import Image
import numpy as np
from numpy import asarray
from pycocotools.coco import COCO
import pycocotools.mask as m
from skimage.util import random_noise
import albumentations as albu
from torch.utils.data import Dataset as BaseDataset


def get_flip():
    return albu.HorizontalFlip(p=1)

def get_training_augmentation():
    train_transform = [
        albu.GaussNoise(p=0.5),
        albu.Perspective(p=0.5),
        albu.OneOf(
            [
                albu.CLAHE(p=1),
                albu.RandomBrightness(p=1),
                albu.RandomGamma(p=1),
            ],
            p=0.9,
        ),
        albu.OneOf(
            [
                albu.Sharpen(p=1),
                albu.Blur(blur_limit=3, p=1),
                albu.MotionBlur(blur_limit=3, p=1),
            ],
            p=0.9,
        ),
        albu.OneOf(
            [
                albu.RandomBrightnessContrast(p=1),
                albu.HueSaturationValue(p=1),
            ],
            p=0.9,
        ),
    ]
    return albu.Compose(train_transform)


def get_validation_augmentation():
    """Add paddings to make image shape divisible by 32"""
    test_transform = [
        albu.PadIfNeeded(384, 480)
    ]
    return albu.Compose(test_transform)


def to_tensor(x,  **kwargs):
    return x.transpose(2, 0, 1).astype('float32')

def to_tensor_mask(x,  **kwargs):
    return x.astype('float32')

def get_preprocessing(preprocessing_fn):
    """Construct preprocessing transform

    Args:
        preprocessing_fn (callbale): data normalization function
            (can be specific for each pretrained neural network)
    Return:
        transform: albumentations.Compose

    """

    _transform = [
        albu.Lambda(image=preprocessing_fn),
        albu.Lambda(image=to_tensor, mask=to_tensor_mask),
        ]
    return albu.Compose(_transform)

def get_preprocessing_eval(preprocessing_fn):
    """Construct preprocessing transform

    Args:
        preprocessing_fn (callbale): data normalization function
            (can be specific for each pretrained neural network)
    Return:
        transform: albumentations.Compose

    """

    _transform = [
        albu.Lambda(image=preprocessing_fn),
        albu.Lambda(image=to_tensor),
        ]
    return albu.Compose(_transform)

class Dataset_SM(BaseDataset):
    """CamVid Dataset. Read images, apply augmentation and preprocessing transformations.

    Args:
        images_dir (str): path to images folder
        masks_dir (str): path to segmentation masks folder
        class_values (list): values of classes to extract from segmentation mask
        augmentation (albumentations.Compose): data transfromation pipeline
            (e.g. flip, scale, etc.)
        preprocessing (albumentations.Compose): data preprocessing
            (e.g. noralization, shape manipulation, etc.)

    """


    def __init__(self, imageDir, annsfile, classes=None, augmentation=None, preprocessing=None, flip=None, trainstate=True, evalstate=False):
        self.my_coco_dataset = COCO(annotation_file=annsfile)
        self.imageList = self.my_coco_dataset.imgs
        self.imageDir = imageDir
        self.class_values = classes

        self.augmentation = augmentation
        self.preprocessing = preprocessing
        self.flip = flip
        self.trainstate = trainstate
        self.evalstate = evalstate

    def __getitem__(self, index):

        preprocess = albu.Resize(384, 288)
        # # read data
        image_path = str(self.imageDir + '/' + self.imageList[index]['file_name'])
        image = Image.open(image_path)
        image = asarray(image)
        original_size = image.shape
        image = np.stack((image, image, image), axis=-1)

        if self.evalstate == True:
            filename = self.imageList[index]['file_name']
            image = preprocess(image=image)
            if self.preprocessing:
                sample = self.preprocessing(image=image['image'])
                image = sample['image']
            return image, original_size, filename

        anns = self.my_coco_dataset.loadAnns(index)
        rle = self.my_coco_dataset.annToRLE(anns[0])
        msk = m.decode(rle)
        sample = preprocess(image=image, mask=msk)
        image, msk = sample['image'], sample['mask']


        if self.flip:
            sample = self.flip(image=image, mask=msk)
            image, msk = sample['image'], sample['mask']

        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image, mask=msk)
            image, msk = sample['image'], sample['mask']

        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=msk)
            image, msk = sample['image'], sample['mask']
        if self.trainstate == False & self.evalstate == False:
            filename = self.imageList[index]['file_name']
            return image, msk, original_size, filename
        else:
            return image, msk

    def __len__(self):
        return len(self.imageList)