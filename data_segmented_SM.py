from PIL import Image
import numpy as np
from numpy import asarray
from pycocotools.coco import COCO
import pycocotools.mask as m
import albumentations as albu
from torch.utils.data import Dataset as BaseDataset
from os import listdir


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


def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')


def to_tensor_mask(x, **kwargs):
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

    def __init__(self, imageDir, annsfile, masksDir=None, classes=None, augmentation=None, preprocessing=None, flip=None,
                 trainstate=True, evalstate=False, image_files=None, greyscale=True):
        self.annsfile = annsfile
        self.masksDir = masksDir
        self.my_coco_dataset = COCO(annotation_file=self.annsfile)
        self.imageList = self.my_coco_dataset.imgs
        self.imageDir = imageDir
        self.class_values = classes

        self.augmentation = augmentation
        self.preprocessing = preprocessing
        self.flip = flip
        self.trainstate = trainstate
        self.evalstate = evalstate
        self.image_files = image_files
        self.greyscale = greyscale
        if self.masksDir:
            self.imageList = listdir(self.imageDir)



    def __getitem__(self, index):

        preprocess = albu.Resize(384, 288)
        # # read data
        if self.image_files:
            filename = self.image_files[index]

        elif self.annsfile is not None:
            filename = self.imageList[index]['file_name']

        elif self.masksDir:
            filename = listdir(self.imageDir)[index]

        else:
            filename = str(index) + '.png'  # for case where filenames are like the index

        # image = Image.open(image_path).convert('L')     # for case of RGB files that should be trained as greyscale !!!
        image_path = str(self.imageDir + '/' + filename)
        image = Image.open(image_path)
        image = asarray(image)
        original_size = image.shape
        if self.greyscale:
            image = np.stack((image, image, image), axis=-1)
        else:
            original_size = image.shape[0:2]       # for case of RGB files

        if self.evalstate:
            return image, original_size, filename
        else:
            if self.annsfile is None:
                mask_path = str(self.masksDir + '/' + filename)
                msk = Image.open(mask_path)
                msk = asarray(msk)
            else:
                anns = self.my_coco_dataset.loadAnns(index)
                rle = self.my_coco_dataset.annToRLE(anns[0])
                msk = m.decode(rle)/255
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
            if self.trainstate == False:
                # filename = self.imageList[index]['file_name']
                return image, msk, original_size, filename
            else:
                return image, msk

    def __len__(self):
        return len(self.imageList)
class Dataset_SM_Simple(BaseDataset):

    def __init__(self, imageDir, annsfile, greyscale=True, preprocessing=None):
        self.annsfile = annsfile
        self.my_coco_dataset = COCO(annotation_file=self.annsfile)
        self.imageList = self.my_coco_dataset.imgs
        self.imageDir = imageDir
        self.greyscale = greyscale
        self.preprocessing = preprocessing  # normalizing...


    def __getitem__(self, index):

        # # read data
        if self.annsfile is not None:
            filename = self.imageList[index]['file_name']
            image_path = str(self.imageDir + '/' + filename)
        else:
            image_path = str(self.imageDir + '/' + str(index) + '.png')  # for case where filenames are like the index
        image = Image.open(image_path)
        image = asarray(image)
        if self.greyscale:
            image = np.stack((image, image, image), axis=-1)

        anns = self.my_coco_dataset.loadAnns(index)
        rle = self.my_coco_dataset.annToRLE(anns[0])
        msk = m.decode(rle)
        filename = filename.rsplit('/', 1)[1]

        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=msk)  # to tensor and normalizing !!
            image, msk = sample['image'], sample['mask']
        return image, msk, filename

    def __len__(self):
        return len(self.imageList)
