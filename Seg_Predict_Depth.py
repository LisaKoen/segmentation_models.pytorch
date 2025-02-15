import torch
from PIL import Image
import numpy as np
import os
import Seg_Predict
import cv2
from torchvision import transforms

import data_segmented_SM as ds
import segmentation_models_pytorch as smp
from numpy import asarray

import gzip     # for saving mask file
from os import listdir
from os.path import isfile, join
import Seg_util as utils


def getcoloredMask(image, mask):
    color_mask = np.zeros_like(image)
    color_mask[:, :, 1] += (mask*250).astype('uint8')
    masked = cv2.addWeighted(image, 0.5, color_mask, 1.0, 0.0)
    return masked
#
# def get_ground_truth_and_predict(image, mask_gt, original_size):
#     postprocess = transforms.Compose([transforms.Resize(original_size)])
#     image_gt = postprocess(image)
#     mask_gt = postprocess(mask_gt)
#     image_gt = image_gt.numpy().transpose(1, 2, 0)
#     msk_gt = mask_gt.numpy().transpose(1, 2, 0)
#
#     with torch.no_grad():
#         model.eval()
#         if torch.cuda.is_available():
#             dev = "cuda:0"
#         else:
#             dev = "cpu"
#         device = torch.device(dev)
#         model.to(device)
#         image = image.to(device).float()
#         msk_pred = model(image.unsqueeze(0)).float()
#
#         ytest = msk_pred[0, 0, :, :].clone().detach().cpu().numpy()
#
#     ytest = ytest.astype('float32')
#
#     kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
#     ytest = cv2.dilate(ytest, kernel, iterations=1)
#     ytest = cv2.morphologyEx(ytest, cv2.MORPH_CLOSE, kernel)
#     ytest = cv2.morphologyEx(ytest, cv2.MORPH_OPEN, kernel)
#
#     ytest = Image.fromarray((ytest * 255).astype(np.uint8), 'L')
#     ytest = postprocess(ytest)
#     ytest = np.array(ytest)/255
#     ytest = ytest.astype('float32')
#
#     msk_gt = msk_gt.squeeze(2)
#     msk_gt = msk_gt.astype('float32')
#
#
#     colmask_gt = getcoloredMask(image_gt, msk_gt)
#     colmask_pred = getcoloredMask(image_gt, ytest)
#
#
#     gt_pred = np.hstack((colmask_gt, colmask_pred))
#     return gt_pred
#
def get_predict(image, original_size):
    postprocess = transforms.Compose([transforms.Resize(original_size)])
    image_gt = postprocess(image)
    image_gt = image_gt.numpy().transpose(1, 2, 0)


    with torch.no_grad():
        model.eval()
        if torch.cuda.is_available():
            dev = "cuda:0"
        else:
            dev = "cpu"
        device = torch.device(dev)
        model.to(device)
        image = image.to(device).float()
        msk_pred = model(image.unsqueeze(0))
        msk_pred = msk_pred.float()

        ytest = msk_pred[0, 0, :, :].clone().detach().cpu().numpy()

    ytest = ytest.astype('float32')

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    # ytest = cv2.dilate(ytest, kernel, iterations=1)
    # ytest = cv2.morphologyEx(ytest, cv2.MORPH_CLOSE, kernel)
    # ytest = cv2.morphologyEx(ytest, cv2.MORPH_OPEN, kernel)

    ytest = Image.fromarray((ytest * 255).astype(np.uint8), 'L')
    ytest = postprocess(ytest)
    ytest = np.array(ytest)
    ytest = (ytest/255).astype('float32')


    colmask_pred = getcoloredMask(image_gt, ytest)
    ytest_max = ytest.max()
    if ytest_max == 0:
        ytest_max = 1e-6
    mask = ytest/ytest_max

    return mask, colmask_pred, ytest

if __name__ == '__main__':

    # Model, loss, optimizer
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    ENCODER = "vgg19_bn"
    ENCODER_WEIGHTS = 'imagenet'
    CLASSES = ['hand']
    ACTIVATION = 'sigmoid'  # could be None for logits or 'softmax2d' for multiclass segmentation
    DEVICE = 'cuda'
    in_channels = 3

    # create segmentation model with pretrained encoder
    model = smp.UnetPlusPlus(
        encoder_name=ENCODER,
        encoder_weights=ENCODER_WEIGHTS,
        classes=len(CLASSES),
        activation=ACTIVATION,
        in_channels=in_channels
    )

    preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)

    # path_to_images = r"C:\Users\lisak\NG\segmentation\hand_bigger\data"
    # path_to_annotations = r"C:\Users\lisak\NG\segmentation\hand_bigger\labels\labels.json"   # for cocodataset
    path_to_images = r"C:\Users\lisak\NG\segmentation\images_dec\Images6"
    # path_to_images = r"C:\Users\lisak\NG\segmentation\finger_marker\test_set_depthC:\Users\lisak\NG\segmentation\finger_marker\test_set_depth"
    path_to_images_depth = r"C:\Users\lisak\NG\segmentation\images_dec\depth_images2"
    # path_to_images = r"C:\Users\lisak\NG\segmentation\finger_marker\data"
    path_to_annotations = None      # for test images only, if no cocodataset
    # path_to_annotations = r"C:\Users\lisak\NG\segmentation\finger_marker\labels\labels.json"

    # path_to_images = r"C:\Users\lisak\NG\segmentation\finger\data"
    # path_to_annotations = r"C:\Users\lisak\NG\segmentation\finger\labels\labels.json"
    image_files = [f for f in listdir(path_to_images) if isfile(join(path_to_images, f))]
    depth_files = [f for f in listdir(path_to_images) if isfile(join(path_to_images_depth, f))]

    # dataset = ds.Dataset_SM(path_to_images, path_to_annotations, classes=CLASSES,
    #                         preprocessing=ds.get_preprocessing(preprocessing_fn), trainstate=False)

    # path_to_weights = r"C:\Users\lisak\NG\segmentation\finger\checkpoints\SM\UnetPlusPlus_vgg19_bn\Transfer\Best_Weights\best_checkpoint.pt"
    path_to_weights = r"C:\Users\lisak\NG\segmentation\finger+finger_marker\checkpoints_2\Best_Weights\checkpoint35.pt"
    ENCODER_WEIGHTS = utils.get_state_dict(path_to_weights)
    model.load_state_dict(ENCODER_WEIGHTS)
    # filenames = next(os.walk(path_to_images))[2]
    # filenames = os.listdir(path_to_images)
    # filenames.sort()

    # path_to_images = r"C:\Users\lisak\NG\segmentation\finger\data\validation_images"
    dataset = ds.Dataset_SM(path_to_images, path_to_annotations, classes=CLASSES,
                            preprocessing=ds.get_preprocessing_eval(preprocessing_fn), trainstate=False, evalstate=True, image_files= image_files, greyscale=False)
    # t1 = image_files[18750]
    save_path = r"C:\Users\lisak\NG\segmentation\images_dec\depth_overlay"
    # for training data to get ground truth overlay and prediction overlayC:\Users\lisak\NG\segmentation\images_dec\depth_overlay
    #
    #     image, mask_gt, original_size, filename = dataset[data]
    #     image = torch.from_numpy(image)
    #     mask_gt = torch.from_numpy(mask_gt)
    #     mask_gt = torch.unsqueeze(mask_gt, 0)
    #
    #     gt_pred = get_ground_truth_and_predict(image, mask_gt, original_size)
    #
    #     # imshow(torch.from_numpy(gt_pred.transpose(2, 0, 1)))
    #
    #     save_path = r"C:\Users\lisak\NG\segmentation\hand_bigger\predictions\SM\UnetPlusPlus_vgg19_bn"
    #     # save_path = r"C:\Users\lisak\NG\segmentation\finger\predictions\SM\UnetPlusPlus_vgg19_bn"
    #     mean = np.array([0.485, 0.456, 0.406])
    #     std = np.array([0.229, 0.224, 0.225])
    #     gt_pred = std * gt_pred + mean
    #     gt_pred = np.clip(gt_pred, 0, 1)
    #
    #     gt_pred = Image.fromarray((gt_pred*255).astype(np.uint8))
    #
    #
    #     try:
    #         filename = re.search('6/(.+?).jpg', filename).group(1)
    #     except AttributeError:
    #         # AAA, ZZZ not found in the original string
    #         filename = 'error in filename'  # apply your error handling
    #
    #     gt_pred.save(str(save_path + '/' + filename + "_gt_pred.png"))
    # for test data to get26346 just prediction overlay filenames from coco dataset or from paths

    for data in range(len(dataset)):

        # image, image_RGB, original_size, filename = dataset[data]
        image, original_size, filename = dataset[data]
        image = torch.from_numpy(image)
        # filename_depth = filename.rsplit('-', 1)[1]
        # filename_depth = filename_depth.rsplit('.', 1)[0]
        # print(data)

        image_path = str(path_to_images_depth + '/' + filename + '.png')
        image_depth = Image.open(image_path)
        image_depth = asarray(image_depth)
        # image_depth = torch.from_numpy(image_depth)


        # mask, overlay = get_predict_depth(image, original_size, image_depth)

        mask, overlay, ytest = get_predict(image, original_size)
        overlay = getcoloredMask(image_depth, ytest)





        # save_path = r"C:\Users\lisak\NG\segmentation\finger_marker\predictions\validation\test""

        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])

        # not needed for depth pictures
        # overlay = std * overlay + mean
        # overlay = np.clip(overlay, 0, 1)

        # overlay = Image.fromarray((overlay * 255).astype(np.uint8))
        overlay = Image.fromarray(overlay.astype(np.uint8))     # for depth pictures


        # try:
        #     filename = re.search('6/(.+?).jpg', filename).group(1)
        # except AttributeError:
        #     # AAA, ZZZ not found in the original string
        #     filename = 'error in filename'  # apply your error handling

        overlay.save(str(save_path + '/overlay/' + filename + '_overlay.png'))
        #
        # mask_filename = str(save_path + '/gz/' + filename + '_mask.npy.gz')
        #
        #
        #
        #
        #
        #
        #
        # with gzip.GzipFile(mask_filename, "wb") as f:
        #     np.save(f, mask)


