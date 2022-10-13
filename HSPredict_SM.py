import torch
from torchvision import models, transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import re
import data_segmented_SM as ds
import segmentation_models_pytorch as smp
import Main_SM




def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose(1,2,0)
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.axis('off')
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(5)  # pause



def getcoloredMask(image, mask):
    color_mask = np.zeros_like(image)
    # color_mask[:, :, 1] += mask.astype('uint8') * 250
    color_mask[:, :, 1] += (mask*250).astype('uint8')
    masked = cv2.addWeighted(image, 1.0, color_mask, 1.0, 0.0)
    return masked

def get_ground_truth_and_predict(image, mask_gt, original_size):
    postprocess = transforms.Compose([transforms.Resize(original_size)])
    image_gt = postprocess(image)
    mask_gt = postprocess(mask_gt)
    image_gt = image_gt.numpy().transpose(1, 2, 0)
    msk_gt = mask_gt.numpy().transpose(1, 2, 0)

    with torch.no_grad():
        model.eval()
        if torch.cuda.is_available():
            dev = "cuda:0"
        else:
            dev = "cpu"
        device = torch.device(dev)
        model.to(device)
        image = image.to(device).float()
        msk_pred = model(image.unsqueeze(0)).float()

        ytest = msk_pred[0, 0, :, :].clone().detach().cpu().numpy()

    ytest = ytest.astype('float32')

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    ytest = cv2.dilate(ytest, kernel, iterations=1)
    ytest = cv2.morphologyEx(ytest, cv2.MORPH_CLOSE, kernel)
    ytest = cv2.morphologyEx(ytest, cv2.MORPH_OPEN, kernel)

    ytest = Image.fromarray((ytest * 255).astype(np.uint8), 'L')
    ytest = postprocess(ytest)
    ytest = np.array(ytest)/255
    ytest = ytest.astype('float32')

    msk_gt = msk_gt.squeeze(2)
    msk_gt = msk_gt.astype('float32')


    colmask_gt = getcoloredMask(image_gt, msk_gt)
    colmask_pred = getcoloredMask(image_gt, ytest)


    gt_pred = np.hstack((colmask_gt, colmask_pred))
    return gt_pred






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

    path_to_images = r"C:\Users\lisak\NG\segmentation\hand_bigger\data"
    path_to_annotations = r"C:\Users\lisak\NG\segmentation\hand_bigger\labels\labels.json"

    # path_to_images = r"C:\Users\lisak\NG\segmentation\finger\data"
    # path_to_annotations = r"C:\Users\lisak\NG\segmentation\finger\labels\labels.json"


    dataset = ds.Dataset_SM(path_to_images, path_to_annotations, classes=CLASSES,
                            preprocessing=ds.get_preprocessing(preprocessing_fn), trainstate=False)

    path_to_weights = r"C:\Users\lisak\NG\segmentation\hand_bigger\checkpoints\SM\UnetPlusPlus_vgg19_bn\Best_Weights\best_checkpoint.pt"
    ENCODER_WEIGHTS = Main_SM.get_state_dict(path_to_weights)
    model.load_state_dict(ENCODER_WEIGHTS)


    for data in range(256):

        image, mask_gt, original_size, filename = dataset[data]
        image = torch.from_numpy(image)
        mask_gt = torch.from_numpy(mask_gt)
        mask_gt = torch.unsqueeze(mask_gt, 0)

        gt_pred = get_ground_truth_and_predict(image, mask_gt, original_size)

        # imshow(torch.from_numpy(gt_pred.transpose(2, 0, 1)))

        save_path = r"C:\Users\lisak\NG\segmentation\hand_bigger\predictions\SM\UnetPlusPlus_vgg19_bn"
        # save_path = r"C:\Users\lisak\NG\segmentation\finger\predictions\SM\UnetPlusPlus_vgg19_bn"
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        gt_pred = std * gt_pred + mean
        gt_pred = np.clip(gt_pred, 0, 1)

        gt_pred = Image.fromarray((gt_pred*255).astype(np.uint8))


        try:
            filename = re.search('6/(.+?).jpg', filename).group(1)
        except AttributeError:
            # AAA, ZZZ not found in the original string
            filename = 'error in filename'  # apply your error handling

        gt_pred.save(str(save_path + '/' + filename + "_gt_pred.png"))





