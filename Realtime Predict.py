import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
from torchvision import datasets, models, transforms
import torchvision
import cv2
# from data.utiils import check_models
import HSPredict_SM as predict
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

def make_image_types(image):
    image = cv2.resize(image, (384, 288))
    # rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_img = np.stack((gray_img, gray_img, gray_img), axis=-1)

    gray_img_pil = Image.fromarray(gray_img)
    gray_img_T = data_transforms(gray_img_pil)
    gray_img_T = torch.unsqueeze(gray_img_T, 0)
    return gray_img_T, gray_img

def set_text_in_image(image, text):
    font = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (20, 20)
    fontScale = 1
    fontColor = (255, 255, 255)
    thickness = 2
    lineType = 2
    cv2.putText(image, text,
                bottomLeftCornerOfText,
                font,
                fontScale,
                fontColor,
                thickness,
                lineType)


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
    path_to_weights = r"C:\Users\lisak\NG\segmentation\finger\checkpoints\SM\UnetPlusPlus_vgg19_bn\Transfer\Best_Weights\best_checkpoint.pt"
    ENCODER_WEIGHTS = Main_SM.get_state_dict(path_to_weights)
    model.load_state_dict(ENCODER_WEIGHTS)

    # save_w_path = 'C:/Users/user/Lisa/NG/weights'
    #
    # feature_ext = False
    # # batch_size = 32
    # model_name = 'googlenet'
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # model = check_models(model_name, feature_ext)
    #
    # checkpoint = torch.load(save_w_path + '/' + 'googlenet_Mon_Jul_25_19_51_08_2022.pt')
    # model.load_state_dict(checkpoint['state_dict'])
    #
    # data_transforms = transforms.Compose([transforms.Resize((224, 224)),
    #                                       transforms.ToTensor()])
    #
    # idx2class = {0: 'NoPoint', 1: 'Point'}
    model.eval()
    cap = cv2.VideoCapture(1)
    i = 0
    while True:
        ret, frame = cap.read()
        img_T, img_show = make_image_types(frame)

        # if i % 5 == 0:
        #     i = 0
        #     prediction = predict.get_predict(img_T, original_size)
        #     # with torch.no_grad():
        #     #     y_pred = model(img_T)
        #
        # # y_pred_tags = torch.log_softmax(y_pred, dim=1)
        # # _, y_pred_tag = torch.max(y_pred_tags, dim=1)
        # # point_state = idx2class[y_pred_tag.item()]
        # #
        # # set_text_in_image(img_show, point_state)
        #
        # cv2.imshow('Realtime', prediction)
        # key = cv2.waitKey(24)
        if key & 0xFF == ord('q'):
            break
        i += 1
    cap.release()
    cv2.destroyAllWindows()

