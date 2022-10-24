import torch
from torchvision import models, transforms
from PIL import Image
import numpy as np
import cv2
import os
import segmentation_models_pytorch as smp
import Main_SM

def make_image_types(image):
    image = cv2.resize(image, (384, 288))
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_img = np.stack((gray_img, gray_img, gray_img), axis=-1)

    gray_img_pil = Image.fromarray(gray_img)
    gray_img_T = data_transforms(gray_img_pil)
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
def getcoloredMask(image, mask):
    color_mask = np.zeros_like(image)
    color_mask[:, :, 1] += (mask*255).astype('uint8')
    masked = cv2.addWeighted(image, 1.0, color_mask, 1.0, 0.0)
    return masked
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

    colmask_pred = getcoloredMask(image_gt, ytest)


    return colmask_pred


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
    seg_subject = input("Enter hand or finger for segmentation  ")
    print(seg_subject + " segmentation")
    if seg_subject=="finger":
        path_to_weights = r"C:\Users\lisak\NG\segmentation\finger\checkpoints\SM\UnetPlusPlus_vgg19_bn\Transfer\Best_Weights\best_checkpoint.pt"
    elif seg_subject=="hand":
        path_to_weights = r"C:\Users\lisak\NG\segmentation\hand_bigger\checkpoints\SM\UnetPlusPlus_vgg19_bn\Best_Weights\best_checkpoint.pt"
    else:
        print("Please enter hand or finger")
        exit()

    print("stand about 1.5 - 2 meters from camera and point your finger")

    ENCODER_WEIGHTS = Main_SM.get_state_dict(path_to_weights)
    model.load_state_dict(ENCODER_WEIGHTS)

    data_transforms = transforms.ToTensor()
    save_video_path = r"C:\Users\lisak\NG\segmentation\finger\predictions\SM\UnetPlusPlus_vgg19_bn\realtime"

    model.eval()
    cap = cv2.VideoCapture(0, cv2.CAP_ANY)
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out = cv2.VideoWriter(save_video_path + "/realtime1.avi", fourcc, 30, (640, 480))
    i = 0

    while True:
        ret, frame = cap.read()
        img_T, img_show = make_image_types(frame)
        original_size = frame.shape[0:2]


        if i % 5 == 0:
            i = 0
            prediction = get_predict(img_T, original_size)

    # for saving the video
        # prediction_s = (prediction * 255).astype('uint8')
        # out.write(prediction_s)
        cv2.imshow('Realtime', prediction)
        key = cv2.waitKey(10)
        if key & 0xFF == ord('q'):
            break
        i += 1
    cap.release()
    out.release()
    cv2.destroyAllWindows()

