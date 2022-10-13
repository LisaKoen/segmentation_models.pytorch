import os
import matplotlib.pyplot as plt
import data_segmented_SM as ds
import torch
import segmentation_models_pytorch as smp
import train_segmented_SM as ts
from segmentation_models_pytorch import utils
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, ConcatDataset, random_split
import ssl

def get_state_dict(weight_path):
    r"""Loads pretrianed weights to model.
    Features::
        - Incompatible layers (unmatched in name or size) will be ignored.
        - Can automatically deal with keys containing "module.".
    Args:
        model (nn.Module): network model.
        weight_path (str): path to pretrained weights.
    Examples::
        # >>> from torchreid.utils import load_pretrained_weights
        # >>> weight_path = 'log/my_model/model-best.pth.tar'
        # >>> load_pretrained_weights(model, weight_path)
    """

    checkpoint = torch.load(weight_path)
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint

    for k, v in state_dict.items():
        if k.startswith('module.' or 'encoder.'):
            # k = k[7:] # discard module.
            k.replace('.module', '')

    return state_dict



def visualize(**images):
    """PLot images in one row."""
    n = len(images)
    plt.figure(figsize=(16, 5))
    for i, (name, image) in enumerate(images.items()):
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(' '.join(name.split('_')).title())
        plt.imshow(image)
    plt.show()


def trainTestSplit(dataset, TTR):

    size = len(dataset)
    n_train = int(size * TTR)
    n_val = size - n_train
    trainDataset, valDataset = random_split(dataset, (n_train, n_val))
    return trainDataset, valDataset


if __name__ == '__main__':
    ssl._create_default_https_context = ssl._create_unverified_context
    # Model, loss, optimizer

    encoder_weights_url = False      # False for transfer learning from your own weights, True if using url
    # os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"


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
        in_channels=in_channels,
    )

    preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, pretrained=ENCODER_WEIGHTS)


    # only needed for transfer learning (here replace url weights with your own weights)
    if encoder_weights_url == False:
        path_to_weights = r"C:\Users\lisak\NG\segmentation\finger\checkpoints\SM\UnetPlusPlus_vgg19_bn\Transfer\Best_Weights\best_checkpoint.pt"
        NEW_ENCODER_WEIGHTS = get_state_dict(path_to_weights)
        model.load_state_dict(NEW_ENCODER_WEIGHTS)


    loss = utils.losses.BCELoss()
    metrics = [
        utils.metrics.IoU(threshold=0.5),
    ]

    optimizer = torch.optim.Adam([
        dict(params=model.parameters(), lr=0.0001),
    ])

    lr_scheduler = optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.8)

    # Datapaths


    path_to_images = r"C:\Users\lisak\NG\segmentation\hand_bigger\data"
    path_to_annotations = r"C:\Users\lisak\NG\segmentation\hand_bigger\labels\labels.json"

    # path_to_images = r"C:\Users\lisak\NG\segmentation\finger\data"
    # path_to_annotations = r"C:\Users\lisak\NG\segmentation\finger\labels\labels.json"


    # Datasets

    dataset = ds.Dataset_SM(path_to_images, path_to_annotations, classes=CLASSES,
                        preprocessing=ds.get_preprocessing(preprocessing_fn))
    dataset_flip = ds.Dataset_SM(path_to_images, path_to_annotations, classes=CLASSES,
                            preprocessing=ds.get_preprocessing(preprocessing_fn), flip=ds.get_flip())


    #### Visualize resulted augmented images and masks
    # image, msk = dataset[4]  # get some sample
    # visualize(
    #     image=image,
    #     cars_mask=msk.squeeze(),
    # )



    augmented_dataset = ds.Dataset_SM(
        path_to_images,
        path_to_annotations,
        augmentation=ds.get_training_augmentation(),
        classes=CLASSES,
        preprocessing=ds.get_preprocessing(preprocessing_fn)
    )
    augmented_dataset_flip = ds.Dataset_SM(
        path_to_images,
        path_to_annotations,
        augmentation=ds.get_training_augmentation(),
        classes=CLASSES,
        preprocessing=ds.get_preprocessing(preprocessing_fn),
        flip=ds.get_flip()
    )
    augmented_dataset_b = ds.Dataset_SM(
        path_to_images,
        path_to_annotations,
        augmentation=ds.get_training_augmentation(),
        classes=CLASSES,
        preprocessing=ds.get_preprocessing(preprocessing_fn)
    )
    augmented_dataset_flip_b = ds.Dataset_SM(
        path_to_images,
        path_to_annotations,
        augmentation=ds.get_training_augmentation(),
        classes=CLASSES,
        preprocessing=ds.get_preprocessing(preprocessing_fn),
        flip=ds.get_flip()
    )
    megaDataset = ConcatDataset([dataset, dataset_flip, augmented_dataset, augmented_dataset_b,
                                 augmented_dataset_flip, augmented_dataset_flip_b])

    batchSize = 2
    train_dataset, valid_dataset = trainTestSplit(megaDataset, 0.9)


    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=4)
    valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=2)

    # create epoch runners
    # it is a simple loop of iterating over dataloader`s samples

    train_epoch = utils.train.TrainEpoch(
        model,
        loss=loss,
        metrics=metrics,
        optimizer=optimizer,
        device=DEVICE,
        verbose=True,
    )

    valid_epoch = utils.train.ValidEpoch(
        model,
        loss=loss,
        metrics=metrics,
        device=DEVICE,
        verbose=True,
    )

    retval = ts.training_loop_SM(60,
                                 optimizer,
                                 lr_scheduler,
                                 train_loader,
                                 valid_loader,
                                 train_epoch,
                                 valid_epoch,
                                 model)
