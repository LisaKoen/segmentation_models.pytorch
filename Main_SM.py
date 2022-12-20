import data_segmented_SM as ds
import torch
import train_segmented_SM as ts
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset, random_split
from Unet_model import UnetPlusPlus
import Seg_util as utils
import train_one_epoch


def trainTestSplit(dataset, TTR):

    size = len(dataset)
    n_train = int(size * TTR)
    n_val = size - n_train
    trainDataset, valDataset = random_split(dataset, (n_train, n_val))
    return trainDataset, valDataset


if __name__ == '__main__':
    # Model, loss, optimizer

    ENCODER = "vgg19_bn"
    ENCODER_WEIGHTS = None
    CLASSES = ['hand']
    ACTIVATION = 'sigmoid'
    DEVICE = 'cuda'
    in_channels = 3


    model = UnetPlusPlus(
        encoder_name=ENCODER,
        encoder_weights=ENCODER_WEIGHTS,
        classes=len(CLASSES),
        activation=ACTIVATION,
        in_channels=in_channels,
    )

    preprocessing_fn = utils.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)

    path_to_weights = r"C:\Users\lisak\NG\segmentation\hand_bigger\checkpoints\SM\UnetPlusPlus_vgg19_bn\Best_Weights\best_checkpoint.pt"
    ENCODER_WEIGHTS = utils.get_state_dict(path_to_weights)
    model.load_state_dict(ENCODER_WEIGHTS)

    loss = utils.BCELoss()
    metrics = [
        utils.IoU(threshold=0.5),
    ]

    optimizer = torch.optim.Adam([
        dict(params=model.parameters(), lr=0.0001),
    ])

    lr_scheduler = optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.8)

    # Datapaths (combines Coco datasets made from Label-Studio)


    path_to_images_A = r"C:\Users\lisak\NG\segmentation\finger_marker\data"
    path_to_annotations_A = r"C:\Users\lisak\NG\segmentation\finger_marker\labels\labels.json"

    path_to_images_B = r"C:\Users\lisak\NG\segmentation\finger_marker\data_edge"
    path_to_annotations_B = r"C:\Users\lisak\NG\segmentation\finger_marker\data_edge_labels\labels.json"

    path_to_images_C = r"C:\Users\lisak\NG\segmentation\finger\data"
    path_to_annotations_C = r"C:\Users\lisak\NG\segmentation\finger\labels\labels.json"

    path_to_images_D = r"C:\Users\lisak\NG\segmentation\finger\data_edge"
    path_to_annotations_D = r"C:\Users\lisak\NG\segmentation\finger\data_edge_labels\labels.json"

    image_path_list = [path_to_images_A, path_to_images_B, path_to_images_C, path_to_images_D]

    annotations_list = [path_to_annotations_A, path_to_annotations_B, path_to_annotations_C, path_to_annotations_D]

    megaDataset_list = []


    # Datasets (creates augmentations and combines all into one dataset, x6 of original size)
    for path_to_images, path_to_annotations in zip(image_path_list, annotations_list):
        dataset = ds.Dataset_SM(path_to_images, path_to_annotations, classes=CLASSES,
                            preprocessing=ds.get_preprocessing(preprocessing_fn))
        dataset_flip = ds.Dataset_SM(path_to_images, path_to_annotations, classes=CLASSES,
                                preprocessing=ds.get_preprocessing(preprocessing_fn), flip=ds.get_flip())


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

        megaDataset_list.append(megaDataset)

    megaDataset = ConcatDataset(megaDataset_list)

    train_dataset, valid_dataset = trainTestSplit(megaDataset, 0.98)


    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=4)
    valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=2)

    path_to_checkpoints = r"C:\Users\lisak\NG\segmentation\finger+finger_marker\checkpoints_debug"
    # create epoch runners
    # it is a simple loop of iterating over dataloader`s samples

    train_epoch = train_one_epoch.TrainEpoch(
        model,
        loss=loss,
        metrics=metrics,
        optimizer=optimizer,
        device=DEVICE,
        verbose=True,
    )

    valid_epoch = train_one_epoch.ValidEpoch(
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
                                 model, path_to_checkpoints)
