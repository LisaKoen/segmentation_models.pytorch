import torch
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'


# def meanIOU(target, predicted):
#     if target.shape != predicted.shape:
#         print("target has dimension", target.shape, ", predicted values have shape", predicted.shape)
#         return
#
#     if target.dim() != 4:
#         print("target has dim", target.dim(), ", Must be 4.")
#         return
#
#     iousum = 0
#     for i in range(target.shape[0]):
#         target_arr = target[i, :, :, :].clone().detach().cpu().numpy().argmax(0)
#         predicted_arr = predicted[i, :, :, :].clone().detach().cpu().numpy().argmax(0)
#
#         intersection = np.logical_and(target_arr, predicted_arr).sum()
#         union = np.logical_or(target_arr, predicted_arr).sum()
#         if union == 0:
#             iou_score = 0
#         else:
#             iou_score = intersection / union
#         iousum += iou_score
#
#     miou = iousum / target.shape[0]
#     return miou
#
#
# def pixelAcc(target, predicted):
#     if target.shape != predicted.shape:
#         print("target has dimension", target.shape, ", predicted values have shape", predicted.shape)
#         return
#
#     if target.dim() != 4:
#         print("target has dim", target.dim(), ", Must be 4.")
#         return
#
#     accsum = 0
#     for i in range(target.shape[0]):
#         target_arr = target[i, :, :, :].clone().detach().cpu().numpy().argmax(0)
#         predicted_arr = predicted[i, :, :, :].clone().detach().cpu().numpy().argmax(0)
#
#         same = (target_arr == predicted_arr).sum()
#         a, b = target_arr.shape
#         total = a * b
#         accsum += same / total
#
#     pixelAccuracy = accsum / target.shape[0]
#     return pixelAccuracy


# train model for n_epochs
def training_loop_SM(n_epochs, optimizer, lr_scheduler, train_loader, valid_loader, train_epoch, valid_epoch, model):

    max_score = 0
    # path_to_checkpoints = r"C:\Users\lisak\NG\segmentation\hand_bigger\checkpoints\SM\FPN_inceptionv4"
    path_to_checkpoints = r"C:\Users\lisak\NG\segmentation\finger\checkpoints\SM\DeepLabv3Plus_vgg19_bn\Transfer"
    # path_to_checkpoints = r"C:\Users\lisak\NG\segmentation\hand_bigger\checkpoints\SM\FPN_vgg19"
    path_to_best_checkpoints = path_to_checkpoints + "/Best_Weights"

    for i in range(0, n_epochs):

        print('\nEpoch: {}'.format(i))
        train_logs = train_epoch.run(train_loader)
        valid_logs = valid_epoch.run(valid_loader)
        checkpoint = {
            'epoch': i + 1,
            'description': "add your description",
            'state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'valid_iou_score': valid_logs['iou_score'],
            'train_iou_score': train_logs['iou_score']
        }

        if max_score < valid_logs['iou_score']:
            max_score = valid_logs['iou_score']
            torch.save(checkpoint, path_to_best_checkpoints + '/best_checkpoint.pt')
            print('Model saved!')

        torch.save(checkpoint, path_to_checkpoints + '/checkpoint' + str(i + 1) + '.pt')
        lr_scheduler.step()

    return max_score

        # try preprocess with 3 channels and then train with 1 channel.