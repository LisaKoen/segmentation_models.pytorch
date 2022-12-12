import torch
import os
# os.environ['KMP_DUPLICATE_LIB_OK']='True'





# train model for n_epochs
def training_loop_SM(n_epochs, optimizer, lr_scheduler, train_loader, valid_loader, train_epoch, valid_epoch, model, path_to_checkpoints):

    max_score = 0
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