import numpy, torch

from torch import nn
from torch.optim import AdamW
from torch.optim import lr_scheduler
from torch.cuda.amp import GradScaler

from copy import deepcopy

from sklearn.metrics import average_precision_score, roc_auc_score

"""
Trains an ensemble of models on a given data loader.

Parameters:
    - models : The models to train.
    - num_epochs : The number of epochs to train each model for.
    - train_loader : The data loader containing the training data.
    - test_loader : The data loader containing the test data.
    - train_func : The function to use to train the individual models.
    - test_func : The function to use to test the individual models.
    - torch_device : The device to store the data in.
    - loss_pos_weight : Amount to weigh the positive labels in loss function.
    - pos_label : What to consider "positive" for APS calculation.
    - lr : The learning rate to use.
    - clip : The gradient clip to use.
    - epoch_threshold : Threshold of epochs to do before stopping if ROC is bad.
    - save_model_path : Where to save the trained models.
    - start_idx : Starting idx for saving models.
"""
def train_ensemble(models, num_epochs, train_loader, test_loader, train_func, 
    test_func, torch_device, loss_pos_weight, pos_label, lr, clip, save_model_path_func=None, 
    start_idx=0):
    # Iterate through models and train each one.
    for idx, model in enumerate(models):
        print()
        print("Training model " + str(idx) + " of " + str(len(models) - 1))

        # Get optimizer and learning rate scheduler.
        optimizer = AdamW(model.parameters(), lr=lr)
        scheduler = lr_scheduler.OneCycleLR(optimizer=optimizer, max_lr=lr, epochs=num_epochs,
            steps_per_epoch=len(train_loader))

        # Scaler for mixed precision training.
        scaler = GradScaler()

        # Loss function to use.
        loss_func = nn.BCEWithLogitsLoss(pos_weight=loss_pos_weight)
        best_model_state_dict = deepcopy(model.state_dict())
        best_aps, best_roc_auc = test_func(model, test_loader, torch_device, pos_label)
        print("Initial results for model: APS=" + str(best_aps) + " ROC AUC=" + str(best_roc_auc))
        print()
        best_epoch = 0

        # Train.
        for epoch in range(num_epochs):
            # Call the training function for this epoch.
            print(str(epoch) + " of " + str(num_epochs - 1))
            train_func(model, train_loader, loss_func, torch_device, optimizer, scheduler, 
                       scaler, clip)
            aps, roc_auc = test_func(model, test_loader, torch_device, pos_label)
            print("APS=" + str(aps) + " ROC AUC=" + str(roc_auc))

            # Save model state if it's the best we have seen so far.
            if (round(roc_auc, 2) > round(best_roc_auc, 2) or 
                (round(roc_auc, 2) == round(best_roc_auc, 2) and round(aps, 2) > round(best_aps, 2))):
                best_roc_auc = roc_auc
                best_aps = aps
                best_epoch = epoch
                best_model_state_dict = deepcopy(model.state_dict())

        # Set model to its best version and save.
        model.load_state_dict(best_model_state_dict)
        if save_model_path_func is not None:
            torch.save(best_model_state_dict, save_model_path_func(idx + start_idx))
        
        print("Best epoch for model: " + str(best_epoch))

    # Return best results for cross-validation using just one model.
    if len(models) == 1:
        return best_aps, best_roc_auc

"""
Tests an ensemble of models on a given data loader.

Parameters:
    - models : The models to test.
    - test_loader : The data loader containing the data to test.
    - prediction_func : The function to use to generate predictions for each model.
    - torch_device : The device to store the data in.
    - pos_label : What to consider "positive" for APS calculation.
"""
def test_ensemble(models, test_loader, prediction_func, torch_device, pos_label):
    summed_preds = numpy.array([])

    # Iterate through each model and get predictions.
    for model in models:
        y_pred, y_true = prediction_func(model, test_loader, torch_device)

        if len(summed_preds) == 0:
            summed_preds = numpy.array(y_pred)
        else:
            summed_preds += y_pred
    
    # Get final predictions.
    predictions = summed_preds / len(models)

    # Get metrics.
    aps = average_precision_score(y_true, predictions, pos_label=pos_label)
    roc_auc = roc_auc_score(y_true, predictions.tolist())
    return aps, roc_auc
