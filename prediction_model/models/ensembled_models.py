import numpy

from torch import nn
from torch.optim import AdamW
from torch.optim import lr_scheduler
from torch.cuda.amp import GradScaler

from copy import deepcopy

from sklearn.metrics import f1_score, roc_auc_score

from chemprop.nn_utils import NoamLR

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
"""
def train_ensemble(models, num_epochs, train_loader, test_loader, train_func, 
    test_func, torch_device):
    # Iterate through models and train each one.
    for idx, model in enumerate(models):
        print()
        print("Training model " + str(idx) + " of " + str(len(models) - 1))
        best_model_state_dict = None
        best_roc_auc = 0

        # Get optimizer and learning rate scheduler.
        optimizer = AdamW([{"params": model.parameters(), "lr": 4e-4, "weight_decay": 0}])
        scheduler = lr_scheduler.OneCycleLR(optimizer=optimizer, max_lr=4e-4, epochs=num_epochs,
            steps_per_epoch=len(train_loader))
        # scheduler = NoamLR(optimizer=optimizer, warmup_epochs=[2.0], total_epochs=[num_epochs],
            # steps_per_epoch=len(train_loader), init_lr=[2e-4], max_lr=[2e-3], final_lr=[2e-4])

        # Scaler for mixed precision training.
        scaler = GradScaler()

        # Loss function to use.
        loss_func = nn.BCEWithLogitsLoss()

        # Train.
        for epoch in range(num_epochs):
            train_func(model, train_loader, loss_func, torch_device, optimizer, scheduler, scaler)
            f1, roc_auc = test_func(model, test_loader, torch_device)
            print("For model " + str(idx) + " epoch " + str(epoch) + 
                " : F1=" + str(f1) + ", ROC AUC=" + str(roc_auc))
            
            # Save model state if it's the best we have seen so far.
            if roc_auc > best_roc_auc:
                best_roc_auc = roc_auc
                best_model_state_dict = deepcopy(model.state_dict())
        
        # Set model to its best version.
        model.load_state_dict(best_model_state_dict)

"""
Tests an ensemble of models on a given data loader.

Parameters:
    - models : The models to test.
    - test_loader : The data loader containing the data to test.
    - prediction_func : The function to use to generate predictions for each model.
    - torch_device : The device to store the data in.
"""
def test_ensemble(models, test_loader, prediction_func, torch_device):
    summed_preds = None

    # Iterate through each model and get predictions.
    for model in models:
        y_pred, _, y_true = prediction_func(model, test_loader, torch_device)

        if summed_preds == None:
            summed_preds = numpy.array(y_pred)
        else:
            summed_preds += y_pred
    
    # Get final predictions.
    predictions = summed_preds / len(models)
    prediction_labels = numpy.round(predictions).tolist()

    # Get metrics.
    f1 = f1_score(y_true, prediction_labels)
    roc_auc = roc_auc_score(y_true, predictions.tolist())
    return f1, roc_auc
