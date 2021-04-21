import numpy

from torch import nn

from copy import deepcopy

from sklearn.metrics import f1_score, roc_auc_score

"""
Trains an ensemble of models on a given data loader.

Parameters:
    - models : The models to train.
    - num_train_epochs : The number of epochs to train each model for.
    - train_loader : The data loader containing the training data.
    - test_loader : The data loader containing the test data.
    - train_func : The function to use to train the individual models.
    - test_func : The function to use to test the individual models.
"""
def train_ensemble(models, num_train_epochs, train_loader, test_loader, train_func, test_func):
    # Loss function to use.
    loss_func = nn.BCEWithLogitsLoss()

    # Iterate through models and train each one.
    for idx, model in enumerate(models):
        best_model_state_dict = None
        best_roc_auc = 0
        for epoch in range(num_train_epochs):
            train_func(model, train_loader, loss_func, num_train_epochs)
            f1, roc_auc = test_func(model, test_loader)
            print("For model " + str(idx) + " epoch " + str(epoch) + 
                " : F1=" + str(f1) + ", ROC AUC=" + str(roc_auc))
            
            # Save model state if it's the best we have seen so far.
            if roc_auc > best_roc_auc:
                best_roc_auc = roc_auc
                best_model_state_dict = deepcopy(model.state_dict())
        
        # Set model to its best version.
        # TODO: test this.
        print(best_model_state_dict)
        model.load_state_dict(best_model_state_dict)
        print(model.state_dict)

"""
Tests an ensemble of models on a given data loader.

Parameters:
    - models : The models to test.
    - test_loader : The data loader containing the data to test.
    - prediction_func : The function to use to generate predictions for each model.
"""
def test_ensemble(models, test_loader, prediction_func):
    summed_preds = None

    # Iterate through each model and get predictions.
    for model in models:
        y_pred, _, y_true = prediction_func(model, test_loader)

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
