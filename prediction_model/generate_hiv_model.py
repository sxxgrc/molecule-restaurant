"""
This script takes care of training the prediction model to classify molecules
as either inhibiting HIV replication or not.
"""

import numpy

from pathlib import Path

from datasets import MoleculeNetFeaturesDataset
from splitters import ScaffoldSplitter
from optimization import get_optimized_hyperparameters
from models import PredictionModel, train_prediction_model, test_prediction_model, get_predictions

from torch_geometric.data import DataLoader

from torch import nn

from sklearn.metrics import f1_score, roc_auc_score

from copy import deepcopy

"""
Data preparation for the HIV dataset.

Produces a train and test data loader, where the data split by molecular scaffold similarity.
"""
def prepare_train_test_data(frac_train=0.8, frac_test=0.2):
    # Data path to store the dataset to train on.
    data_path = str(Path().absolute()) + "/prediction_model/data/torch-geometric"

    # Load the HIV dataset we will be using for this training.
    dataset = MoleculeNetFeaturesDataset(data_path, name="HIV")
    
    # Split the dataset into train and test datasets and create data loaders for them.
    scaffold_splitter = ScaffoldSplitter(dataset)
    train_dataset, test_dataset = scaffold_splitter.split(frac_train, frac_test)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32)

    return train_loader, test_loader

"""
Trains the full HIV classifier model on a given train data loader.
"""
def train_hiv_model(models, num_train_epochs, train_loader, test_loader):
    # Loss function to use.
    loss_func = nn.BCEWithLogitsLoss(reduction="none")

    # Iterate through models and train each one.
    for idx, model in enumerate(models):
        best_model_state_dict = None
        best_roc_auc = 0
        for epoch in range(num_train_epochs):
            train_prediction_model(model, train_loader, loss_func, num_train_epochs)
            f1, roc_auc = test_prediction_model(model, test_loader)
            print("For model " + str(idx) + " epoch " + str(epoch) + 
                " : F1=" + str(f1) + ", ROC AUC=" + str(roc_auc))
            
            # Save model state if it's the best we have seen so far.
            if roc_auc > best_roc_auc:
                best_roc_auc = roc_auc
                best_model_state_dict = deepcopy(model.state_dict())
        
        # Set model to its best version.
        # TODO: test this.
        model.load_state_dict(best_model_state_dict)

"""
Tests the full HIV classifier model on a given data loader.
"""
def test_hiv_model(models, test_loader):
    summed_preds = None

    # Iterate through each model and get predictions.
    for model in models:
        y_pred, _, y_true = get_predictions(model, test_loader)

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

"""
Generates all of the prediction models to use for the ensembled final
model.
This will optimize the hyperparameters for the model before creating them.
"""
def generate_hiv_models(ensemble_size, atom_dim, bond_dim, features_dim, torch_device,
                        train_loader, test_loader, num_opt_iters):
    # Get the optimized hyperparameters for the models.
    model_args = get_optimized_hyperparameters(ensemble_size, atom_dim, bond_dim,
                                                features_dim, torch_device, train_loader,
                                                test_loader, num_opt_iters)

"""
Main method for generating the fully trained HIV classifier.
"""
def generate_hiv_classifier(num_train_epochs, ensemble_size, torch_device, num_opt_iters):
    print("Generating a new HIV replication inhibition classifier.")

    # Prepare the train and test data.
    train_loader, test_loader = prepare_train_test_data()

    # Generate the models.
    data_view = next(iter(train_loader))
    atom_dim = data_view.x.shape[1]
    bond_dim = data_view.edge_attr.shape[1]
    features_dim = data_view.features.shape[1]
    models = generate_hiv_models(ensemble_size, atom_dim, bond_dim, 
                                 features_dim, torch_device, num_opt_iters)

    # Train the ensembled models.


    # Save the models in the trained-models directory.

generate_hiv_classifier(30, 1, None)