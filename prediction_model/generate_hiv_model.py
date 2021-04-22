"""
This script takes care of training the prediction model to classify molecules
as either inhibiting HIV replication or not.
"""

import torch

from pathlib import Path

from os import path

from prediction_model.datasets import MoleculeNetFeaturesDataset
from prediction_model.splitters import ScaffoldSplitter
from prediction_model.optimization import get_optimized_hyperparameters
from prediction_model.models import (
    HIVClassifier,
    PredictionModel, 
    train_prediction_model, 
    test_prediction_model, 
    get_predictions,
    train_ensemble,
    test_ensemble)

from torch_geometric.data import DataLoader

# The path for the saved model.
SAVE_PATH = str(Path().absolute()) + "/prediction_model/trained_models/"

"""
Data preparation for the HIV dataset.

Produces a train and test data loader, where the data split by molecular scaffold similarity.
"""
def prepare_train_test_data(frac_train=0.8, frac_test=0.2, batch_size=32):
    # Data path to store the dataset to train on.
    data_path = str(Path().absolute()) + "/prediction_model/data/torch-geometric"

    # Load the HIV dataset we will be using for this training.
    dataset = MoleculeNetFeaturesDataset(data_path, name="HIV")

    # TODO: Testing
    dataset = dataset[:640]

    # Split the dataset into train and test datasets and create data loaders for them.
    if (frac_train == 1.0):
        # No reason to split.
        train_loader = DataLoader(dataset, batch_size=batch_size)
        test_loader = None
    else:
        print("Splitting the data...")
        scaffold_splitter = ScaffoldSplitter(dataset)
        train_dataset, test_dataset = scaffold_splitter.split(frac_train, frac_test)
        print("Finished splitting.")

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
            num_workers=4, pin_memory=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=4, pin_memory=True)

    return train_loader, test_loader

"""
Generates all of the prediction models to use for the ensembled final
model.
This will optimize the hyperparameters for the model before creating them.

Parameters:
    - ensemble_size : The amount of models to use in the ensemble.
    - atom_dim : The dimension of the atom features in the dataset.
    - bond_dim : The dimension of the bond features in the dataset.
    - features_dim : The dimension of the molecule features in the dataset.
    - torch_device : The device being used for PyTorch.
    - train_loader : Data loader containing the train dataset.
    - test_loader : Data loader containing the test dataset.
    - num_opt_iters : The amount of iterations to optimize the hyperparameters for.
    - num_epochs : The number of epochs to train each model for.
"""
def generate_initial_hiv_models(ensemble_size, atom_dim, bond_dim, features_dim, torch_device,
                        train_loader, test_loader, num_opt_iters, num_epochs):
    # Get the optimized hyperparameters for the models.
    model_args = get_optimized_hyperparameters(ensemble_size, atom_dim, bond_dim,
                                                features_dim, torch_device, train_loader,
                                                test_loader, num_opt_iters, num_epochs // 2)
    
    # Generate the models.
    models = [
        PredictionModel(model_args, atom_dim, bond_dim, features_dim, torch_device).to(torch_device)
        for _ in range(ensemble_size)]
    
    return models

"""
Gets dimension information from the HIV dataset.

Parameters:
    - data_loader : The data loader for the HIV dataset. If this is None, we create one.

Returns:
    - atom_dim : The dimension of the atom features in the dataset.
    - bond_dim : The dimension of the bond features in the dataset.
    - features_dim : The dimension of the molecule features in the dataset.
"""
def get_dimensions(data_loader):
    # Generate a data loader for the HIV dataset if none were given.
    if data_loader == None:
        data_loader, _ = prepare_train_test_data(frac_train=1.0, frac_test=0)
    
    # Compute the dimension values based off the dataset.
    data_view = next(iter(data_loader))
    atom_dim = data_view.x.shape[1]
    bond_dim = data_view.edge_attr.shape[1]
    features_dim = data_view.features.shape[1]
    return atom_dim, bond_dim, features_dim

"""
Helper method for getting the path to a model's saved state dict.
"""
def get_model_state_dict_path(model_idx):
    return SAVE_PATH + "hiv_" + str(model_idx) + ".pt"

"""
Main method for generating the fully trained HIV classifier.
"""
def generate_hiv_models(num_train_epochs, ensemble_size, torch_device, num_opt_iters, batch_size):
    print()
    print("Generating a new HIV replication inhibition classifier. Mind the noise!")

    # Prepare the train and test data.
    train_loader, test_loader = prepare_train_test_data(batch_size=batch_size)

    # Generate the models.
    print("Initializing the HIV models...")
    atom_dim, bond_dim, features_dim = get_dimensions(train_loader)
    models = generate_initial_hiv_models(ensemble_size, atom_dim, bond_dim, features_dim, torch_device, 
                                 train_loader, test_loader, num_opt_iters, num_train_epochs)
    print("Done initializing the models.")

    # Train the ensembled models.
    print("Training the ensemble...")
    train_ensemble(models, num_train_epochs, train_loader, test_loader, train_prediction_model,
                   test_prediction_model, torch_device)

    # Test the ensembled model.
    f1, roc_auc = test_ensemble(models, test_loader, get_predictions, torch_device)
    print("Results of final ensembled model: F1=" + str(f1) + ", ROC AUC=" + str(roc_auc))

    # Save each of the models to the save path.
    for idx, model in enumerate(models):
        torch.save(model.state_dict(), get_model_state_dict_path(idx))

"""
Gets a trained HIV classifier. If the model already exists on disk, simply
loads it. Otherwise, we create the model and train it from scratch.

Parameters:
    - num_train_epochs : The amount of epochs to train the model for.
    - ensemble_size : The amount of models to use in the ensemble for the final model.
    - torch_device : The PyTorch device used for the models.
    - num_opt_iters : The amount of iterations to optimize the hyperparameters for.
    - bath_size : The size of the batches to use for the dataset.
"""
def get_hiv_classifier(num_train_epochs, ensemble_size, torch_device, num_opt_iters, batch_size):
    # Check if all of the models for ensemble exist, if not create all of them.
    for model_idx in range(ensemble_size):
        model_path = get_model_state_dict_path(model_idx)
        if not path.exists(model_path) or not path.getsize(model_path) > 0:
            generate_hiv_models(num_train_epochs, ensemble_size, torch_device, num_opt_iters, batch_size)
            break

    # Initialize models. Everything should exist already so don't need data loaders.
    atom_dim, bond_dim, features_dim = get_dimensions(None)
    models = generate_initial_hiv_models(ensemble_size, atom_dim, bond_dim, features_dim, 
                                         torch_device, None, None, num_opt_iters, num_train_epochs)
    
    # Load in each model's state dict.
    for idx, model in enumerate(models):
        model.load_state_dict(torch.load(get_model_state_dict_path(idx)))

    # Create HIV classifier model object and return.
    return HIVClassifier(models)
