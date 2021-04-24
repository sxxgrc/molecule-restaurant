"""
This script takes care of training the prediction model to classify molecules
as either inhibiting HIV replication or not.
"""

import torch

from pathlib import Path

from os import path

from prediction_model.datasets import MoleculeNetFeaturesDataset, MoleculeDataset
from prediction_model.dataloaders import ExtendedDataLoader
from prediction_model.splitters import ScaffoldSplitter
from prediction_model.optimization import get_optimized_hyperparameters
from prediction_model.models import (
    HIVClassifier,
    create_prediction_model,
    train_prediction_model, 
    test_prediction_model, 
    get_predictions,
    train_ensemble,
    test_ensemble
)

# The path for the saved model.
SAVE_PATH = str(Path().absolute()) + "/prediction_model/trained_models/"

"""
Gets the data for training the HIV model.

Returns:
    - dataset : The HIV dataset to use for training.
    - loss_pos : The loss positive weight to use for computing the loss.
    - atom_dim : The dimension of the atom features in the dataset.
    - bond_dim : The dimension of the bond features in the dataset.
    - features_dim : The dimension of the molecule features in the dataset.
"""
def get_data():
    # Data path to store the dataset to train on.
    data_path = str(Path().absolute()) + "/prediction_model/data/torch-geometric"

    # Load the HIV dataset we will be using for this training.
    dataset = MoleculeNetFeaturesDataset(data_path, name="HIV")

    # Compute loss positive weight.
    num_pos = sum([data.y for data in dataset]).detach().item()
    num_neg = len(dataset) - num_pos
    loss_pos_weight = num_neg / num_pos

    # Compute dimensions.
    atom_dim = dataset[0].x.shape[1]
    bond_dim = dataset[0].edge_attr.shape[1]
    features_dim = dataset[0].features.shape[0]

    return dataset, loss_pos_weight, atom_dim, bond_dim, features_dim

"""
Data preparation for the HIV dataset.

Produces a train and test data loader, where the data split by molecular scaffold similarity.
"""
def prepare_train_test_data(dataset, frac_train=0.8, frac_test=0.2, batch_size=32):
    # Check if train and test data already exist.
    main_path = str(Path().absolute()) + "/prediction_model/data/torch-geometric/hiv/normalized"
    train_path = main_path + "/train.pt"
    test_path = main_path + "/test.pt"
    if (not path.exists(train_path) or path.getsize(train_path) == 0) or (not path.exists(test_path)
        or path.getsize(test_path) == 0):
        # Split the dataset into train and test datasets by scaffold.
        print("Splitting the data...")
        scaffold_splitter = ScaffoldSplitter()
        train_dataset, test_dataset = scaffold_splitter.split(dataset, frac_train, frac_test)
        print("Finished splitting.")

        # Create datasets for the train and test data which normalize them.
        print("Creating train and test datasets and normalizing them...")
        train_dataset = MoleculeDataset(root=main_path, name="train", dataset=train_dataset)
        test_dataset = MoleculeDataset(root=main_path, name="test", dataset=test_dataset)
    else:
        # Get datasets.
        train_dataset = MoleculeDataset(root=main_path, name="train")
        test_dataset = MoleculeDataset(root=main_path, name="test")

    # Create data loaders for train and test data.
    train_loader = ExtendedDataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
        num_workers=4, pin_memory=True)
    test_loader = ExtendedDataLoader(test_dataset, batch_size=batch_size, num_workers=4, pin_memory=True)
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
    - loss_pos_weight : Amount to weigh the positive labels in loss function.
    - pos_label : What to consider "positive" for F1 calculation.
"""
def generate_initial_hiv_models(ensemble_size, atom_dim, bond_dim, features_dim, torch_device,
                        train_loader, test_loader, num_opt_iters, num_epochs, loss_pos_weight,
                        pos_label):
    # Get the optimized hyperparameters for the models.
    model_args = get_optimized_hyperparameters(atom_dim, bond_dim, features_dim, torch_device, 
                                                train_loader, test_loader, num_opt_iters, 
                                                num_epochs // 2, loss_pos_weight, pos_label)
    
    # Generate the models.
    models = [
        create_prediction_model(model_args, atom_dim, bond_dim, features_dim, torch_device)
        for _ in range(ensemble_size)]
    
    return models

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
    data, loss_pos_weight, atom_dim, bond_dim, features_dim = get_data()
    train_loader, test_loader = prepare_train_test_data(data, batch_size=batch_size)

    # Convert loss pos weight to tensor.
    loss_pos_weight = torch.tensor(loss_pos_weight, device=torch_device)

    # Generate the models.
    print("Initializing the HIV models...")
    models = generate_initial_hiv_models(ensemble_size, atom_dim, bond_dim, features_dim, torch_device, 
                                 train_loader, test_loader, num_opt_iters, num_train_epochs, 
                                 loss_pos_weight, 0)
    print("Done initializing the models.")

    # Train the ensembled models.
    print("Training the ensemble...")
    train_ensemble(models, num_train_epochs, train_loader, test_loader, train_prediction_model,
                   test_prediction_model, torch_device, loss_pos_weight, 0)

    # Test the ensembled model.
    f1, roc_auc = test_ensemble(models, test_loader, get_predictions, torch_device, 0)
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
    _, _, atom_dim, bond_dim, features_dim = get_data()
    models = generate_initial_hiv_models(ensemble_size, atom_dim, bond_dim, features_dim, 
                                         torch_device, None, None, num_opt_iters, num_train_epochs)
    
    # Load in each model's state dict.
    for idx, model in enumerate(models):
        model.load_state_dict(torch.load(get_model_state_dict_path(idx)))

    # Create HIV classifier model object and return.
    return HIVClassifier(models).to(torch_device)
