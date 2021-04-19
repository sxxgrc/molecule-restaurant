"""
This script takes care of training the prediction model to classify molecules
as either inhibiting HIV replication or not.
"""

from pathlib import Path

from datasets import MoleculeNetFeaturesDataset
from splitters import ScaffoldSplitter

from torch_geometric.data import DataLoader

# Data path to store the dataset to train on.
DATA_PATH = str(Path().absolute()) + "/prediction_model/data"

"""
Data preparation for the HIV dataset.

Produces a train and test data loader, where the data split by molecular scaffold similarity.
"""
def prepare_train_test_data(frac_train=0.8, frac_test=0.2):
    # Load the HIV dataset we will be using for this training.
    dataset = MoleculeNetFeaturesDataset(DATA_PATH + "/torch-geometric", name="HIV")
    
    # Split the dataset into train and test datasets and create data loaders for them.
    scaffold_splitter = ScaffoldSplitter(dataset)
    train_dataset, test_dataset = scaffold_splitter.split(frac_train, frac_test)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32)

    return train_loader, test_loader


"""
Tests the D-MPNN model using the given test dataloader.
"""
def test_model(test_loader, model_parameters):
    print("hi")



"""
Trains the D-MPNN model using the given train dataloader.
"""
def train_model(train_loader, model_parameters):
    print("hi")


"""
Main method for generating the fully trained HIV classifier.
"""
def generate_hiv_classifier():
    # Prepare the train and test data.
    train_loader, test_loader = prepare_train_test_data()

    # max num bonds = 9

generate_hiv_classifier()