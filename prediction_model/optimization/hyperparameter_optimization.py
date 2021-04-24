import json

from pathlib import Path

from os import path

from prediction_model.models.arguments import ModelArgs
from prediction_model.models import (
    create_prediction_model,
    train_prediction_model, 
    test_prediction_model, 
    get_predictions,
    train_ensemble,
    test_ensemble
)

from hyperopt import hp, fmin, tpe

from copy import deepcopy

from numpy.random import RandomState

# We are using the same space as used in chemprop, along with a few changes.
SPACE = {
    "dropout_prob": hp.quniform("dropout_prob", low=0.0, high=0.4, q=0.05),
    "message_passing_depth": hp.quniform("message_passing_depth", low=3, high=7, q=1),
    "hidden_size": hp.quniform("hidden_size", low=300, high=2400, q=100),
    "num_ffn_layers": hp.quniform("num_ffn_layers", low=2, high=4, q=1),
    "ffn_hidden_size": hp.quniform("ffn_hidden_size", low=300, high=2400, q=100),
    "ffn_dropout_prob": hp.quniform("ffn_dropout_prob", low=0.0, high=0.4, q=0.05)
}

# The integer parameters.
INT_KEYS = ["message_passing_depth", "hidden_size", "num_ffn_layers", "ffn_hidden_size"]

# Path of saved parameters JSON file.
SAVE_PATH = (str(Path().absolute()) + 
             "/prediction_model/optimization/saved_hyperparameters/hyperparameters.json")

"""
Runs hyperparameter optimization on the ensembled PredictionModel.
This will optimize all of the parameters present in the ModelArgs class.

Parameters:
    - atom_dim : The dimension of the atom features in the dataset.
    - bond_dim : The dimension of the bond features in the dataset.
    - features_dim : The dimension of the molecule features in the dataset.
    - torch_device : The device being used for PyTorch.
    - train_loader : Data loader containing the train dataset.
    - test_loader : Data loader containing the test dataset.
    - num_optimization_iters : The amount of iterations to optimize the hyperparameters for.
    - num_epochs : The number of epochs to train each model for.
    - loss_pos_weight : Amount to weigh the positive labels in loss function.
    - pos_label : What to consider "positive" for F1 calculation.
    - lr : The learning rate to use.
"""
def optimize_hyperparameters(atom_dim, bond_dim, features_dim, torch_device, 
                             train_loader, test_loader, num_optimization_iters, num_epochs,
                             loss_pos_weight, pos_label, lr):
    print("Optimizing hyperparameters...")
    
    # Initialize model arguments.
    model_args = ModelArgs(dropout_prob=0.0, message_passing_depth=3, hidden_size=300,
        num_ffn_layers=2, ffn_hidden_size=300, ffn_dropout_prob=0.0)

    # Define the method to optimize.
    results = []
    def objective(hyperparams):
        # Convert hyperparameters from float to int when necessary.
        for key in INT_KEYS:
            hyperparams[key] = int(hyperparams[key])
        
        # TODO:
        # # Generate model arguments from hyperparams.
        # hyper_args = deepcopy(model_args)

        # for key, value in hyperparams.items():
        #     setattr(hyper_args, key, value)
        hyper_args = ModelArgs(dropout_prob=0.1, message_passing_depth=4, hidden_size=500,
                               num_ffn_layers=2, ffn_hidden_size=500, ffn_dropout_prob=0.1)

        print("Trying following hyperparameters:")
        print("MPNN Dropout Prob=" + str(hyper_args.dropout_prob))
        print("Message Passing Depth=" + str(hyper_args.message_passing_depth))
        print("MPNN Hidden Size=" + str(hyper_args.hidden_size))
        print("Number of FFN Layers=" + str(hyper_args.num_ffn_layers))
        print("FFN Hidden Size=" + str(hyper_args.ffn_hidden_size))
        print("FFN Dropout Prob=" + str(hyper_args.ffn_dropout_prob))

        # Create, train, and test the models.
        model = [create_prediction_model(hyper_args, atom_dim, bond_dim, features_dim, torch_device)]
        train_ensemble(model, num_epochs, train_loader, test_loader, train_prediction_model, 
                       test_prediction_model, torch_device, loss_pos_weight, pos_label, lr)
        f1, roc_auc = test_ensemble(model, test_loader, get_predictions, torch_device, pos_label)
        print("Results of model: F1=" + str(f1) + ", ROC AUC=" + str(roc_auc))

        # Record results.
        results.append({"f1": f1, "roc_auc": roc_auc, "hyperparams": hyperparams})

        # Return metric we will minimize for optimization.
        return -1 * roc_auc

    # Optimize hyperparameters.
    fmin(objective, SPACE, algo=tpe.suggest, max_evals=num_optimization_iters, rstate=RandomState())

    # Get best results.
    best_result = min(results, key=lambda result: -1 * result["roc_auc"])
    print("Best result from hyperparameter optimization has F1=" + str(best_result["f1"]) +
          " and ROC AUC=" + str(best_result["roc_auc"]))
    
    # Save best hyperparameters in JSON file.
    with open(SAVE_PATH, "w") as file:
        json.dump(best_result["hyperparams"], file, indent=4, sort_keys=True)


"""
Gets the optimized hyperparameters for the full ensembled prediction model.
"""
def get_optimized_hyperparameters(atom_dim, bond_dim, features_dim, torch_device, 
                             train_loader, test_loader, num_optimization_iters, num_epochs, 
                             loss_pos_weight, pos_label, lr):
    # Check if parameters are in file, creating them if not.
    if not path.exists(SAVE_PATH) or not path.getsize(SAVE_PATH) > 0:
        optimize_hyperparameters(atom_dim, bond_dim, features_dim, torch_device,
                                 train_loader, test_loader, num_optimization_iters, num_epochs, 
                                 loss_pos_weight, pos_label, lr)
    
    # Load the parameters from the JSON file.
    with open(SAVE_PATH, "r") as json_file:
        hyper_params = json.load(json_file)
    
    # Generate the model arguments from stored parameters.
    model_args = ModelArgs(dropout_prob=0.0, message_passing_depth=3, hidden_size=300,
        num_ffn_layers=2, ffn_hidden_size=300, ffn_dropout_prob=0.0)
    for key, value in hyper_params.items():
            setattr(model_args, key, value)
    
    return model_args
