import json, pickle, numpy

from pathlib import Path

from os import path

from prediction_model.models.arguments import ModelArgs
from prediction_model.models import (
    create_prediction_model,
    train_prediction_model, 
    test_prediction_model, 
    train_ensemble
)

from hyperopt import hp, fmin, tpe, Trials

from numpy.random import RandomState

from functools import partial

# We are using the same space as used in chemprop, along with a few changes.
SPACE = {
    "dropout_prob": hp.quniform("dropout_prob", low=0.0, high=0.3, q=0.05),
    "message_passing_depth": hp.quniform("message_passing_depth", low=4, high=7, q=1),
    "hidden_size": hp.quniform("hidden_size", low=400, high=2400, q=100),
    "num_ffn_layers": hp.quniform("num_ffn_layers", low=2, high=4, q=1),
    "ffn_hidden_size": hp.quniform("ffn_hidden_size", low=400, high=2400, q=100),
    "ffn_dropout_prob": hp.quniform("ffn_dropout_prob", low=0.0, high=0.3, q=0.05),
    "learning_rate": hp.loguniform("learning_rate", numpy.log(0.0001), numpy.log(0.01))
}

# The integer parameters.
INT_KEYS = ["message_passing_depth", "hidden_size", "num_ffn_layers", "ffn_hidden_size"]

# Path of saved parameters JSON file.
SAVE_PATH = (str(Path().absolute()) + 
             "/prediction_model/optimization/saved_hyperparameters/hyperparameters.json")

# Path of saved trials.
TRIALS_PATH = str(Path().absolute()) + "/prediction_model/optimization/saved_hyperparameters/trials.pkl"

"""
Runs hyperparameter optimization on the ensembled PredictionModel.
This will optimize all of the parameters present in the ModelArgs class.

Parameters:
    - atom_dim : The dimension of the atom features in the dataset.
    - bond_dim : The dimension of the bond features in the dataset.
    - features_dim : The dimension of the molecule features in the dataset.
    - torch_device : The device being used for PyTorch.
    - validation_loader : Data loader containing the validation dataset.
    - test_loader : Data loader containing the test dataset.
    - num_optimization_iters : The amount of iterations to optimize the hyperparameters for.
    - num_epochs : The number of epochs to train each model for.
    - loss_pos_weight : Amount to weigh the positive labels in loss function.
    - pos_label : What to consider "positive" for F1 calculation.
"""
def optimize_hyperparameters(atom_dim, bond_dim, features_dim, torch_device, 
                             train_loader, validation_loader, num_optimization_iters, num_epochs,
                             loss_pos_weight, pos_label):
    print("Optimizing hyperparameters...")

    # Define the method to optimize.
    def objective(hyperparams):
        # Convert hyperparameters from float to int when necessary.
        for key in INT_KEYS:
            hyperparams[key] = int(hyperparams[key])
        
        # Generate model arguments from hyperparams.
        hyper_args = ModelArgs(0, 0, 0, 0, 0, 0)
        for key, value in hyperparams.items():
            if key != "learning_rate":
                setattr(hyper_args, key, value)
        lr = hyperparams["learning_rate"]

        hyper_args = ModelArgs(dropout_prob=0.2, ffn_dropout_prob=0.2, ffn_hidden_size=1600, hidden_size=2100,
            message_passing_depth=5, num_ffn_layers=2)
        lr = 0.0075455
        
        print("Trying following hyperparameters:")
        print("MPNN Dropout Prob=" + str(hyper_args.dropout_prob))
        print("Message Passing Depth=" + str(hyper_args.message_passing_depth))
        print("MPNN Hidden Size=" + str(hyper_args.hidden_size))
        print("Number of FFN Layers=" + str(hyper_args.num_ffn_layers))
        print("FFN Hidden Size=" + str(hyper_args.ffn_hidden_size))
        print("FFN Dropout Prob=" + str(hyper_args.ffn_dropout_prob))
        print("Learning Rate=" + str(lr))

        # Create, train, and test the models.
        model = [create_prediction_model(hyper_args, atom_dim, bond_dim, features_dim, torch_device)]
        f1, roc_auc = train_ensemble(model, num_epochs, train_loader, validation_loader, train_prediction_model, 
                       test_prediction_model, torch_device, loss_pos_weight, pos_label, lr)
        print("Best results for model: F1=" + str(f1) + ", ROC AUC=" + str(roc_auc))

        # Return metric we will minimize for optimization.
        return -1 * roc_auc

    # Our model takes a while to run so use less number of random states for TPE algorithm.
    # Original value is 20.
    n_startup_jobs = 15
    algo = partial(tpe.suggest, n_startup_jobs=n_startup_jobs)

    # Import any older trials we have run.
    if path.exists(TRIALS_PATH) and path.getsize(TRIALS_PATH) > 0:
        trials = pickle.load(open(TRIALS_PATH, "rb"))
    else:
        trials = Trials()

    # Optimize hyperparameters.
    i = 0
    while i < num_optimization_iters:
        print()
        print("Running hyperoptimization trial " + str(i + 1) + " of " + str(num_optimization_iters))

        # Run minimization for one more iteration.
        best = fmin(objective, SPACE, algo=algo, max_evals=len(trials) + 1, trials=trials, 
                    rstate=RandomState())

        # Save trials so far.
        pickle.dump(trials, open(TRIALS_PATH, "wb"))

        # If last trial was bad then add another iteration.
        if trials.losses()[-1] == -0.5:
            num_optimization_iters += 1

            # Would be helpful to have more random states for TPE.
            if len(trials) < n_startup_jobs + 1:
                n_startup_jobs += 1
                algo = partial(tpe.suggest, n_startup_jobs=n_startup_jobs)

        # Iterate.
        i += 1

    # Get best results.
    best_result = trials.best_trial["result"]["loss"] * -1
    print("Best result from hyperparameter optimization has ROC AUC=" + str(best_result))
    
    # Save best hyperparameters in JSON file.
    with open(SAVE_PATH, "w") as file:
        json.dump(best, file, indent=4, sort_keys=True)

"""
Gets the optimized hyperparameters for the full ensembled prediction model.
"""
def get_optimized_hyperparameters(atom_dim, bond_dim, features_dim, torch_device, 
                             train_loader, validation_loader, num_optimization_iters, num_epochs, 
                             loss_pos_weight, pos_label):
    # Check if parameters are in file, creating them if not.
    if not path.exists(SAVE_PATH) or not path.getsize(SAVE_PATH) > 0:
        optimize_hyperparameters(atom_dim, bond_dim, features_dim, torch_device,
                                 train_loader, validation_loader, num_optimization_iters, num_epochs, 
                                 loss_pos_weight, pos_label)
    
    # Load the parameters from the JSON file.
    with open(SAVE_PATH, "r") as json_file:
        hyper_params = json.load(json_file)
    
    # Generate the model arguments from stored parameters.
    model_args = ModelArgs(dropout_prob=0.0, message_passing_depth=3, hidden_size=300,
        num_ffn_layers=2, ffn_hidden_size=300, ffn_dropout_prob=0.0)
    for key, value in hyper_params.items():
        if key != "learning_rate":
            setattr(model_args, key, value)
    
    return model_args, hyper_params["learning_rate"]
