"""
Primary script for MoleculeRestaurant to generate molecules.
"""

import torch

from prediction_model import get_hiv_classifier

# Grab the device to use for the training.
if torch.cuda.is_available():
    dev = "cuda:0"
else:
    dev = "cpu"

device = torch.device(dev)
print()
print("Using PyTorch device : " + str(device))

# Generate the HIV classifier model.
hiv_classifier = get_hiv_classifier(num_train_epochs=30, ensemble_size=5, torch_device=device, 
                                    num_opt_iters=50, batch_size=512)

# Generate the molecule chef model which will select reactants for generating molecules.

# Generate the molecular transformer model which will generate molecules from the reactants.

# Train the latent space of molecule chef using QED scores and predictions from the HIV classifier.

# Build overall final model.

# Generate molecules and output them.

# TODO: For predicting molecules, always normalize the input data by the model's training dataset scaler.
