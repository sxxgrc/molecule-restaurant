"""
Primary script for MoleculeRestaurant to generate molecules.
"""

import torch

from prediction_model import get_hiv_classifier

################################################################
# Initialization of Environment
################################################################

# Grab the device to use for the training.
if torch.cuda.is_available():
    dev = "cuda:0"
else:
    dev = "cpu"

device = torch.device(dev)
print()
print("Using PyTorch device : " + str(device))

hiv_classifier = get_hiv_classifier(30, 5, device, 30, 512)

# For predicting molecules, always normalize the input data by the model's training dataset scaler.
