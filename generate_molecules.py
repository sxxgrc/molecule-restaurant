"""
Primary script for MoleculeRestaurant to generate molecules.
"""

import torch

################################################################
# Initialization of Environment
################################################################

# Grab the device to use for the training.
if torch.cuda.is_available():
    dev = "cuda:0"
else:
    dev = "cpu"

device = torch.device(dev)
