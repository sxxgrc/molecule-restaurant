from torch import nn

"""
The full classifier model which uses a D-MPNN model to create an embedding for the
input molecule, as well as a set of provided molecular features, to predict the classification
of that molecule.
"""
class PredictionModel(nn.Module):
    print("hi")
