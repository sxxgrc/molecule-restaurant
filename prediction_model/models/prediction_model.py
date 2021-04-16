"""
The primary script for creating and training the D-MPNN for classifying molecules
against some specific property.

At the end, this script outputs a model, trained with the entirety of the dataset, to the
trained-models directory, which can then be used by the overall MoleculeRestaurant network.
"""

from torch import nn

"""
A class for the arguments used throughout the prediction model.
Allows for more readability and easier use of the arguments.
"""
class ModelArgs():
    """
    Initializes a ModelArgs object.

    Parameters:
        - bias : The bias to use for the model.
        - dropout_prob : The probability to use for dropout layers in the model.
        - message_passing_depth : The amount of message passing to do in the MPNN model.
        - torch_device : The PyTorch device to use with the model.
        - bond_message_vec_size : The size of the bond message vectors being created in the MPNN model.
        - num_ffn_layers : The number of feedforward neural networks to use in the primary prediction model.
    """
    def __init__(self, bias, dropout_prob, message_passing_depth, torch_device, 
                 bond_message_vec_size, num_ffn_layers):
        self.bias = bias
        self.dropout_prob = dropout_prob
        self.message_passing_depth = message_passing_depth
        self.torch_device = torch_device
        self.bond_message_vec_size = bond_message_vec_size
        self.num_ffn_layers = num_ffn_layers


"""
The directed message passing neural network, which takes in molecular graphs and outputs
an embedding for that graph.
"""
class DMPNNEncoder(nn.Module):
    """
    Initializes an object instance of a DMPNNEncoder model.

    Parameters:
        - args : An instance of the ModelArgs class which contains various parameters for the model.
    """
    def __init__(self, args):
        super(DMPNNEncoder, self).__init__()
        


"""
The full classifier model which uses the D-MPNN model to create an embedding for the
input molecule, as well as a set of provided molecular features, to predict the classification
of that molecule.
"""
class PredictionModel(nn.Module):
    print("hi")
