"""
A class for the arguments used throughout the prediction model.
Allows for more readability and easier use of the arguments.
"""
class ModelArgs():
    """
    Initializes a ModelArgs object.

    Parameters:
        - bias : Whether to use a bias for the model.
        - dropout_prob : The probability to use for dropout layers in the model.
        - message_passing_depth : The amount of message passing to do in the MPNN model.
        - torch_device : The PyTorch device to use with the model.
        - hidden_size : The size of the hidden vectors being created in the MPNN model.
        - num_ffn_layers : The number of feedforward neural networks to use in the primary prediction model.
    """
    def __init__(self, bias, dropout_prob, message_passing_depth, torch_device, 
                 hidden_size, num_ffn_layers):
        self.bias = bias
        self.dropout_prob = dropout_prob
        self.message_passing_depth = message_passing_depth
        self.torch_device = torch_device
        self.hidden_size = hidden_size
        self.num_ffn_layers = num_ffn_layers
