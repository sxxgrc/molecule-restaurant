import numpy, torch

from torch import nn

from .directed_mpnn import DMPNNEncoder

from sklearn.metrics import f1_score, roc_auc_score

"""
The full classifier model which uses a D-MPNN model to create an embedding for the
input molecule, as well as a set of provided molecular features, to predict the classification
of that molecule.
"""
class PredictionModel(nn.Module):
    """
    Initializes an object instance of a PredictionModel.

    Parameters:
        - args : An instance of the ModelArgs class which contains various parameters for the model.
        - atom_dim : Atom feature vector dimension.
        - bond_dim : Bond feature vector dimension.
        - features_dim : The dimension of the additional feature vector.
        - torch_device : The PyTorch device to use with the model.
    """
    def __init__(self, args, atom_dim, bond_dim, features_dim, torch_device):
        super(PredictionModel, self).__init__()

        # Encoder model, which will encode the input molecule.
        self.encoder = DMPNNEncoder(args, atom_dim, bond_dim, torch_device).to(torch_device)

        # Activation function for the feed-forward neural networks which will compute prediction.
        self.sigmoid = nn.Sigmoid().to(torch_device)

        # Dropout layer to use for feed-forward neural networks.
        dropout = nn.Dropout(args.ffn_dropout_prob).to(torch_device)

        # The layers of feed-forward neural networks which will compute the property 
        # prediction from the embedding.
        if (args.num_ffn_layers == 1):
            ffn = [dropout, nn.Linear(args.hidden_size + features_dim, 1).to(torch_device)]
        else:
            # First layer.
            ffn = [dropout, nn.Linear(args.hidden_size + features_dim, args.ffn_hidden_size).to(torch_device)]

            # Middle layers.
            for _ in range(args.num_ffn_layers - 2):
                ffn.extend([self.sigmoid, dropout, 
                    nn.Linear(args.ffn_hidden_size, args.ffn_hidden_size).to(torch_device)])
            
            # Final layer.
            ffn.extend([self.sigmoid, dropout, nn.Linear(args.ffn_hidden_size, 1).to(torch_device)])
        
        # Create final FFN model.
        self.ffn = nn.Sequential(*ffn).to(torch_device)

        # Initialize the weights for the model.
        for param in self.parameters():
            if param.dim() == 1:
                nn.init.constant_(param, 0)
            else:
                nn.init.xavier_normal_(param)

    """
    Predicts whether a given molecule has a specific property or not.

    Parameters:
        - atom_features : Tensor mapping each atom of each molecule to its features.
        - bond_features : Tensor mapping each bond (in both directions) of each molecule to its features.
        - bond_index : Tensor containing the atoms that make up each bond (one row for origin and one for target).
        - molecule_features : Tensor mapping each molecule to its features.
        - atom_to_molecule : Tensor mapping each atom to its molecule.
    """
    def forward(self, atom_features, bond_features, bond_index, molecule_features, atom_to_molecule):
        # Compute prediction.
        output = self.encoder(atom_features, bond_features, bond_index, molecule_features, atom_to_molecule)
        output = self.ffn(output)
            
        # Only apply sigmoid to output when not training, as we will use BCEWithLogitsLoss
        # for training which will apply a sigmoid to the output.
        if not self.training:
            output = self.sigmoid(output)
        
        return output

"""
Helper method for getting model arguments from batch. 
Sends all of the data to the desired torch device.

Parameters:
    - batch : The batch of data to separate.
    - torch_device : The device to store the data in.

Returns:
    - atom_features : Tensor mapping each atom of each molecule to its features.
    - bond_features : Tensor mapping each bond (in both directions) of each molecule to its features.
    - bond_index : Tensor containing the atoms that make up each bond (one row for origin and one for target).
    - molecule_features : Tensor mapping each molecule to its features.
    - atom_to_molecule : Tensor mapping each atom to its molecule.
    - true_y : Tensor containing the actual label for each molecule.
"""
def get_model_args_from_batch(batch, torch_device):
    atom_features = batch.x.to(torch_device)
    bond_features = batch.edge_attr.to(torch_device)
    bond_index = batch.edge_index.to(torch_device)
    molecule_features = batch.features.to(torch_device)
    atom_to_molecule = batch.batch.to(torch_device)
    true_y = batch.y.to(torch_device)

    return atom_features, bond_features, bond_index, molecule_features, atom_to_molecule, true_y

"""
Trains the prediction model for a given data loader.
"""
def train_prediction_model(model, data_loader, criterion, torch_device, optimizer, scheduler, scaler):
    # Train.
    model.train()
    torch.set_grad_enabled(True)

    loss_sum = 0
    print()
    for data in data_loader:
        # Get separated data.
        atom_features, bond_features, bond_index, molecule_features, atom_to_molecule, true_y = \
            get_model_args_from_batch(data, torch_device)

        # Set gradient to zero for iteration.
        optimizer.zero_grad(set_to_none=True)

        # Get output and loss.
        with torch.cuda.amp.autocast():
            y_hat = model(atom_features, bond_features, bond_index, 
                molecule_features, atom_to_molecule)
            print(y_hat.detach().cpu().numpy())
            loss = criterion(y_hat, true_y)

        # Perform back propagation and optimization.
        scaler.scale(loss).backward()
        loss_sum += loss.detach().item() # Get loss item after back propagation.
        scaler_result = scaler.step(optimizer)
        scaler.update()

        # When this is none, the scaler created NaN or inf gradients and optimizer step was skipped.
        if scaler_result != None:
            scheduler.step()
    
    print("Post-training prediction model loss: " + str(loss_sum / len(data_loader)))


"""
Helper method which gets prediction results for a given model and data loader.
"""
def get_predictions(model, data_loader, torch_device):
    model.eval()
    torch.set_grad_enabled(False)

    # Get predictions.
    y_pred = []
    y_pred_labels = []
    y_true = []
    for data in data_loader:
        # Get separated data.
        atom_features, bond_features, bond_index, molecule_features, atom_to_molecule, true_y = \
            get_model_args_from_batch(data, torch_device)
        
        # Get predictions and true values.
        y_hat = model(atom_features, bond_features, bond_index, molecule_features, 
            atom_to_molecule).detach().cpu().numpy()
        y_pred += y_hat.tolist()
        y_pred_labels += numpy.round(y_hat).tolist()
        y_true += true_y.detach().cpu().numpy().tolist()
    
    print(y_pred)
    print(y_pred_labels)
    print(y_true)
    print()
    return y_pred, y_pred_labels, y_true

"""
Tests the prediction model on a given data loader.
"""
def test_prediction_model(model, data_loader, torch_device):
    # Get predictions.
    y_pred, y_pred_labels, y_true = get_predictions(model, data_loader, torch_device)

    # Compute metrics.
    f1 = f1_score(y_true, y_pred_labels)
    roc_auc = roc_auc_score(y_true, y_pred)
    return f1, roc_auc
