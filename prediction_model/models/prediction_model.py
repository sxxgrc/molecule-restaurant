import numpy, torch

from torch import nn

from .directed_mpnn import DMPNNEncoder

from sklearn.metrics import average_precision_score, roc_auc_score

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

        # Activation function for the middle feed forward neural networks.
        relu = nn.ReLU(inplace=True).to(torch_device)

        # Activation function for the final prediction computation.
        self.sigmoid = nn.Sigmoid().to(torch_device)

        # Dropout layer to use for feed-forward neural networks.
        dropout = nn.Dropout(args.dropout_prob).to(torch_device)

        # The layers of feed-forward neural networks which will compute the property 
        # prediction from the embedding.
        if (args.num_ffn_layers == 1):
            ffn = [dropout, nn.Linear(args.hidden_size + features_dim, 1)]
        else:
            # First layer.
            ffn = [dropout, nn.Linear(args.hidden_size + features_dim, args.hidden_size)]

            # Middle layers.
            for _ in range(args.num_ffn_layers - 2):
                ffn.extend([relu, dropout, 
                    nn.Linear(args.hidden_size, args.hidden_size)])
            
            # Final layer.
            ffn.extend([relu, dropout, nn.Linear(args.hidden_size, 1)])
        
        # Create final FFN model.
        self.ffn = nn.Sequential(*ffn).to(torch_device)

    """
    Predicts whether a given molecule has a specific property or not.

    Parameters:
        - atom_features : Tensor mapping each atom of each molecule to its features.
        - bond_features : Tensor mapping each bond (in both directions) of each molecule to its features.
        - bond_index : Tensor containing the atoms that make up each bond (one row for origin and one for target).
        - molecule_features : Tensor mapping each molecule to its features.
        - atom_incoming_bond_map : Tensor mapping each atom to the indices of its incoming bonds.
        - bond_reverse_map : Tensor mapping each bond to the index of its reverse.
        - num_bonds_per_atom : List containing the number of bonds for each atom.
        - num_atoms_per_mol : List containing the number of atoms for each molecule.
    """
    def forward(self, atom_features, bond_features, bond_index, molecule_features,
                atom_incoming_bond_map, bond_reverse_map, num_bonds_per_atom, num_atoms_per_mol):
        # Compute prediction.
        output = self.encoder(atom_features, bond_features, bond_index, molecule_features, 
                              atom_incoming_bond_map, bond_reverse_map, num_bonds_per_atom,
                              num_atoms_per_mol)
        output = self.ffn(output).squeeze()
            
        # Only apply sigmoid to output when not training, as we will use BCEWithLogitsLoss
        # for training which will apply a sigmoid to the output.
        if not self.training:
            output = self.sigmoid(output)
        
        return output

"""
Creates a PredictionModel and initializes its parameters.
"""
def create_prediction_model(model_args, atom_dim, bond_dim, features_dim, torch_device):
    model = PredictionModel(model_args, atom_dim, bond_dim, features_dim, torch_device)

    # Initialize the weights for the model.
    for param in model.parameters():
        if param.dim() == 1:
            nn.init.constant_(param, 0)
        else:
            nn.init.xavier_normal_(param)
    
    # Send model to device.
    model.to(torch_device)
    
    return model

"""
Helper method for getting model arguments from batch. 
Sends all of the data to the desired torch device.

Parameters:
    - extended_batch : An ExtendedBatch object containing a torch_geometric batch.
    - torch_device : The device to send the tensors to.
"""
def get_model_args_from_batch(extended_batch, torch_device):
    # Send items from torch geometric batch to torch device.
    atom_features = extended_batch.batch.x.to(torch_device)
    bond_features = extended_batch.batch.edge_attr.to(torch_device)
    bond_origins = extended_batch.batch.edge_index[0].to(torch_device)
    molecule_features = extended_batch.batch.features.to(torch_device)
    true_y = extended_batch.batch.y.to(torch_device).squeeze()

    # Send items from external batch to torch device.
    atom_incoming_bond_map = extended_batch.atom_incoming_bond_map.to(torch_device)
    bond_reverse_map = extended_batch.bond_reverse_map.to(torch_device)

    return atom_features, bond_features, bond_origins, molecule_features, atom_incoming_bond_map, \
           bond_reverse_map, extended_batch.batch.num_bonds_per_atom, \
           extended_batch.batch.num_atoms_per_mol, true_y

"""
Trains the prediction model for a given data loader.
"""
def train_prediction_model(model, data_loader, criterion, torch_device, optimizer, scheduler, 
                           scaler, clip):
    # Train.
    model.train()
    torch.set_grad_enabled(True)

    loss_sum = 0
    for data in data_loader:
        # Get separated data.
        atom_features, bond_features, bond_origins, molecule_features, atom_incoming_bond_map,\
            bond_reverse_map, num_bonds_per_atom, num_atoms_per_mol, true_y = \
            get_model_args_from_batch(data, torch_device)

        # Set gradient to zero for iteration.
        optimizer.zero_grad(set_to_none=True)

        # Get output and loss.
        with torch.cuda.amp.autocast():
            y_hat = model(atom_features, bond_features, bond_origins, molecule_features, 
                          atom_incoming_bond_map, bond_reverse_map, num_bonds_per_atom,
                          num_atoms_per_mol)
            loss = criterion(y_hat, true_y)

        # Perform back propagation and optimization.
        scaler.scale(loss).backward()
        nn.utils.clip_grad_value_(model.parameters(), clip_value=clip)
        loss_sum += loss.detach().item() # Get loss item after back propagation.
        scaler.step(optimizer)
        original_scale = scaler.get_scale()
        scaler.update()

        # Takes care of times when optimizer did not step because of NaN or inf
        # issues from scaler.
        if original_scale == scaler.get_scale():
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
        atom_features, bond_features, bond_origins, molecule_features, atom_incoming_bond_map,\
            bond_reverse_map, num_bonds_per_atom, num_atoms_per_mol, true_y = \
            get_model_args_from_batch(data, torch_device)
        
        # Get predictions and true values.
        y_hat = model(atom_features, bond_features, bond_origins, molecule_features, 
                          atom_incoming_bond_map, bond_reverse_map, num_bonds_per_atom,
                          num_atoms_per_mol).detach().cpu().numpy()
        y_pred += y_hat.tolist()
        y_true += true_y.detach().cpu().numpy().tolist()
    return y_pred, y_true

"""
Tests the prediction model on a given data loader.
"""
def test_prediction_model(model, data_loader, torch_device, pos_label):
    # Get predictions.
    y_pred, y_true = get_predictions(model, data_loader, torch_device)

    # Compute metrics.
    aps = average_precision_score(y_true, y_pred, pos_label=pos_label)
    roc_auc = roc_auc_score(y_true, y_pred)
    return aps, roc_auc
