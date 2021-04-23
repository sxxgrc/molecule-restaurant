import numpy, torch

from threading import Thread

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

        # Activation function for the middle feed forward neural networks.
        relu = nn.LeakyReLU(inplace=True).to(torch_device)

        # Activation function for the final prediction computation.
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
                ffn.extend([relu, dropout, 
                    nn.Linear(args.ffn_hidden_size, args.ffn_hidden_size).to(torch_device)])
            
            # Final layer.
            ffn.extend([relu, dropout, nn.Linear(args.ffn_hidden_size, 1).to(torch_device)])
        
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
        - atom_incoming_bond_map : Tensor mapping each atom to the indices of its incoming bonds.
        - bond_reverse_map : Tensor mapping each bond to the index of its reverse.
        - atom_chunk_indices : List of masks which contain the indices of each atom's bonds.
        - molecule_chunk_indices : List of masks which contain indices of each molecule's atoms.
    """
    def forward(self, atom_features, bond_features, bond_index, molecule_features,
                atom_incoming_bond_map, bond_reverse_map, atom_chunk_indices, molecule_chunk_indices):
        # Compute prediction.
        output = self.encoder(atom_features, bond_features, bond_index, molecule_features, 
                              atom_incoming_bond_map, bond_reverse_map, atom_chunk_indices, 
                              molecule_chunk_indices)
        output = self.ffn(output).squeeze()
            
        # Only apply sigmoid to output when not training, as we will use BCEWithLogitsLoss
        # for training which will apply a sigmoid to the output.
        if not self.training:
            output = self.sigmoid(output)
        
        return output

"""
Generates a mapping of each atom to the indices of the bonds coming into that atom.
Parameters:
    - num_atoms : The total number of atoms in the input.
    - incoming_edge_points : List of the end indices for all edges in the input.
    - results : Array to place value into.
    - results_idx : Index for result.
"""
def get_atom_incoming_bond_map(num_atoms, incoming_edge_points, results, results_idx):
    # Convert edge tensor to numpy.
    incoming_edge_points = incoming_edge_points.detach().cpu().numpy()

    # Initialize mapping.
    a2b = [[] for _ in range(num_atoms)]

    # Find all of the indices where the atom is the end point of an edge. We add 1
    # to the index to account for padding.
    max_bonds = 0
    for edge_idx in range(len(incoming_edge_points)):
        a2b[incoming_edge_points[edge_idx]].append(edge_idx + 1)
        max_bonds = max(max_bonds, len(a2b[incoming_edge_points[edge_idx]]))
    
    # Pad the mapping with zeros and convert to tensor.
    results[results_idx] = [numpy.pad(a2b[idx], (0, max_bonds - len(a2b[idx])), 'constant') 
        for idx in range(num_atoms)]

"""
Generates a mapping of each bond to the index of its reverse bond.
That is, for the bond {0, 1}, this function will generate a tensor where the initial
index for {0, 1} will contain the index of the bond {1, 0}.
"""
def get_bond_reverse_map(edge_index, num_edges, results, results_idx):
    # Generate mapping of each edge to its index.
    edge_index = edge_index.detach().cpu().numpy()
    edge_index_map = {(edge_index[0][idx], edge_index[1][idx]) : idx
        for idx in range(num_edges)}

    # Generate the bond reverse map.
    results[results_idx] = [edge_index_map[(edge_index[1][idx], edge_index[0][idx])] 
        for idx in range(num_edges)]

"""
Gets a mask for each atom's bonds.
Specifically, this returns a list of lists, where each inner list corresponds to an atom, and
the list is a boolean mask for the positions of the atom's bonds.

Parameters:
    - num_atoms : The number of atoms in the current batch.
    - bond_to_atom_map : Mapping of each bond to the index of the atom it originates from.
    - results : Array to place value into.
    - results_idx : Index for result.
"""
def get_atom_chunk_mask(num_atoms, bond_to_atom_map, results, results_idx):
    results[results_idx] = [[bond_to_atom_map == i] for i in range(num_atoms)]

"""
Gets a mask for each molecule's atoms.
Specifically, this returns a list of lists, where each inner list corresponds to a molecule,
and the list is a boolean mask for the positions of the molecule's atoms.

Parameters:
    - atom_to_molecule : Mapping of each atom to the index of its molecule.
    - num_molecules : The number of molecules in the batch.
    - results : Array to place value into.
    - results_idx : Index for result.
"""
def get_molecule_chunk_mask(atom_to_molecule, num_molecules, results, results_idx):
    results[results_idx] = [[atom_to_molecule == i] for i in range(num_molecules)]

"""
Helper method for getting model arguments from batch. 
Sends all of the data to the desired torch device.

Parameters:
    - batch : The batch of data to separate.
    - torch_device : The device to store the data in.

Returns:
    - atom_features : Tensor mapping each atom of each molecule to its features.
    - bond_features : Tensor mapping each bond (in both directions) of each molecule to its features.
    - bond_index : Tensor containing atoms that make up each bond (one row for origin one for target).
    - molecule_features : Tensor mapping each molecule to its features.
    - atom_incoming_bond_map : Tensor mapping each atom to the indices of its incoming bonds.
    - bond_reverse_map : Tensor mapping each bond to the index of its reverse.
    - atom_chunk_mask : List of masks which contain the indices of each atom's bonds.
    - molecule_chunk_mask : List of masks which contain indices of each molecule's atoms.
    - true_y : Tensor containing the actual label for each molecule.
"""
def get_model_args_from_batch(batch, torch_device):
    # Do CPU intensive computations first.
    threads = [None] * 4
    results = [None] * 4

    threads[0] = Thread(target=get_atom_incoming_bond_map, args=(batch.x.shape[0], batch.edge_index[1],
                                                                 results, 0))
    threads[0].start()
    threads[1] = Thread(target=get_bond_reverse_map, args=(batch.edge_index, batch.edge_index.shape[1],
                                                           results, 1))
    threads[1].start()
    threads[2] = Thread(target=get_atom_chunk_mask, args=(batch.x.shape[0], batch.edge_index[0], results, 2))
    threads[2].start()
    threads[3] = Thread(target=get_molecule_chunk_mask, args=(batch.batch, batch.y.shape[0], results, 3))
    threads[3].start()

    # Collect threads.
    for i in range(len(threads)):
        threads[i].join()

    # Send all to correct device.
    atom_features = batch.x.to(torch_device)
    bond_features = batch.edge_attr.to(torch_device)
    bond_index = batch.edge_index.to(torch_device)
    molecule_features = batch.features.to(torch_device)
    true_y = batch.y.to(torch_device).squeeze()
    atom_incoming_bond_map = torch.as_tensor(results[0], device=torch_device, dtype=torch.int)
    bond_reverse_map = torch.as_tensor(results[1], device=torch_device, dtype=torch.int)

    return atom_features, bond_features, bond_index, molecule_features, atom_incoming_bond_map, \
           bond_reverse_map, results[2], results[3], true_y

"""
Trains the prediction model for a given data loader.
"""
def train_prediction_model(model, data_loader, criterion, torch_device, optimizer, scheduler, scaler):
    # Train.
    model.train()
    torch.set_grad_enabled(True)

    loss_sum = 0
    for data in data_loader:
        # Get separated data.
        atom_features, bond_features, bond_index, molecule_features, atom_incoming_bond_map,\
            bond_reverse_map, atom_chunk_mask, molecule_chunk_mask, true_y = \
            get_model_args_from_batch(data, torch_device)

        # Set gradient to zero for iteration.
        optimizer.zero_grad(set_to_none=True)

        # Get output and loss.
        with torch.cuda.amp.autocast():
            y_hat = model(atom_features, bond_features, bond_index, molecule_features, 
                          atom_incoming_bond_map, bond_reverse_map, atom_chunk_mask,
                          molecule_chunk_mask)
            loss = criterion(y_hat, true_y)

        # Perform back propagation and optimization.
        scaler.scale(loss).backward()
        nn.utils.clip_grad_value_(model.parameters(), clip_value=5.0)
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
        atom_features, bond_features, bond_index, molecule_features, atom_incoming_bond_map,\
            bond_reverse_map, atom_chunk_mask, molecule_chunk_mask, true_y = \
            get_model_args_from_batch(data, torch_device)
        
        # Get predictions and true values.
        y_hat = model(atom_features, bond_features, bond_index, molecule_features, 
                          atom_incoming_bond_map, bond_reverse_map, atom_chunk_mask,
                          molecule_chunk_mask).detach().cpu().numpy()
        y_pred += y_hat.tolist()
        y_pred_labels += numpy.round(y_hat).tolist()
        y_true += true_y.detach().cpu().numpy().tolist()
    return y_pred, y_pred_labels, y_true

"""
Tests the prediction model on a given data loader.
"""
def test_prediction_model(model, data_loader, torch_device, pos_label):
    # Get predictions.
    y_pred, y_pred_labels, y_true = get_predictions(model, data_loader, torch_device)

    # Compute metrics.
    f1 = f1_score(y_true, y_pred_labels, pos_label=pos_label)
    roc_auc = roc_auc_score(y_true, y_pred)
    return f1, roc_auc
