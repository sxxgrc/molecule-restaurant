import torch, numpy

from torch import nn

"""
The directed message passing neural network, which takes in molecular graphs and outputs
an embedding for that graph.
This process is entirely based off the D-MPNN described in the paper
Analyzing Learned Molecular Representations for Property Prediction by Yang et al.
"""
class DMPNNEncoder(nn.Module):
    """
    Initializes an object instance of a DMPNNEncoder model.

    Parameters:
        - args : An instance of the ModelArgs class which contains various parameters for the model.
        - atom_dim : Atom feature vector dimension.
        - bond_dim : Bond feature vector dimension.
        - torch_device : The PyTorch device to use.
    """
    def __init__(self, args, atom_dim, bond_dim, torch_device):
        super(DMPNNEncoder, self).__init__()

        # Model arguments.
        self.torch_device = torch_device
        self.message_passing_depth = args.message_passing_depth

        # Cached zero vector to use for atoms without bonds.
        self.cached_zero_vector = nn.Parameter(torch.zeros(args.hidden_size), requires_grad=False)

        # Dropout layer.
        self.dropout_layer = nn.Dropout(args.dropout_prob)

        # Activation function for the linear networks in the model.
        self.relu = nn.ReLU(inplace=True)

        # Batch normalization for training.
        self.bn_i = nn.BatchNorm1d(args.hidden_size)
        self.bn_m = nn.BatchNorm1d(args.hidden_size)
        self.bn_a = nn.BatchNorm1d(args.hidden_size)

        # Input linear model W_i used to calculate h0.
        self.W_i = nn.Linear(atom_dim + bond_dim, args.hidden_size, bias=False)

        # Inner hidden linear model W_m used to calculate h{t+1}.
        self.W_m = nn.Linear(args.hidden_size, args.hidden_size, bias=False)

        # Output linear model W_a used to calculate h{v}, the molecular representation of the result.
        self.W_a = nn.Linear(atom_dim + args.hidden_size, args.hidden_size, bias=False)

    """
    Computes the next bond message for the given last hidden state.
    Specifically computes: m_{t+1} for bond {v, w} = sum of h_t for all incoming bonds to atom v
    except for the bond {w, v}
    """
    def compute_next_bond_message(self, h_t, max_num_bonds, bond_incoming_bond_map, b2rev):
        # Add a zero vector to top of h_t to allow for padding.
        padded_h_t = torch.cat(
            (torch.as_tensor([[0]], device=self.torch_device).expand(1, h_t.shape[1]), h_t), dim=0)

        # Get all of the rows for each bond that correspond to its incoming bonds.
        all_incoming_bonds = torch.index_select(padded_h_t, 0, bond_incoming_bond_map.view(-1))

        # Split the previous tensor into chunks so that each chunk corresponds to one bond, containing
        # all of the rows needed for the sum (i.e. the incoming bond vectors).
        chunked_incoming_bonds = torch.split(all_incoming_bonds, max_num_bonds)

        # Sum up each chunk and stack them back together.
        summed_chunks = torch.stack([torch.sum(chunk, dim=0) for chunk in chunked_incoming_bonds])

        # Get the rows for the reverse bonds for each bond so we can subtract them from the sum.
        reverse_bond_rows = torch.index_select(h_t, 0, b2rev)

        # Subtract and output the final m_{t+1} result.
        return summed_chunks - reverse_bond_rows

    """
    Encodes a batch of molecules.

    Parameters:
        - atom_features : Tensor mapping each atom of each molecule to its features.
        - bond_features : Tensor mapping each bond (in both directions) of each molecule to its features.
        - bond_origins : Tensor containing the atoms that originate each bond.
        - molecule_features : Tensor mapping each molecule to its features.
        - atom_incoming_bond_map : Tensor mapping each atom to the indices of its incoming bonds.
        - bond_reverse_map : Tensor mapping each bond to the index of its reverse.
        - num_bonds_per_atom : List containing the number of bonds for each atom.
        - num_atoms_per_mol : List containing the number of atoms for each molecule.
    """
    def forward(self, atom_features, bond_features, bond_origins, molecule_features, atom_incoming_bond_map,
                bond_reverse_map, num_bonds_per_atom, num_atoms_per_mol):
        if (bond_features.shape[0] > 0):
            # Concatenate the edge feature matrix with the node feature matrix (edge vw concats with node v).
            expanded_x = torch.index_select(atom_features, 0, bond_origins)
            edge_node_feat_mat = torch.cat((bond_features, expanded_x), dim=1).float()

            # Generate h_0 for each edge.
            h_0 = self.relu(self.bn_i(self.W_i(edge_node_feat_mat)))
            h_t = h_0

            # Expand the atom to incoming bond map for all of the bonds in the dataset.
            bond_incoming_bond_map = torch.index_select(atom_incoming_bond_map, 0, bond_origins)

            # Message passing phase.
            for _ in range(self.message_passing_depth):
                # Compute the next message for each edge.
                m_t = self.compute_next_bond_message(h_t, atom_incoming_bond_map.shape[1], 
                                                    bond_incoming_bond_map, bond_reverse_map)

                # Compute the next hidden state for each edge.
                h_t = self.relu(h_0 + self.bn_m(self.W_m(m_t)))

                # Add dropout layer to not overtrain.
                h_t = self.dropout_layer(h_t)

            # Get atom-wise representation of messages. Some atoms have 0 bonds so we deal with that as well.
            atom_chunks = torch.split(h_t, num_bonds_per_atom)
            m_v = torch.stack([torch.sum(atom_chunk, 0) if atom_chunk.numel() else 
                                self.cached_zero_vector for atom_chunk in atom_chunks])
        else:
            m_v = (torch.as_tensor([0], device=self.torch_device, dtype=torch.float)
                        .expand(atom_features.shape[0], self.cached_zero_vector.shape[0]))

        # Get the atom-wise representation of hidden states by concating node features to node messages.
        node_feat_message = torch.cat((atom_features, m_v), dim=1)
        h_v = self.relu(self.bn_a(self.W_a(node_feat_message)))
        h_v = self.dropout_layer(h_v)

        # Readout phase, which creates the molecule representation from the atom representations.
        molecule_chunks = torch.split(h_v, num_atoms_per_mol)
        h = torch.stack([torch.sum(molecule_chunk, 0) / molecule_chunk.shape[0] for 
                         molecule_chunk in molecule_chunks])

        # Concatenate molecular representation with external features and output.
        final_representation = torch.cat((h, molecule_features), dim=1)
        return final_representation
