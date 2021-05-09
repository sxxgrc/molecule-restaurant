import numpy

from torch import nn

"""
The full HIV replication inhibition classifier model.
This model is built from an ensemble of PredictionModel objects which were trained on HIV replication
inhibition data.
Note that this model is built from other trained models, so this model does not need to be trained.

Predicts whether a molecule will inhibit HIV replication (0) or not (1).
"""
class HIVClassifier(nn.Module):
    """
    Initializes an object instance of an HIVClassifier model.

    Parameters:
        - models: The set of models to use in an ensembled form for the prediction.
    """
    def __init__(self, models):
        super(HIVClassifier, self).__init__()
        self.models = models
    
    """
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
        predictions = None

        # Get predictions for each model.
        for model in self.models:
            model.eval()
            y_pred = model(atom_features, bond_features, bond_index, molecule_features,
                           atom_incoming_bond_map, bond_reverse_map, num_bonds_per_atom, 
                           num_atoms_per_mol).detach().numpy()

            if predictions == None:
                predictions = numpy.array(y_pred)
            else:
                predictions += y_pred

        # Average predictions and return.
        predictions /= len(self.models)
        return predictions
