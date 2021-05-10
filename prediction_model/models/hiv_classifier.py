import numpy, torch

from rdkit import Chem

from chemprop.features import rdkit_2d_normalized_features_generator as features_generator

from prediction_model.datasets.molecule_net_features_dataset import (
    generate_atom_features, generate_bond_features
)

from prediction_model.dataloaders import get_bond_reverse_map, get_atom_incoming_bond_map

"""
The full HIV replication inhibition classifier model.
This model is built from an ensemble of PredictionModel objects which were trained on HIV replication
inhibition data.

Predicts probability that a molecule will not inhibit HIV replication.
"""
class HIVClassifier():
    """
    Initializes an object instance of an HIVClassifier model.

    Parameters:
        - models: The set of models to use in an ensembled form for the prediction.
        - molecule_scaler : Normalization scaler for molecule features based on the training dataset for the models.
        - bond_scaler : Normalization scaler for bond features based on the training dataset for the models.
        - atom_scaler : Normalization scaler for atom features based on the training dataset for the models.
        - torch_device : The PyTorch device used for the model.
    """
    def __init__(self, models, molecule_scaler, bond_scaler, atom_scaler, torch_device):
        self.models = models
        self.molecule_scaler = molecule_scaler
        self.bond_scaler = bond_scaler
        self.atom_scaler = atom_scaler
        self.torch_device = torch_device

    """
    Converts the given molecule SMILES string into a valid input into the internal prediction models.
    The features are normalized using the model's scalers.

    Returns:
        - atom_features : Tensor mapping each atom to its features, normalized.
        - num_atoms : The number of atoms in the molecule.
        - bond_origins : Tensor containing the atoms that start each bond.
        - bond_features : Tensor mapping each bond (in both directions) to its features, normalized.
        - num_bonds_per_atom : List containing the number of bonds for each atom.
        - molecule_features : Tensor of the molecule's features, normalized.
        - atom_incoming_bond_map : Tensor mapping each atom to the indices of its incoming bonds.
        - bond_reverse_map : Tensor mapping each bond to the index of its reverse.
    """
    def convert_to_input(self, molecule_smiles):
        # Convert SMILES to actual molecule.
        mol = Chem.MolFromSmiles(molecule_smiles)

        # Get atom features.
        atom_features, num_atoms = generate_atom_features(mol)
        num_atoms = num_atoms.tolist()

        # Normalize the atom features.
        xs = []
        for x in atom_features:
            xs.append(torch.as_tensor(self.atom_scaler.transform(x.reshape(1, -1)[0]), dtype=torch.long))
        atom_features = torch.stack(xs).to(self.torch_device)

        # Get bond features.
        bond_index, bond_features, num_bonds_per_atom = generate_bond_features(mol, atom_features)
        bond_origins = bond_index[0].to(self.torch_device)
        num_bonds_per_atom = num_bonds_per_atom.tolist()

        # Normalize the bond features.
        edge_attrs = []
        for edge_attr in bond_features:
            edge_attrs.append(torch.as_tensor(self.bond_scaler.transform(edge_attr.reshape(1, -1)[0]), 
                             dtype=torch.long))
        if len(edge_attrs) > 0:
            bond_features = torch.stack(edge_attrs).to(self.torch_device)
        else:
            bond_features = torch.tensor([]).to(self.torch_device)

        # Get molecule features.
        molecule_features = torch.as_tensor(features_generator(molecule_smiles))

        # Normalize the molecule features.
        molecule_features = torch.as_tensor(self.molecule_scaler.transform(molecule_features.reshape(1, -1)[0]),
                             dtype=torch.long).unsqueeze(0)
        molecule_features = molecule_features.to(self.torch_device)

        # Get atom incoming bond map.
        atom_incoming_bond_map = get_atom_incoming_bond_map(atom_features, bond_index)
        atom_incoming_bond_map = atom_incoming_bond_map.to(self.torch_device)

        # Get bond reverse map.
        bond_reverse_map = get_bond_reverse_map(bond_index)
        bond_reverse_map = bond_reverse_map.to(self.torch_device)

        return (atom_features, bond_features, bond_origins, molecule_features,
                atom_incoming_bond_map, bond_reverse_map, num_bonds_per_atom, num_atoms)
    
    """
    Generates a prediction for the HIV replication inhibition label for the given molecule.

    Parameters:
        - molecule_smiles : The SMILES string for the input molecule.
    """
    def predict(self, molecule_smiles):
        # Convert the molecule into the desired arguments.
        model_args = self.convert_to_input(molecule_smiles)

        # Get predictions for each model.
        prediction = 0
        for model in self.models:
            model.eval()
            y_pred = model(*model_args).detach().item()
            prediction += y_pred

        # Average predictions and return.
        prediction /= len(self.models)
        return prediction
