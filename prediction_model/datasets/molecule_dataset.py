import torch, pickle

from torch_geometric.data import InMemoryDataset, Data

from chemprop.data import StandardScaler

from numpy import vstack

from os import path

"""
Generic molecule dataset which serves to hold subsets of a MoleculeNetFeaturesDataset.
This will normalize the atom, edge, and molecule features within that dataset and create
a new copy of it.
"""
class MoleculeDataset(InMemoryDataset):
    def __init__(self, root, name, dataset=None, molecule_scaler=None, bond_scaler=None, atom_scaler=None):
        self.dataset = dataset
        self.name = name
        self.molecule_scaler = molecule_scaler
        self.bond_scaler = bond_scaler
        self.atom_scaler = atom_scaler
        super(MoleculeDataset, self).__init__(root, None, None, None)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_dir(self):
        return self.root

    @property
    def raw_file_names(self):
        return self.name + ".pt"
    
    def download(self):
        # No-Op.
        return

    @property
    def processed_dir(self):
        return self.root
    
    @property
    def processed_file_names(self):
        return self.name + ".pt"

    @property
    def scalers_file_paths(self):
        base_path = self.root + "/"
        return {"molecule": base_path + "molecule_scaler.pkl", "bond": base_path + "bond_scaler.pkl",
                "atom": base_path + "atom_scaler.pkl"}

    """
    Generates the molecule, bond, and atom scalers for the dataset.
    """
    def generate_scalers(self):
        print("Creating new molecule, atom, and bond scalers for dataset.")

        # Get all data features for normalization.
        all_molecule_features = []
        all_atom_features = []
        all_bond_features = []
        for d in self.dataset:
            all_molecule_features.append(d.features)
            all_atom_features += d.x
            all_bond_features += d.edge_attr
        
        # Generate and save molecule feature scaler for data.
        molecule_scaler = StandardScaler(replace_nan_token=0)
        molecule_scaler.fit(vstack(all_molecule_features))
        self.molecule_scaler = molecule_scaler
        with open(self.scalers_file_paths["molecule"], "wb") as f:
            pickle.dump(molecule_scaler, f, pickle.HIGHEST_PROTOCOL)

        # Generate and save atom feature scaler for data.
        atom_scaler = StandardScaler(replace_nan_token=0)
        atom_scaler.fit(vstack(all_atom_features))
        self.atom_scaler = atom_scaler
        with open(self.scalers_file_paths["atom"], "wb") as f:
            pickle.dump(atom_scaler, f, pickle.HIGHEST_PROTOCOL)

        # Generate and save bond feature scaler for data.
        bond_scaler = StandardScaler(replace_nan_token=0)
        bond_scaler.fit(vstack(all_bond_features))
        self.bond_scaler = bond_scaler
        with open(self.scalers_file_paths["bond"], "wb") as f:
            pickle.dump(bond_scaler, f, pickle.HIGHEST_PROTOCOL)

    """
    Gets the scalers being used by this dataset.
    """
    def get_scalers(self):
        # If any of the scalers are None they must all be None so we create them.
        if self.molecule_scaler is None:
            if (not path.exists(self.scalers_file_paths["molecule"]) 
                or not path.getsize(self.scalers_file_paths["molecule"]) > 0):
                self.generate_scalers()
            else:
                with open(self.scalers_file_paths["molecule"], 'rb') as input:
                    self.molecule_scaler = pickle.load(input)
                with open(self.scalers_file_paths["bond"], 'rb') as input:
                    self.bond_scaler = pickle.load(input)
                with open(self.scalers_file_paths["atom"], 'rb') as input:
                    self.atom_scaler = pickle.load(input)
            
        return self.molecule_scaler, self.bond_scaler, self.atom_scaler
    
    """
    Processes the provided dataset if it has not been saved in the processed_dir/processed_file_names.
    Specifically, this will copy the data from the given dataset, normalizing the atom, edge, and molecular
    features in the process.

    We only create scalers for the main training dataset, which are then used for the validation and
    testing datasets.
    """
    def process(self):
        # If any of the scalers are None they must all be None so we create them.
        if self.molecule_scaler is None:
            self.generate_scalers()

        # Generate output dataset.
        data_list = []

        for d in self.dataset:
            # Normalize each set of features.
            features = (torch.as_tensor(
                self.molecule_scaler.transform(d.features.reshape(1, -1)[0]), 
                dtype=torch.long).unsqueeze(0))
            
            xs = []
            for x in d.x:
                xs.append(torch.as_tensor(self.atom_scaler.transform(x.reshape(1, -1)[0]),
                    dtype=torch.long))
            x = torch.stack(xs)

            edge_attrs = []
            for edge_attr in d.edge_attr:
                edge_attrs.append(torch.as_tensor(
                    self.bond_scaler.transform(edge_attr.reshape(1, -1)[0]),
                    dtype=torch.long))
            edge_attr = torch.stack(edge_attrs)
            
            # Create new data element using original values and normalized values.
            data = Data(x=x, edge_index=d.edge_index, edge_attr=edge_attr, y=d.y, smiles=d.smiles,
                        features=features, num_bonds_per_atom=d.num_bonds_per_atom,
                        num_atoms_per_mol=d.num_atoms_per_mol)
            data_list.append(data)

        # Save into processed path.
        torch.save(self.collate(data_list), self.processed_paths[0])
