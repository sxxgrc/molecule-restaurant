import torch

from torch_geometric.data import InMemoryDataset, Data

from chemprop.data import StandardScaler

from numpy import vstack

"""
Generic molecule dataset which serves to hold subsets of a MoleculeNetFeaturesDataset.
This will normalize the atom, edge, and molecule features within that dataset and create
a new copy of it.
"""
class MoleculeDataset(InMemoryDataset):
    def __init__(self, root, name, dataset=None):
        self.dataset = dataset
        self.name = name
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
    
    """
    Processes the provided dataset if it has not been saved in the processed_dir/processed_file_names.
    Specifically, this will copy the data from the given dataset, normalizing the atom, edge, and molecular
    features in the process.
    """
    def process(self):
        # Get all data features for normalization.
        all_molecule_features = []
        all_atom_features = []
        all_edge_features = []
        for d in self.dataset:
            all_molecule_features.append(d.features)
            all_atom_features += d.x
            all_edge_features += d.edge_attr
        
        # Normalize the molecule features in the data.
        molecule_features_scaler = StandardScaler(replace_nan_token=0)
        molecule_features_scaler.fit(vstack(all_molecule_features))

        # Normalize the atom features in the data.
        atom_features_scaler = StandardScaler(replace_nan_token=0)
        atom_features_scaler.fit(vstack(all_atom_features))

        # Normalize the edge features in the data.
        edge_features_scaler = StandardScaler(replace_nan_token=0)
        edge_features_scaler.fit(vstack(all_edge_features))

        # Generate output dataset.
        data_list = []

        for d in self.dataset:
            # Normalize each set of features.
            features = (torch.as_tensor(
                molecule_features_scaler.transform(d.features.reshape(1, -1)[0]), 
                dtype=torch.long).unsqueeze(0))
            
            xs = []
            for x in d.x:
                xs.append(torch.as_tensor(atom_features_scaler.transform(x.reshape(1, -1)[0]),
                    dtype=torch.long))
            x = torch.stack(xs)

            edge_attrs = []
            for edge_attr in d.edge_attr:
                edge_attrs.append(torch.as_tensor(
                    edge_features_scaler.transform(edge_attr.reshape(1, -1)[0]),
                    dtype=torch.long))
            edge_attr = torch.stack(edge_attrs)
            
            # Create new data element using original values and normalized values.
            data = Data(x=x, edge_index=d.edge_index, edge_attr=edge_attr, y=d.y, smiles=d.smiles,
                        features=features, num_bonds_per_atom=d.num_bonds_per_atom,
                        num_atoms_per_mol=d.num_atoms_per_mol)
            data_list.append(data)

        # Save into processed path.
        torch.save(self.collate(data_list), self.processed_paths[0])
