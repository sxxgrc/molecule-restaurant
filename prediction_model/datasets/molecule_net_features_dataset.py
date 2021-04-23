import torch, re, os

from torch_geometric.datasets import MoleculeNet
from torch_geometric.data import Data, download_url, extract_gz

from chemprop.features import morgan_binary_features_generator
from chemprop.data import StandardScaler

from rdkit import Chem
from numpy import vstack

x_map = {
    'atomic_num':
    list(range(0, 119)),
    'chirality': [
        'CHI_UNSPECIFIED',
        'CHI_TETRAHEDRAL_CW',
        'CHI_TETRAHEDRAL_CCW',
        'CHI_OTHER',
    ],
    'degree':
    list(range(0, 11)),
    'formal_charge':
    list(range(-5, 7)),
    'num_hs':
    list(range(0, 9)),
    'num_radical_electrons':
    list(range(0, 5)),
    'hybridization': [
        'UNSPECIFIED',
        'S',
        'SP',
        'SP2',
        'SP3',
        'SP3D',
        'SP3D2',
        'OTHER',
    ],
    'is_aromatic': [False, True],
    'is_in_ring': [False, True],
}

e_map = {
    'bond_type': [
        'misc',
        'SINGLE',
        'DOUBLE',
        'TRIPLE',
        'AROMATIC',
    ],
    'stereo': [
        'STEREONONE',
        'STEREOZ',
        'STEREOE',
        'STEREOCIS',
        'STEREOTRANS',
        'STEREOANY',
    ],
    'is_conjugated': [False, True],
}


"""
New dataset class to use for storing the data. This data will wrap around
the generic torch-geometric dataset, providing additional feature vectors
for each molecule as well.
"""
class MoleculeNetFeaturesDataset(MoleculeNet):
    def __init__(self, root, name):
        super(MoleculeNetFeaturesDataset, self).__init__(root, name)
    
    @property
    def raw_file_names(self):
        return f'{self.names[self.name][2]}.csv'

    @property
    def processed_file_names(self):
        return 'data.pt'

    def download(self):
        url = self.url.format(self.names[self.name][1])
        path = download_url(url, self.raw_dir)
        if self.names[self.name][1][-2:] == 'gz':
            extract_gz(path, self.raw_dir)
            os.unlink(path)

    # We make some small revisions to the original process method which is why we copy it here.
    def process(self):
        with open(self.raw_paths[0], 'r') as f:
            dataset = f.read().split('\n')[1:-1]
            dataset = [x for x in dataset if len(x) > 0]  # Filter empty lines.

        data_list = []
        for line in dataset:
            line = re.sub(r'\".*\"', '', line)  # Replace ".*" strings.
            line = line.split(',')

            smiles = line[self.names[self.name][3]]
            ys = line[self.names[self.name][4]]
            ys = ys if isinstance(ys, list) else [ys]

            ys = [float(y) if len(y) > 0 else float('NaN') for y in ys]
            y = torch.tensor(ys, dtype=torch.float).view(1, -1)

            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                continue

            xs = []
            for atom in mol.GetAtoms():
                x = []
                x.append(x_map['atomic_num'].index(atom.GetAtomicNum()))
                x.append(x_map['chirality'].index(str(atom.GetChiralTag())))
                x.append(x_map['degree'].index(atom.GetTotalDegree()))
                x.append(x_map['formal_charge'].index(atom.GetFormalCharge()))
                x.append(x_map['num_hs'].index(atom.GetTotalNumHs()))
                x.append(x_map['num_radical_electrons'].index(
                    atom.GetNumRadicalElectrons()))
                x.append(x_map['hybridization'].index(
                    str(atom.GetHybridization())))
                x.append(x_map['is_aromatic'].index(atom.GetIsAromatic()))
                x.append(x_map['is_in_ring'].index(atom.IsInRing()))
                xs.append(x)

            x = torch.tensor(xs, dtype=torch.long).view(-1, 9)
            num_atoms_per_mol = torch.tensor([len(mol.GetAtoms())], dtype=torch.int)

            num_bonds_per_atom = [0] * len(mol.GetAtoms())
            edge_indices, edge_attrs = [], []
            for bond in mol.GetBonds():
                i = bond.GetBeginAtomIdx()
                j = bond.GetEndAtomIdx()

                e = []
                e.append(e_map['bond_type'].index(str(bond.GetBondType())))
                e.append(e_map['stereo'].index(str(bond.GetStereo())))
                e.append(e_map['is_conjugated'].index(bond.GetIsConjugated()))

                edge_indices += [[i, j], [j, i]]
                edge_attrs += [e, e]

                num_bonds_per_atom[i] += 1
                num_bonds_per_atom[j] += 1

            edge_index = torch.tensor(edge_indices)
            edge_index = edge_index.t().to(torch.long).view(2, -1)
            edge_attr = torch.tensor(edge_attrs, dtype=torch.long).view(-1, 3)
            num_bonds_per_atom = torch.tensor(num_bonds_per_atom, dtype=torch.int)

            # Sort indices.
            if edge_index.numel() > 0:
                perm = (edge_index[0] * x.size(0) + edge_index[1]).argsort()
                edge_index, edge_attr = edge_index[:, perm], edge_attr[perm]

            # Generate feature vector for molecule.
            features = torch.as_tensor(morgan_binary_features_generator(mol)).unsqueeze(0)

            # Create data item for this molecule.
            data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y,
                        smiles=smiles, features=features, num_bonds_per_atom=num_bonds_per_atom,
                        num_atoms_per_mol=num_atoms_per_mol)

            if self.pre_filter is not None and not self.pre_filter(data):
                continue

            if self.pre_transform is not None:
                data = self.pre_transform(data)

            data_list.append(data)

        torch.save(self.collate(data_list), self.processed_paths[0])

    def get_len(self):
        return len(self)

"""
Normalizes the features for a given dataset.
Specifically, normalizes the atom, edge, and molecule features for the dataset.
"""
def normalize_features(dataset):
    # Get all data features for normalization.
    all_molecule_features = []
    all_atom_features = []
    all_edge_features = []
    for d in dataset:
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

    # Add in new values for data features.
    for idx, d in enumerate(dataset):
        dataset[idx].features = (torch.as_tensor(
            molecule_features_scaler.transform(d.features.reshape(1, -1)[0]), 
            dtype=torch.long).unsqueeze(0))
        
        xs = []
        for x in d.x:
            xs.append(torch.as_tensor(
                atom_features_scaler.transform(x.reshape(1, -1)[0]),
                dtype=torch.long))
        dataset[idx].x = torch.stack(xs)

        edge_attrs = []
        for edge_attr in d.edge_attr:
            edge_attrs.append(torch.as_tensor(
                edge_features_scaler.transform(edge_attr.reshape(1, -1)[0]),
                dtype=torch.long))
        dataset[idx].edge_attr = torch.stack(edge_attrs)

    print(dataset[0].features.shape)
