import torch, re, os

from torch_geometric.datasets import MoleculeNet
from torch_geometric.data import Data, download_url, extract_gz

from chemprop.features import rdkit_2d_normalized_features_generator as features_generator

from rdkit import Chem

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
Generates atom features for a given molecule.

Returns:
    - x : A tensor containing the features for the atoms in the molecule.
    - num_atoms_per_mol : The number of atoms in the molecule.
"""
def generate_atom_features(mol):
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
    num_atoms = torch.tensor([len(mol.GetAtoms())], dtype=torch.int)

    return x, num_atoms

"""
Generates the bond features for a given molecule.

Returns:
    - edge_index : List of each atom in the edges
    - edge_attr : The features for each edge
    - num_bonds_per_atom : Mapping of the number of bonds in each atom
"""
def generate_bond_features(mol):
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

    return edge_index, edge_attr, num_bonds_per_atom

"""
New dataset class to use for storing the data. This data will wrap around
the generic torch-geometric dataset, providing additional feature vectors
for each molecule as well.
"""
class MoleculeNetFeaturesDataset(MoleculeNet):
    def __init__(self, root, name):
        super(MoleculeNetFeaturesDataset, self).__init__(root, name)

    # Copied over from MoleculeNet.
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
            # Get the SMILES string for the molecule.
            line = re.sub(r'\".*\"', '', line)  # Replace ".*" strings.
            line = line.split(',')
            smiles = line[self.names[self.name][3]]

            # Get actual label for molecule.
            ys = line[self.names[self.name][4]]
            ys = ys if isinstance(ys, list) else [ys]
            ys = [float(y) if len(y) > 0 else float('NaN') for y in ys]
            y = torch.tensor(ys, dtype=torch.float).view(1, -1)

            # Get rdkit molecule version from SMILES.
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                continue

            # Generate the atom features for the molecule.
            x, num_atoms_per_mol = generate_atom_features(mol)

            # Generate the edge features for the molecule.
            edge_index, edge_attr, num_bonds_per_atom = generate_bond_features(mol)

            # Generate feature vector for molecule.
            features = torch.as_tensor(features_generator(mol))

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
