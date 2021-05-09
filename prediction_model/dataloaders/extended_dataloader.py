import torch, numpy

from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from torch.utils.data.sampler import WeightedRandomSampler
from torch._six import container_abcs, string_classes, int_classes

from torch_geometric.data import Data, Batch

"""
Generates a mapping of each bond to the index of its reverse bond.
That is, for the bond {0, 1}, this function will generate a tensor where the initial
index for {0, 1} will contain the index of the bond {1, 0}.
"""
def get_bond_reverse_map(edge_index):
    edge_index = edge_index.numpy()
    num_edges = edge_index.shape[1]

    # Generate mapping of each edge to its index.
    edge_index_map = {(edge_index[0][idx], edge_index[1][idx]) : idx
        for idx in range(num_edges)}

    # Generate the bond reverse map.
    b2rev = [edge_index_map[(edge_index[1][idx], edge_index[0][idx])] 
        for idx in range(num_edges)]
    return torch.as_tensor(b2rev, dtype=torch.int)

"""
Generates a mapping of each atom to the indices of the bonds coming into that atom.
"""
def get_atom_incoming_bond_map(x, edge_index):
    # Get necessary values.
    num_atoms = x.shape[0]
    incoming_edge_points = edge_index[1].numpy()

    # Initialize mapping.
    a2b = [[] for _ in range(num_atoms)]

    # Find all of the indices where the atom is the end point of an edge. We add 1
    # to the index to account for padding.
    max_bonds = 0
    for edge_idx in range(len(incoming_edge_points)):
        a2b[incoming_edge_points[edge_idx]].append(edge_idx + 1)
        max_bonds = max(max_bonds, len(a2b[incoming_edge_points[edge_idx]]))
    
    # Pad the mapping with zeros and convert to tensor.
    a2b = [numpy.pad(a2b[idx], (0, max_bonds - len(a2b[idx])), 'constant') 
        for idx in range(num_atoms)]
    return torch.as_tensor(a2b, dtype=torch.int)

"""
Batch object which wraps around the torch geometric batch object.
This object provides the following additional values:
    - atom_incoming_bond_map : Tensor mapping each atom to the indices of its incoming bonds.
    - bond_reverse_map : Tensor mapping each bond to the index of its reverse.
    - atom_chunk_mask : List of masks which contain the indices of each atom's bonds.
    - molecule_chunk_mask : List of masks which contain indices of each molecule's atoms.
"""
class ExtendedBatch():
    def __init__(self, torch_geometric_batch):
        self.batch = torch_geometric_batch
        self.atom_incoming_bond_map = get_atom_incoming_bond_map(self.batch.x, self.batch.edge_index)
        self.bond_reverse_map = get_bond_reverse_map(self.batch.edge_index)

        # Convert count tensors into lists.
        self.batch.num_bonds_per_atom = self.batch.num_bonds_per_atom.tolist()
        self.batch.num_atoms_per_mol = self.batch.num_atoms_per_mol.tolist()

"""
Collater which collates the separate molecules into one batch.
Just an extended version of the Collater used by torch geometric.
"""
class ExtendedCollater(object):
    def torch_geometric_collate(self, batch):
        elem = batch[0]
        if isinstance(elem, Data):
            return Batch.from_data_list(batch, [], [])
        elif isinstance(elem, torch.Tensor):
            return default_collate(batch)
        elif isinstance(elem, float):
            return torch.tensor(batch, dtype=torch.float)
        elif isinstance(elem, int_classes):
            return torch.tensor(batch)
        elif isinstance(elem, string_classes):
            return batch
        elif isinstance(elem, container_abcs.Mapping):
            return {key: self.torch_geometric_collate([d[key] for d in batch]) for key in elem}
        elif isinstance(elem, tuple) and hasattr(elem, '_fields'):
            return type(elem)(*(self.torch_geometric_collate(s) for s in zip(*batch)))
        elif isinstance(elem, container_abcs.Sequence):
            return [self.torch_geometric_collate(s) for s in zip(*batch)]

        raise TypeError('DataLoader found invalid type: {}'.format(type(elem)))

    def collate(self, batch):
        torch_geometric_batch = self.torch_geometric_collate(batch)
        return ExtendedBatch(torch_geometric_batch)

    def __call__(self, batch):
        return self.collate(batch)

"""
Dataloader which wraps around a MoleculeDataset and uses the ExtendedCollater.
Also an extended version of the Dataloader used by torch geometric.
"""
class ExtendedDataLoader(DataLoader):
    """
    Creates a weighted random sampler for the dataset, which attributes weights
    to each class in the data. Works to diminish the issue of imbalance in datasets.
    """
    def create_weighted_sampler(self, dataset):
        # Get weight for each class according to how many times it appears.
        num_pos = sum([data.y for data in dataset]).detach().item()
        num_neg = len(dataset) - num_pos
        weight_per_class = [len(dataset) / num_neg, len(dataset) / num_pos]

        # Get weights for each data item.
        weight_per_item = [weight_per_class[data.y.int()] for data in dataset]
        weights = torch.as_tensor(weight_per_item, dtype=torch.double)

        # Create sampler.
        return WeightedRandomSampler(weights, len(dataset))

    """
    Parameters:
        - dataset : The dataset from which to load the data.
        - sampler : Whether to use a weighted random sampler or not.
        - batch_size : How many samples per batch to load.
        - shuffle : If set to True, the data will be reshuffled at every epoch.
    """
    def __init__(self, dataset, sampler=False, batch_size=1, shuffle=False, **kwargs):
        weighted_sampler = None
        if (sampler):
            weighted_sampler = self.create_weighted_sampler(dataset)

        super(ExtendedDataLoader, self).__init__(dataset, batch_size, shuffle, collate_fn=ExtendedCollater(),
                                                 sampler=weighted_sampler, **kwargs)
