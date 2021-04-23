import torch

from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from torch._six import container_abcs, string_classes, int_classes

from torch_geometric.data import Data, Batch

"""
Batch object which wraps around the torch geometric batch object.
This object provides the following additional values:
    - atom_incoming_bond_map : Tensor mapping each atom to the indices of its incoming bonds.
    - bond_reverse_map : Tensor mapping each bond to the index of its reverse.
    - atom_chunk_mask : List of masks which contain the indices of each atom's bonds.
    - molecule_chunk_mask : List of masks which contain indices of each molecule's atoms.
"""
class ExtendedBatch():
    def __init__(self):
        print("hji")

"""
Collater which collates the separate molecules into one batch.
Just an extended version of the Collater used by torch geometric.
"""
class ExtendedCollater(object):
    def torch_geometric_collate(self, batch):
        elem = batch[0]
        if isinstance(elem, Data):
            return Batch.from_data_list(batch, self.follow_batch,
                                        self.exclude_keys)
        elif isinstance(elem, torch.Tensor):
            return default_collate(batch)
        elif isinstance(elem, float):
            return torch.tensor(batch, dtype=torch.float)
        elif isinstance(elem, int_classes):
            return torch.tensor(batch)
        elif isinstance(elem, string_classes):
            return batch
        elif isinstance(elem, container_abcs.Mapping):
            return {key: self.collate([d[key] for d in batch]) for key in elem}
        elif isinstance(elem, tuple) and hasattr(elem, '_fields'):
            return type(elem)(*(self.collate(s) for s in zip(*batch)))
        elif isinstance(elem, container_abcs.Sequence):
            return [self.collate(s) for s in zip(*batch)]

        raise TypeError('DataLoader found invalid type: {}'.format(type(elem)))

    def collate(self, batch):
        torch_geometric_batch = self.torch_geometric_collate(batch)
        print("Hi ^_^")
        return ExtendedBatch(torch_geometric_batch)

    def __call__(self, batch):
        return self.collate(batch)

"""
Dataloader which wraps around a MoleculeNetFeaturesDataset and uses the ExtendedCollater.
Also an extended version of the Dataloader used by torch geometric.
"""
class ExtendedDataLoader(DataLoader):
    """
    Parameters:
        dataset : The dataset from which to load the data.
        batch_size : How many samples per batch to load.
        shuffle : If set to True, the data will be reshuffled at every epoch.
    """
    def __init__(self, dataset, batch_size=1, shuffle=False, **kwargs):
        super(ExtendedDataLoader, self).__init__(dataset, batch_size, shuffle,
                                        collate_fn=ExtendedCollater(), **kwargs)
