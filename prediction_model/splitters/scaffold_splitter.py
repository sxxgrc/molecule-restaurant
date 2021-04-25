from rdkit.Chem import MolFromSmiles
from rdkit.Chem.Scaffolds.MurckoScaffold import MurckoScaffoldSmiles

from numpy.testing import assert_almost_equal

from random import Random

"""
A generic dataset splitter which splits the dataset into a train set and 
test set according to the similarity of the dataset's molecule's scaffolds.

This follows the same process as the deepchem ScaffoldSplitter class found here:
github.com/deepchem/deepchem/blob/master/deepchem/splits/splitters.py
"""
class ScaffoldSplitter():
    def __init__(self, scaffold_generator=MurckoScaffoldSmiles):
        self.scaffold_generator = scaffold_generator
    
    """
    Splits the dataset into train and test datasets according to the percentages
    provided.
    """
    def split(self, dataset, frac_train, frac_test):
        assert_almost_equal(frac_train + frac_test, 1.)

        # Get the sorted scaffold sets.
        scaffold_sets = self.generate_scaffolds(dataset)

        # Balance scaffolds so that those larger than half of test or val go to train.
        random = Random()
        big_index_sets = []
        small_index_sets = []
        test_size = frac_test * len(dataset)
        for index_set in list(scaffold_sets.values()):
            # Test here will be split in half for final test and val.
            if len(index_set) > test_size / 4:
                big_index_sets.append(index_set)
            else:
                small_index_sets.append(index_set)
        random.shuffle(big_index_sets)
        random.shuffle(small_index_sets)
        scaffold_sets = big_index_sets + small_index_sets

        # Generate the split indices for each dataset.
        train_cutoff = frac_train * len(dataset)
        train_indices = []
        test_indices = []

        for scaffold_set in scaffold_sets:
            if len(train_indices) + len(scaffold_set) > train_cutoff:
                test_indices += scaffold_set
            else:
                train_indices += scaffold_set

        # Return the split datasets.
        return dataset[train_indices], dataset[test_indices]
    
    """
    Generates a list of lists containing the indices of each scaffold in the dataset.
    Specifically, this will find all of the scaffolds in the dataset, all of the molecules
    that have those scaffolds, and split them up into separate lists (using the indices of the
    molecules). The scaffolds are then sorted from largest to smallest sets.
    """
    def generate_scaffolds(self, dataset):
        scaffolds = {}

        # Get all of the scaffolds for the molecules in the dataset and separate them.
        for idx in range(len(dataset)):
            smiles = dataset[idx].smiles
            scaffold = self.generate_scaffold(smiles)
            if scaffold not in scaffolds:
                scaffolds[scaffold] = [idx]
            else:
                scaffolds[scaffold].append(idx)

        return scaffold_sets

    """
    Computes the scaffold for a molecule represented by the given SMILES string.
    """
    def generate_scaffold(self, smiles):
        mol = MolFromSmiles(smiles)
        return self.scaffold_generator(mol=mol)
