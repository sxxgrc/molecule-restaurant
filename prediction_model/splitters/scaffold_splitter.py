from rdkit.Chem import MolFromSmiles
from rdkit.Chem.Scaffolds.MurckoScaffold import MurckoScaffoldSmiles

from numpy.testing import assert_almost_equal

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
        
        # Sort the scaffolds from largest to smallest scaffold sets.
        scaffolds = {key: sorted(value) for key, value in scaffolds.items()}
        scaffold_sets = [
            scaffold_set for (_, scaffold_set) in sorted(scaffolds.items(),
                key=lambda x: (len(x[1]), x[1][0]), reverse=True)
        ]
        return scaffold_sets

    """
    Computes the scaffold for a molecule represented by the given SMILES string.
    """
    def generate_scaffold(self, smiles):
        mol = MolFromSmiles(smiles)
        return self.scaffold_generator(mol=mol)
