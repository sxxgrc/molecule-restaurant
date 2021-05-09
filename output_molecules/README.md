# Output Molecules

Contains all of the files corresponding to the output of the molecule generation.

Specifically will contain the following files and directories:
- `tokenized_reactants.pt` : Contains the optimized reactants created by Molecule Chef
- `tokenized_products.pt` : Contains the products created by MolecularTransformer from the tokenized reactants
- `molecule_smiles.pt` : Contains the SMILES string form of the tokenized products
- `metrics/` : Contains the metrics computed from the products
