# Molecule Restaurant

## Overview

Code used for our paper "A Deep Learning Approach to De Novo Design of Synthesizable Molecules with Specific Molecular Properties". 

Serves to build two things: a molecule classifier for HIV replication inhibition, and a version of the Molecule Chef model, build by Bradshaw et al., which is optimized towards generating molecules that inhibit HIV replication.

This code uses, as submodules, a fork of the original code for the Molecule Chef model by Bradshaw et al., and the Molecular Transformer model built by Schwaller et al.

&nbsp;

## Hardware / Software Used for Testing

We used Pytorch with an Nvidia GPU on a Debian system to test this code. We recommend using this configuration in order to ensure everything works as expected, as no other configurations were tested.

&nbsp;

## Setup

### Pre-Requisites

This program requires an existing version of Anaconda, and of the Unzip program.

Anaconda can be installed following https://docs.anaconda.com/anaconda/install/ for your specific OS.

The Unzip program can be installed by `sudo apt-get install unzip` or a similar command for your specific OS.

&nbsp;

### Installation

You can install all of the required packages for this program using the provided `setup_molecule_restaurant.sh` script. Specifically, you should do the following in order to get everything ready.

Set up a new conda environment and activate it:

```
conda create -n molecule-restaurant
conda activate molecule-restaurant
```

Install the required libraries for the program:

```
./setup_molecule_restaurant.sh
```

&nbsp;

### Python Path

As we use various submodules along with the main program, the Python path for your current shell needs to be updated to include these files. This can be done after setup has been completed by running the following:

```
./setup_python_paths.sh
```

&nbsp;

## Usage

The main entrypoint for generating molecules using the program is the `generate_molecules.py` script which can be called by:

```
python generate_molecules.py
```

One important changeable parameter within this script is the `final` boolean parameter used in the call to `get_hiv_classifier`. This paremeter will specify whether the HIV classifier will be tested against part of the dataset during training or not. If this is present, the original HIV dataset will have an 8:1:1 split for training, validation, and testing. Otherwise, the dataset will have a 9:1 split for training and validation.

&nbsp;

### Using Pre-Existing Parameters for Models

In order to save the time that it would take to train the various models used for this program, we have provided some of the parameters for the models. Specifically, we have provided trained weights for an ensemble of 5 models for the HIV classifier, optimized hyperparameters for the HIV classifier models, and trained weights for the Molecular Transformer model.

These parameters will be used automatically, such that you can simply run the `generate_molecules.py` script and it will use these, skipping the training portion of the program.

&nbsp;

### Running Everything from Scratch

If you do want to run the entire program from scratch, including training, you can remove the provided parameters for the HIV classifier model, by running:

```
rm prediction_model/optimization/saved_hyperparameters/hyperparameters.json
rm prediction_model/optimization/saved_hyperparameters/trials.pkl
rm prediction_model/trained_models/hiv*
```

You can then run the `generate_molecules.py` script and it will run both the hyperparameter optimization and full ensemble training for the HIV classifier.

If a broken pipe error occurs during the hyperparameter optimization, you can just restart the optimization by stopping and re-running the script.

&nbsp;

### Viewing Results

The `generate_molecules.py` script will create a set of output files for the molecules it generates, with these files being stored in the `output_molecules` directory. The `README.md` file in that directory explains what will be provided but to summarize, the following files will be generated:

- `tokenized_reactants.pt` : Generated reactants
- `tokenized_products.pt` : Generated products
- `molecule_smiles.pt` : SMILES strings for the generated products
- `metrics/molecule_chef_metrics.pt` : The tabulated metrics created by Molecule Chef
- `metrics/qed_values.pt` : The sorted QED values associated with each product molecule
- `metrics/hiv_replication_inhibition_probability.pt` : The sorted HIV replication inhibition probability for each product molecule

