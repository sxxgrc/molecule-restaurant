# Molecule Restaurant

## Overview

Code used for our paper "A Deep Learning Approach to De Novo Design of Synthesizable Molecules with Specific Molecular Properties". 

Serves to build two things: 
- A molecule classifier for HIV replication inhibition, based off the D-MPNN created by Yang et al.
- A modified version of the Molecule Chef model built by Bradshaw et al., which is optimized towards generating molecules that inhibit HIV replication

This code uses, as submodules, a fork of the original code for the Molecule Chef model by Bradshaw et al., and the Molecular Transformer model built by Schwaller et al.

The original code for the directed MPNN, by Yang et al., which our HIV classifier is based off, can be found at https://github.com/chemprop/chemprop.

&nbsp;

## Topics

- [Overview](#Overview)
  * [Topics](#Topics)
- [Hardware / Software Used for Testing](#Hardware-/-Software-Used-for-Testing)
- [Setup](#Setup)
    * [Prerequisites](#Prerequisites)
    * [Installation](#Installation)
- [Usage](#Usage)
    * [Setting Up Python Path for Submodules](#Setting-Up-Python-Path-for-Submodules)
    * [Main Script for Generating Molecules](#Main-Script-for-Generating-Molecules)
    * [Using Preexisting Parameters for Models](#Using-Preexisting-Parameters-for-Models)
    * [Running Everything from Scratch](#Running-Everything-from-Scratch)
- [Viewing Results](#Viewing-Results)

&nbsp;

## Hardware / Software Used for Testing

We used Pytorch with an Nvidia GPU, via CUDA, on a Debian system to test this code. We recommend using a similar configuration in order to ensure everything works as expected, as no other configurations were tested.

&nbsp;

## Setup

### Prerequisites

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

## Usage

### Setting Up Python Path for Submodules

As we use various submodules along with the main program, the Python path for your current shell needs to be updated to include these files. This can be done after setup has been completed by running the following:

```
./setup_python_paths.sh
```

This needs to be run every time you launch a new shell instance and intend to run this program.

&nbsp;

### Main Script for Generating Molecules

The main entrypoint for generating molecules using the model is the `generate_molecules.py` script which can be called by:

```
python generate_molecules.py
```

One important changeable parameter within this script is the `final` boolean parameter used in the call to `get_hiv_classifier`. This parameter will specify whether the HIV classifier will be tested against part of the dataset during training (`False`) or not (`True`). Specifically, if the value is set to `False` the original HIV dataset will have an 8:1:1 split for training, validation, and testing. Otherwise, the dataset will have a 9:1 split for training and validation, with the validation being used to select the best epoch from training.

&nbsp;

### Using Preexisting Parameters for Models

In order to save the time that it would take to train the various models used for this program, we have provided some of the parameters for the models. 

Specifically, we have provided:
  - Trained weights for an ensemble of 5 models for the HIV D-MPNN 
  - Optimized hyperparameters for the HIV D-MPNN
  - Trained weights for the Molecule Chef model and property prediction model it uses
  - Trained weights for the Molecular Transformer model (as selected by Bradshaw et al.)

These parameters will be used automatically, such that you can simply run the `generate_molecules.py` script and it will use these, skipping the training portion of the program. 

Initial downloading and processing of the HIV and Molecule Chef data will still be done, so the generation process will still initially take a bit of time (albeit a lot less than it would without the provided parameters). The processed data will be stored on your machine however, so further runs will have a substantial decrease in run time.

&nbsp;

### Running Everything from Scratch

If you do want to run the entire program from scratch, including training and hyperparameter optimization, you can remove the provided parameters for the HIV classifier model and the Molecule Chef model as follows:

```
rm prediction_model/optimization/saved_hyperparameters/hyperparameters.json
rm prediction_model/optimization/saved_hyperparameters/trials.pkl
rm prediction_model/trained_models/hiv*
rm molecule_chef_model_best.pth.pick
```

You can then run the `generate_molecules.py` script and it will run both the hyperparameter optimization and full ensemble training for the HIV classifier. This will take a while (on the scale of 1-3 days) so be prepared, and make sure to use a capable system for this.

If a broken pipe error happens to occur during the hyperparameter optimization, you can just restart the optimization by stopping and re-running the script. It will pick up near where it left off.

&nbsp;

## Viewing Results

The `generate_molecules.py` script will create a set of output files for the molecules it generates, with these files being stored in the `output_molecules` directory. The `README.md` file in that directory explains what will be provided but, to summarize, the following files will be generated:

- `tokenized_reactants.pt` : Generated reactants
- `tokenized_products.pt` : Generated products
- `molecule_smiles.pt` : SMILES strings for the generated products
- `metrics/molecule_chef_metrics.pt` : The tabulated metrics created by Molecule Chef
- `metrics/qed_values.pt` : The sorted QED values associated with each product molecule
- `metrics/hiv_replication_inhibition_probability.pt` : The sorted HIV replication inhibition probability for each product molecule
