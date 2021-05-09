# Molecule Restaurant

## Overview

Code used for our paper "A Deep Learning Approach to De Novo Design of Synthesizable Molecules with Specific Molecular Properties". 

Serves to build two things: a molecule classifier for HIV replication inhibition, and a version of the Molecule Chef model, build by Bradshaw et al., which is optimized towards generating molecules that inhibit HIV replication.

This code uses, as submodules, a fork of the original code for the Molecule Chef model by Bradshaw et al., and the Molecular Transformer model built by Schwaller et al.

## Hardware / Software Used for Testing

We used Pytorch with an Nvidia GPU on a Debian system to test this code. We recommend using these as well in order to ensure everything works as planned, as no other configurations were tested.

## Installation

All of the requirements for using this code can be installed through the `setup_molecule_restaurant.sh` script. An existing Anaconda environment is required for this, so please make sure to install a version of Anaconda before using this.

The Unzip shell program is also necessary to run this code. This can be installed by `sudo apt-get install unzip` or a similar command for your specific OS.

Once you have made sure that Anaconda and Unzip are installed, 

requirements:
anaconda
if u want to make new conda env for all this:
conda create -n molecule-restaurant
conda activate molecule-restaurant
unzip
run ./setup_molecule_restaurant.sh

## Usage

### Running Everything from Scratch

if broken pipe from hyperparameter optimization just restart, cancel with ctrl c and start again

### Using Pre-Existing Weights for Models


explain difference in testing vs final
