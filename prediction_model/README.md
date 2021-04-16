# Prediction Model

Primary script for creating and training the end prediction model for MoleculeRestaurant.
Specifically, for a specific graph dataset, this will train a deep learning model which will serve
as a classifier for molecules according to the main parameter specified for the dataset.
The end goal of this model is to accompany the MoleculeChef model in order to generate
synthesizable molecules that have the unique property specified by the dataset.

This model is a directed message passing neural network that uses the graphical structure of
the given molecules to generate this prediction.
