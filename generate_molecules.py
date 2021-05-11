"""
Primary script for MoleculeRestaurant to generate molecules.
"""

import torch, subprocess, json, os, numpy, importlib

from pathlib import Path

from prediction_model import get_hiv_classifier

from rdkit import Chem

molecule_chef_train = importlib.import_module("external_models.molecule-chef.scripts.train")

# Grab the device to use for the training.
if torch.cuda.is_available():
    dev = "cuda:0"
else:
    dev = "cpu"

device = torch.device(dev)
print()
print("Using PyTorch device : " + str(device))

# Generate the HIV classifier model.
print("Generating HIV replication inhibition predictor model...")
hiv_classifier = get_hiv_classifier(num_train_epochs=30, ensemble_size=5, torch_device=device, 
                                    num_opt_iters=50, batch_size=512, final=True)
print("Done!")
print()

# Generate and train the molecule chef model which will select reactants for generating molecules.
molecule_chef_data_path = (str(Path().absolute()) + 
                           "/external_models/molecule-chef/processed_data/train_products.txt")
if not path.exists(molecule_chef_data_path) or not path.getsize(molecule_chef_data_path) > 0:
    print("Getting data for Molecule Chef model...")
    subprocess.run(["python", "external_models/molecule-chef/scripts/prepare_data/prepare_datasets.py"])
    print("Done!")
    print()

best_molecule_chef_weights = (str(Path().absolute()) + "/molecule_chef_model_best.pth.pick")
if not path.exists(best_molecule_chef_weights) or not path.getsize(best_molecule_chef_weights) > 0:
    print("Generating and training the Molecule Chef model for creating reactants...")
    molecule_chef_train.train_molecule_chef_qed_hiv(hiv_classifier, predictor_label_to_optimize=0)
    print("Done!")
    print()

# Optimize the latent space of molecule chef using QED scores and predictions from the HIV classifier.
output_folder = str(Path().absolute()) + "/output_molecules/"
reactants_path = output_folder + "tokenized_reactants.pt"
print("Optimizing the latent space of Molecule Chef using QED and HIV classifier...")
subprocess.run(["python", "external_models/molecule-chef/scripts/evaluate/optimize/run_local_latent_search.py",
                best_molecule_chef_weights, reactants_path])
print("Done!")
print()

# Generate molecules from reactants.
products_path = output_folder + "tokenized_products.pt"
smiles_path = output_folder + "molecule_smiles.pt"
print("Generating products from optimized reactants...")
subprocess.run(["python", "external_models/MolecularTransformer/translate.py", 
                "-model", str(Path().absolute()) + "/molecular_transformer_weights.pt",
                "-src", reactants_path, "-output", products_path, "-batch_size", "300", 
                "-replace_unk",  "-max_length",  "500", "-fast", "-gpu", "1", "-n_best", "5"])
subprocess.run(["python", "external_models/molecule-chef/scripts/evaluate/" +
                "put_together_molecular_transformer_predictions.py", products_path, smiles_path,
                "--nbest=5"])
print("Done!")
print()
exit

# Compute metrics for molecules.
metrics_config = str(Path().absolute()) + "/external_models/molecule_metrics.json"
metrics_output = output_folder + "metrics/"
molecule_chef_metrics_path = metrics_output + "molecule_chef_metrics.pt"
qed_path = metrics_output + "qed_values.pt"
hiv_path = metrics_output + "hiv_replication_inhibition_probability.pt"
print("Computing metrics for molecules...")

# Modify molecule chef metrics config file to use actual products.
with open(metrics_config, "r+") as f:
    data = json.load(f)
    data["data_dir"] = output_folder
os.remove(metrics_config)
with open(metrics_config, 'w') as f:
    json.dump(data, f, indent=4)

# Compute molecule chef metrics.
with open(molecule_chef_metrics_path, "w") as f:
    subprocess.run(["python", "external_models/molecule-chef/scripts/evaluate/generation/metrics/" +
                    "evaluate_metrics.py", metrics_config], stdout=f, stderr=subprocess.STDOUT)

# Compute QED and HIV metrics.
with open(smiles_path, "r") as molecules_file, open(qed_path, "w") as qed_file, open(hiv_path, "w") as hiv_file:
    smiles = []
    hiv_results = []
    qed_results = []
    for molecule_smiles in molecules_file:
        smiles.append(molecule_smiles)

        # Compute metrics.
        hiv_results.append(hiv_classifier.predict(molecule_smiles))
        molecule = Chem.MolFromSmiles(molecule_smiles)
        qed_results.append(Chem.QED.qed(molecule))

    # Sort the metrics and output to respective files.
    hiv_sort_indices = numpy.argsort(hiv_results)[::-1]
    qed_sort_indices = numpy.argsort(qed_results)

    for idx in range(len(smiles)):
        hiv_file.write(smiles[hiv_sort_indices[idx]] + " " + str(hiv_results[hiv_sort_indices[idx]]) + "\n")
        qed_file.write(smiles[qed_sort_indices[idx]] + " " + str(qed_results[qed_sort_indices[idx]]) + "\n")

print("Done!")
