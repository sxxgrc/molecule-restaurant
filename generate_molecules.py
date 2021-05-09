"""
Primary script for MoleculeRestaurant to generate molecules.
"""

import torch, subprocess, json, os

from pathlib import Path

from prediction_model import get_hiv_classifier

from scripts.train import train_molecule_chef_qed_hiv

from rdkit import Chem

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
                                    num_opt_iters=50, batch_size=512)
print("Done!")
print()

exit

# Generate and train the molecule chef model which will select reactants for generating molecules.
print("Generating the Molecule Chef model for creating reactants...")
subprocess.run(["python", "external_models/molecule-chef/scripts/prepare_data/prepare_datasets.py"])
train_molecule_chef_qed_hiv(hiv_classifier, predictor_label_to_optimize=0)
best_molecule_chef_weights = str(Path().absolute()) + "/external_models/molecule-chef/scripts/train/chkpts/model_best.pth.pick"
print("Done!")
print()

# Optimize the latent space of molecule chef using QED scores and predictions from the HIV classifier.
print("Optimizing the latent space of Molecule Chef using QED and HIV classifier...")
subprocess.run(["python", "external_models/molecule-chef/scripts/evaluate/optimize/run_local_latent_search.py",
                best_molecule_chef_weights])
print("Done!")
print()

# Generate molecule reactants.
output_folder = str(Path().absolute()) + "/output_molecules/"
reactants_path = output_folder + "tokenized_reactants.pt"
print("Generating reactants...")
subprocess.run(["python", "external_models/molecule-chef/scripts/evaluate/generation/" + 
                "generate_for_mchef/create_reactant_bags.py", best_molecule_chef_weights,
                reactants_path])
print("Done!")
print()

# Generate molecules from reactants.
products_path = output_folder + "tokenized_products.pt"
smiles_path = output_folder + "molecule_smiles.pt"
print("Generating products from reactants...")
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
    for molecule_smiles in molecules_file:
        # Compute metrics.
        hiv_result = hiv_classifier(molecule_smiles)
        molecule = Chem.MolFromSmiles(molecule_smiles)
        qed_result = Chem.QED.qed(molecule)

        # Write to files.
        qed_file.write(molecule_smiles + " " + str(qed_result) + "\n")
        hiv_file.write(molecule_smiles + " " + str(hiv_result) + "\n")

print("Done!")
