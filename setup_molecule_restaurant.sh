#!/bin/bash

# Make sure anaconda is installed.
if ! command -v conda &> /dev/null; then
    echo "Anaconda is not installed. Please install it before running this!"
    exit
fi

# Make sure unzip is installed.
if ! command -v unzip &> /dev/null; then
    echo "Unzip is not installed. Please install it before running this!"
    exit
fi

# Create anaconda environment.
echo "Creating conda environment, this will take a while..."
conda install -y pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch -c nvidia >setup_log 2>&1
conda install -y pip >>setup_log 2>&1
conda install -y -c rdkit rdkit >>setup_log 2>&1
pip install git+https://github.com/bp-kelley/descriptastorus >>setup_log 2>&1
pip install chemprop >>setup_log 2>&1
pip install gdown >>setup_log 2>&1

# Intall torch_geometric.
torch_version=$(python -c "import torch; print(torch.__version__)")
torch_version="${torch_version%?}0"
cuda_version=$(python -c "import torch; print(torch.version.cuda)")

if [[ ${cuda_version}  == "None" ]]; then
    cuda_version="cpu"
else
    cuda_version="cu${cuda_version//./}"
fi

pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-"${torch_version}"+"${cuda_version}".html >>setup_log 2>&1
pip install torch-sparse -f https://pytorch-geometric.com/whl/torch-"${torch_version}"+"${cuda_version}".html >>setup_log 2>&1
pip install torch-cluster -f https://pytorch-geometric.com/whl/torch-"${torch_version}"+"${cuda_version}".html >>setup_log 2>&1
pip install torch-spline-conv -f https://pytorch-geometric.com/whl/torch-"${torch_version}"+"${cuda_version}".html >>setup_log 2>&1
pip install torch-geometric >>setup_log 2>&1
echo "Done!"

# Init submodules.
echo "Setting up submodules..."
git submodule update --init --recursive >>setup_log 2>&1
git submodule update --recursive >>setup_log 2>&1
pip install -e external_models/MolecularTransformer >>setup_log 2>&1
unzip -o external_models/molecule-chef/data.zip -d external_models/molecule-chef/ >>setup_log 2>&1
gdown https://drive.google.com/u/0/uc?id=1ogXzAg71BOs9SBrVt-umgcdc1_0ijUvU >>setup_log 2>&1
echo "Done!"

# Download the HIV model weights.
echo "Downloading pretrained D-MPNN weights for 5 ensemble models..."
gdown https://drive.google.com/u/0/uc?id=13D7mageHsCApEqkdC4CI71EcWJdqxMJZ >>setup_log 2>&1
gdown https://drive.google.com/u/0/uc?id=1MoqMRV-gFShIYJfmazbhobtoN-CaoTmu >>setup_log 2>&1
gdown https://drive.google.com/u/0/uc?id=1B5KSHNcPcGMSp5Gm5PLyTrBJW2uAbVgp >>setup_log 2>&1
gdown https://drive.google.com/u/0/uc?id=1p0AUUEBGM3mos4QxxumAutN0u9U7RkpA >>setup_log 2>&1
gdown https://drive.google.com/u/0/uc?id=1NjHhkpWMFKo0T2gfoFAML-a5uIFTeZJY >>setup_log 2>&1
mv hiv_* prediction_model/trained_models/
echo "Done!"
