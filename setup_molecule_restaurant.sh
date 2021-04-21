#!/bin/bash

# Make sure anaconda is installed.
if ! command -v conda &> /dev/null; then
    echo "Anaconda is not installed. Please install it before running this!"
    exit
fi

# Make sure PyTorch is installed.
python -c "import torch; print(torch.__version__)" > /dev/null 2>&1
ret=$?
if [ $ret -ne 0 ]; then
    echo "PyTorch is not installed. Please install it before running this!"
    exit
fi

# Create anaconda environment.
echo "Creating conda environment, this will take a while..."
conda install pip
conda install -c conda-forge rdkit
pip install git+https://github.com/bp-kelley/descriptastorus
pip install chemprop

# Intall torch_geometric.
torch_version=$(python -c "import torch; print(torch.__version__)")
torch_version="${torch_version%?}0"
cuda_version=$(python -c "import torch; print(torch.version.cuda)")

if [[ ${cuda_version}  == "None" ]]; then
    cuda_version="cpu"
else
    cuda_version="cu${cuda_version//./}"
fi

pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-"${torch_version}"+"${cuda_version}".html
pip install torch-sparse -f https://pytorch-geometric.com/whl/torch-"${torch_version}"+"${cuda_version}".html
pip install torch-cluster -f https://pytorch-geometric.com/whl/torch-"${torch_version}"+"${cuda_version}".html
pip install torch-spline-conv -f https://pytorch-geometric.com/whl/torch-"${torch_version}"+"${cuda_version}".html
pip install torch-geometric 
