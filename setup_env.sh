#!/bin/bash

# Create conda environment
conda create -n mlsec-project python=3.8 -y

# Activate environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate mlsec-project

# Install PyTorch with pip (this will get the M1 compatible version)
pip install torch torchvision

# Install other requirements
pip install scipy scikit-learn numpy opencv-python Pillow tensorboardX

# Additional dependencies that might be needed
pip install tqdm 