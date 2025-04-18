#!/bin/bash

# Activate the conda environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate mlsec-project || { echo 'Failed to activate conda environment'; exit 1; }

# Install the datasets library if not already installed
pip install datasets --quiet || { echo 'Failed to install datasets library'; exit 1; }

# Python script to download and organize the dataset
python - <<EOF
from datasets import load_dataset
import os
import time

try:
    # Load the dataset
    print('Loading dataset...')
    dataset = load_dataset('prithivMLmods/Deepfake-vs-Real-v2')

    # Create directories for train and test sets
    os.makedirs('dataset/train/real', exist_ok=True)
    os.makedirs('dataset/train/deepfake', exist_ok=True)
    os.makedirs('dataset/test/real', exist_ok=True)
    os.makedirs('dataset/test/deepfake', exist_ok=True)

    # Save train set
    print('Saving train set...')
    train_set = dataset['train']
    for i, example in enumerate(train_set):
        image = example['image']
        label = example['label']
        label_dir = 'real' if label == 1 else 'deepfake'
        image.save(f'dataset/train/{label_dir}/{i}.jpg')
        if i % 100 == 0:
            time.sleep(1)  # Add delay every 100 images

    # For simplicity, use a portion of the train set as the test set
    print('Saving test set...')
    for i, example in enumerate(train_set.select(range(1000))):  # Select first 1000 for test
        image = example['image']
        label = example['label']
        label_dir = 'real' if label == 1 else 'deepfake'
        image.save(f'dataset/test/{label_dir}/{i}.jpg')
        if i % 100 == 0:
            time.sleep(1)  # Add delay every 100 images

    print('Dataset download and organization complete.')
except Exception as e:
    print(f'An error occurred: {e}')
EOF 