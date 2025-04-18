from datasets import load_dataset
import os
from PIL import Image
from tqdm import tqdm

# Ensure real directory exists
os.makedirs('dataset/train/real', exist_ok=True)

# Load the dataset
print('Loading Deepfake-vs-Real-v2 dataset...')
dataset = load_dataset('prithivMLmods/Deepfake-vs-Real-v2')
train_set = dataset['train']

# Filter and save only real images (label=1)
print('Downloading real images...')
real_count = 0
for idx, example in enumerate(tqdm(train_set)):
    if example['label'] == 1:  # Real images
        image = example['image']
        # Convert RGBA to RGB if needed
        if image.mode == 'RGBA':
            image = image.convert('RGB')
        save_path = f'dataset/train/real/{real_count}.jpg'
        image.save(save_path, 'JPEG')
        real_count += 1

print(f'Downloaded {real_count} real images') 