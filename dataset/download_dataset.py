import os
import gc
from datasets import load_dataset
from PIL import Image
import time
from tqdm import tqdm
import numpy as np
import shutil
import random

def create_directories():
    dirs = [
        'dataset/train/real',
        'dataset/train/deepfake',
        'dataset/val/real',
        'dataset/val/deepfake',
        'dataset/test/real',
        'dataset/test/deepfake'
    ]
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)
        print(f"Created directory: {dir_path}")

def save_image(img_data, path):
    try:
        if os.path.exists(path):
            # Skip if file already exists
            return True
        if isinstance(img_data, np.ndarray):
            image = Image.fromarray(img_data)
        elif isinstance(img_data, Image.Image):
            image = img_data
        else:
            print(f"Unexpected image type: {type(img_data)}")
            return False
        if image.mode != 'RGB':
            image = image.convert('RGB')
        image.save(path, 'JPEG', quality=95)
        return True
    except Exception as e:
        print(f"Error saving image {path}: {e}")
        print(f"Image type: {type(img_data)}")
        if isinstance(img_data, np.ndarray):
            print(f"Array shape: {img_data.shape}, dtype: {img_data.dtype}")
        return False

def count_files(folder):
    return len([f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))])

def main():
    try:
        create_directories()
        print("Loading Deepfake-vs-Real-v2 dataset (streaming)...")
        dataset = load_dataset("prithivMLmods/Deepfake-vs-Real-v2", split="train", streaming=True)

        # First pass: count class totals, but skip if a checkpoint file exists
        checkpoint_file = 'dataset/split_checkpoint.txt'
        if os.path.exists(checkpoint_file):
            with open(checkpoint_file, 'r') as f:
                real_count, fake_count = map(int, f.read().strip().split(','))
            print(f"Loaded split counts from checkpoint: real={real_count}, deepfake={fake_count}")
        else:
            print("Counting class totals (first pass)...")
            real_count, fake_count = 0, 0
            for example in tqdm(dataset, desc="Counting", mininterval=2.0):
                if example['label'] == 1:
                    real_count += 1
                else:
                    fake_count += 1
            with open(checkpoint_file, 'w') as f:
                f.write(f"{real_count},{fake_count}")
            print(f"Saved split counts to checkpoint: real={real_count}, deepfake={fake_count}")

        real_train_max = int(0.7 * real_count)
        real_val_max = int(0.15 * real_count)
        fake_train_max = int(0.7 * fake_count)
        fake_val_max = int(0.15 * fake_count)
        real_test_max = real_count - real_train_max - real_val_max
        fake_test_max = fake_count - fake_train_max - fake_val_max
        print(f"Train split: real={real_train_max}, deepfake={fake_train_max}")
        print(f"Val split: real={real_val_max}, deepfake={fake_val_max}")
        print(f"Test split: real={real_test_max}, deepfake={fake_test_max}")

        # Count already saved images in each folder
        real_train_idx = count_files('dataset/train/real')
        fake_train_idx = count_files('dataset/train/deepfake')
        real_val_idx = count_files('dataset/val/real')
        fake_val_idx = count_files('dataset/val/deepfake')
        real_test_idx = count_files('dataset/test/real')
        fake_test_idx = count_files('dataset/test/deepfake')
        print(f"Resuming from: train/real={real_train_idx}, train/deepfake={fake_train_idx}, val/real={real_val_idx}, val/deepfake={fake_val_idx}, test/real={real_test_idx}, test/deepfake={fake_test_idx}")

        # Second pass: stream again and save images if not already present
        dataset = load_dataset("prithivMLmods/Deepfake-vs-Real-v2", split="train", streaming=True)
        real_seen = fake_seen = 0
        for example in tqdm(dataset, desc="Saving", mininterval=2.0):
            label = example['label']
            if label == 1:
                if real_train_idx < real_train_max:
                    save_path = f'dataset/train/real/{real_train_idx}.jpg'
                    if not os.path.exists(save_path):
                        if save_image(example['image'], save_path):
                            real_train_idx += 1
                    else:
                        real_train_idx += 1
                elif real_val_idx < real_val_max:
                    save_path = f'dataset/val/real/{real_val_idx}.jpg'
                    if not os.path.exists(save_path):
                        if save_image(example['image'], save_path):
                            real_val_idx += 1
                    else:
                        real_val_idx += 1
                elif real_test_idx < real_test_max:
                    save_path = f'dataset/test/real/{real_test_idx}.jpg'
                    if not os.path.exists(save_path):
                        if save_image(example['image'], save_path):
                            real_test_idx += 1
                    else:
                        real_test_idx += 1
                real_seen += 1
            else:
                if fake_train_idx < fake_train_max:
                    save_path = f'dataset/train/deepfake/{fake_train_idx}.jpg'
                    if not os.path.exists(save_path):
                        if save_image(example['image'], save_path):
                            fake_train_idx += 1
                    else:
                        fake_train_idx += 1
                elif fake_val_idx < fake_val_max:
                    save_path = f'dataset/val/deepfake/{fake_val_idx}.jpg'
                    if not os.path.exists(save_path):
                        if save_image(example['image'], save_path):
                            fake_val_idx += 1
                    else:
                        fake_val_idx += 1
                elif fake_test_idx < fake_test_max:
                    save_path = f'dataset/test/deepfake/{fake_test_idx}.jpg'
                    if not os.path.exists(save_path):
                        if save_image(example['image'], save_path):
                            fake_test_idx += 1
                    else:
                        fake_test_idx += 1
                fake_seen += 1
            # Stop if all splits are complete
            if (real_train_idx >= real_train_max and fake_train_idx >= fake_train_max and
                real_val_idx >= real_val_max and fake_val_idx >= fake_val_max and
                real_test_idx >= real_test_max and fake_test_idx >= fake_test_max):
                print("All splits complete. Exiting early.")
                break
        print(f"Saved: train/real={real_train_idx}, train/deepfake={fake_train_idx}, val/real={real_val_idx}, val/deepfake={fake_val_idx}, test/real={real_test_idx}, test/deepfake={fake_test_idx}")
        print("\nDataset processing complete!")
    except Exception as e:
        print(f"An error occurred: {e}")
        raise

if __name__ == "__main__":
    main()