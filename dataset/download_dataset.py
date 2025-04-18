import os
import gc
from datasets import load_dataset
from PIL import Image
import time
from tqdm import tqdm
import numpy as np

def create_directories():
    """Create necessary directories for dataset organization."""
    dirs = [
        'dataset/train/real',
        'dataset/train/deepfake',
        'dataset/test/real',
        'dataset/test/deepfake'
    ]
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)
        print(f"Created directory: {dir_path}")

def save_image(img_data, path):
    """Save image data to the specified path."""
    try:
        # Convert numpy array to PIL Image if needed
        if isinstance(img_data, np.ndarray):
            image = Image.fromarray(img_data)
        elif isinstance(img_data, Image.Image):
            image = img_data
        else:
            print(f"Unexpected image type: {type(img_data)}")
            return False

        # Convert to RGB if needed
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

def process_batch(examples, start_idx, is_train=True):
    """Process a batch of examples."""
    prefix = 'train' if is_train else 'test'
    for i, example in enumerate(examples):
        try:
            idx = start_idx + i
            label_dir = 'real' if example['label'] == 1 else 'deepfake'
            save_path = f'dataset/{prefix}/{label_dir}/{idx}.jpg'
            save_image(example['image'], save_path)
            
            # Free up memory
            if i % 10 == 0:
                gc.collect()
        except Exception as e:
            print(f"Error processing example {idx}: {e}")
            continue

def main():
    try:
        # Create directories
        create_directories()
        
        # Load dataset with streaming enabled
        print("Loading Deepfake-vs-Real-v2 dataset...")
        dataset = load_dataset("prithivMLmods/Deepfake-vs-Real-v2", streaming=True)
        train_stream = dataset['train']
        
        # Process in smaller batches
        batch_size = 50  # Reduced batch size
        total_processed = 0
        train_size = 4000  # Reduced dataset size for initial testing
        test_size = 1000
        
        print("\nProcessing training set...")
        batch = []
        for i, example in enumerate(tqdm(train_stream, total=train_size)):
            if i >= train_size:
                break
                
            batch.append(example)
            if len(batch) >= batch_size:
                process_batch(batch, total_processed, is_train=True)
                total_processed += len(batch)
                batch = []
                time.sleep(0.5)  # Increased delay between batches
                
        # Process remaining training examples
        if batch:
            process_batch(batch, total_processed, is_train=True)
            total_processed += len(batch)
            
        # Reset for test set
        print("\nProcessing test set...")
        batch = []
        total_processed = 0
        for i, example in enumerate(tqdm(train_stream, total=test_size)):
            if i >= test_size:
                break
                
            batch.append(example)
            if len(batch) >= batch_size:
                process_batch(batch, total_processed, is_train=False)
                total_processed += len(batch)
                batch = []
                time.sleep(0.5)  # Increased delay between batches
                
        # Process remaining test examples
        if batch:
            process_batch(batch, total_processed, is_train=False)
            
        print("\nDataset processing complete!")
        
    except Exception as e:
        print(f"An error occurred: {e}")
        raise

if __name__ == "__main__":
    main() 