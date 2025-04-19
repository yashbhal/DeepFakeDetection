import os
import gc
import logging
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm
import numpy as np

# ---------------- Setup ---------------- #

LOG_FILE = 'dataset_preparation.log'
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filemode='w'
)

SPLIT_RATIOS = {
    'train': 0.7,
    'val': 0.15,
    'test': 0.15
}

LABEL_MAP = {
    1: 'real',
    0: 'deepfake'
}

# ---------------- Functions ---------------- #

def create_directories():
    for split in SPLIT_RATIOS.keys():
        for label in LABEL_MAP.values():
            os.makedirs(f'dataset/{split}/{label}', exist_ok=True)
            logging.info(f"Ensured directory: dataset/{split}/{label}")

def save_image(img_data, path):
    try:
        if os.path.exists(path):
            return True
        if isinstance(img_data, np.ndarray):
            image = Image.fromarray(img_data)
        elif isinstance(img_data, Image.Image):
            image = img_data
        else:
            logging.warning(f"Unexpected image type: {type(img_data)}")
            return False
        if image.mode != 'RGB':
            image = image.convert('RGB')
        image.save(path, 'JPEG', quality=95)
        return True
    except Exception as e:
        logging.error(f"Failed to save image {path}: {e}")
        return False

def count_files(folder):
    return len([f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))])

def main():
    try:
        create_directories()
        logging.info("🚀 Loading dataset (streaming mode)...")
        dataset = load_dataset("prithivMLmods/Deepfake-vs-Real-v2", split="train", streaming=True)

        # Check if counts are cached
        checkpoint_path = 'dataset/split_checkpoint.txt'
        if os.path.exists(checkpoint_path):
            with open(checkpoint_path, 'r') as f:
                real_count, fake_count = map(int, f.read().strip().split(','))
            logging.info(f"Loaded checkpoint: real={real_count}, deepfake={fake_count}")
        else:
            # Count instances
            real_count, fake_count = 0, 0
            logging.info("Counting real/fake samples...")
            for example in tqdm(dataset, desc="Counting"):
                label = example['label']
                real_count += (label == 1)
                fake_count += (label == 0)
            with open(checkpoint_path, 'w') as f:
                f.write(f"{real_count},{fake_count}")
            logging.info(f"Saved checkpoint: real={real_count}, deepfake={fake_count}")

        # Compute split maxes
        real_split_max = {
            'train': int(SPLIT_RATIOS['train'] * real_count),
            'val': int(SPLIT_RATIOS['val'] * real_count),
            'test': real_count - int(SPLIT_RATIOS['train'] * real_count) - int(SPLIT_RATIOS['val'] * real_count)
        }
        fake_split_max = {
            'train': int(SPLIT_RATIOS['train'] * fake_count),
            'val': int(SPLIT_RATIOS['val'] * fake_count),
            'test': fake_count - int(SPLIT_RATIOS['train'] * fake_count) - int(SPLIT_RATIOS['val'] * fake_count)
        }

        logging.info(f"📊 Split max counts: {real_split_max=} {fake_split_max=}")

        # Count existing files
        indices = {
            'real': {split: count_files(f'dataset/{split}/real') for split in SPLIT_RATIOS},
            'deepfake': {split: count_files(f'dataset/{split}/deepfake') for split in SPLIT_RATIOS}
        }

        logging.info(f"📂 Resuming from file counts: {indices}")

        # Reload streaming dataset for saving
        dataset = load_dataset("prithivMLmods/Deepfake-vs-Real-v2", split="train", streaming=True)

        # Save loop
        for example in tqdm(dataset, desc="Saving images"):
            label = example['label']
            label_str = LABEL_MAP[label]

            split = None
            for s in SPLIT_RATIOS:
                if indices[label_str][s] < (real_split_max[s] if label == 1 else fake_split_max[s]):
                    split = s
                    break

            if split is None:
                continue  # All splits filled for this label

            save_path = f'dataset/{split}/{label_str}/{indices[label_str][split]}.jpg'
            if save_image(example['image'], save_path):
                indices[label_str][split] += 1

            # Stop early if all done
            done = all(
                indices['real'][s] >= real_split_max[s] and
                indices['deepfake'][s] >= fake_split_max[s]
                for s in SPLIT_RATIOS
            )
            if done:
                logging.info("✅ All splits complete.")
                break

        logging.info(f"✅ Final counts: {indices}")
        print("✅ Dataset successfully prepared. Check log file for details.")

    except Exception as e:
        logging.error(f"❌ Dataset preparation failed: {e}")
        raise

if __name__ == "__main__":
    main()
