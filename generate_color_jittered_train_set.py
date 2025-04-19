import os
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

jitter = transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1)

transform = transforms.Compose([
    transforms.ToTensor(),
    jitter,
    transforms.ToPILImage()
])

def jitter_split(split):
    print(f"Generating jittered images for {split} set...")
    for cls in ['real', 'deepfake']:
        input_dir = f'dataset/{split}/{cls}'
        output_dir = f'dataset_jittered/{split}/{cls}'
        os.makedirs(output_dir, exist_ok=True)

        for fname in tqdm(os.listdir(input_dir), desc=f"{split}/{cls}"):
            if not fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue
            in_path = os.path.join(input_dir, fname)
            out_path = os.path.join(output_dir, fname)
            try:
                image = Image.open(in_path).convert('RGB')
                jittered = transform(image)
                jittered.save(out_path, quality=95)
            except Exception as e:
                print(f"Failed on {in_path}: {e}")

if __name__ == "__main__":
    for split in ['train', 'val']:
        jitter_split(split)
    print("✅ Jittered train/val sets generated.")
