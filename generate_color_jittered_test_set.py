import os
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

# Color jitter transform (simulate social media filters)
jitter = transforms.ColorJitter(
    brightness=0.4,
    contrast=0.4,
    saturation=0.4,
    hue=0.1
)

transform = transforms.Compose([
    transforms.ToTensor(),
    jitter,
    transforms.ToPILImage()
])

def apply_jitter_and_save(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    for fname in tqdm(files, desc=f"Jittering {input_dir}"):
        try:
            image = Image.open(os.path.join(input_dir, fname)).convert('RGB')
            jittered = transform(image)
            jittered.save(os.path.join(output_dir, fname), quality=95)
        except Exception as e:
            print(f"Error processing {fname}: {e}")

def main():
    print("📸 Generating color-jittered test set...")

    classes = ['real', 'deepfake']
    for cls in classes:
        input_dir = f'dataset/test/{cls}'
        output_dir = f'dataset_jittered/test/{cls}'
        apply_jitter_and_save(input_dir, output_dir)

    print("✅ Done. Jittered test set saved to `dataset_jittered/test/`.")

if __name__ == "__main__":
    main()
