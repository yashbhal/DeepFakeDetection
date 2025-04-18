import os
import torch
from torchvision import datasets, transforms
from PIL import Image
from networks.resnet import resnet50

# --- 1. Print class-to-index mapping ---
test_dir = 'dataset/test'
dataset = datasets.ImageFolder(test_dir)
print('Class to index mapping:', dataset.class_to_idx)

# --- 2. Print evaluation transforms ---
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
print('Evaluation transforms:', preprocess)

# --- 3. Run a sample prediction for one real and one fake image ---
# Find one image from each class
def find_sample_image(class_name):
    class_dir = os.path.join(test_dir, class_name)
    for fname in os.listdir(class_dir):
        if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
            return os.path.join(class_dir, fname)
    return None

real_img_path = find_sample_image('real')
fake_img_path = find_sample_image('deepfake')

print('Sample real image:', real_img_path)
print('Sample fake image:', fake_img_path)

# Load model
model_path = 'weights/blur_jpg_prob0.5.pth'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = resnet50(num_classes=1)
state_dict = torch.load(model_path, map_location=device)
model.load_state_dict(state_dict['model'])
model = model.to(device)
model.eval()

for img_path, label_name in [(real_img_path, 'real'), (fake_img_path, 'deepfake')]:
    img = Image.open(img_path).convert('RGB')
    img_tensor = preprocess(img).unsqueeze(0).to(device)
    with torch.no_grad():
        prob = model(img_tensor).sigmoid().item()
    print(f'Predicted probability for {label_name} image ({img_path}): {prob:.4f} (closer to 1 = fake, 0 = real)')
