
from torchvision.datasets import ImageFolder
from torch.utils.data import random_split, DataLoader

def datapreprocess():
    data_dir = "/Users/deepandee/Desktop/ISL_SIGNLANG_CNN/DATA/ISL_TRAINING-1"
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    dataset = ImageFolder(root=data_dir, transform=transform)
    class_names = dataset.classes[:35]
    class_to_idx = {k: v for k, v in dataset.class_to_idx.items() if v < 35}
    train_size = int(0.8 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)
    test_loader = DataLoader(test_dataset, batch_size=32)
    return train_loader, val_loader, test_loader, class_names, class_to_idx

import torch
import torchvision.transforms as transforms
from PIL import Image
from model import SignLanguageCNN
import os

train_loader, val_loader, test_loader, class_names, class_to_idx = datapreprocess()
print("Class Names and their corresponding indices:")
for class_name, idx in class_to_idx.items():
    print(f"{class_name}: {idx} (Label: {class_name})")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SignLanguageCNN(num_classes=len(class_names)).to(device)
model.load_state_dict(torch.load("isl_sign_model.pt", map_location=device))
model.eval()

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

image_path = "test3.png"
image = Image.open(image_path).convert("RGB")
image_tensor = transform(image).unsqueeze(0).to(device)

with torch.no_grad():
    output = model(image_tensor)
    _, predicted = torch.max(output, 1)
    predicted_class = class_names[predicted.item()]
print(f"Predicted Class: {predicted_class}")