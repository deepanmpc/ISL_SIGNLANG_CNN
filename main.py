#!/usr/bin/env python3
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from model import SignLanguageCNN

class SignLanguageDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.classes = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.images = []
        self.labels = []
        for class_name in self.classes:
            class_dir = os.path.join(data_dir, class_name)
            class_idx = self.class_to_idx[class_name]
            for img_name in os.listdir(class_dir):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(class_dir, img_name)
                    self.images.append(img_path)
                    self.labels.append(class_idx)
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label

def setup_data_transforms():
    train_transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.RandomHorizontalFlip(p=0.3),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    test_transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return train_transform, test_transform

def create_data_loaders(data_dir, batch_size=32, train_split=0.8):
    train_transform, test_transform = setup_data_transforms()
    full_dataset = SignLanguageDataset(data_dir, transform=train_transform)
    train_size = int(train_split * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])
    val_dataset.dataset.transform = test_transform
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    return train_loader, val_loader, len(full_dataset.classes)

def train_model(model, train_loader, val_loader, optimizer_name='Adam', num_epochs=50, learning_rate=0.001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    if optimizer_name.lower() == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    elif optimizer_name.lower() == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    elif optimizer_name.lower() == 'rmsprop':
        optimizer = optim.RMSprop(model.parameters(), lr=learning_rate)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)
    train_losses, val_losses, train_accuracies, val_accuracies = [], [], [], []
    best_val_acc = 0.0
    model_save_path = f'best_model_{optimizer_name}.pth'
    print(f"\n🚀 Starting training with {optimizer_name} for {num_epochs} epochs...")
    print("=" * 60)
    for epoch in range(num_epochs):
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
        train_accuracy = 100 * train_correct / train_total
        avg_train_loss = train_loss / len(train_loader)
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        val_accuracy = 100 * val_correct / val_total
        avg_val_loss = val_loss / len(val_loader)
        scheduler.step(avg_val_loss)
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        train_accuracies.append(train_accuracy)
        val_accuracies.append(val_accuracy)
        print(f"Epoch {epoch+1}/{num_epochs} | Val Acc: {val_accuracy:.2f}% | Val Loss: {avg_val_loss:.4f}")
        if val_accuracy > best_val_acc:
            best_val_acc = val_accuracy
            torch.save(model.state_dict(), model_save_path)
            print(f"  💾 New best model for {optimizer_name} saved! (Val Acc: {val_accuracy:.2f}%)")
    print(f"\n✅ Training with {optimizer_name} completed! Best validation accuracy: {best_val_acc:.2f}%")
    return train_losses, val_losses, train_accuracies, val_accuracies, best_val_acc

def visualize_feature_maps(model, image_path, device):
    model.eval()
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)
    outputs = []
    def hook(module, input, output):
        outputs.append(output)
    hook_handles = []
    hook_handles.append(model.conv1.register_forward_hook(hook))
    hook_handles.append(model.conv2.register_forward_hook(hook))
    with torch.no_grad():
        model(image_tensor)
    for handle in hook_handles:
        handle.remove()
    layer_names = ['Conv Layer 1', 'Conv Layer 2']
    for i, layer_output in enumerate(outputs):
        num_features = min(layer_output.shape[1], 16)
        fig, axes = plt.subplots(4, 4, figsize=(12, 12))
        fig.suptitle(f'Feature Maps - {layer_names[i]}', fontsize=16)
        for j in range(num_features):
            row, col = divmod(j, 4)
            ax = axes[row, col]
            feature_map = layer_output[0, j, :, :].cpu().numpy()
            ax.imshow(feature_map, cmap='viridis')
            ax.axis('off')
        plt.tight_layout()
        plt.savefig(f'feature_maps_visualization_{i+1}.png', dpi=300)
        plt.show()

def main():
    print("=" * 80)
    print("🤟 SIGN LANGUAGE RECOGNITION CNN - MAIN SYSTEM")
    print("=" * 80)
    DATA_DIR = "../DATA/ISL_TRAINING-1"
    BATCH_SIZE = 32
    NUM_EPOCHS = 10
    LEARNING_RATE = 0.001
    OPTIMIZERS = ['Adam', 'SGD', 'RMSprop']
    if not os.path.exists(DATA_DIR):
        print(f"❌ Data directory not found: {DATA_DIR}")
        return
    print(f"📁 Data directory: {DATA_DIR}")
    train_loader, val_loader, num_classes = create_data_loaders(DATA_DIR, BATCH_SIZE)
    print(f"✅ Dataset loaded successfully! ({num_classes} classes)")
    results = {}
    for optimizer_name in OPTIMIZERS:
        print(f"\n" + "="*30 + f" Testing Optimizer: {optimizer_name} " + "="*30)
        model = SignLanguageCNN(num_classes=num_classes)
        _, _, _, _, best_acc = train_model(
            model, train_loader, val_loader, optimizer_name, NUM_EPOCHS, LEARNING_RATE
        )
        results[optimizer_name] = best_acc
    print("\n" + "="*30 + " Optimizer Comparison Results " + "="*30)
    best_optimizer = ''
    best_overall_acc = 0.0
    for optimizer_name, acc in results.items():
        print(f"  - {optimizer_name}: {acc:.2f}% validation accuracy")
        if acc > best_overall_acc:
            best_overall_acc = acc
            best_optimizer = optimizer_name
    print(f"\n🏆 Best performing optimizer: {best_optimizer} with {best_overall_acc:.2f}% accuracy")
    print("="*80)
    print(f"\n🔍 Loading best model (trained with {best_optimizer}) for final evaluation...")
    model = SignLanguageCNN(num_classes=num_classes)
    model.load_state_dict(torch.load(f'best_model_{best_optimizer}.pth'))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    sample_found = False
    for class_name in os.listdir(DATA_DIR):
        class_dir = os.path.join(DATA_DIR, class_name)
        if os.path.isdir(class_dir):
            for img_name in os.listdir(class_dir):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    sample_image_path = os.path.join(class_dir, img_name)
                    sample_found = True
                    break
            if sample_found:
                break
    if sample_found:
        print(f"\n🎨 Visualizing feature maps using sample image: {sample_image_path}")
        visualize_feature_maps(model, sample_image_path, device)
        print("✅ Feature map visualizations saved as 'feature_maps_visualization_*.png'")
    print(f"\n🎉 System finished!")

if __name__ == "__main__":
    main()
