#!/usr/bin/env python3
"""
Main Script for Sign Language Recognition CNN
Handles training, testing, and prediction for the ISL CNN model
"""

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
    """Custom dataset for sign language images"""
    
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.classes = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
        self.images = []
        self.labels = []
        
        # Load all images and labels
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
        
        # Load and preprocess image
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

def setup_data_transforms():
    """Setup data transformations for training and testing"""
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
    """Create training and validation data loaders"""
    train_transform, test_transform = setup_data_transforms()
    
    # Create full dataset
    full_dataset = SignLanguageDataset(data_dir, transform=train_transform)
    
    # Split into train and validation
    train_size = int(train_split * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])
    
    # Update transforms for validation
    val_dataset.dataset.transform = test_transform
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    return train_loader, val_loader, len(full_dataset.classes)

def train_model(model, train_loader, val_loader, num_epochs=50, learning_rate=0.001):
    """Train the CNN model"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)
    
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    
    best_val_acc = 0.0
    
    print(f"\n🚀 Starting training for {num_epochs} epochs...")
    print("=" * 60)
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_idx, (images, labels) in enumerate(train_loader):
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
            
            if batch_idx % 50 == 0:
                print(f"Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}/{len(train_loader)}, "
                      f"Loss: {loss.item():.4f}")
        
        train_accuracy = 100 * train_correct / train_total
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
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
        
        # Update learning rate
        scheduler.step(avg_val_loss)
        
        # Store metrics
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        train_accuracies.append(train_accuracy)
        val_accuracies.append(val_accuracy)
        
        # Print epoch results
        print(f"Epoch {epoch+1}/{num_epochs}:")
        print(f"  Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.2f}%")
        print(f"  Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.2f}%")
        print(f"  Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
        print("-" * 60)
        
        # Save best model
        if val_accuracy > best_val_acc:
            best_val_acc = val_accuracy
            torch.save(model.state_dict(), 'best_model.pth')
            print(f"  💾 New best model saved! (Val Acc: {val_accuracy:.2f}%)")
    
    print(f"\n✅ Training completed!")
    print(f"Best validation accuracy: {best_val_acc:.2f}%")
    
    return train_losses, val_losses, train_accuracies, val_accuracies

def evaluate_model(model, test_loader):
    """Evaluate the trained model"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    
    correct = 0
    total = 0
    class_correct = {}
    class_total = {}
    
    print("\n🔍 Evaluating model...")
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Per-class accuracy
            for i in range(len(labels)):
                label = labels[i].item()
                pred = predicted[i].item()
                
                if label not in class_correct:
                    class_correct[label] = 0
                    class_total[label] = 0
                
                class_total[label] += 1
                if label == pred:
                    class_correct[label] += 1
    
    overall_accuracy = 100 * correct / total
    print(f"Overall Accuracy: {overall_accuracy:.2f}%")
    
    # Print per-class accuracy
    print("\nPer-class Accuracy:")
    for class_idx in sorted(class_correct.keys()):
        class_acc = 100 * class_correct[class_idx] / class_total[class_idx]
        print(f"  Class {class_idx}: {class_acc:.2f}%")
    
    return overall_accuracy

def predict_sign(model, image_path, class_names):
    """Predict sign language from a single image"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    
    # Load and preprocess image
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    # Make prediction
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][predicted_class].item()
    
    return predicted_class, confidence, probabilities[0]

def plot_training_history(train_losses, val_losses, train_accuracies, val_accuracies):
    """Plot training history"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot losses
    ax1.plot(train_losses, label='Train Loss')
    ax1.plot(val_losses, label='Validation Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Plot accuracies
    ax2.plot(train_accuracies, label='Train Accuracy')
    ax2.plot(val_accuracies, label='Validation Accuracy')
    ax2.set_title('Training and Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """Main function to run the sign language recognition system"""
    print("=" * 80)
    print("🤟 SIGN LANGUAGE RECOGNITION CNN - MAIN SYSTEM")
    print("=" * 80)
    
    # Configuration
    DATA_DIR = "../DATA/ISL_TRAINING-1"  # Adjust path as needed
    BATCH_SIZE = 32
    NUM_EPOCHS = 30
    LEARNING_RATE = 0.001
    
    # Check if data directory exists
    if not os.path.exists(DATA_DIR):
        print(f"❌ Data directory not found: {DATA_DIR}")
        print("Please ensure the data directory path is correct.")
        return
    
    print(f"📁 Data directory: {DATA_DIR}")
    print(f"⚙️  Configuration:")
    print(f"   - Batch size: {BATCH_SIZE}")
    print(f"   - Epochs: {NUM_EPOCHS}")
    print(f"   - Learning rate: {LEARNING_RATE}")
    
    try:
        # Create data loaders
        print("\n📊 Loading dataset...")
        train_loader, val_loader, num_classes = create_data_loaders(DATA_DIR, BATCH_SIZE)
        print(f"✅ Dataset loaded successfully!")
        print(f"   - Number of classes: {num_classes}")
        print(f"   - Training samples: {len(train_loader.dataset)}")
        print(f"   - Validation samples: {len(val_loader.dataset)}")
        
        # Initialize model
        print(f"\n🏗️  Initializing model...")
        model = SignLanguageCNN(num_classes=num_classes)
        print(f"✅ Model initialized with {num_classes} output classes")
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"   - Total parameters: {total_params:,}")
        print(f"   - Trainable parameters: {trainable_params:,}")
        
        # Train model
        print(f"\n🚀 Starting training...")
        train_losses, val_losses, train_accuracies, val_accuracies = train_model(
            model, train_loader, val_loader, NUM_EPOCHS, LEARNING_RATE
        )
        
        # Plot training history
        print(f"\n📈 Plotting training history...")
        plot_training_history(train_losses, val_losses, train_accuracies, val_accuracies)
        
        # Load best model for evaluation
        print(f"\n🔍 Loading best model for evaluation...")
        model.load_state_dict(torch.load('best_model.pth'))
        
        # Evaluate model
        overall_accuracy = evaluate_model(model, val_loader)
        
        # Test prediction on a sample image
        print(f"\n🎯 Testing prediction...")
        # Find a sample image from the dataset
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
            class_names = sorted([d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))])
            predicted_class, confidence, all_probabilities = predict_sign(model, sample_image_path, class_names)
            
            print(f"Sample image: {sample_image_path}")
            print(f"Predicted class: {predicted_class} ({class_names[predicted_class]})")
            print(f"Confidence: {confidence:.2%}")
            
            # Show top 5 predictions
            top_5_indices = torch.topk(all_probabilities, 5).indices
            print(f"\nTop 5 predictions:")
            for i, idx in enumerate(top_5_indices):
                prob = all_probabilities[idx].item()
                print(f"  {i+1}. Class {idx} ({class_names[idx]}): {prob:.2%}")
        
        print(f"\n" + "=" * 80)
        print(f"🎉 SIGN LANGUAGE RECOGNITION SYSTEM COMPLETED!")
        print(f"=" * 80)
        print(f"📊 Final Results:")
        print(f"   - Best validation accuracy: {max(val_accuracies):.2f}%")
        print(f"   - Model saved as: best_model.pth")
        print(f"   - Training history saved as: training_history.png")
        print(f"   - Model ready for inference!")
        
    except Exception as e:
        print(f"❌ Error occurred: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
