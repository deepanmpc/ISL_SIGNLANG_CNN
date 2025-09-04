import torch
from torch.utils.data import DataLoader
from torchmetrics.classification import MulticlassAccuracy
import torch.nn as nn
import torch.optim as optim

def train(model, train_loader, val_loader, epochs, device):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    acc_metric = MulticlassAccuracy(num_classes=35).to(device)

    model.to(device)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        total_acc = 0.0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            total_acc += acc_metric(outputs, labels)

        avg_loss = running_loss / len(train_loader)
        avg_acc = total_acc / len(train_loader)
        print(f"[Epoch {epoch+1}] Loss: {avg_loss:.4f}, Accuracy: {avg_acc:.4f}")
        model.eval()
        with torch.no_grad():
            val_acc = 0.0
            for val_images, val_labels in val_loader:
                val_images, val_labels = val_images.to(device), val_labels.to(device)
                val_outputs = model(val_images)
                val_acc += acc_metric(val_outputs, val_labels)
            print(f"Validation Accuracy: {val_acc / len(val_loader):.4f}")