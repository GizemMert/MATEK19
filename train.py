import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import DataLoader
from DataLoader import get_data_loaders


batch_size = 128
num_epochs = 150
learning_rate = 0.001

# Data loaders
train_loader, val_loader, _ = get_data_loaders('/lustre/groups/labs/marr/qscd01/datasets/191024_AML_Matek/train_val_test', batch_size=batch_size, num_workers=0)

# of classes in training data
unique_labels = set()
for _, label in train_loader.dataset:
    unique_labels.add(label)

num_classes = len(unique_labels)

#pre-trained ResNet-50 model
model = models.resnet50(pretrained=True, progress=True)
model.fc = nn.Linear(model.fc.in_features, num_classes)

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

train_losses = [] 
train_accuracies = [] 

best_val_loss = float('inf') 
no_improvement = 0  # non-improvement counter
early_stopping_patience = 5  # to wait before early stopping


for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    correct_train = 0
    total_train = 0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total_train += labels.size(0)
        correct_train += (predicted == labels).sum().item()

    train_accuracy = 100 * correct_train / total_train  
    train_loss /= len(train_loader)

    train_losses.append(train_loss) 
    train_accuracies.append(train_accuracy / 100.0)  

    # Validation loop
    model.eval()
    val_loss = 0.0
    correct_val = 0
    total_val = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            val_loss += loss.item() 
            _, predicted = torch.max(outputs.data, 1)
            total_val += labels.size(0)
            correct_val += (predicted == labels).sum().item()

    val_accuracy = 100 * correct_val / total_val  
    val_loss /= len(val_loader)

    print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%, "
          f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%")

    # Early stopping 
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        no_improvement = 0
    else:
        no_improvement += 1

    if no_improvement >= early_stopping_patience:
        print("Early stopping due to no improvement in validation loss.")
        break


overall_train_loss = sum(train_losses) / len(train_losses)
overall_train_accuracy = sum(train_accuracies) / len(train_accuracies)

print(f"Overall Average Train Loss: {overall_train_loss:.4f}, Overall Average Train Accuracy: {overall_train_accuracy:.2f}")


torch.save(model.state_dict(), 'resnet_model.pth')



