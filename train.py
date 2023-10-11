import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from sklearn.metrics import accuracy_score, f1_score
from torchvision.models import ResNet50_Weights
from Model_Custom import CustomNetwork
from DataLoader import get_data_loaders
import matplotlib.pyplot as plt


batch_size = 128
num_epochs = 50
learning_rate = 0.001

# Data loaders
train_loader, val_loader, _ = get_data_loaders('/lustre/groups/labs/marr/qscd01/datasets/191024_AML_Matek'
                                               '/train_val_test',
                                               batch_size=batch_size, num_workers=0)

# of classes in training data
unique_labels = set()
for _, label in train_loader.dataset:
    unique_labels.add(label)

num_classes = len(unique_labels)

# load custom network
model = CustomNetwork(num_classes)

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# save the results of metrics
results = open("training_results.txt", "w")

# Training loop
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

train_losses = []
train_accuracies = []
train_f1s = []

val_losses = []
val_accuracies = []
val_f1s = []

best_val_f1 = 0.0
best_epoch = 0

try:
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        correct_train = 0
        total_train = 0
        true_labels_train = []
        predicted_labels_train = []

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

            true_labels_train.extend(labels.cpu().numpy())
            predicted_labels_train.extend(predicted.cpu().numpy())

        train_loss /= len(train_loader)
        train_accuracy = accuracy_score(true_labels_train, predicted_labels_train)
        train_f1 = f1_score(true_labels_train, predicted_labels_train, average='weighted')

        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)
        train_f1s.append(train_f1)

        # Validation loop
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0
        true_labels_val = []
        predicted_labels_val = []

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

                true_labels_val.extend(labels.cpu().numpy())
                predicted_labels_val.extend(predicted.cpu().numpy())

        val_loss /= len(val_loader)
        val_accuracy = accuracy_score(true_labels_val, predicted_labels_val)
        val_f1 = f1_score(true_labels_val, predicted_labels_val, average='weighted')

        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)
        val_f1s.append(val_f1)

        print(f"Epoch {epoch + 1}/{num_epochs}, "
              f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2%}, Train F1: {train_f1:.4f}, "
              f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2%}, Val F1: {val_f1:.4f}")

        # Save results to the file
        results.write(f"Epoch {epoch + 1}/{num_epochs}\n")
        results.write(
            f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2%}, Train F1: {train_f1:.4f}\n")
        results.write(
            f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2%}, Val F1: {val_f1:.4f}\n")
        results.write("\n")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch = epoch + 1
            torch.save(model.state_dict(), 'resnet_model.pth')

    results.close()

    print(f"Best Validation F1 score: {best_val_f1:.4f} at Epoch {best_epoch}")

except KeyboardInterrupt:
    print("Training interrupted. Saving the best model.")
    print(f"Best Validation F1 score: {best_val_f1:.4f} at Epoch {best_epoch}")
    torch.save(model.state_dict(), 'resnet_model.pth')

# Plot loss
plt.figure(figsize=(8, 6))
plt.plot(range(1, num_epochs + 1), train_losses, label='Train Loss', marker='o', linestyle='-', color='b')
plt.plot(range(1, num_epochs + 1), val_losses, label='Validation Loss', marker='o', linestyle='-', color='r')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Train and Validation Loss Over Epochs')
plt.legend()
plt.grid(True)
plt.xticks(range(1, num_epochs + 1))
plt.minorticks_on()
plt.grid(which='minor', linestyle='--', linewidth=0.5, alpha=0.7)
plt.tight_layout()
plt.savefig('loss_plot.png', dpi=300)
