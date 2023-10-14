import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from sklearn.metrics import accuracy_score, f1_score
from torchvision.models import ResNet50_Weights
from Model_Custom import CustomNetwork
from DataLoader import get_data_loaders
import matplotlib.pyplot as plt
from Autoencoder import Autoencoder


batch_size = 64
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

model = Autoencoder()

# Loss function and optimizer
criterion = nn.MSELoss()  # Use Mean Squared Error (MSE) loss
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

train_losses = []
val_losses = []
best_val_loss = float('inf')

# Open a file for saving the training results
results_file = open("training_results_autoencoder.txt", "w")

try:
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0

        for images, _ in train_loader:  # Don't use labels (_)
            images = images.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, images)  # Compare outputs to input images
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        # Validation loop
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for images, _ in val_loader:  # Don't use labels (_)
                images = images.to(device)
                outputs = model(images)
                loss = criterion(outputs, images)  # Compare outputs to input images
                val_loss += loss.item()

        val_loss /= len(val_loader)
        val_losses.append(val_loss)

        # Print training and validation losses on the same line
        print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")

        # Save results to the file
        results_file.write(f"Epoch {epoch + 1}/{num_epochs}\n")
        results_file.write(f"Train Loss: {train_loss:.4f}\n")
        results_file.write(f"Validation Loss: {val_loss:.4f}\n\n")

        if val_loss < best_val_loss:  # Check if the current validation loss is the best
            best_val_loss = val_loss  # Update the best validation loss
            torch.save(model.state_dict(), 'best_autoencoder_model.pth')

    results_file.close()

except KeyboardInterrupt:
    print("Training interrupted.")
    torch.save(model.state_dict(), 'best_autoencoder_model.pth')


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
