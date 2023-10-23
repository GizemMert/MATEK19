import torch
import os
import numpy as np
from PIL import Image
import torch.nn as nn
import torch.optim as optim
from DataLoader import get_data_loaders
from Autoencoder import Autoencoder
from torchmetrics.image import StructuralSimilarityIndexMeasure


batch_size = 64
num_epochs = 30
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

# Training loop
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-3)
ssim_metric = StructuralSimilarityIndexMeasure().to(device)


train_losses = []
val_losses = []
best_loss_val = float('inf')

results_file = open("training_results_autoencoder.txt", "w")

try:
    for epoch in range(num_epochs):

        model.train()
        loss_train = 0.0
        mse_loss_train = 0.0
        ssim_loss_train = 0.0

        for images, _ in train_loader:
            images = images.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            mse = criterion(outputs, images)
            ssim = 1 - ssim_metric(outputs, images)
            train_loss = mse + ssim
            train_loss.backward()
            optimizer.step()

            loss_train += train_loss.item()
            mse_loss_train += mse.item()
            ssim_loss_train += ssim.item()

        loss_train /= len(train_loader)
        mse_loss_train /= len(train_loader)
        ssim_loss_train /= len(train_loader)
        train_losses.append(loss_train)

        # Validation loop
        model.eval()
        loss_val = 0.0
        mse_loss_val = 0.0
        ssim_loss_val = 0.0

        with torch.no_grad():
            for images, _ in val_loader:
                images = images.to(device)
                outputs = model(images)
                mse = criterion(outputs, images)
                ssim = 1 - ssim_metric(outputs, images)
                val_loss = mse + ssim

                loss_val += val_loss.item()
                mse_loss_val += mse.item()
                ssim_loss_val += ssim.item()

            loss_val /= len(val_loader)
            mse_loss_val /= len(val_loader)
            ssim_loss_val /= len(val_loader)
            val_losses.append(loss_val)

        print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {loss_train:.4f}, Validation Loss: {loss_val:.4f}, "
              f"SSIM Train Loss:{ssim_loss_train:.4f}, SSIM Val Loss:{ssim_loss_val:.4f}, "
              f"MSE Train: {mse_loss_train:.4f}, MSE Val: {mse_loss_val:.4f}")

        results_file.write(f"Epoch {epoch + 1}/{num_epochs}\n")
        results_file.write(f"Train Loss: {loss_train:.4f}\n")
        results_file.write(f"Validation Loss: {loss_val:.4f}\n\n")
        results_file.write(f"SSIM Train Loss:{ssim_loss_train:.4f}\n\n")
        results_file.write(f"SSIM Val Loss:{ssim_loss_val:.4f}\n\n")

        if loss_val < best_loss_val:
            best_val_loss = loss_val
            torch.save(model.state_dict(), 'best_autoencoder_mod.pth')

    results_file.close()

except KeyboardInterrupt:
    print("Training interrupted.")
    torch.save(model.state_dict(), 'best_autoencoder_mod.pth')
