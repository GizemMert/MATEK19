import torch
import torch.nn as nn
from DataLoader import get_data_loaders
from sklearn.metrics import accuracy_score, f1_score
from Model_Custom import CustomNetwork
from Autoencoder import Autoencoder
batch_size = 128

# Data loader
_, _, test_loader = get_data_loaders('/lustre/groups/labs/marr/qscd01/datasets/191024_AML_Matek'
                                     '/train_val_test',
                                     batch_size=batch_size, num_workers=0)

# Number of classes in test data
unique_labels = set()
for _, label in test_loader.dataset:
    unique_labels.add(label)

num_classes = len(unique_labels)

model = Autoencoder()  # Create an instance of your Autoencoder model

# Load the trained weights
model.load_state_dict(torch.load('best_autoencoder_model.pth'))

# Loss function (MSE for autoencoder)
criterion = nn.MSELoss()

# Evaluation loop
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

test_loss = 0.0

with torch.no_grad():
    for images, _ in test_loader:  # No labels needed for testing
        images = images.to(device)
        outputs = model(images)
        loss = criterion(outputs, images)  # Calculate loss using MSE
        test_loss += loss.item()

test_loss /= len(test_loader)

print(f"Test Loss (MSE): {test_loss:.4f}")
