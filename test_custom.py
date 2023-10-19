import torch
import torch.nn as nn
from DataLoader import get_data_loaders
from sklearn.metrics import accuracy_score, f1_score
from Model_Custom import CustomNetwork

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

# Create an instance of your custom network
model = CustomNetwork(num_classes)

# Load the trained weights
model.load_state_dict(torch.load('custom_model.pth'))

# Loss function
criterion = nn.CrossEntropyLoss()

# Evaluation loop
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

test_loss = 0.0
correct = 0
total = 0
true_labels = []
predicted_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)

        test_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        true_labels.extend(labels.cpu().numpy())
        predicted_labels.extend(predicted.cpu().numpy())

test_accuracy = accuracy_score(true_labels, predicted_labels)
test_loss /= len(test_loader)
test_f1 = f1_score(true_labels, predicted_labels, average='weighted')

print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2%}, Test F1 Score: {test_f1:.4f}")
