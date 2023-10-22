import os
import random
import torch
import torch.nn as nn
from DataLoader import get_data_loaders
from sklearn.metrics import accuracy_score, f1_score
from Model_Custom import CustomNetwork
from Autoencoder import Autoencoder
from torchmetrics.image import StructuralSimilarityIndexMeasure
from torchvision.utils import save_image

batch_size = 128

_, _, test_loader = get_data_loaders('/lustre/groups/labs/marr/qscd01/datasets/191024_AML_Matek/train_val_test',
                                     batch_size=batch_size, num_workers=0)

unique_labels = set()
for _, label in test_loader.dataset:
    unique_labels.add(label)

num_classes = len(unique_labels)

model = Autoencoder()
model.load_state_dict(torch.load('best_autoencoder_model.pth'))

criterion = nn.MSELoss()
ssim_metric = StructuralSimilarityIndexMeasure()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

test_loss = 0.0
ssim_loss_test = 0.0

output_folder = "reconstructed"
os.makedirs(output_folder, exist_ok=True)
images_to_compare = []

with torch.no_grad():
    for i, (images, _) in enumerate(test_loader):
        images = images.to(device)
        outputs = model(images)
        loss = criterion(outputs, images)
        test_loss += loss.item()

        for image, output in zip(images, outputs):
            image = image.unsqueeze(0).cpu()
            output = output.unsqueeze(0).cpu()
            ssim = 1 - ssim_metric(image, output)
            ssim_loss_test += ssim.item()

        if i < 10:
            images = images * 255
            outputs = outputs * 255
            image = torch.cat([images[i], outputs[i]], dim=2).cpu()
            images_to_compare.append(image)

test_loss /= len(test_loader)
ssim_loss_test /= len(test_loader)

print(f"Test Loss (MSE): {test_loss:.4f}")
print(f"Average SSIM Loss: {ssim_loss_test:.4f}")

for i in range(10):
    save_image(images_to_compare[i], os.path.join(output_folder, f'reconstructed_{i}.png'), nrow=1)
