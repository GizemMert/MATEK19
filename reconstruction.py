import torch
import os
import numpy as np
from PIL import Image
from DataLoader import get_data_loaders
from Autoencoder import Autoencoder
from torchvision.utils import save_image

batch_size = 64
_, val_loader, _ = get_data_loaders('/lustre/groups/labs/marr/qscd01/datasets/191024_AML_Matek'
                                    '/train_val_test',
                                    batch_size=batch_size, num_workers=0)

unique_labels = set()
for _, label in val_loader.dataset:
    unique_labels.add(label)

num_classes = len(unique_labels)

model = Autoencoder()
model.load_state_dict(torch.load('best_autoencoder_mod.pth'))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

output_folder = "ten_reconstruct"
os.makedirs(output_folder, exist_ok=True)
images_to_compare = []

with torch.no_grad():
    for i, (images, _) in enumerate(val_loader):
        images = images.to(device)
        outputs = model(images)

        if i < 10:
            images = images * 255
            outputs = outputs * 255
            image = torch.cat([images[i], outputs[i]], dim=2).cpu()
            images_to_compare.append(image)

for i in range(10):
    save_image(images_to_compare[i], os.path.join(output_folder, f'reconstructed_{i}.png'), nrow=1)
