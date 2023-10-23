import torch
import os
import numpy as np
from PIL import Image
from DataLoader import get_data_loaders
from Autoencoder import Autoencoder


_, val_loader, _ = get_data_loaders('/lustre/groups/labs/marr/qscd01/datasets/191024_AML_Matek'
                                    '/train_val_test',
                                    batch_size=1, num_workers=0)

model = Autoencoder()
model.load_state_dict(torch.load('best_autoencoder_model.pth'))
model.eval()

save_folder = "outputsvsimages"
os.makedirs(save_folder, exist_ok=True)

with torch.no_grad():
    for i, (images, _) in enumerate(val_loader):
        images = images.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        outputs = model(images)

        images = (images * 255).cpu().detach().numpy().astype(np.uint8)
        outputs = (outputs * 255).cpu().detach().numpy().astype(np.uint8)
        concatenated_images = np.concatenate((images, outputs), axis=2)

        concatenated_image = Image.fromarray(concatenated_images[0].transpose(1, 2, 0))
        concatenated_image.save(os.path.join(save_folder, f'image_{i}.jpg'))

        if i >= 10:
            break
