import os
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from sklearn.preprocessing import LabelEncoder


class MatekDataLoader(Dataset):
    def __init__(self, data_dir, split='train', transform=None):
        self.data_dir = data_dir
        self.split = split
        self.transform = transform

        # Define the directory for the specified split
        self.split_dir = os.path.join(data_dir, split)

        self.image_files = [f for f in os.listdir(self.split_dir) if f.endswith('.tiff')]

        self.label_encoder = LabelEncoder()

        # Extract and encode labels
        self.labels = [os.path.basename(f).split('_')[0] for f in self.image_files]
        self.encoded_labels = self.label_encoder.fit_transform(self.labels)

        self.images = {}

        for img_filename in self.image_files:

            # read
            img_path = os.path.join(self.split_dir, img_filename)
            image = Image.open(img_path).convert('RGB')

            if self.transform:
                image = self.transform(image)

            self.images[img_filename] = image

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_filename = self.image_files[idx]
        image = self.images[image_filename]

        label = self.encoded_labels[idx]

        return image, label


def get_data_loaders(data_dir, batch_size, num_workers):
    transform = transforms.Compose([
        transforms.Resize((128, 128)),  # Resize
        transforms.ToTensor(),
        Normalize()
    ])

    train_dataset = MatekDataLoader(data_dir, split='train', transform=transform)
    val_dataset = MatekDataLoader(data_dir, split='val', transform=transform)
    test_dataset = MatekDataLoader(data_dir, split='test', transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader


class Normalize(object):
    def __call__(self, image):
        return image / 255.0
    