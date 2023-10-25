import torch.nn as nn


class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 128, kernel_size=2, stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 64, kernel_size=2, stride=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 20, kernel_size=2, stride=1),
            nn.BatchNorm2d(20),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(15680, 1000),
            nn.ReLU(),
        )

        self.decoder = nn.Sequential(
            nn.Linear(1000, 15680),
            nn.Unflatten(1, (20, 28, 28)),

            nn.ConvTranspose2d(20, 64, kernel_size=2, stride=1),
            nn.ReLU(),

            nn.ConvTranspose2d(64, 128, kernel_size=2, stride=1),
            nn.ReLU(),

            nn.ConvTranspose2d(128, 256, kernel_size=2, stride=2),
            nn.ReLU(),

            nn.ConvTranspose2d(256, 200, kernel_size=3, stride=1),
            nn.ReLU(),

            nn.ConvTranspose2d(200, 150, kernel_size=4, stride=2),
            nn.ReLU(),

            nn.ConvTranspose2d(150, 3, kernel_size=3, stride=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x