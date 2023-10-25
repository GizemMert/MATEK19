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
            nn.Conv2d(128, 256, kernel_size=3, stride=2),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 128, kernel_size=2, stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 60, kernel_size=2, stride=2),
            nn.BatchNorm2d(60),
            nn.ReLU(),
            nn.Conv2d(60, 50, kernel_size=2, stride=2),
            nn.BatchNorm2d(50),
            nn.ReLU(),
            nn.Conv2d(50, 40, kernel_size=2),
            nn.BatchNorm2d(40),
            nn.ReLU(),
        )

        self.decoder = nn.Sequential(

            nn.ConvTranspose2d(40, 50, kernel_size=3),
            nn.ReLU(),

            nn.ConvTranspose2d(50, 60, kernel_size=2, stride=2),
            nn.ReLU(),

            nn.ConvTranspose2d(60, 128, kernel_size=2, stride=2),
            nn.ReLU(),

            nn.ConvTranspose2d(128, 256, kernel_size=2, stride=2),
            nn.ReLU(),

            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            nn.ReLU(),

            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.ReLU(),

            nn.ConvTranspose2d(64, 3, kernel_size=1, stride=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
