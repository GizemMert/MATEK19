import torch.nn as nn


class Autoencoder1(nn.Module):
    def __init__(self):
        super(Autoencoder1, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=1, stride=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=1, stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=1, stride=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 100, kernel_size=1, stride=2),
            nn.BatchNorm2d(100),
            nn.ReLU(),
            nn.Conv2d(100, 30, kernel_size=1, stride=1),
            nn.BatchNorm2d(30),
            nn.ReLU(),
            nn.Conv2d(30, 5, kernel_size=1, stride=1),
            nn.BatchNorm2d(5),
            nn.ReLU(),
            # nn.Flatten(),
            # nn.Linear(7840, 1000)
        )

        self.decoder = nn.Sequential(
            # nn.Linear(1000, 7840),

            # nn.Unflatten(1, (10, 28, 28)),
            nn.ConvTranspose2d(5, 30, kernel_size=1, stride=1),
            nn.ReLU(),

            nn.ConvTranspose2d(30, 100, kernel_size=1, stride=1),
            nn.ReLU(),

            nn.ConvTranspose2d(100, 256, kernel_size=1, stride=2),
            nn.ReLU(),

            nn.ConvTranspose2d(256, 200, kernel_size=2, stride=1),
            nn.ReLU(),

            nn.ConvTranspose2d(200, 150, kernel_size=2, stride=2),
            nn.ReLU(),

            nn.ConvTranspose2d(150, 3, kernel_size=1, stride=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
