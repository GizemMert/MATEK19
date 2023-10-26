import torch
import torch.nn as nn


class CustomNetwork(nn.Module):
    def __init__(self, num_classes):
        super(CustomNetwork, self).__init__()

        # Conv1
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU()

        # Conv2
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.relu2 = nn.ReLU()

        # Max pool
        self.maxpool1 = nn.MaxPool2d(2, 2)

        # Conv3
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.relu3 = nn.ReLU()

        # Conv4
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(512)
        self.relu4 = nn.ReLU()

        # Max pool
        self.maxpool2 = nn.MaxPool2d(2, 2)

        # Conv5
        self.conv5 = nn.Conv2d(512, 1024, kernel_size=3, stride=2, padding=1)
        self.bn5 = nn.BatchNorm2d(1024)
        self.relu5 = nn.ReLU()

        # Conv6
        self.conv6 = nn.Conv2d(1024, 2048, kernel_size=3, stride=2, padding=1)
        self.bn6 = nn.BatchNorm2d(2048)
        self.relu6 = nn.ReLU()

        # Global average pool
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))

        # Fully connected layer
        self.fc1 = nn.Linear(2048, 512)
        self.relu7 = nn.ReLU()

        # Output layer
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)

        x = self.maxpool1(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)

        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu4(x)

        x = self.maxpool2(x)

        x = self.conv5(x)
        x = self.bn5(x)
        x = self.relu5(x)

        x = self.conv6(x)
        x = self.bn6(x)
        x = self.relu6(x)

        x = self.avgpool(x)

        x = x.view(-1, 2048)

        x = self.fc1(x)
        x = self.relu7(x)

        x = self.fc2(x)

        return x
