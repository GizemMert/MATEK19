import torch
from Model_Custom import CustomNetwork
from torchsummary import summary
from Autoencoder import Autoencoder

model = Autoencoder()

sample_input = torch.randn(128, 3, 224, 224)
# Print the model summary
summary(model, input_size=sample_input.shape[1:])

