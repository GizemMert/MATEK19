import torch
from Model_Custom import CustomNetwork


model = CustomNetwork(14)

# Create a random input tensor with the same dimensions as your expected input (e.g., batch size 128, 3 channels, 224x224)
sample_input = torch.randn(128, 3, 224, 224)  # Adjust the batch size as needed

# Pass the input through the model
output = model(sample_input)

# Print the size of the output tensor
print("Output size:", output.size())
