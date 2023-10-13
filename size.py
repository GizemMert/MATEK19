from PIL import Image
import numpy as np

# Define the path to an example image
image_path = "/lustre/groups/labs/marr/qscd01/datasets/191024_AML_Matek/train_val_test/train/BAS_0023.tiff"

# Open the image
image = Image.open(image_path)

# Get original size and channels
original_size = image.size  # (width, height)
original_channels = len(np.array(image).shape)

# Convert the image to a NumPy array
image_array = np.array(image)

# Calculate the minimum and maximum pixel values
min_pixel_value = np.min(image_array)
max_pixel_value = np.max(image_array)

print("Original Image Size:", original_size)
print("Original Image Channels:", original_channels)
print("Minimum Pixel Value:", min_pixel_value)
print("Maximum Pixel Value:", max_pixel_value)
