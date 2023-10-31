import os
import glob


directory_path = '/lustre/groups/labs/marr/qscd01/datasets/armingruber/_Domains/Acevedo_cropped'

total_images = 0

for subfolder in os.listdir(directory_path):
    subfolder_path = os.path.join(directory_path, subfolder)

    if os.path.isdir(subfolder_path):
        tiff_files = glob.glob(os.path.join(subfolder_path, '*.jpg'))
        num_images = len(tiff_files)

        total_images += num_images

output_file = "total_images_count_Acevedo_cropped.txt"
with open(output_file, "w") as file:
    file.write(f"Total number of images in the directory: {total_images}")

print(f"Total number of images in the directory: {total_images}")
print(f"Result saved to {output_file}")
