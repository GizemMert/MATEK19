import os
from torchvision import datasets

# Define the directory paths
directory_paths = [
    '/lustre/groups/labs/marr/qscd01/datasets/armingruber/_Domains/Matek_cropped',
    '/lustre/groups/labs/marr/qscd01/datasets/armingruber/_Domains/MLL_20221220',
    '/lustre/groups/labs/marr/qscd01/datasets/armingruber/_Domains/Acevedo_cropped'
]

total_images_counts = []

for directory_path in directory_paths:
    total_images = 0
    for subfolder in os.listdir(directory_path):
        subfolder_path = os.path.join(directory_path, subfolder)

        if os.path.isdir(subfolder_path) and not subfolder == 'ipynb_checkpoints':
            dataset = datasets.ImageFolder(subfolder_path)
            num_images = len(dataset)
            total_images += num_images

    total_images_counts.append((directory_path, total_images))


for directory_path, total_images in total_images_counts:
    output_file = f"total_images_count_{os.path.basename(directory_path)}.txt"
    with open(output_file, "w") as file:
        file.write(f"Total number of images for {os.path.basename(directory_path)}: {total_images}")

