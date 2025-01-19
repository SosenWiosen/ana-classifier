import os
import shutil

# File containing dataset information
dataset_file = "/Users/sosen/UniProjects/eng-thesis/I3A single cell/gt_training.csv"  # Change this to the path of your file

# Root directory where images are located
source_directory = "/Users/sosen/UniProjects/eng-thesis/I3A single cell/images"  # Change this to the directory containing your images

# Root directory to place the class-based folders
destination_directory = "/Users/sosen/UniProjects/eng-thesis/I3A single cell/images_grouped"  # Root folder to organize files

# Ensure destination directory exists
os.makedirs(destination_directory, exist_ok=True)

# Read the file and process each line
with open(dataset_file, "r") as file:
    for line in file:
        # Split the line into components (filename, mask, class name, other details)
        parts = line.strip().split(",")
        if len(parts) < 3:
            continue  # Ignore lines that don't have enough columns

        # Extract file names and class name
        image_file = parts[0]  # First column: Image file
        mask_file = parts[1]  # Second column: Mask file
        class_name = parts[2]  # Third column: Class name

        # Create a folder corresponding to the class name
        class_folder = os.path.join(destination_directory, class_name)
        os.makedirs(class_folder, exist_ok=True)

        # Move or copy the image and mask files into the appropriate folder
        image_path = os.path.join(source_directory, image_file)
        mask_path = os.path.join(source_directory, mask_file)

        # Check if the files exist before copying
        if os.path.exists(image_path):
            shutil.copy(image_path, class_folder)  # Use shutil.move() if you want to move instead of copy
        if os.path.exists(mask_path):
            shutil.copy(mask_path, class_folder)  # Use shutil.move() if you want to move instead of copy

print("Files organized successfully.")