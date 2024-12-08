import os
import shutil

# Path to the parent directory
parent_dir = "/Users/sosen/UniProjects/eng-thesis/data/data-uncompressed/temp"
# Path to the combined directory
combined_dir = "/Users/sosen/UniProjects/eng-thesis/data/data-uncompressed/2D"

# Create the combined directory if it doesn't exist
os.makedirs(combined_dir, exist_ok=True)

# Iterate over each item in the parent directory
for root, _, files in os.walk(parent_dir):
    for file in files:
        if file.endswith(".tiff"):
            # Determine the source file path
            source_file = os.path.join(root, file)
            
            # Calculate the relative path of the file (from the parent directory)
            relative_path = os.path.relpath(source_file, parent_dir)
            
            # Create the target directory structure within the combined directory
            target_dir = os.path.join(combined_dir, os.path.dirname(relative_path))
            os.makedirs(target_dir, exist_ok=True)
            
            # Copy the file to the target directory
            shutil.copy2(source_file, target_dir)