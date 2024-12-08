import os
import shutil

# Define source and destination directories
src = "/Users/sosen/UniProjects/eng-thesis/data/data-uncompressed/2D"
dst = "/Users/sosen/UniProjects/eng-thesis/data/data-uncompressed/2D-tiff"

# Walk through the source directory
for root, _, files in os.walk(src):
    for file in files:
        if file.endswith(".tiff"):
            # Construct the full file path
            file_path = os.path.join(root, file)
            
            # Determine the relative path from the source directory
            relative_path = os.path.relpath(root, src)
            
            # Determine the destination directory
            dest_dir = os.path.join(dst, relative_path)
            
            # Ensure the destination directory exists
            os.makedirs(dest_dir, exist_ok=True)
            
            # Copy the file to the destination directory
            shutil.copy(file_path, dest_dir)

print("Copying of TIFF files completed.")