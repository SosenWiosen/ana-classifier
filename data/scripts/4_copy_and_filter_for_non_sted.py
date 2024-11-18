import os
import shutil

# Define the source and destination directories
source_dir = "/Users/sosen/UniProjects/eng-thesis/data/data-uncompressed/2D-tiff-grouped-png/"
destination_dir = "/Users/sosen/UniProjects/eng-thesis/data/data-uncompressed/2D-tiff-grouped-filtered/"


# Walk through the source directory
for dirpath, dirs, files in os.walk(source_dir):
    for file in files:
        # Exclude files that contain 'sted' or 'tissue'
        if 'sted'  not in file and 'STED' not in file and 'tissue' not in file:
            # Generate full file paths
            source_file = os.path.join(dirpath, file)
            relative_path = os.path.relpath(dirpath, source_dir)
            destination_file = os.path.join(destination_dir, relative_path, file)

            # Make sure that destination directories exist
            os.makedirs(os.path.dirname(destination_file), exist_ok=True)

            # Copy file
            shutil.copy2(source_file, destination_file)