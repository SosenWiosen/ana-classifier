import os
import shutil
import random
import string

# Define the source and destination directories
source_dir = "/Users/sosen/UniProjects/eng-thesis/data/data-uncompressed/2D-tiff-grouped-filtered/"
destination_dir = "/Users/sosen/UniProjects/eng-thesis/data/data-uncompressed/2D-tiff-grouped-random/"

# Function to generate a random string with specified length
def random_string(length):
    letters = string.ascii_letters
    return ''.join(random.choice(letters) for _ in range(length))

# Walk through the source directory
for dirpath, dirs, files in os.walk(source_dir):
    for file in files:
        # Exclude files that contain 'sted' or 'STED' and 'tissue'
        if 'sted' not in file and 'STED' not in file and 'tissue' not in file:
            # Generate full file paths
            source_file = os.path.join(dirpath, file)
            relative_path = os.path.relpath(dirpath, source_dir)
            
            # Generate a new randomized filename
            new_filename = random_string(5) + '_' + file  # Use a random prefix
            
            destination_file = os.path.join(destination_dir, relative_path, new_filename)

            # Make sure that destination directories exist
            os.makedirs(os.path.dirname(destination_file), exist_ok=True)

            # Copy file
            shutil.copy2(source_file, destination_file)