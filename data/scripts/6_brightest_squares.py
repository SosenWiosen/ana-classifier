# Define the source and destination directories
import os

import cv2

from brightest_square import find_brightest_square


source_dir = "/Users/sosen/UniProjects/eng-thesis/data/data-uncompressed/2D-tiff-grouped-filtered/"
destination_dir = "/Users/sosen/UniProjects/eng-thesis/data/data-uncompressed/2D-tiff-brightest-areas-128/"
square_size = 128  # Define the size of the square you want to extract

# Ensure the destination directory exists
os.makedirs(destination_dir, exist_ok=True)

# Walk through the source directory
for dirpath, dirs, files in os.walk(source_dir):
    for file in files:
        # Only process image files, you can add more extensions if needed
        if file.endswith(('.tif', '.tiff', '.jpg', '.png')):
            # Generate full file paths
            source_file = os.path.join(dirpath, file)
            relative_path = os.path.relpath(dirpath, source_dir)
            destination_file_dir = os.path.join(destination_dir, relative_path)
            destination_file = os.path.join(destination_file_dir, file)
            
            # Make sure that destination directories exist
            os.makedirs(destination_file_dir, exist_ok=True)
            
            try:
                # Find the brightest square in the current image
                brightest_square = find_brightest_square(source_file, square_size)
                
                # Save the brightest square to the destination
                cv2.imwrite(destination_file, brightest_square)
                # print(f"Processed and copied: {source_file} to {destination_file}")
            except Exception as e:
                print(f"Failed to process {source_file}: {e}")

print("All files have been processed successfully.")