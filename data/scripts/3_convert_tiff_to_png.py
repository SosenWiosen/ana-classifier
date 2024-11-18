import os
from PIL import Image

def convert_tiff_to_png(source_dir, destination_dir):
    for root, dirs, files in os.walk(source_dir):
        for file in files:
            if file.lower().endswith('.tiff') or file.lower().endswith('.tif'):
                # Build full file path
                full_path = os.path.join(root, file)
                
                # Construct relative directory path
                relative_dir = os.path.relpath(root, source_dir)
                
                # Create corresponding destination directory if it doesn't exist
                dest_dir = os.path.join(destination_dir, relative_dir)
                os.makedirs(dest_dir, exist_ok=True)
                
                # Load and convert image
                with Image.open(full_path) as img:
                    # Extract file name without extension and append .png
                    file_name, _ = os.path.splitext(file)
                    png_file_path = os.path.join(dest_dir, file_name + '.png')
                    
                    # Save image as PNG
                    img.save(png_file_path, 'PNG')
                # print(f"Converted {full_path} to {png_file_path}")

source_directory = '/Users/sosen/UniProjects/eng-thesis/data/data-uncompressed/2D-tiff-grouped'
destination_directory = '/Users/sosen/UniProjects/eng-thesis/data/data-uncompressed/2D-tiff-grouped-png/'

convert_tiff_to_png(source_directory, destination_directory)