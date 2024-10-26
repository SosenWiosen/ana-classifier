#!/bin/zsh

# Path to the parent directory
parent_dir="/Users/sosen/UniProjects/eng-thesis/data/data-uncompressed/temp"
# Path to the combined directory
combined_dir="/Users/sosen/UniProjects/eng-thesis/data/data-uncompressed/2D"

# Create the combined directory if it doesn't exist
mkdir -p "$combined_dir"

# Iterate over each subdirectory in the parent directory
for subdir in "$parent_dir"/*; do
    # Check if it's a directory
    if [ -d "$subdir" ]; then
        # Find all .tiff files within the current subdirectory recursively
        find "$subdir" -type f -name "*.tiff" | while read file; do
            # Get the relative path of the file relative to the top directory inside "$parent_dir"
            relative_path="${file#$subdir/}"
            # Create the target directory structure inside "$combined_dir"
            target_dir="$combined_dir/$(dirname "$relative_path")"
            mkdir -p "$target_dir"
            # Copy the .tiff file to the target directory
            cp -Rnp "$file" "$target_dir/"
        done
    fi
done