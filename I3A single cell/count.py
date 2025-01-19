import os

# Define a list of image file extensions to look for
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tiff", ".webp"}

def count_images_by_directory(root_directory):
    """
    Recursively counts image files in each directory and subdirectory.

    Args:
        root_directory (str): The root directory to start counting from.

    Returns:
        dict: A dictionary where keys are directory paths and values are counts of image files.
    """
    image_count_by_dir = {}

    # Walk through the directory tree
    for current_dir, _, files in os.walk(root_directory):
        # Count how many image files are in the current directory
        image_count = sum(1 for file in files if os.path.splitext(file.lower())[1] in IMAGE_EXTENSIONS)
        image_count_by_dir[current_dir] = image_count

    return image_count_by_dir


def print_image_counts(image_count_by_dir):
    """
    Prints the image counts for each directory.

    Args:
        image_count_by_dir (dict): Dictionary with directory paths as keys and image counts as values.
    """
    for directory, count in image_count_by_dir.items():
        print(f"Directory: {directory}, Image Files: {count}")


# Main script to execute the function
if __name__ == "__main__":
    # Define the root directory to start counting (change this to your desired directory)
    root_directory = "/Users/sosen/UniProjects/eng-thesis/I3A single cell/images_grouped"  # Example: "./images"

    # Get the count of image files by directory
    image_counts = count_images_by_directory(root_directory)

    # Print the results
    print_image_counts(image_counts)