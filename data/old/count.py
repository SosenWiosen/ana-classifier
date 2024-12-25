import os

def count_png_files_in_subdirectories(directory_path):
    # Dictionary to store the count of PNG files for each subdirectory
    png_file_counts = {}

    # Walk through the directory
    for root, dirs, files in os.walk(directory_path):
        # Filter out the PNG files
        png_files = [file for file in files if file.lower().endswith('.png')]
        
        # Calculate the number of PNG files in the current directory
        png_file_count = len(png_files)
        
        # Store the count in the dictionary with the current directory path as the key
        png_file_counts[root] = png_file_count

    return png_file_counts

def main():
    # Path to the main directory
    directory_path = '/Users/sosen/UniProjects/eng-thesis/data/old/manual'
    
    # Get the PNG file count for each subdirectory
    png_file_counts = count_png_files_in_subdirectories(directory_path)
    
    # Print the results
    for subdir, count in png_file_counts.items():
        print(f'{subdir}: {count} PNG files')

if __name__ == '__main__':
    main()