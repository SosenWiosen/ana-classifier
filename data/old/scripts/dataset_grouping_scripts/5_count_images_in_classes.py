import os;
def count_images(dst_path):
    result = {}

    for dirpath,_,filenames in os.walk(dst_path):
        # Store count of base filenames
        if dirpath != dst_path:
            base_filename_count = {}
            for f in filenames:
                # Check if file is a .tiff file and doesn't contain 'tissue'
                if f.endswith(".png") and 'tissue' not in f:
                    # Split the filename from the last underscore (_)
                    base_name = "_".join(f.split("_")[:-1])
                    # Update count
                    base_filename_count[base_name] = base_filename_count.get(base_name, 0) + 1

            classname = os.path.basename(dirpath)
            # Count unique images; each unique base filename represents a unique image
            result[classname] = len(base_filename_count)
    
    return result

dst_path = "/Users/sosen/UniProjects/eng-thesis/data/data-uncompressed/2D-tiff-grouped-filtered"

# Calling the count function
image_count = count_images(dst_path)



# Sort the dictionary by its keys (i.e., class names) and print the result
for classname, count in sorted(image_count.items()):
    print(f'Number of unique images in class {classname}: {count}')