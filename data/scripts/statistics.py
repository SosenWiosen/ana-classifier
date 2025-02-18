from PIL import Image
import numpy as np
import os

def compute_red_channel_stats(image_dir):
    image_sums = 0
    image_sq_sums = 0
    num_pixels = 0
    
    # Traverse directories recursively
    for root, dirs, files in os.walk(image_dir):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_path = os.path.join(root, file)
                try:
                    with Image.open(image_path) as img:
                        # Convert to RGB (discards alpha if present)
                        img = img.convert('RGB')
                        img = img.resize((256, 256))  # Resize uniformly
                        img_array = np.array(img, dtype=np.float32) / 255.0
                        
                        red_channel = img_array[:, :, 0]
                        
                        image_sums += np.sum(red_channel)  # Sum the values in the red channel
                        image_sq_sums += np.sum(np.square(red_channel))  # Sum the squares of the red channel
                        num_pixels += red_channel.size  # Count total number of red channel pixels processed

                except IOError:
                    print(f"Error opening or processing {image_path}")

    if num_pixels == 0:
        raise ValueError("No images processed. Check your image directory path and file access permissions.")
    
    mean_red = image_sums / num_pixels
    std_dev_red = np.sqrt(image_sq_sums / num_pixels - mean_red**2)

    return mean_red, std_dev_red

# Specify the directory containing your images
image_directory = '/Users/sosen/UniProjects/eng-thesis/data/manual/CROPPED/NON-STED/'
mean_red, std_dev_red = compute_red_channel_stats(image_directory)
print(f"Mean of the red channel: {mean_red}")
print(f"Standard deviation of the red channel: {std_dev_red}")

