from PIL import Image
import numpy as np

def calculate_channel_contribution(image_path):
    with Image.open(image_path) as img:
        # Ensure the image is in RGB mode
        img = img.convert('RGB')
        # Convert the image to a NumPy array
        img_np = np.array(img)
        
        # Calculate the sum of each channel
        red_sum = np.sum(img_np[:, :, 0])
        green_sum = np.sum(img_np[:, :, 1])
        blue_sum = np.sum(img_np[:, :, 2])
        
        # Calculate total pixels intensity
        total_sum = red_sum + green_sum + blue_sum

        # Calculate contribution as a percentage
        red_percentage = (red_sum / total_sum) * 100
        green_percentage = (green_sum / total_sum) * 100
        blue_percentage = (blue_sum / total_sum) * 100

        return red_percentage, green_percentage, blue_percentage

# Example usage
image_path = '/Users/sosen/UniProjects/eng-thesis/data/data-uncompressed/2D-tiff-grouped-filtered-sted/AC4/2D 21_Sample 141_AC4_Plytka23_Image 14_STED.png'
red_perc, green_perc, blue_perc = calculate_channel_contribution(image_path)

print(f"Red Channel Contribution: {red_perc:.2f}%")
print(f"Green Channel Contribution: {green_perc:.2f}%")
print(f"Blue Channel Contribution: {blue_perc:.2f}%")