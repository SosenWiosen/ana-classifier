import cv2
import numpy as np

def find_brightest_square(image_path, square_size):
    # Load the image
    image = cv2.imread(image_path)
    
    if image is None:
        raise FileNotFoundError(f"The image path '{image_path}' is invalid or the file does not exist.")
    
    # Convert to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Dimensions of the image
    h, w = gray_image.shape
    max_intensity_sum = None
    brightest_top_left = (0, 0)
    
    # Use integral image for efficient sum calculation over a window
    integral_image = cv2.integral(gray_image)
    
    # Slide the window over the image
    for i in range(h - square_size + 1):
        for j in range(w - square_size + 1):
            # Calculate the sum of intensities within the current window
            x, y = j, i
            xx, yy = j + square_size, i + square_size
            current_sum = integral_image[yy, xx] - integral_image[yy, x] - integral_image[y, xx] + integral_image[y, x]
            
            if max_intensity_sum is None or current_sum > max_intensity_sum:
                max_intensity_sum = current_sum
                brightest_top_left = (x, y)
    
    # Extract the image region corresponding to the brightest square
    x, y = brightest_top_left
    brightest_square = image[y:y + square_size, x:x + square_size]
    
    return brightest_square