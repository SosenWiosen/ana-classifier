import cv2 as cv
import numpy as np
import os

def normalize_image_contrast(img):
    # Convert the image to grayscale
    
    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv.createCLAHE(clipLimit=5.0, tileGridSize=(8, 8))
    clahe_img = clahe.apply(img)
    
    # Normalize the intensity levels to the range [0, 255]
    normalized_img = cv.normalize(clahe_img, None, alpha=100000, norm_type=cv.NORM_L2)
    
    return normalized_img

def preprocess_image(img):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    cv.imshow('Grayscale Image', gray)
    cv.waitKey(0)

    normalized_img = normalize_image_contrast(gray)
    cv.imshow('Normalized Image', normalized_img)
    cv.waitKey(0)

    # Apply adaptive thresholding to get a binary image
    max_value = 255
    block_size = 51  # Tune this value (must be odd)
    C = 15  # Tune this value
    adaptive_thresh = cv.adaptiveThreshold(
        normalized_img, max_value, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, block_size, C)
    cv.imshow('Adaptive Threshold Image', adaptive_thresh)
    cv.waitKey(0)

    # Apply morphological closing to close small gaps in the binary image
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (10, 10))
    morph = cv.morphologyEx(adaptive_thresh, cv.MORPH_CLOSE, kernel)
    cv.imshow('Morphological Close Image', morph)
    cv.waitKey(0)

    # Apply additional smoothing after morphological closing
    smoothed = cv.medianBlur(morph, 51)  # Apply median blur to smooth the image
    cv.imshow('Median Blur Image', smoothed)
    cv.waitKey(0)
    
    return smoothed

def find_and_draw_contours(img):
    # Preprocess the image to highlight dense regions
    processed_image = preprocess_image(img)

    # Find contours in the preprocessed image
    contours, _ = cv.findContours(processed_image, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    # Draw contours on the original image
    contour_img = img.copy()  # Preserve original image for drawing contours
    cv.drawContours(contour_img, contours, -1, (0, 255, 0), 2)  # (0, 255, 0) is green, 2 is the thickness of the contours
    return contour_img

def process_single_image(input_path, output_path=None):
    # Read the image from the input path
    img = cv.imread(input_path)
    if img is None:
        print(f"Error: Unable to load image at {input_path}")
        return

    # Process the image
    processed_img = find_and_draw_contours(img)
    
    if output_path:
        # Save the processed image to the output path
        cv.imwrite(output_path, processed_img)
        print(f"Processed image saved to {output_path}")
    else:
        # Display the processed image
        cv.imshow('Processed Image', processed_img)
        cv.waitKey(0)
        cv.destroyAllWindows()
if __name__ == "__main__":
    # input_dir = '/Users/sosen/UniProjects/eng-thesis/data/data-uncompressed/2D-tiff-grouped-filtered'
    # output_dir = '/Users/sosen/UniProjects/eng-thesis/data/data-uncompressed/2D-cropped'
    # process_directory(input_dir, output_dir)
    input_img ='/Users/sosen/UniProjects/eng-thesis/data/data-uncompressed/2D-tiff-grouped-filtered-sted/AC4/2D 21_Sample154_AC4_Plytka24_Image 6_STED.tiff'
    process_single_image(input_img)
    # input_img ='/Users/sosen/UniProjects/eng-thesis/data/data-uncompressed/2D-tiff-grouped-filtered/AC5/Sample_197_AC5_largespeckled_nRNP_Plytka29_Image 5_STEDtiff.tiff'
    # process_single_image(input_img)