import cv2 as cv
import numpy as np
import os

def preprocess_image(img):
# Convert the image to grayscale
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    cv.imshow('Grayscale Image', gray)
    cv.waitKey(0)

    # Apply a Gaussian Blur to reduce noise and improve contour detection
    blurred = cv.GaussianBlur(gray, (5, 5), 0)
    cv.imshow('Blurred Image', blurred)
    cv.waitKey(0)

    # Apply a threshold to get a binary image
    _, thresh = cv.threshold(blurred, 127, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)
    inverted_thresh = cv.bitwise_not(thresh)

    cv.imshow('Threshold Image', inverted_thresh)
    cv.waitKey(0)

    # Apply morphological closing to close small gaps in the binary image
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (35, 35))
    morph = cv.morphologyEx(inverted_thresh, cv.MORPH_CLOSE, kernel)
    cv.imshow('Morphological Close Image', morph)
    cv.waitKey(0)
    # Invert the image to make cells white and background black

    # Apply additional smoothing after morphological closing
    smoothed = cv.medianBlur(morph, 111)  # Apply median blur to smooth the image
    cv.imshow('Median Blur Image', smoothed)
    cv.waitKey(0)
    
    return smoothed
    
    return morph


def find_and_draw_contours(img):
    # Preprocess the image to highlight dense regions
    processed_image = preprocess_image(img)

    # Find contours in the preprocessed image
    contours, _ = cv.findContours(processed_image, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    # Draw contours on the original image
    cv.drawContours(img, contours, -1, (0, 255, 0), 2)  # (0, 255, 0) is the color (green), 2 is the thickness of the contours
    return img

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
    # input_img ='/Users/sosen/UniProjects/eng-thesis/data/data-uncompressed/2D-tiff-grouped-filtered/AC2/Sample_139_AC2_Plytka23_Image 4_STED.tiff'
    input_img ='/Users/sosen/UniProjects/eng-thesis/data/data-uncompressed/2D-tiff-grouped-filtered/AC5/Sample_197_AC5_largespeckled_nRNP_Plytka29_Image 5_STEDtiff.tiff'
    process_single_image(input_img)
    