import cv2

# Read the image
image = cv2.imread('/Users/sosen/UniProjects/eng-thesis/data/data-uncompressed/2D-tiff-grouped-filtered/AC2/2D 37_Sample167_AC2_DFS_2D_Image 11.tiff')

# Extract the red channel
red_channel = image[:, :, 2]

# Convert to grayscale (though this is essentially just using the red channel)
grayscale_image = red_channel
# Apply Otsu's binarization
_, binary_mask = cv2.threshold(grayscale_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
# Define a kernel for morphological operations
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

# Perform morphological operations
# First, remove small noise
mask_cleaned = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)

# Then, close small holes inside the cells
mask_final = cv2.morphologyEx(mask_cleaned, cv2.MORPH_CLOSE, kernel)
cv2.imwrite('masked_image.jpg', mask_final)
