import cv2
import os
import numpy as np

# Reading source and output paths from the console
sourcePath = input("Enter the source directory path: ")
outputPath = input("Enter the destination directory path: ")

# Ensure output directory exists
if not os.path.exists(outputPath):
    os.makedirs(outputPath)

# Iterate over each file in the source directory
for filename in os.listdir(sourcePath):
    file_path = os.path.join(sourcePath, filename)
    
    # Skip if it is a directory
    if os.path.isdir(file_path):
        continue

    # Read the image
    src = cv2.imread(file_path, cv2.IMREAD_COLOR)

    # Check if the image is loaded successfully
    if src is None:
        print(f"Error loading image: {file_path}")
        continue

    # Convert BGR image to Lab color space
    lab = cv2.cvtColor(src, cv2.COLOR_BGR2Lab)

    # Split the Lab channels
    lab_channels = cv2.split(lab)
    A = lab_channels[1]

    # Apply Otsu's thresholding on the 'A' channel
    _, thresh = cv2.threshold(A, 70, 235, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Morphological operations
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    result_open = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel_open)

    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    result_close = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel_close)

    # Combine the results of the open and close operations
    morphed_result = cv2.bitwise_or(result_open, result_close)

    # Further improve the result with additional open and close operations
    improve_with_open = cv2.morphologyEx(morphed_result, cv2.MORPH_OPEN, kernel_open)
    final_result = cv2.morphologyEx(improve_with_open, cv2.MORPH_CLOSE, kernel_close)

    # Apply the final mask to the original image
    final_src = cv2.bitwise_and(src, src, mask=final_result)

    # Save the processed image to the output directory
    output_file_path = os.path.join(outputPath, filename)
    if not cv2.imwrite(output_file_path, final_src):
        print(f"Error: Could not save the image {output_file_path}")

print("Processing completed.")