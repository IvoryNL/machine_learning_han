# source_folder = r"D:\Data\0. Machine Learning\0. Mini_project_finger_counting\0. Data\1. New data\Dataset Ivan"
import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import RandomRotation, RandomZoom, RandomTranslation
import random

# Define the source folder
source_folder = r"D:\Data\0. Machine Learning\0. Mini_project_finger_counting\0. Data\1. New data\Dataset Ivan V2"

# Define augmentation layers
rotation_layer = RandomRotation(factor=(-15 / 360, 15 / 360), fill_mode='constant', fill_value=0)
scaling_layer = RandomZoom(height_factor=(-0.1, 0.1), width_factor=(-0.1, 0.1), fill_mode='constant', fill_value=0)
translation_layer = RandomTranslation(height_factor=0.1, width_factor=0.1, fill_mode='constant', fill_value=0)

# Number of augmentations to generate per image
num_augmentations = 5

# Helper function to ensure directory existence
def ensure_dir_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)

# Function to extract the prefix from an image filename
def get_prefix(filename):
    return '_'.join(filename.split('_')[:-1]) + '_'

# Function to resize the image to 67x50
def resize_image(image, target_size=(67, 50)):
    """Resize the image to the specified size."""
    return cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)

# Function to apply min-max scaling
def min_max_scale(image):
    """Normalize the image using min-max scaling to range [0, 1]."""
    return image / 255.0

# Function to convert the image to grayscale
def convert_to_grayscale(image):
    """Convert the image to grayscale."""
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Function to detect background color (most common color in the corners)
def detect_background_color(image):
    """Detect the background color based on the corner pixels."""
    corners = [
        image[0, 0],  # Top-left corner
        image[0, -1],  # Top-right corner
        image[-1, 0],  # Bottom-left corner
        image[-1, -1]  # Bottom-right corner
    ]
    background_color = np.median(corners, axis=0)  # Use the median color of the corners
    return background_color.astype(np.uint8)

# Function to fill gaps with the background color
def fill_gaps_with_background(image, mask, background_color):
    """Fill the gaps in the grayscale augmented image using the background color."""
    image[mask == 0] = background_color  # Replace gaps with background color
    return image

# Function to apply random rotation with background fill
def apply_rotation(image):
    """Apply random rotation to grayscale images using Keras RandomRotation."""
    background_color = detect_background_color(image)
    img_norm = tf.convert_to_tensor(image, dtype=tf.float32) / 255.0  # Normalize to [0, 1]
    rotated = rotation_layer(tf.expand_dims(tf.expand_dims(img_norm, -1), 0))  # Add channel and batch dimensions
    rotated = tf.squeeze(rotated).numpy() * 255  # Remove dimensions and scale back to [0, 255]
    rotated = rotated.astype(np.uint8)
    mask = rotated > 0  # Binary mask for non-zero pixels
    return fill_gaps_with_background(rotated, mask, background_color)

# Function to apply scaling augmentation
def apply_scaling(image):
    """Apply random scaling to grayscale images using Keras RandomZoom."""
    background_color = detect_background_color(image)
    img_norm = tf.convert_to_tensor(image, dtype=tf.float32) / 255.0  # Normalize to [0, 1]
    scaled = scaling_layer(tf.expand_dims(tf.expand_dims(img_norm, -1), 0))  # Add channel and batch dimensions
    scaled = tf.squeeze(scaled).numpy() * 255  # Remove dimensions and scale back to [0, 255]
    scaled = scaled.astype(np.uint8)
    mask = scaled > 0  # Binary mask for non-zero pixels
    return fill_gaps_with_background(scaled, mask, background_color)

# Function to apply translation augmentation
def apply_translation(image):
    """Apply random translation to grayscale images using Keras RandomTranslation."""
    background_color = detect_background_color(image)
    img_norm = tf.convert_to_tensor(image, dtype=tf.float32) / 255.0  # Normalize to [0, 1]
    translated = translation_layer(tf.expand_dims(tf.expand_dims(img_norm, -1), 0))  # Add channel and batch dimensions
    translated = tf.squeeze(translated).numpy() * 255  # Remove dimensions and scale back to [0, 255]
    translated = translated.astype(np.uint8)
    mask = translated > 0  # Binary mask for non-zero pixels
    return fill_gaps_with_background(translated, mask, background_color)

# Function to apply brightness augmentation
def apply_brightness(image):
    """Apply random brightness adjustment (±15%)."""
    factor = random.uniform(0.85, 1.15)  # Brightness factor between 85% and 115%
    image = np.clip(image * factor, 0, 255).astype(np.uint8)
    return image

# Function to apply sharpen or Gaussian blur augmentation
def apply_focus(image):
    """Randomly sharpen or blur the image."""
    if random.choice([True, False]):
        # Apply Gaussian blur
        return cv2.GaussianBlur(image, (5, 5), 0)
    else:
        # Apply sharpening kernel
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        return cv2.filter2D(image, -1, kernel)

# Function to perform data augmentation with the new steps integrated
def augment_data(source_folder, num_augmentations):
    for label_folder in os.listdir(source_folder):
        label_path = os.path.join(source_folder, label_folder)
        if not os.path.isdir(label_path):
            continue

        # Create the augmented folder inside the label folder
        output_folder = os.path.join(label_path, "augmented")
        ensure_dir_exists(output_folder)

        count = 1  # Start counting images from 1
        for image_file in os.listdir(label_path):
            image_path = os.path.join(label_path, image_file)
            if not image_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue

            # Extract the prefix from the original image
            prefix = get_prefix(image_file)

            # Read the image
            image = cv2.imread(image_path)
            if image is None:
                continue

            # Step 1: Resize the image
            image = resize_image(image)

            # Step 2: Convert to grayscale
            image = convert_to_grayscale(image)

            # Save the original grayscale image in the augmented folder
            original_output_path = os.path.join(output_folder, f"{prefix}{count}.jpg")
            cv2.imwrite(original_output_path, image)
            count += 1

            # Perform multiple random augmentations for this image
            for i in range(num_augmentations):
                augmented_image = image.copy()

                # Randomly apply rotation
                if random.choice([True, False]):
                    augmented_image = apply_rotation(augmented_image)

                # Randomly apply scaling
                if random.choice([True, False]):
                    augmented_image = apply_scaling(augmented_image)

                # Randomly apply translation
                if random.choice([True, False]):
                    augmented_image = apply_translation(augmented_image)

                # Randomly apply brightness adjustment
                if random.choice([True, False]):
                    augmented_image = apply_brightness(augmented_image)

                # Randomly apply focus adjustment (sharpen or blur)
                if random.choice([True, False]):
                    augmented_image = apply_focus(augmented_image)

                # Step 3: Apply min-max scaling
                augmented_image = min_max_scale(augmented_image)

                # Save the augmented image
                augmented_output_path = os.path.join(output_folder, f"{prefix}{count}.jpg")
                cv2.imwrite(augmented_output_path, (augmented_image * 255).astype(np.uint8))
                count += 1

# Execute the augmentation
augment_data(source_folder, num_augmentations)