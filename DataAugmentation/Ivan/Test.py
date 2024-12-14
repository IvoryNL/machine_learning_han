import os
import cv2
import tensorflow as tf
from tensorflow.keras.layers import RandomRotation

# Define the source folder
source_folder = r"D:\Data\0. Machine Learning\0. Mini_project_finger_counting\0. Data\1. New data\Dataset IvanV1"

# Define the rotation layer with a random range
rotation_layer = RandomRotation(factor=(10 / 360, 60 / 360))
# rotation_layer = RandomRotation(factor=(-15 / 360, 15 / 360), fill_mode='constant', fill_value=0)

# Helper function to ensure directory existence
def ensure_dir_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)

# Function to extract the prefix from an image filename
def get_prefix(filename):
    return '_'.join(filename.split('_')[:-1]) + '_'

# Function to apply rotation augmentation
def rotate_image(image):
    """Apply random rotation using Keras RandomRotation layer."""
    image = tf.convert_to_tensor(image, dtype=tf.float32) / 255.0  # Normalize to [0, 1]
    augmented = rotation_layer(tf.expand_dims(image, 0))  # Add batch dimension
    augmented = tf.squeeze(augmented).numpy() * 255  # Remove batch and scale back to [0, 255]
    return augmented.astype('uint8')

# Function to perform data augmentation
def augment_data(source_folder):
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

            # Save the original image in the augmented folder
            original_output_path = os.path.join(output_folder, f"{prefix}{count}.jpg")
            cv2.imwrite(original_output_path, image)
            count += 1

            # Perform augmentation (rotation)
            rotated_image = rotate_image(image)

            # Save the augmented image
            augmented_output_path = os.path.join(output_folder, f"{prefix}{count}.jpg")
            cv2.imwrite(augmented_output_path, rotated_image)
            count += 1

# Execute the augmentation
augment_data(source_folder)
