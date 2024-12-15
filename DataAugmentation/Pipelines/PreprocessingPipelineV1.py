import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import RandomRotation, RandomZoom, RandomTranslation
import random

class PreprocessingPipeline:
    # Define augmentation layers
    rotation_layer = RandomRotation(factor=(-15 / 360, 15 / 360), fill_mode='constant', fill_value=0)
    scaling_layer = RandomZoom(height_factor=(-0.1, 0.1), width_factor=(-0.1, 0.1), fill_mode='constant', fill_value=0)
    translation_layer = RandomTranslation(height_factor=0.1, width_factor=0.1, fill_mode='constant', fill_value=0)

    # Function to resize the image to 67x50
    def resize_image(self, image):
        """Resize the image to the specified size."""
        target_size = (67, 50)
        return cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)

    # Function to convert the image to grayscale
    def convert_to_grayscale(self, image):
        """Convert the image to grayscale."""
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Function to apply random rotation with background fill
    def apply_rotation(self, image):
        """Apply random rotation to grayscale images using Keras RandomRotation."""
        background_color = self.detect_background_color(image)
        img_norm = tf.convert_to_tensor(image, dtype=tf.float32) / 255.0  # Normalize to [0, 1]
        rotated = self.rotation_layer(tf.expand_dims(tf.expand_dims(img_norm, -1), 0))  # Add channel and batch dimensions
        rotated = tf.squeeze(rotated).numpy() * 255  # Remove dimensions and scale back to [0, 255]
        rotated = rotated.astype(np.uint8)
        mask = rotated > 0  # Binary mask for non-zero pixels
        return self.fill_gaps_with_background(rotated, mask, background_color)

    # Function to apply scaling augmentation
    def apply_scaling(self, image):
        """Apply random scaling to grayscale images using Keras RandomZoom."""
        background_color = self.detect_background_color(image)
        img_norm = tf.convert_to_tensor(image, dtype=tf.float32) / 255.0  # Normalize to [0, 1]
        scaled = self.scaling_layer(
            tf.expand_dims(tf.expand_dims(img_norm, -1), 0))  # Add channel and batch dimensions
        scaled = tf.squeeze(scaled).numpy() * 255  # Remove dimensions and scale back to [0, 255]
        scaled = scaled.astype(np.uint8)
        mask = scaled > 0  # Binary mask for non-zero pixels
        return self.fill_gaps_with_background(scaled, mask, background_color)

    # Function to apply translation augmentation
    def apply_translation(self, image):
        """Apply random translation to grayscale images using Keras RandomTranslation."""
        background_color = self.detect_background_color(image)
        img_norm = tf.convert_to_tensor(image, dtype=tf.float32) / 255.0  # Normalize to [0, 1]
        translated = self.translation_layer(
            tf.expand_dims(tf.expand_dims(img_norm, -1), 0))  # Add channel and batch dimensions
        translated = tf.squeeze(translated).numpy() * 255  # Remove dimensions and scale back to [0, 255]
        translated = translated.astype(np.uint8)
        mask = translated > 0  # Binary mask for non-zero pixels
        return self.fill_gaps_with_background(translated, mask, background_color)

    # Function to apply brightness augmentation
    def apply_brightness(self, image):
        """Apply random brightness adjustment (±15%)."""
        factor = random.uniform(0.85, 1.15)  # Brightness factor between 85% and 115%
        image = np.clip(image * factor, 0, 255).astype(np.uint8)
        return image

    # Function to apply sharpen or Gaussian blur augmentation
    def apply_focus(self, image):
        """Randomly sharpen or blur the image."""
        if random.choice([True, False]):
            # Apply Gaussian blur
            return cv2.GaussianBlur(image, (5, 5), 0)
        else:
            # Apply sharpening kernel
            kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
            return cv2.filter2D(image, -1, kernel)

    # Function to apply min-max scaling
    def min_max_scale(self, image):
        """Normalize the image using min-max scaling to range [0, 1]."""
        return image / 255.0

    # Function to detect background color (most common color in the corners)
    def detect_background_color(self, image):
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
    def fill_gaps_with_background(self, image, mask, background_color):
        """Fill the gaps in the grayscale augmented image using the background color."""
        image[mask == 0] = background_color  # Replace gaps with background color
        return image

    def process(self, source_folder, desired_augmented_images):
        """Generates multiple augmented images per input image."""

        augmented_batches = []

        for label_folder in os.listdir(source_folder):
            print(label_folder)
            label_path = os.path.join(source_folder, label_folder)
            if not os.path.isdir(label_path):
                continue

            augmented_images  = []

            for image_file in os.listdir(label_path):
                image_path = os.path.join(label_path, image_file)
                if not image_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    continue

                # Read the image
                image = cv2.imread(image_path)
                if image is None:
                    continue

                # Step 1: Resize the image
                image = self.resize_image(image)

                # Step 2: Convert to grayscale
                image = self.convert_to_grayscale(image)

                # Perform multiple random augmentations for this image
                for i in range(desired_augmented_images):
                    augmented_image = image.copy()

                    # Randomly apply rotation
                    if random.choice([True, False]):
                        augmented_image = self.apply_rotation(augmented_image)

                    # Randomly apply scaling
                    if random.choice([True, False]):
                        augmented_image = self.apply_scaling(augmented_image)

                    # Randomly apply translation
                    if random.choice([True, False]):
                        augmented_image = self.apply_translation(augmented_image)

                    # Randomly apply brightness adjustment
                    if random.choice([True, False]):
                        augmented_image = self.apply_brightness(augmented_image)

                    # Randomly apply focus adjustment (sharpen or blur)
                    if random.choice([True, False]):
                        augmented_image = self.apply_focus(augmented_image)

                    # Step 3: Apply min-max scaling
                    augmented_image = self.min_max_scale(augmented_image)
                    augmented_images.append(augmented_image)

                augmented_batches.append(augmented_images)

        return augmented_batches
