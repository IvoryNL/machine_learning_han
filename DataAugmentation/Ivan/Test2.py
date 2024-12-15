import os
import numpy as np
import tensorflow as tf

class PreprocessingPipeline:
    def create_augmentation_pipeline(self):
        """Defines the augmentation pipeline with custom rotation and background fill."""
        return tf.keras.Sequential([
            tf.keras.layers.Lambda(self.apply_rotation_with_fill),  # Custom rotation with background fill
            tf.keras.layers.RandomZoom(0.1, 0.1),  # Random zoom
            tf.keras.layers.RandomTranslation(0.1, 0.1),  # Random translation
            tf.keras.layers.Rescaling(1.0 / 255),  # Normalize to [0, 1]
            tf.keras.layers.Lambda(lambda x: tf.image.rgb_to_grayscale(x)),  # Convert to grayscale
        ])

    def detect_background_color(self, image):
        """Detect the background color based on the four corner pixels."""
        # Sample the corner pixels
        corners = [
            image[0, 0],       # Top-left corner
            image[0, -1],      # Top-right corner
            image[-1, 0],      # Bottom-left corner
            image[-1, -1],     # Bottom-right corner
        ]
        # Average the corner values to estimate the background color
        background_color = np.mean(corners, axis=0)
        return background_color.astype(np.uint8)

    def apply_rotation_with_fill(self, image):
        """Applies random rotation and fills the black gaps with the background color."""
        # Convert Tensor to NumPy for pixel manipulation
        image_np = tf.keras.utils.array_to_img(image).convert("RGB")
        image_np = np.array(image_np, dtype=np.uint8)

        # Step 1: Detect the background color
        background_color = self.detect_background_color(image_np)

        # Step 2: Rotate the image
        angle = np.random.uniform(-10, 10)  # Random rotation angle in degrees
        rotated_image = tf.keras.preprocessing.image.random_rotation(
            image_np, angle, row_axis=0, col_axis=1, channel_axis=2, fill_mode='constant', cval=0
        )

        # Step 3: Replace black gaps (0) with background color
        mask = np.all(rotated_image == [0, 0, 0], axis=-1)  # Identify black pixels
        rotated_image[mask] = background_color  # Replace with background color

        # Step 4: Convert back to Tensor
        return tf.convert_to_tensor(rotated_image, dtype=tf.float32)

    def load_dataset(self, source_folder, batch_size):
        """Loads the dataset from a folder."""
        dataset = tf.keras.utils.image_dataset_from_directory(
            source_folder,
            labels=None,  # No explicit labels since folder name is the label
            image_size=(50, 67),  # Ensure images are resized to 50x67
            batch_size=batch_size,
            color_mode="rgb"  # Load in RGB format
        )
        return dataset

    def process(self, source_folder, batch_size, desired_augmented_images):
        """Generates multiple augmented images per input image."""
        dataset = self.load_dataset(source_folder, batch_size)
        augmentation_pipeline = self.create_augmentation_pipeline()

        augmented_batches = []

        for images in dataset:
            for image in images:  # Process one image at a time
                augmented_images = []
                for _ in range(desired_augmented_images):
                    # Apply augmentation pipeline multiple times per image
                    augmented_image = augmentation_pipeline(tf.expand_dims(image, axis=0))
                    augmented_images.append(augmented_image[0])

                # Append the augmented images with labels (folder name)
                label = os.path.basename(source_folder)
                augmented_batches.append((augmented_images, [label] * desired_augmented_images))

        return augmented_batches
